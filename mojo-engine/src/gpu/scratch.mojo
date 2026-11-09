# scratch.mojo - ScratchPool: the decode step's pooled
# device scratch (Wave F, Lane F1-A item 3).
#
# The problem, counted in the Wave E record and re-counted
# here against the 1.07 tree: one decode token's resident
# layer chain allocates about 27 fresh device buffers per
# layer (norm outputs, projection outputs, rope outputs,
# attention output and partials, gate/swiglu/residual
# intermediates - every _alloc_f32 in gpu_layer.mojo plus the
# raw creates in the attention launch and the layer-boundary
# copies), about 1,404 allocations per token across the 52
# layers, each one a driver allocation and a driver free a
# step later. The kernels' math does not care: every consumer
# of these buffers WRITES EVERY ELEMENT before any read (the
# full-write audit lives on _alloc_f32 in gpu_layer.mojo),
# so a buffer's previous contents are irrelevant.
#
# The pool: free lists of DeviceBuffer[DType.float32] keyed
# by EXACT element count (the step's sizes repeat exactly -
# hidden, q width, kv width, ffn width, vocab - so exact
# classes hit ~always). acquire() pops a parked buffer of
# the requested size or creates a fresh one; release() parks
# a buffer for the next acquire of its size. Reuse is safe
# on the context's single in-order queue: a released buffer
# is only rewritten by launches enqueued AFTER every launch
# that read it, which is exactly the order the chain enqueues
# them. Buffers are NEVER zeroed here (the zero-fill launch
# per buffer was the overhead Wave E already removed once);
# a caller needing zeros uses create_zero_f32, not the pool.
#
# Bounds: at most SCRATCH_POOL_MAX_FREE_PER_CLASS buffers
# park per size class; releases past the cap destroy the
# buffer instead (its device memory returns to the driver),
# so a pathological caller cannot pin unbounded memory in
# the pool. All pooled bytes live inside GpuRuntime's
# 512 MiB scratch envelope (GPU_SCRATCH_ENVELOPE_BYTES, the
# budget plan figure these allocations were always charged
# against); the pool changes WHEN buffers are created, not
# the envelope they are accounted under.
#
# Ownership law for callers: a DeviceBuffer obtained from
# acquire_f32 is owned by the caller until it is handed back
# with release_f32 (same element count it was acquired with)
# or destroyed. Double release is a caller bug the pool
# cannot detect (the buffer would be handed out twice); the
# release points in the resident chain are written so each
# buffer has exactly one, at its last use. Buffers that are
# views of memory owned elsewhere (the KV cache's raw
# buffers, weight slots) must NEVER be released here.
#
# The policy (class lookup, the park cap, the counters) is
# pure integer bookkeeping in ScratchPoolState, VM-testable
# without a device; ScratchPool itself only adds the buffer
# lists. The split mirrors gpu/context.mojo's budget_release
# pattern. Diag seam: phase `gpu`, through src/diag.mojo.
# Named refusal: gpu_scratch_bad_size (non-positive acquire
# or a release whose size disagrees with its class is the
# caller's error path - release carries the size it was
# acquired with, checked against nothing but positivity:
# a wrong size would park the buffer in the wrong class and
# hand a later acquire a short buffer, so the resident
# chain's release sites pass the same expression they
# acquired with).

from max.gpu.host import DeviceBuffer, DeviceContext

from diag import log_debug

comptime SCRATCH_POOL_MAX_FREE_PER_CLASS = 64


# ---------------------------------------------------------------------------
# Pure pool bookkeeping (VM-testable, no device)
# ---------------------------------------------------------------------------


struct ScratchPoolState(Copyable, Movable):
    """Counters and the class table for one pool.

    sizes[i] is the element count of class i; free_counts[i]
    is how many buffers of that class are currently parked.
    created counts driver allocations, reused counts acquires
    served from a parked buffer, live counts buffers checked
    out right now, dropped counts releases destroyed at the
    park cap. The state never touches a buffer; ScratchPool
    drives it and performs the matching list operation.
    """

    var sizes: List[Int]
    var free_counts: List[Int]
    var created: Int
    var reused: Int
    var live: Int
    var dropped: Int

    def __init__(out self):
        self.sizes = List[Int]()
        self.free_counts = List[Int]()
        self.created = 0
        self.reused = 0
        self.live = 0
        self.dropped = 0

    def find_class(self, n: Int) -> Int:
        """Index of the class for exactly n elements, or -1."""
        for i in range(len(self.sizes)):
            if self.sizes[i] == n:
                return i
        return -1

    def note_acquire(mut self, n: Int) -> Bool:
        """Book one acquire of n elements. Returns True when a
        parked buffer serves it (the caller pops the class
        list), False when the caller must create a fresh
        buffer (a new class is registered as a side effect
        when n is unseen)."""
        var idx = self.find_class(n)
        if idx == -1:
            self.sizes.append(n)
            self.free_counts.append(0)
            idx = len(self.sizes) - 1
        self.live += 1
        if self.free_counts[idx] > 0:
            self.free_counts[idx] -= 1
            self.reused += 1
            return True
        self.created += 1
        return False

    def note_release(mut self, n: Int) -> Bool:
        """Book one release of an n-element buffer. Returns
        True when the buffer parks (the caller pushes it on
        the class list), False when the park cap destroys it
        instead. Releasing a size the pool never acquired
        registers the class (a foreign buffer parks like any
        other - the layer boundary's upload buffer joins the
        pool this way); the counters keep the event visible."""
        var idx = self.find_class(n)
        if idx == -1:
            self.sizes.append(n)
            self.free_counts.append(0)
            idx = len(self.sizes) - 1
        self.live -= 1
        if self.free_counts[idx] >= SCRATCH_POOL_MAX_FREE_PER_CLASS:
            self.dropped += 1
            return False
        self.free_counts[idx] += 1
        return True

    def total_free(self) -> Int:
        var total = 0
        for i in range(len(self.free_counts)):
            total += self.free_counts[i]
        return total


# ---------------------------------------------------------------------------
# The pool: buffer lists driven by the pure state
# ---------------------------------------------------------------------------


struct ScratchPool(Movable):
    """Size-classed free lists of float32 device buffers."""

    var state: ScratchPoolState
    var free: List[List[DeviceBuffer[DType.float32]]]

    def __init__(out self):
        self.state = ScratchPoolState()
        self.free = List[List[DeviceBuffer[DType.float32]]]()

    def acquire(
        mut self, mut ctx: DeviceContext, n: Int
    ) raises -> DeviceBuffer[DType.float32]:
        """A float32 device buffer of exactly n elements,
        contents UNDEFINED (see the module header's full-write
        law). Parks are popped most-recent-first; a miss
        creates a fresh buffer through the context."""
        if n <= 0:
            raise Error(
                "gpu_scratch_bad_size: acquire length must be positive, got "
                + String(n)
            )
        while len(self.free) < len(self.state.sizes):
            self.free.append(List[DeviceBuffer[DType.float32]]())
        if self.state.note_acquire(n):
            var idx = self.state.find_class(n)
            var buf = self.free[idx].pop()
            _ = log_debug(
                "gpu",
                "scratch reuse n="
                + String(n)
                + " live="
                + String(self.state.live),
            )
            return buf^
        var buf = ctx.enqueue_create_buffer[DType.float32](n)
        _ = log_debug(
            "gpu",
            "scratch create n="
            + String(n)
            + " live="
            + String(self.state.live),
        )
        return buf^

    def release(mut self, var buf: DeviceBuffer[DType.float32], n: Int) raises:
        """Park buf (acquired with n elements) for reuse, or
        destroy it when its class is at the park cap."""
        if n <= 0:
            raise Error(
                "gpu_scratch_bad_size: release length must be positive, got "
                + String(n)
            )
        while len(self.free) < len(self.state.sizes):
            self.free.append(List[DeviceBuffer[DType.float32]]())
        if self.state.note_release(n):
            var idx = self.state.find_class(n)
            self.free[idx].append(buf^)
            _ = log_debug(
                "gpu",
                "scratch park n="
                + String(n)
                + " live="
                + String(self.state.live),
            )
        else:
            _ = log_debug(
                "gpu",
                "scratch drop at park cap n="
                + String(n)
                + " live="
                + String(self.state.live),
            )
