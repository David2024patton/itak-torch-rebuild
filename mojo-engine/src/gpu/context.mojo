# context.mojo - GpuRuntime: MAX DeviceContext ownership,
# budget accounting, and the copy helpers every GPU lane uses
# (Wave C lane C1; the seam contract is WAVE_C_INTERFACES.md).
#
# Budget law: the resident total is weights bytes + KV bytes +
# the scratch envelope, checked against the cap
# (MOJO_ENGINE_GPU_BUDGET_BYTES in src/config.mojo, default
# 24,696,061,952 B = the RX 7900 XTX's 25,769,803,776 B total
# minus a 1 GiB reserve) BEFORE the allocation it guards:
# reserve_*() refuses by name first, and only then does the
# caller create buffers. The scratch envelope is accounted
# from birth (it is a plan figure, 512 MiB, never allocated as
# one buffer; the largest single activation is the 808,192 B
# logits row).
#
# Release law (Wave E, Lane E3; owed since Wave C): a
# reservation is returned with unreserve_*() when the
# structure it guarded is torn down - DeviceWeights and
# DeviceKvCache expose release_reservation() for exactly
# that, and GpuGlimmerModel.release_reservations() releases
# both. Before this API existed the ledger only ever grew in
# a long-lived runtime, so a second build in the same process
# double-counted and refused gpu_budget_exceeded on a card
# that had the room (the Wave C layer proof reset the ledger
# field directly to work around it; that hack is gone).
#
# Named refusals: gpu_no_device (DeviceContext creation finds
# no usable device), gpu_budget_exceeded (a reservation would
# push the resident total over the cap, or a size overflows),
# gpu_copy_shape (empty / negative / length-mismatched copies).
#
# Wave F (Lane F1-A, item 4): the runtime also owns the two
# SHARED RoPE position buffers the layer steps used to rebuild
# per call - rope_positions (one absolute position repeated,
# the decode step's shape) and rope_positions_span (a run of
# consecutive positions, the prefill batch's shape). One host
# buffer, one device buffer, and one copy per POSITION CHANGE
# now serve every rope launch in a step; see the methods.
#
# Wave F (Lane F1-A, item 3): the runtime also owns the
# step's SCRATCH POOL (gpu/scratch.mojo): acquire_f32 /
# release_f32 hand out and park the resident chain's
# intermediate buffers, so a decode step creates its working
# set once and reuses it across all 52 layers and every
# later token instead of allocating per op.
#
# Diag seam: phase `gpu`, through src/diag.mojo as everywhere.

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from diag import log_debug
from gpu.scratch import ScratchPool

comptime GPU_SCRATCH_ENVELOPE_BYTES = 536870912
comptime _ZERO_BLOCK = 256


# ---------------------------------------------------------------------------
# Pure budget arithmetic (VM-testable, no device needed)
# ---------------------------------------------------------------------------


def gpu_budget_total(weights: Int, kv: Int, scratch: Int) raises -> Int:
    """weights + kv + scratch with overflow refused by name."""
    if weights < 0 or kv < 0 or scratch < 0:
        raise Error(
            "gpu_budget_exceeded: budget components must not be negative,"
            " got weights="
            + String(weights)
            + " kv="
            + String(kv)
            + " scratch="
            + String(scratch)
        )
    var total = weights + kv
    if total < weights or total + scratch < total:
        raise Error("gpu_budget_exceeded: resident total overflows")
    return total + scratch


def check_gpu_budget(weights: Int, kv: Int, scratch: Int, cap: Int) raises:
    """Refuse by name when the resident total exceeds the cap."""
    if cap <= 0:
        raise Error(
            "gpu_budget_exceeded: budget cap must be positive, got "
            + String(cap)
        )
    var total = gpu_budget_total(weights, kv, scratch)
    if total > cap:
        raise Error(
            "gpu_budget_exceeded: resident total "
            + String(total)
            + " B (weights "
            + String(weights)
            + " + kv "
            + String(kv)
            + " + scratch "
            + String(scratch)
            + ") exceeds the cap "
            + String(cap)
            + " B"
        )


def budget_release(current: Int, amount: Int, what: String) raises -> Int:
    """current - amount for one ledger bucket, refused by name.

    Pure, so the release arithmetic is VM-testable without a
    device; GpuRuntime.unreserve_* delegates here. A negative
    amount, or releasing more than the bucket currently holds
    (a double release, or releasing a reservation that was
    never made), refuses gpu_budget_exceeded instead of
    silently driving the ledger negative - a negative bucket
    would let a later reserve pass against room that does not
    exist.
    """
    if amount < 0:
        raise Error(
            "gpu_budget_exceeded: unreserve amount for "
            + what
            + " must not be negative, got "
            + String(amount)
        )
    if amount > current:
        raise Error(
            "gpu_budget_exceeded: unreserve of "
            + String(amount)
            + " B for "
            + what
            + " exceeds its reserved "
            + String(current)
            + " B"
        )
    return current - amount


# ---------------------------------------------------------------------------
# Shared RoPE position buffers: pure policy (VM-testable, no
# device). The runtime methods below delegate every decision
# to these three functions; the scenario the policy encodes is
# in test_gpu_plumbing.mojo.
# ---------------------------------------------------------------------------


def rope_pos_needs_grow(cached_capacity: Int, vectors: Int) -> Bool:
    """True when a cached position buffer of cached_capacity
    entries cannot serve a launch of `vectors` positions."""
    return vectors > cached_capacity


def rope_pos_grown_capacity(cached_capacity: Int, vectors: Int) -> Int:
    """Capacity after a grow: at least `vectors`, at least
    double the old capacity. Capacities here are head counts
    (tens of entries), so doubling stays trivially small and
    amortizes regrowth to a handful of events per process."""
    var doubled = cached_capacity * 2
    return vectors if vectors > doubled else doubled


def rope_pos_needs_fill(
    cached_position: Int, position: Int, grew: Bool
) -> Bool:
    """True when the single-position buffer must be (re)filled:
    after any grow (fresh buffers hold no values) or when the
    cached position differs from the requested one.
    cached_position is -1 before the first fill."""
    return grew or cached_position != position


def rope_span_needs_fill(
    cached_start: Int,
    cached_vectors: Int,
    cached_per: Int,
    start: Int,
    vectors: Int,
    per: Int,
    grew: Bool,
) -> Bool:
    """The span buffer's fill rule: refill after a grow or when
    the (start, vectors, vectors-per-position) key changes.
    Inside one prefill, every layer's two rope launches share
    one key, so the span is filled once per prefill."""
    return (
        grew
        or cached_start != start
        or cached_vectors != vectors
        or cached_per != per
    )


# ---------------------------------------------------------------------------
# Zero-fill kernels (one exact grid, one element per thread)
# ---------------------------------------------------------------------------


def zero_f32_kernel(buf: Pointer[Float32, MutAnyOrigin], n: Int32):
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx < Int(n):
        buf[unsafe_offset=idx] = Float32(0.0)


def zero_u8_kernel(buf: Pointer[UInt8, MutAnyOrigin], n: Int32):
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx < Int(n):
        buf[unsafe_offset=idx] = UInt8(0)


# ---------------------------------------------------------------------------
# The runtime
# ---------------------------------------------------------------------------


struct GpuRuntime(Movable):
    """Owns the MAX DeviceContext and the budget ledger.

    weights_bytes / kv_bytes are RESERVED figures accumulated
    through reserve_weights / reserve_kv (each checks the new
    total against the cap before applying it) and returned
    through unreserve_weights / unreserve_kv on teardown;
    scratch_bytes starts at the full envelope. Callers reserve
    the exact computed size of a structure BEFORE allocating
    any of its buffers, so an over-cap build fails before the
    card fills.
    """

    var ctx: DeviceContext
    var budget_cap_bytes: Int
    var weights_bytes: Int
    var kv_bytes: Int
    var scratch_bytes: Int
    # Shared RoPE position buffers (Wave F, item 4): the host
    # and device buffers sit in one-element lists (empty = not
    # built yet) because the struct is constructed before any
    # rope launch sizes are known; capacities grow-only.
    # rope_pos_* caches one repeated absolute position (the
    # decode step); rope_pos_position is -1 before the first
    # fill. rope_span_* caches a run of consecutive positions
    # keyed by (start, vectors, vectors-per-position).
    var rope_pos_host: List[HostBuffer[DType.int32]]
    var rope_pos_dev: List[DeviceBuffer[DType.int32]]
    var rope_pos_capacity: Int
    var rope_pos_position: Int
    var rope_span_host: List[HostBuffer[DType.int32]]
    var rope_span_dev: List[DeviceBuffer[DType.int32]]
    var rope_span_capacity: Int
    var rope_span_start: Int
    var rope_span_vectors: Int
    var rope_span_per: Int
    # The decode step's scratch pool (Wave F, item 3); LAST
    # field and LAST __init__ parameter, the positional law.
    var scratch: ScratchPool

    def __init__(
        out self,
        var ctx: DeviceContext,
        budget_cap_bytes: Int,
        weights_bytes: Int,
        kv_bytes: Int,
        scratch_bytes: Int,
        var rope_pos_host: List[HostBuffer[DType.int32]],
        var rope_pos_dev: List[DeviceBuffer[DType.int32]],
        rope_pos_capacity: Int,
        rope_pos_position: Int,
        var rope_span_host: List[HostBuffer[DType.int32]],
        var rope_span_dev: List[DeviceBuffer[DType.int32]],
        rope_span_capacity: Int,
        rope_span_start: Int,
        rope_span_vectors: Int,
        rope_span_per: Int,
        var scratch: ScratchPool,
    ):
        self.ctx = ctx^
        self.budget_cap_bytes = budget_cap_bytes
        self.weights_bytes = weights_bytes
        self.kv_bytes = kv_bytes
        self.scratch_bytes = scratch_bytes
        self.rope_pos_host = rope_pos_host^
        self.rope_pos_dev = rope_pos_dev^
        self.rope_pos_capacity = rope_pos_capacity
        self.rope_pos_position = rope_pos_position
        self.rope_span_host = rope_span_host^
        self.rope_span_dev = rope_span_dev^
        self.rope_span_capacity = rope_span_capacity
        self.rope_span_start = rope_span_start
        self.rope_span_vectors = rope_span_vectors
        self.rope_span_per = rope_span_per
        self.scratch = scratch^

    @staticmethod
    def _make_context() raises -> DeviceContext:
        try:
            return DeviceContext()
        except e:
            raise Error(
                "gpu_no_device: MAX DeviceContext creation failed: "
                + String(e)
            )

    @staticmethod
    def create(budget_cap_bytes: Int) raises -> GpuRuntime:
        """Create a runtime on the default device under a cap.

        The cap itself is validated through the budget check
        with the scratch envelope alone, so a non-positive or
        sub-envelope cap refuses by name before any device
        buffer exists.
        """
        check_gpu_budget(0, 0, GPU_SCRATCH_ENVELOPE_BYTES, budget_cap_bytes)
        var ctx = Self._make_context()
        _ = log_debug(
            "gpu",
            "runtime created cap="
            + String(budget_cap_bytes)
            + " scratch_envelope="
            + String(GPU_SCRATCH_ENVELOPE_BYTES),
        )
        return GpuRuntime(
            ctx^,
            budget_cap_bytes,
            0,
            0,
            GPU_SCRATCH_ENVELOPE_BYTES,
            List[HostBuffer[DType.int32]](),
            List[DeviceBuffer[DType.int32]](),
            0,
            -1,
            List[HostBuffer[DType.int32]](),
            List[DeviceBuffer[DType.int32]](),
            0,
            -1,
            0,
            0,
            ScratchPool(),
        )

    def total_reserved(self) raises -> Int:
        return gpu_budget_total(
            self.weights_bytes, self.kv_bytes, self.scratch_bytes
        )

    def reserve_weights(mut self, additional: Int) raises:
        """Check-then-apply a weights reservation."""
        check_gpu_budget(
            self.weights_bytes + additional,
            self.kv_bytes,
            self.scratch_bytes,
            self.budget_cap_bytes,
        )
        self.weights_bytes += additional
        _ = log_debug(
            "gpu",
            "reserve weights +"
            + String(additional)
            + " total_weights="
            + String(self.weights_bytes),
        )

    def reserve_kv(mut self, additional: Int) raises:
        """Check-then-apply a KV reservation."""
        check_gpu_budget(
            self.weights_bytes,
            self.kv_bytes + additional,
            self.scratch_bytes,
            self.budget_cap_bytes,
        )
        self.kv_bytes += additional
        _ = log_debug(
            "gpu",
            "reserve kv +"
            + String(additional)
            + " total_kv="
            + String(self.kv_bytes),
        )

    def unreserve_weights(mut self, amount: Int) raises:
        """Return a weights reservation (teardown path).

        The mirror of reserve_weights: callers release the
        exact figure they reserved when the structure it
        guarded is destroyed, so a later build in the same
        runtime reserves against the true free room.
        """
        self.weights_bytes = budget_release(
            self.weights_bytes, amount, "weights"
        )
        _ = log_debug(
            "gpu",
            "unreserve weights -"
            + String(amount)
            + " total_weights="
            + String(self.weights_bytes),
        )

    def unreserve_kv(mut self, amount: Int) raises:
        """Return a KV reservation (teardown path); see
        unreserve_weights."""
        self.kv_bytes = budget_release(self.kv_bytes, amount, "kv")
        _ = log_debug(
            "gpu",
            "unreserve kv -"
            + String(amount)
            + " total_kv="
            + String(self.kv_bytes),
        )

    def rope_positions(
        mut self, position: Int, vectors: Int
    ) raises -> DeviceBuffer[DType.int32]:
        """The shared single-position RoPE buffer: a device
        int32 buffer whose first `vectors` entries all equal
        `position`, ready for rope_kernel. Built on first use,
        grown (never shrunk) when a wider launch arrives, and
        refilled ONLY when the position changes: inside one
        decode step all 52 layers' K and Q ropes read the same
        buffer with no refill and no copy (Wave F, Lane F1-A
        item 4; the per-call construction this replaces -
        one host buffer, one device buffer, one fill, one
        copy per rope call - ran 2 x 52 times per token).
        The returned buffer is a view of the cached one;
        callers must not hold it across a call that refills
        it with a different position (none does: a step's
        ropes all share the step's one position)."""
        if vectors <= 0:
            raise Error(
                "gpu_copy_shape: rope positions vector count must be"
                " positive, got "
                + String(vectors)
            )
        if position < 0:
            raise Error(
                "gpu_copy_shape: rope position must not be negative, got "
                + String(position)
            )
        var grew = rope_pos_needs_grow(self.rope_pos_capacity, vectors)
        if grew:
            var capacity = rope_pos_grown_capacity(
                self.rope_pos_capacity, vectors
            )
            self.rope_pos_host = List[HostBuffer[DType.int32]]()
            self.rope_pos_host.append(
                self.ctx.enqueue_create_host_buffer[DType.int32](capacity)
            )
            self.rope_pos_dev = List[DeviceBuffer[DType.int32]]()
            self.rope_pos_dev.append(
                self.ctx.enqueue_create_buffer[DType.int32](capacity)
            )
            self.rope_pos_capacity = capacity
        if rope_pos_needs_fill(self.rope_pos_position, position, grew):
            var host = self.rope_pos_host[0].copy()
            for i in range(self.rope_pos_capacity):
                host[i] = Int32(position)
            self.ctx.enqueue_copy(
                src_buf=host, dst_buf=self.rope_pos_dev[0]
            )
            self.rope_pos_position = position
            _ = log_debug(
                "gpu",
                "rope positions filled position="
                + String(position)
                + " capacity="
                + String(self.rope_pos_capacity),
            )
        return self.rope_pos_dev[0].copy()

    def rope_positions_span(
        mut self, start_position: Int, vectors: Int, vectors_per_batch: Int
    ) raises -> DeviceBuffer[DType.int32]:
        """The shared span RoPE buffer (prefill batch shape):
        entry v equals start_position + v // vectors_per_batch,
        the per-row positions _rope_buf_batch builds. Same
        caching law as rope_positions, keyed by (start,
        vectors, vectors_per_batch): inside one prefill every
        layer's K and Q ropes share one key, so the span is
        built and filled once per prefill instead of twice
        per layer. The returned buffer is a view of the
        cached one, under the same hold-across-refill rule."""
        if vectors <= 0 or vectors_per_batch <= 0:
            raise Error(
                "gpu_copy_shape: rope span vectors / vectors-per-batch"
                " must be positive, got "
                + String(vectors)
                + " / "
                + String(vectors_per_batch)
            )
        if start_position < 0:
            raise Error(
                "gpu_copy_shape: rope span start must not be negative, got "
                + String(start_position)
            )
        var grew = rope_pos_needs_grow(self.rope_span_capacity, vectors)
        if grew:
            var capacity = rope_pos_grown_capacity(
                self.rope_span_capacity, vectors
            )
            self.rope_span_host = List[HostBuffer[DType.int32]]()
            self.rope_span_host.append(
                self.ctx.enqueue_create_host_buffer[DType.int32](capacity)
            )
            self.rope_span_dev = List[DeviceBuffer[DType.int32]]()
            self.rope_span_dev.append(
                self.ctx.enqueue_create_buffer[DType.int32](capacity)
            )
            self.rope_span_capacity = capacity
        if rope_span_needs_fill(
            self.rope_span_start,
            self.rope_span_vectors,
            self.rope_span_per,
            start_position,
            vectors,
            vectors_per_batch,
            grew,
        ):
            var host = self.rope_span_host[0].copy()
            for v in range(vectors):
                host[v] = Int32(start_position + v // vectors_per_batch)
            self.ctx.enqueue_copy(
                src_buf=host, dst_buf=self.rope_span_dev[0]
            )
            self.rope_span_start = start_position
            self.rope_span_vectors = vectors
            self.rope_span_per = vectors_per_batch
            _ = log_debug(
                "gpu",
                "rope span filled start="
                + String(start_position)
                + " vectors="
                + String(vectors)
                + " capacity="
                + String(self.rope_span_capacity),
            )
        return self.rope_span_dev[0].copy()

    def acquire_f32(
        mut self, n: Int
    ) raises -> DeviceBuffer[DType.float32]:
        """A scratch float32 buffer of n elements from the
        runtime's pool (Wave F, item 3): contents UNDEFINED,
        for full-write consumers only - the contract and the
        audit live in gpu/scratch.mojo and on _alloc_f32 in
        gpu_layer.mojo. Ownership passes to the caller until
        release_f32 (same n) or destruction."""
        return self.scratch.acquire(self.ctx, n)

    def release_f32(
        mut self, var buf: DeviceBuffer[DType.float32], n: Int
    ) raises:
        """Park a scratch buffer acquired with n elements for
        the next acquire of that size (or destroy it at the
        park cap). NEVER call this on a buffer the pool did
        not hand out unless it is a plain owned allocation of
        exactly n float32 elements with no other owner - and
        never on a view of weights or KV memory."""
        self.scratch.release(buf^, n)


# ---------------------------------------------------------------------------
# Copy helpers (the contract's surface; rt is mut because the
# MAX context enqueues work)
# ---------------------------------------------------------------------------


def upload_u8(
    mut rt: GpuRuntime, data: List[UInt8]
) raises -> DeviceBuffer[DType.uint8]:
    """Host bytes -> a new device uint8 buffer of the same length."""
    if len(data) == 0:
        raise Error("gpu_copy_shape: upload_u8 of an empty list")
    var dev = rt.ctx.enqueue_create_buffer[DType.uint8](len(data))
    var host = rt.ctx.enqueue_create_host_buffer[DType.uint8](len(data))
    for i in range(len(data)):
        host[i] = data[i]
    rt.ctx.enqueue_copy(src_buf=host, dst_buf=dev)
    _ = log_debug("gpu", "upload_u8 bytes=" + String(len(data)))
    return dev^


def upload_f32(
    mut rt: GpuRuntime, data: List[Float32]
) raises -> DeviceBuffer[DType.float32]:
    """Host values -> a new device float32 buffer of the same length."""
    if len(data) == 0:
        raise Error("gpu_copy_shape: upload_f32 of an empty list")
    var dev = rt.ctx.enqueue_create_buffer[DType.float32](len(data))
    var host = rt.ctx.enqueue_create_host_buffer[DType.float32](len(data))
    for i in range(len(data)):
        host[i] = data[i]
    rt.ctx.enqueue_copy(src_buf=host, dst_buf=dev)
    _ = log_debug("gpu", "upload_f32 elems=" + String(len(data)))
    return dev^


def download_f32(
    mut rt: GpuRuntime, buf: DeviceBuffer[DType.float32], n: Int
) raises -> List[Float32]:
    """The first n elements of a device float32 buffer, host-side."""
    if n < 0:
        raise Error(
            "gpu_copy_shape: download_f32 length must not be negative, got "
            + String(n)
        )
    var out = List[Float32](capacity=n)
    if n == 0:
        return out^
    var host = rt.ctx.enqueue_create_host_buffer[DType.float32](n)
    rt.ctx.enqueue_copy(src_buf=buf, dst_buf=host)
    rt.ctx.synchronize()
    out.resize(n, Float32(0.0))
    for i in range(n):
        out[i] = host[i]
    return out^


def download_u8(
    mut rt: GpuRuntime, buf: DeviceBuffer[DType.uint8], n: Int
) raises -> List[UInt8]:
    """The first n bytes of a device uint8 buffer, host-side.

    Additive to the contract's list: the C1 weights proof
    hashes a device buffer's bytes against the loader's, which
    needs the byte route, not a float reinterpretation.
    """
    if n < 0:
        raise Error(
            "gpu_copy_shape: download_u8 length must not be negative, got "
            + String(n)
        )
    var out = List[UInt8](capacity=n)
    if n == 0:
        return out^
    var host = rt.ctx.enqueue_create_host_buffer[DType.uint8](n)
    rt.ctx.enqueue_copy(src_buf=buf, dst_buf=host)
    rt.ctx.synchronize()
    out.resize(n, UInt8(0))
    for i in range(n):
        out[i] = host[i]
    return out^


def create_zero_f32(
    mut rt: GpuRuntime, n: Int
) raises -> DeviceBuffer[DType.float32]:
    """A zero-filled device float32 buffer of n elements."""
    if n <= 0:
        raise Error(
            "gpu_copy_shape: zero float32 buffer length must be positive,"
            " got "
            + String(n)
        )
    var dev = rt.ctx.enqueue_create_buffer[DType.float32](n)
    var grid = (n + _ZERO_BLOCK - 1) // _ZERO_BLOCK
    rt.ctx.enqueue_function[zero_f32_kernel](
        dev, Int32(n), grid_dim=grid, block_dim=_ZERO_BLOCK
    )
    _ = log_debug("gpu", "zero_f32 elems=" + String(n))
    return dev^


def create_zero_u8(
    mut rt: GpuRuntime, n: Int
) raises -> DeviceBuffer[DType.uint8]:
    """A zero-filled device uint8 buffer of n bytes."""
    if n <= 0:
        raise Error(
            "gpu_copy_shape: zero uint8 buffer length must be positive, got "
            + String(n)
        )
    var dev = rt.ctx.enqueue_create_buffer[DType.uint8](n)
    var grid = (n + _ZERO_BLOCK - 1) // _ZERO_BLOCK
    rt.ctx.enqueue_function[zero_u8_kernel](
        dev, Int32(n), grid_dim=grid, block_dim=_ZERO_BLOCK
    )
    _ = log_debug("gpu", "zero_u8 bytes=" + String(n))
    return dev^
