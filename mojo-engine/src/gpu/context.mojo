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
# Diag seam: phase `gpu`, through src/diag.mojo as everywhere.

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext, DeviceBuffer

from diag import log_debug

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

    def __init__(
        out self,
        var ctx: DeviceContext,
        budget_cap_bytes: Int,
        weights_bytes: Int,
        kv_bytes: Int,
        scratch_bytes: Int,
    ):
        self.ctx = ctx^
        self.budget_cap_bytes = budget_cap_bytes
        self.weights_bytes = weights_bytes
        self.kv_bytes = kv_bytes
        self.scratch_bytes = scratch_bytes

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
            ctx^, budget_cap_bytes, 0, 0, GPU_SCRATCH_ENVELOPE_BYTES
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
