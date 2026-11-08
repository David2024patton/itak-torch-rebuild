# context.mojo - GpuRuntime: the MAX DeviceContext, the budget
# accounting, and the host<->device copy helpers (Lane C1).
#
# =====================================================================
# PLACEHOLDER / STAND-IN (Lane C3, 2026-11-07). This file exists
# in the lane/mojo-c3-gen worktree ONLY so Lane C3's
# src/model/gpu_model.mojo compiles against the Wave C contract
# (mojo-engine/WAVE_C_INTERFACES.md) before Lane C1 lands.
# DELETE AT INTEGRATION: the coordinator replaces this whole
# file with Lane C1's real src/gpu/context.mojo. Do not ship
# this stand-in as the C1 implementation: its budget accounting
# is a plain host-side counter, its copy helpers stage through
# host buffers exactly as tests/gpu_proof_ops.mojo does, and it
# has no VM unit tests of its own (C1 owes those).
# =====================================================================
#
# Contract surface coded against (WAVE_C_INTERFACES.md, Lane C1):
#   GpuRuntime owns the MAX DeviceContext, the budget accounting
#   (weights bytes + KV bytes + scratch envelope vs cap, refusal
#   above), and the copy helpers every other lane uses:
#     upload_u8(rt, List[UInt8]) -> DeviceBuffer[DType.uint8]
#     upload_f32(rt, List[Float32]) -> DeviceBuffer[DType.float32]
#     download_f32(rt, DeviceBuffer[DType.float32], n)
#         -> List[Float32]
#   plus zero-filled device buffer constructors by dtype and
#   element count. Named refusals: gpu_no_device,
#   gpu_budget_exceeded, gpu_copy_shape. Diag lines through the
#   existing seam (`gpu` phase).
#
# Budget numbers from the contract's memory budget section:
#   card total 25,769,803,776 B (RX 7900 XTX, 24 GiB); cap =
#   card total minus a 1 GiB reserve = 24,696,061,952 B by
#   default (overridable via MOJO_ENGINE_GPU_BUDGET_BYTES, the
#   config addition Lane C1 owns in src/config.mojo; this
#   stand-in takes the cap as a constructor argument so C3
#   never reads that env var itself); scratch and activations
#   envelope 536,870,912 B (512 MiB).

from max.gpu.host import DeviceContext, DeviceBuffer

from diag import log_debug

comptime DEFAULT_GPU_BUDGET_BYTES = 24_696_061_952
comptime GPU_SCRATCH_ENVELOPE_BYTES = 536_870_912


struct GpuRuntime(Movable):
    """Owns the device context and the resident-byte counter.

    Construction refuses by name (gpu_no_device) when no device
    is present: on a GPU-less host DeviceContext creation fails
    and that failure is converted here, never swallowed.
    """

    var ctx: DeviceContext
    var budget_bytes: Int
    var resident_bytes: Int

    def __init__(out self, budget_bytes: Int) raises:
        if budget_bytes <= 0:
            raise Error(
                "gpu_budget_exceeded: budget cap must be positive, got "
                + String(budget_bytes)
            )
        try:
            self.ctx = DeviceContext()
        except e:
            raise Error("gpu_no_device: DeviceContext creation failed: " + String(e))
        self.budget_bytes = budget_bytes
        self.resident_bytes = 0
        _ = log_debug(
            "gpu",
            "runtime created budget_bytes=" + String(budget_bytes),
        )

    def account(mut self, n_bytes: Int, what: String) raises:
        """Charge n_bytes against the cap BEFORE allocating.

        Refuses by name (gpu_budget_exceeded) when the charge
        would push the resident total over the cap; the counter
        is unchanged on refusal.
        """
        if n_bytes < 0:
            raise Error(
                "gpu_copy_shape: negative byte charge for " + what
            )
        if self.resident_bytes + n_bytes > self.budget_bytes:
            raise Error(
                "gpu_budget_exceeded: "
                + what
                + " needs "
                + String(n_bytes)
                + " B, resident "
                + String(self.resident_bytes)
                + " B, cap "
                + String(self.budget_bytes)
                + " B"
            )
        self.resident_bytes += n_bytes
        _ = log_debug(
            "gpu",
            "account "
            + what
            + " +"
            + String(n_bytes)
            + " B resident="
            + String(self.resident_bytes),
        )


def upload_u8(
    rt: GpuRuntime, data: List[UInt8]
) raises -> DeviceBuffer[DType.uint8]:
    """Copy a host byte list into a fresh device buffer."""
    var dev = rt.ctx.enqueue_create_buffer[DType.uint8](len(data))
    if len(data) > 0:
        var host = rt.ctx.enqueue_create_host_buffer[DType.uint8](len(data))
        for i in range(len(data)):
            host[i] = data[i]
        rt.ctx.enqueue_copy(src_buf=host, dst_buf=dev)
        rt.ctx.synchronize()
    return dev^


def upload_f32(
    rt: GpuRuntime, data: List[Float32]
) raises -> DeviceBuffer[DType.float32]:
    """Copy a host float list into a fresh device buffer."""
    var dev = rt.ctx.enqueue_create_buffer[DType.float32](len(data))
    if len(data) > 0:
        var host = rt.ctx.enqueue_create_host_buffer[DType.float32](len(data))
        for i in range(len(data)):
            host[i] = data[i]
        rt.ctx.enqueue_copy(src_buf=host, dst_buf=dev)
        rt.ctx.synchronize()
    return dev^


def download_f32(
    rt: GpuRuntime, dev: DeviceBuffer[DType.float32], n: Int
) raises -> List[Float32]:
    """Copy n floats back from a device buffer into a host list.

    Refuses by name (gpu_copy_shape) on a negative count.
    """
    if n < 0:
        raise Error(
            "gpu_copy_shape: download count must not be negative, got "
            + String(n)
        )
    var out = List[Float32](capacity=n)
    out.resize(n, Float32(0.0))
    if n > 0:
        var host = rt.ctx.enqueue_create_host_buffer[DType.float32](n)
        rt.ctx.enqueue_copy(src_buf=dev, dst_buf=host)
        rt.ctx.synchronize()
        for i in range(n):
            out[i] = host[i]
    return out^


def create_zero_f32(
    rt: GpuRuntime, n: Int
) raises -> DeviceBuffer[DType.float32]:
    """A zero-filled float32 device buffer of n elements."""
    if n < 0:
        raise Error(
            "gpu_copy_shape: buffer element count must not be negative, got "
            + String(n)
        )
    var dev = rt.ctx.enqueue_create_buffer[DType.float32](n)
    if n > 0:
        var zeros = List[Float32](capacity=n)
        zeros.resize(n, Float32(0.0))
        var host = rt.ctx.enqueue_create_host_buffer[DType.float32](n)
        for i in range(n):
            host[i] = zeros[i]
        rt.ctx.enqueue_copy(src_buf=host, dst_buf=dev)
        rt.ctx.synchronize()
    return dev^


def create_zero_u8(
    rt: GpuRuntime, n: Int
) raises -> DeviceBuffer[DType.uint8]:
    """A zero-filled uint8 device buffer of n elements."""
    if n < 0:
        raise Error(
            "gpu_copy_shape: buffer element count must not be negative, got "
            + String(n)
        )
    var dev = rt.ctx.enqueue_create_buffer[DType.uint8](n)
    if n > 0:
        var host = rt.ctx.enqueue_create_host_buffer[DType.uint8](n)
        for i in range(n):
            host[i] = UInt8(0)
        rt.ctx.enqueue_copy(src_buf=host, dst_buf=dev)
        rt.ctx.synchronize()
    return dev^
