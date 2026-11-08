# context.mojo - STAND-IN GpuRuntime for Lane C2's proof.
#
# Minimal stand-in for Lane C1's src/gpu/context.mojo (see
# tests/gpu_standin/gpu/__init__.mojo): owns a DeviceContext
# and provides the contract's copy helpers as free functions:
#   upload_u8(rt, List[UInt8]) -> DeviceBuffer[DType.uint8]
#   upload_f32(rt, List[Float32]) -> DeviceBuffer[DType.float32]
#   download_f32(rt, DeviceBuffer[DType.float32], n) -> List[Float32]
# Budget accounting is C1's real job; this stand-in records
# the cap and resident counters but enforces nothing beyond
# passing them through (the proof prints them as stand-in
# values, never as C1 verdicts).
#
# Named refusals (contract names): gpu_no_device,
# gpu_copy_shape.

from max.gpu.host import DeviceContext, DeviceBuffer

from diag import log_debug


struct GpuRuntime(Movable):
    var ctx: DeviceContext
    var budget_cap_bytes: Int
    var resident_bytes: Int

    def __init__(
        out self,
        var ctx: DeviceContext,
        budget_cap_bytes: Int,
        resident_bytes: Int,
    ):
        self.ctx = ctx^
        self.budget_cap_bytes = budget_cap_bytes
        self.resident_bytes = resident_bytes

    @staticmethod
    def create(budget_cap_bytes: Int) raises -> GpuRuntime:
        """Stand-in construction: a bare DeviceContext. On a
        GPU-less host DeviceContext creation fails; that is
        surfaced as the contract refusal gpu_no_device."""
        try:
            var ctx = DeviceContext()
            return GpuRuntime(ctx^, budget_cap_bytes, 0)
        except:
            raise Error(
                "gpu_no_device: DeviceContext creation found no device"
            )


def upload_u8(
    rt: GpuRuntime, data: List[UInt8]
) raises -> DeviceBuffer[DType.uint8]:
    """Host byte list -> fresh device buffer (stand-in)."""
    var dev = rt.ctx.enqueue_create_buffer[DType.uint8](len(data))
    if len(data) > 0:
        var host = rt.ctx.enqueue_create_host_buffer[DType.uint8](len(data))
        for i in range(len(data)):
            host[i] = data[i]
        rt.ctx.enqueue_copy(src_buf=host, dst_buf=dev)
    _ = log_debug("gpu.standin", "upload_u8 bytes=" + String(len(data)))
    return dev^


def upload_f32(
    rt: GpuRuntime, data: List[Float32]
) raises -> DeviceBuffer[DType.float32]:
    """Host float list -> fresh device buffer (stand-in)."""
    var dev = rt.ctx.enqueue_create_buffer[DType.float32](len(data))
    if len(data) > 0:
        var host = rt.ctx.enqueue_create_host_buffer[DType.float32](len(data))
        for i in range(len(data)):
            host[i] = data[i]
        rt.ctx.enqueue_copy(src_buf=host, dst_buf=dev)
    _ = log_debug("gpu.standin", "upload_f32 n=" + String(len(data)))
    return dev^


def download_f32(
    rt: GpuRuntime, buf: DeviceBuffer[DType.float32], n: Int
) raises -> List[Float32]:
    """Device buffer -> host float list of length n (stand-in).
    Synchronizes before reading, like every proof driver in
    this tree does by hand."""
    if n < 0:
        raise Error(
            "gpu_copy_shape: download length must be >= 0, got " + String(n)
        )
    var out = List[Float32](capacity=n)
    out.resize(n, Float32(0.0))
    if n > 0:
        var host = rt.ctx.enqueue_create_host_buffer[DType.float32](n)
        rt.ctx.enqueue_copy(src_buf=buf, dst_buf=host)
        rt.ctx.synchronize()
        for i in range(n):
            out[i] = host[i]
    return out^
