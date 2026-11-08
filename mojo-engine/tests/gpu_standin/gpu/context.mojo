# context.mojo - STAND-IN GpuRuntime mirroring C1's surface.
#
# Same names and signatures as src/gpu/context.mojo (Lane
# C1) for everything Lane C2 uses: GpuRuntime.create(cap),
# reserve_weights / reserve_kv, upload_u8 / upload_f32 /
# download_f32 (mut rt; empty uploads refuse gpu_copy_shape),
# create_zero_f32 / create_zero_u8. Internals are simpler:
# the budget check is a plain sum against the cap.

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext, DeviceBuffer

from diag import log_debug

comptime GPU_SCRATCH_ENVELOPE_BYTES = 536870912
comptime _ZERO_BLOCK = 256


def zero_f32_kernel(buf: Pointer[Float32, MutAnyOrigin], n: Int32):
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx < Int(n):
        buf[unsafe_offset=idx] = Float32(0.0)


def zero_u8_kernel(buf: Pointer[UInt8, MutAnyOrigin], n: Int32):
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx < Int(n):
        buf[unsafe_offset=idx] = UInt8(0)


struct GpuRuntime(Movable):
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
    def create(budget_cap_bytes: Int) raises -> GpuRuntime:
        if budget_cap_bytes <= GPU_SCRATCH_ENVELOPE_BYTES:
            raise Error(
                "gpu_budget_exceeded: cap "
                + String(budget_cap_bytes)
                + " does not cover the scratch envelope"
            )
        try:
            var ctx = DeviceContext()
            return GpuRuntime(
                ctx^, budget_cap_bytes, 0, 0, GPU_SCRATCH_ENVELOPE_BYTES
            )
        except:
            raise Error(
                "gpu_no_device: DeviceContext creation found no device"
            )

    def reserve_weights(mut self, additional: Int) raises:
        if (
            self.weights_bytes + additional + self.kv_bytes
            + self.scratch_bytes
            > self.budget_cap_bytes
        ):
            raise Error(
                "gpu_budget_exceeded: weights reservation over the cap"
            )
        self.weights_bytes += additional

    def reserve_kv(mut self, additional: Int) raises:
        if (
            self.weights_bytes + self.kv_bytes + additional
            + self.scratch_bytes
            > self.budget_cap_bytes
        ):
            raise Error("gpu_budget_exceeded: kv reservation over the cap")
        self.kv_bytes += additional


def upload_u8(
    mut rt: GpuRuntime, data: List[UInt8]
) raises -> DeviceBuffer[DType.uint8]:
    if len(data) == 0:
        raise Error("gpu_copy_shape: upload_u8 of an empty list")
    var dev = rt.ctx.enqueue_create_buffer[DType.uint8](len(data))
    var host = rt.ctx.enqueue_create_host_buffer[DType.uint8](len(data))
    for i in range(len(data)):
        host[i] = data[i]
    rt.ctx.enqueue_copy(src_buf=host, dst_buf=dev)
    _ = log_debug("gpu.standin", "upload_u8 bytes=" + String(len(data)))
    return dev^


def upload_f32(
    mut rt: GpuRuntime, data: List[Float32]
) raises -> DeviceBuffer[DType.float32]:
    if len(data) == 0:
        raise Error("gpu_copy_shape: upload_f32 of an empty list")
    var dev = rt.ctx.enqueue_create_buffer[DType.float32](len(data))
    var host = rt.ctx.enqueue_create_host_buffer[DType.float32](len(data))
    for i in range(len(data)):
        host[i] = data[i]
    rt.ctx.enqueue_copy(src_buf=host, dst_buf=dev)
    _ = log_debug("gpu.standin", "upload_f32 n=" + String(len(data)))
    return dev^


def download_f32(
    mut rt: GpuRuntime, buf: DeviceBuffer[DType.float32], n: Int
) raises -> List[Float32]:
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


def create_zero_f32(
    mut rt: GpuRuntime, n: Int
) raises -> DeviceBuffer[DType.float32]:
    if n <= 0:
        raise Error("gpu_copy_shape: zero f32 length must be positive")
    var dev = rt.ctx.enqueue_create_buffer[DType.float32](n)
    var grid = (n + _ZERO_BLOCK - 1) // _ZERO_BLOCK
    rt.ctx.enqueue_function[zero_f32_kernel](
        dev, Int32(n), grid_dim=grid, block_dim=_ZERO_BLOCK
    )
    return dev^


def create_zero_u8(
    mut rt: GpuRuntime, n: Int
) raises -> DeviceBuffer[DType.uint8]:
    if n <= 0:
        raise Error("gpu_copy_shape: zero u8 length must be positive")
    var dev = rt.ctx.enqueue_create_buffer[DType.uint8](n)
    var grid = (n + _ZERO_BLOCK - 1) // _ZERO_BLOCK
    rt.ctx.enqueue_function[zero_u8_kernel](
        dev, Int32(n), grid_dim=grid, block_dim=_ZERO_BLOCK
    )
    return dev^
