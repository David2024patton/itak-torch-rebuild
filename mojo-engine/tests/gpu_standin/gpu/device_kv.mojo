# device_kv.mojo - STAND-IN DeviceKvCache mirroring C1's
# surface (see tests/gpu_standin/gpu/__init__.mojo).
#
# Same names and signatures as src/gpu/device_kv.mojo (Lane
# C1) for everything Lane C2 uses: DeviceKvCache.build(mut
# rt, pattern, num_kv_heads, head_dim, context_len),
# append(mut rt, layer, k_row, v_row, position) with C1's
# refusal names, stored_len / first_visible_position /
# key_buffer / value_buffer. Layout is B1's: per layer K and
# V float32 buffers of capacity (window or context) x
# kv_heads x head_dim; append stages one row and scatters it
# into its slot with a small copy kernel.

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceBuffer

from diag import log_debug
from gpu.context import GpuRuntime, create_zero_f32, upload_f32
from kv.cache import LayerKind


def kv_scatter_kernel(
    src: Pointer[Float32, MutAnyOrigin],
    dst: Pointer[Float32, MutAnyOrigin],
    offset: Int32,
    n: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        dst[unsafe_offset=Int(offset) + i] = src[unsafe_offset=i]


struct DeviceKvCache(Movable):
    var key_bufs: List[DeviceBuffer[DType.float32]]
    var val_bufs: List[DeviceBuffer[DType.float32]]
    var capacities: List[Int]
    var windows: List[Int]
    var appended: List[Int]
    var num_kv_heads: Int
    var head_dim: Int
    var context_len: Int
    var total_bytes: Int

    def __init__(
        out self,
        var key_bufs: List[DeviceBuffer[DType.float32]],
        var val_bufs: List[DeviceBuffer[DType.float32]],
        var capacities: List[Int],
        var windows: List[Int],
        var appended: List[Int],
        num_kv_heads: Int,
        head_dim: Int,
        context_len: Int,
        total_bytes: Int,
    ):
        self.key_bufs = key_bufs^
        self.val_bufs = val_bufs^
        self.capacities = capacities^
        self.windows = windows^
        self.appended = appended^
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.context_len = context_len
        self.total_bytes = total_bytes

    @staticmethod
    def build(
        mut rt: GpuRuntime,
        pattern: List[LayerKind],
        num_kv_heads: Int,
        head_dim: Int,
        context_len: Int,
    ) raises -> DeviceKvCache:
        if len(pattern) == 0 or num_kv_heads <= 0 or head_dim <= 0:
            raise Error(
                "gpu_kv_zero_dimension: DeviceKvCache needs a non-empty"
                " pattern and positive kv geometry"
            )
        if context_len <= 0:
            raise Error(
                "gpu_kv_zero_dimension: context_len must be positive, got "
                + String(context_len)
            )
        var row = num_kv_heads * head_dim
        var total = 0
        for i in range(len(pattern)):
            var cap = context_len if pattern[i].is_global else pattern[i].window
            total += cap * row * 4 * 2
        rt.reserve_kv(total)
        var key_bufs = List[DeviceBuffer[DType.float32]]()
        var val_bufs = List[DeviceBuffer[DType.float32]]()
        var capacities = List[Int]()
        var windows = List[Int]()
        var appended = List[Int]()
        for i in range(len(pattern)):
            var cap = context_len if pattern[i].is_global else pattern[i].window
            if cap <= 0:
                raise Error(
                    "gpu_kv_invalid_window: layer "
                    + String(i)
                    + " has non-positive capacity "
                    + String(cap)
                )
            key_bufs.append(create_zero_f32(rt, cap * row))
            val_bufs.append(create_zero_f32(rt, cap * row))
            capacities.append(cap)
            windows.append(0 if pattern[i].is_global else pattern[i].window)
            appended.append(0)
        _ = log_debug(
            "gpu.standin",
            "DeviceKvCache stand-in layers=" + String(len(pattern)),
        )
        return DeviceKvCache(
            key_bufs^, val_bufs^, capacities^, windows^, appended^,
            num_kv_heads, head_dim, context_len, total,
        )

    def layer_count(self) -> Int:
        return len(self.key_bufs)

    def row_len(self) -> Int:
        return self.num_kv_heads * self.head_dim

    def resident_bytes(self) -> Int:
        return self.total_bytes

    def _check_layer(self, layer: Int) raises:
        if layer < 0 or layer >= len(self.key_bufs):
            raise Error(
                "gpu_kv_invalid_layer: layer "
                + String(layer)
                + " out of range, cache has "
                + String(len(self.key_bufs))
                + " layers"
            )

    def stored_len(self, layer: Int) raises -> Int:
        self._check_layer(layer)
        var a = self.appended[layer]
        var cap = self.capacities[layer]
        return a if a < cap else cap

    def first_visible_position(self, layer: Int) raises -> Int:
        self._check_layer(layer)
        return self.appended[layer] - self.stored_len(layer)

    def key_buffer(self, layer: Int) raises -> DeviceBuffer[DType.float32]:
        self._check_layer(layer)
        return self.key_bufs[layer].copy()

    def value_buffer(self, layer: Int) raises -> DeviceBuffer[DType.float32]:
        self._check_layer(layer)
        return self.val_bufs[layer].copy()

    def append(
        mut self,
        mut rt: GpuRuntime,
        layer: Int,
        k_row: List[Float32],
        v_row: List[Float32],
        position: Int,
    ) raises:
        self._check_layer(layer)
        var row = self.row_len()
        if len(k_row) != row or len(v_row) != row:
            raise Error(
                "gpu_kv_shape_mismatch: k/v rows have "
                + String(len(k_row))
                + "/"
                + String(len(v_row))
                + " elements, expected "
                + String(row)
            )
        if position != self.appended[layer]:
            raise Error(
                "gpu_kv_out_of_order_position: layer "
                + String(layer)
                + " expects position "
                + String(self.appended[layer])
                + ", got "
                + String(position)
            )
        if self.windows[layer] > 0 and position >= self.windows[layer]:
            raise Error(
                "gpu_kv_window_wrapped: layer "
                + String(layer)
                + " position "
                + String(position)
                + " would wrap its window "
                + String(self.windows[layer])
            )
        if position >= self.capacities[layer]:
            raise Error(
                "gpu_kv_capacity_exceeded: layer "
                + String(layer)
                + " position "
                + String(position)
                + " beyond capacity "
                + String(self.capacities[layer])
            )
        var k_stage = upload_f32(rt, k_row)
        var v_stage = upload_f32(rt, v_row)
        var grid = (row + 255) // 256
        rt.ctx.enqueue_function[kv_scatter_kernel](
            k_stage, self.key_bufs[layer], Int32(position * row), Int32(row),
            grid_dim=grid, block_dim=256,
        )
        rt.ctx.enqueue_function[kv_scatter_kernel](
            v_stage, self.val_bufs[layer], Int32(position * row), Int32(row),
            grid_dim=grid, block_dim=256,
        )
        rt.ctx.synchronize()
        self.appended[layer] = position + 1
