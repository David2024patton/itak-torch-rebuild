# device_kv.mojo - STAND-IN DeviceKvCache for Lane C2's proof.
#
# Minimal stand-in for Lane C1's src/gpu/device_kv.mojo (see
# tests/gpu_standin/gpu/__init__.mojo). Layout mirrors B1's
# KvCache byte for byte: per layer, K and V float32 device
# buffers of capacity_positions x kv_heads x head_dim, where
# capacity is the layer's window (2,048) for windowed layers
# and the configured context length for global layers; a
# position's row is [kv_heads][head_dim] at slot = position
# (pre-wrap; windowed appends that would wrap refuse, the
# Wave C limit in the contract).
#
# Append takes host K/V rows (already rotated for sliding
# layers - rotation is the model layer's job), stages them,
# and scatters them into the slot with a small copy kernel:
# whole-buffer copies are never used. Read path for
# attention: key_buffer(layer) / value_buffer(layer) expose
# the raw buffers; stored_len(layer) and
# first_visible_position(layer) carry B1's meanings.
#
# Named refusals: gpu_kv_invalid_layer, gpu_kv_out_of_order,
# gpu_kv_bad_shape, gpu_kv_out_of_range,
# gpu_kv_window_wrapped (the contract's Wave C limit name).

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceBuffer, DeviceContext

from diag import log_debug
from gpu.context import GpuRuntime, upload_f32
from kv.cache import LayerKind


def kv_scatter_kernel(
    src: Pointer[Float32, MutAnyOrigin],
    dst: Pointer[Float32, MutAnyOrigin],
    offset: Int32,
    n: Int32,
):
    """Copy one staged row into a KV buffer at a slot offset:
    one thread per element."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        dst[unsafe_offset=Int(offset) + i] = src[unsafe_offset=i]


struct DeviceKvCache(Movable):
    var ctx: DeviceContext
    var key_bufs: List[DeviceBuffer[DType.float32]]
    var val_bufs: List[DeviceBuffer[DType.float32]]
    var capacities: List[Int]
    var windows: List[Int]
    var appended: List[Int]
    var num_kv_heads: Int
    var head_dim: Int
    var context_len: Int

    def __init__(
        out self,
        var ctx: DeviceContext,
        var key_bufs: List[DeviceBuffer[DType.float32]],
        var val_bufs: List[DeviceBuffer[DType.float32]],
        var capacities: List[Int],
        var windows: List[Int],
        var appended: List[Int],
        num_kv_heads: Int,
        head_dim: Int,
        context_len: Int,
    ):
        self.ctx = ctx^
        self.key_bufs = key_bufs^
        self.val_bufs = val_bufs^
        self.capacities = capacities^
        self.windows = windows^
        self.appended = appended^
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.context_len = context_len

    @staticmethod
    def create(
        rt: GpuRuntime,
        kinds: List[LayerKind],
        num_kv_heads: Int,
        head_dim: Int,
        context_len: Int,
    ) raises -> DeviceKvCache:
        """Stand-in construction: one K and one V buffer per
        layer at B1 capacity (window or context length)."""
        if len(kinds) == 0 or num_kv_heads <= 0 or head_dim <= 0:
            raise Error(
                "gpu_kv_bad_shape: DeviceKvCache needs a non-empty"
                " pattern and positive kv geometry"
            )
        if context_len <= 0:
            raise Error(
                "gpu_kv_out_of_range: context length must be"
                " positive, got "
                + String(context_len)
            )
        var row = num_kv_heads * head_dim
        var key_bufs = List[DeviceBuffer[DType.float32]]()
        var val_bufs = List[DeviceBuffer[DType.float32]]()
        var capacities = List[Int]()
        var windows = List[Int]()
        var appended = List[Int]()
        for i in range(len(kinds)):
            var cap = context_len if kinds[i].is_global else kinds[i].window
            if cap <= 0:
                raise Error(
                    "gpu_kv_bad_shape: layer "
                    + String(i)
                    + " has non-positive capacity "
                    + String(cap)
                )
            key_bufs.append(
                rt.ctx.enqueue_create_buffer[DType.float32](cap * row)
            )
            val_bufs.append(
                rt.ctx.enqueue_create_buffer[DType.float32](cap * row)
            )
            capacities.append(cap)
            windows.append(0 if kinds[i].is_global else kinds[i].window)
            appended.append(0)
        _ = log_debug(
            "gpu.standin",
            "DeviceKvCache stand-in layers=" + String(len(kinds)),
        )
        return DeviceKvCache(
            rt.ctx.copy(),
            key_bufs^,
            val_bufs^,
            capacities^,
            windows^,
            appended^,
            num_kv_heads,
            head_dim,
            context_len,
        )

    def row_len(self) -> Int:
        return self.num_kv_heads * self.head_dim

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

    def key_buffer(
        self, layer: Int
    ) raises -> DeviceBuffer[DType.float32]:
        self._check_layer(layer)
        return self.key_bufs[layer].copy()

    def value_buffer(
        self, layer: Int
    ) raises -> DeviceBuffer[DType.float32]:
        self._check_layer(layer)
        return self.val_bufs[layer].copy()

    def append(
        mut self,
        layer: Int,
        k_row: List[Float32],
        v_row: List[Float32],
        position: Int,
    ) raises:
        """Scatter one position's K/V rows into their slot.
        The stand-in holds its own copy of the device context
        (DeviceContext copies share one context), so this
        signature matches the contract's append exactly."""
        self._check_layer(layer)
        var row = self.row_len()
        if len(k_row) != row or len(v_row) != row:
            raise Error(
                "gpu_kv_bad_shape: append rows must be "
                + String(row)
                + " floats, got K "
                + String(len(k_row))
                + " V "
                + String(len(v_row))
            )
        if position != self.appended[layer]:
            raise Error(
                "gpu_kv_out_of_order: layer "
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
                + "; the CPU path is the fallback beyond the window"
            )
        if position >= self.capacities[layer]:
            raise Error(
                "gpu_kv_out_of_range: layer "
                + String(layer)
                + " position "
                + String(position)
                + " beyond capacity "
                + String(self.capacities[layer])
            )
        var slot = position
        var rt = GpuRuntime(self.ctx.copy(), 0, 0)
        var k_stage = upload_f32(rt, k_row)
        var v_stage = upload_f32(rt, v_row)
        var grid = (row + 255) // 256
        self.ctx.enqueue_function[kv_scatter_kernel](
            k_stage,
            self.key_bufs[layer],
            Int32(slot * row),
            Int32(row),
            grid_dim=grid,
            block_dim=256,
        )
        self.ctx.enqueue_function[kv_scatter_kernel](
            v_stage,
            self.val_bufs[layer],
            Int32(slot * row),
            Int32(row),
            grid_dim=grid,
            block_dim=256,
        )
        self.ctx.synchronize()
        self.appended[layer] = position + 1
