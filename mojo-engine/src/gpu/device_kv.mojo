# device_kv.mojo - DeviceKvCache: B1's KV cache layout on
# device (Wave C lane C1; the seam contract is
# WAVE_C_INTERFACES.md, the CPU twin is src/kv/cache.mojo).
#
# Layout, byte for byte B1: per layer, two float32 device
# buffers (K and V), each capacity_positions x kv_heads x
# head_dim, where a row is one position's [kv_heads][head_dim]
# flat row, the slot for a position is position % window for
# windowed layers and the position itself for global layers,
# and capacity_positions is the layer's window (2,048 on
# Glimmer) for windowed layers and the configured context
# length for global layers. Host-side per-layer state
# (appended count, ring slot_position for windowed layers)
# mirrors B1's LayerCache, so stored_len and
# first_visible_position carry B1's exact meanings.
#
# Append path: the caller passes one position's K and V rows
# as host lists (already rotated for sliding layers; rotation
# is the model layer's job, exactly as in the CPU step). The
# row is staged in a small device buffer and scattered into
# its slot by kv_scatter_kernel. Whole-buffer copies on the
# append path are FORBIDDEN (they would move gigabytes per
# token). Read-back for proofs goes the other way through
# kv_gather_kernel into a one-row staging buffer.
#
# Refusals mirror B1's names with the gpu_kv_ prefix:
# gpu_kv_zero_dimension / gpu_kv_invalid_window /
# gpu_kv_size_overflow / gpu_kv_invalid_layer /
# gpu_kv_shape_mismatch / gpu_kv_out_of_order_position /
# gpu_kv_position_evicted / gpu_kv_position_not_written.
#
# WAVE C LIMIT (the contract's stated deviation from B1, and
# the one place this cache does NOT mirror the CPU twin):
# B1's windowed append wraps the ring and evicts the oldest
# position. Here a windowed append whose position would wrap
# past the window refuses gpu_kv_window_wrapped, because the
# Wave C attention kernel consumes visible rows in slot order
# from offset 0, which equals position order only before a
# wrap. The CPU path remains the fallback beyond 2,048
# positions on windowed layers. A global append past the
# configured context capacity refuses gpu_kv_capacity_exceeded
# (B1's global store grows; a device buffer cannot).

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceBuffer

from diag import log_debug
from gpu.context import (
    GpuRuntime,
    create_zero_f32,
    download_f32,
    upload_f32,
)
from kv.cache import LayerKind, kv_size_accounting

comptime _KV_BLOCK = 256
comptime INT_MAX = 9223372036854775807


# ---------------------------------------------------------------------------
# Pure layout / size arithmetic (VM-testable, no device)
# ---------------------------------------------------------------------------


def kv_capacity_positions(kind: LayerKind, context_len: Int) raises -> Int:
    """Buffer capacity in positions for one layer: the window
    for windowed layers, the configured context for globals."""
    if context_len <= 0:
        raise Error(
            "gpu_kv_zero_dimension: context_len must be positive, got "
            + String(context_len)
        )
    if kind.is_global:
        return context_len
    if kind.window <= 0:
        raise Error(
            "gpu_kv_invalid_window: sliding window must be positive, got "
            + String(kind.window)
        )
    return kind.window


def kv_slot_of(kind: LayerKind, position: Int) -> Int:
    """Storage row index for an absolute position (B1's rule)."""
    if kind.is_global:
        return position
    return position % kind.window


def _checked_mul(a: Int, b: Int, what: String) raises -> Int:
    if a < 0 or b < 0:
        raise Error(
            "gpu_kv_zero_dimension: "
            + what
            + " must not be negative, got "
            + String(a)
            + " * "
            + String(b)
        )
    if a != 0 and b > INT_MAX // a:
        raise Error(
            "gpu_kv_size_overflow: "
            + what
            + " ("
            + String(a)
            + " * "
            + String(b)
            + ") overflows the addressable size"
        )
    return a * b


def kv_buffer_bytes(capacity: Int, kv_heads: Int, head_dim: Int) raises -> Int:
    """Bytes in ONE of a layer's two buffers (K or V):
    capacity x kv_heads x head_dim float32."""
    if kv_heads <= 0 or head_dim <= 0:
        raise Error(
            "gpu_kv_zero_dimension: kv_heads and head_dim must be positive,"
            " got "
            + String(kv_heads)
            + " / "
            + String(head_dim)
        )
    if capacity <= 0:
        raise Error(
            "gpu_kv_zero_dimension: capacity must be positive, got "
            + String(capacity)
        )
    var elems = _checked_mul(
        _checked_mul(capacity, kv_heads, "kv buffer elements"),
        head_dim,
        "kv buffer elements",
    )
    return _checked_mul(elems, 4, "kv buffer bytes")


def kv_total_bytes(
    pattern: List[LayerKind], kv_heads: Int, head_dim: Int, context_len: Int
) raises -> Int:
    """The full device KV byte total (K and V together),
    window-capped exactly as B1's kv_size_accounting computes
    it: (13 x context + 39 x 2048) positions x 2,048 B on
    Glimmer at context 131,072 = 3,653,238,784 B. Delegates to
    the proven B1 accounting and re-names its refusals with
    the gpu_kv_ prefix."""
    if len(pattern) == 0:
        raise Error("gpu_kv_zero_dimension: layer pattern must not be empty")
    if kv_heads <= 0 or head_dim <= 0 or context_len <= 0:
        raise Error(
            "gpu_kv_zero_dimension: kv_heads, head_dim and context_len must"
            " be positive, got "
            + String(kv_heads)
            + " / "
            + String(head_dim)
            + " / "
            + String(context_len)
        )
    for i in range(len(pattern)):
        if not pattern[i].is_global and pattern[i].window <= 0:
            raise Error(
                "gpu_kv_invalid_window: sliding window must be positive,"
                " got "
                + String(pattern[i].window)
            )
    try:
        var acct = kv_size_accounting(pattern, kv_heads, head_dim, context_len)
        return acct.total_bytes
    except e:
        var msg = String(e)
        if msg.find("kv_size_overflow") != -1:
            raise Error(
                "gpu_kv_size_overflow: KV size overflows (from B1"
                " accounting: "
                + msg
                + ")"
            )
        raise Error("gpu_kv_zero_dimension: " + msg)


def kv_append_refusal(
    kind: LayerKind,
    capacity: Int,
    appended: Int,
    position: Int,
    k_len: Int,
    v_len: Int,
    row_len: Int,
) -> String:
    """The refusal NAME an append with these facts earns, or
    "" when the append is legal. Pure, so the refusal surface
    (including the Wave C wrap limit) is VM-testable without a
    device. Check order mirrors B1: shape, then order, then
    this wave's wrap / capacity limits."""
    if k_len != row_len or v_len != row_len:
        return "gpu_kv_shape_mismatch"
    if position != appended:
        return "gpu_kv_out_of_order_position"
    if not kind.is_global and position >= kind.window:
        return "gpu_kv_window_wrapped"
    if kind.is_global and position >= capacity:
        return "gpu_kv_capacity_exceeded"
    return ""


# ---------------------------------------------------------------------------
# Scatter / gather kernels: one row between a staging buffer
# and its slot in a resident buffer. One block, threads stride
# the row; rows are at most kv_heads x head_dim = 256 floats
# on Glimmer.
# ---------------------------------------------------------------------------


def kv_scatter_kernel(
    dst: Pointer[Float32, MutAnyOrigin],
    src: Pointer[Float32, MutAnyOrigin],
    row_len: Int32,
    dst_offset: Int32,
):
    var i = Int(thread_idx.x)
    while i < Int(row_len):
        dst[unsafe_offset=Int(dst_offset) + i] = src[unsafe_offset=i]
        i += _KV_BLOCK


def kv_gather_kernel(
    src: Pointer[Float32, MutAnyOrigin],
    dst: Pointer[Float32, MutAnyOrigin],
    row_len: Int32,
    src_offset: Int32,
):
    var i = Int(thread_idx.x)
    while i < Int(row_len):
        dst[unsafe_offset=i] = src[unsafe_offset=Int(src_offset) + i]
        i += _KV_BLOCK


# ---------------------------------------------------------------------------
# One layer's device storage plus its B1 host-side state
# ---------------------------------------------------------------------------


struct DeviceKvLayer(Movable):
    var kind: LayerKind
    var capacity: Int
    var k_buf: DeviceBuffer[DType.float32]
    var v_buf: DeviceBuffer[DType.float32]
    # Windowed layers only: which absolute position each ring
    # slot currently holds, -1 when never written (B1's field).
    var slot_position: List[Int]
    var appended: Int

    def __init__(
        out self,
        kind: LayerKind,
        capacity: Int,
        var k_buf: DeviceBuffer[DType.float32],
        var v_buf: DeviceBuffer[DType.float32],
        var slot_position: List[Int],
    ):
        self.kind = kind.copy()
        self.capacity = capacity
        self.k_buf = k_buf^
        self.v_buf = v_buf^
        self.slot_position = slot_position^
        self.appended = 0

    def stored(self) -> Int:
        """Positions currently stored and visible (B1's rule)."""
        if self.kind.is_global:
            return self.appended
        if self.appended < self.kind.window:
            return self.appended
        return self.kind.window

    def first_visible_position(self) -> Int:
        """Absolute position of the oldest stored entry."""
        return self.appended - self.stored()


struct DeviceKvCache(Movable):
    """A device KV cache for one sequence: one K and one V
    device buffer per layer, B1 layout and (except the stated
    wrap limit) B1 semantics."""

    var layers: List[DeviceKvLayer]
    var num_kv_heads: Int
    var head_dim: Int
    var context_len: Int
    var total_bytes: Int

    def __init__(
        out self,
        var layers: List[DeviceKvLayer],
        num_kv_heads: Int,
        head_dim: Int,
        context_len: Int,
        total_bytes: Int,
    ):
        self.layers = layers^
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
        """Reserve the exact KV total, then allocate every
        layer's two zero-filled buffers. Raises
        gpu_budget_exceeded (via the runtime) before the first
        buffer exists when the total is over cap."""
        var total = kv_total_bytes(
            pattern, num_kv_heads, head_dim, context_len
        )
        rt.reserve_kv(total)
        var row = num_kv_heads * head_dim
        var layers = List[DeviceKvLayer](capacity=len(pattern))
        for i in range(len(pattern)):
            var kind = pattern[i].copy()
            var capacity = kv_capacity_positions(kind, context_len)
            var elems = capacity * row
            var k_buf = create_zero_f32(rt, elems)
            var v_buf = create_zero_f32(rt, elems)
            var slot_position = List[Int]()
            if not kind.is_global:
                slot_position.resize(kind.window, -1)
            layers.append(
                DeviceKvLayer(kind, capacity, k_buf^, v_buf^, slot_position^)
            )
        _ = log_debug(
            "gpu",
            "device kv built layers="
            + String(len(pattern))
            + " kv_bytes="
            + String(total),
        )
        return DeviceKvCache(layers^, num_kv_heads, head_dim, context_len, total)

    # -- inventory ---------------------------------------------------

    def layer_count(self) -> Int:
        return len(self.layers)

    def row_len(self) -> Int:
        """Floats in one position's row: num_kv_heads * head_dim."""
        return self.num_kv_heads * self.head_dim

    def resident_bytes(self) -> Int:
        return self.total_bytes

    def _check_layer(self, layer: Int) raises:
        if layer < 0 or layer >= len(self.layers):
            raise Error(
                "gpu_kv_invalid_layer: layer "
                + String(layer)
                + " out of range, cache has "
                + String(len(self.layers))
                + " layers"
            )

    def layer_kind(self, layer: Int) raises -> LayerKind:
        self._check_layer(layer)
        return self.layers[layer].kind.copy()

    def capacity_positions(self, layer: Int) raises -> Int:
        self._check_layer(layer)
        return self.layers[layer].capacity

    def stored_len(self, layer: Int) raises -> Int:
        """Positions currently stored (visible) in one layer."""
        self._check_layer(layer)
        return self.layers[layer].stored()

    def next_position(self, layer: Int) raises -> Int:
        """The next position an append to this layer must carry."""
        self._check_layer(layer)
        return self.layers[layer].appended

    def first_visible_position(self, layer: Int) raises -> Int:
        """Absolute position of the oldest stored entry in one
        layer: appended - stored_len (B1's meaning)."""
        self._check_layer(layer)
        return self.layers[layer].first_visible_position()

    def key_buffer(
        self, layer: Int
    ) raises -> DeviceBuffer[DType.float32]:
        """The layer's raw K device buffer, for attention."""
        self._check_layer(layer)
        return self.layers[layer].k_buf.copy()

    def value_buffer(
        self, layer: Int
    ) raises -> DeviceBuffer[DType.float32]:
        """The layer's raw V device buffer, for attention."""
        self._check_layer(layer)
        return self.layers[layer].v_buf.copy()

    # -- append --------------------------------------------------------

    def append(
        mut self,
        mut rt: GpuRuntime,
        layer: Int,
        k_row: List[Float32],
        v_row: List[Float32],
        position: Int,
    ) raises:
        """Append one position's K and V rows to a layer via
        one-row staging + scatter. Refuses by name on a wrong
        row length (gpu_kv_shape_mismatch), an out-of-range
        layer (gpu_kv_invalid_layer), an out-of-order position
        (gpu_kv_out_of_order_position), a windowed wrap
        (gpu_kv_window_wrapped, the Wave C limit), and a global
        append past capacity (gpu_kv_capacity_exceeded)."""
        var row = self.row_len()
        if len(k_row) != row or len(v_row) != row:
            raise Error(
                "gpu_kv_shape_mismatch: k/v rows have "
                + String(len(k_row))
                + "/"
                + String(len(v_row))
                + " elements, expected "
                + String(row)
                + " (num_kv_heads * head_dim)"
            )
        self._check_layer(layer)
        var refusal = kv_append_refusal(
            self.layers[layer].kind,
            self.layers[layer].capacity,
            self.layers[layer].appended,
            position,
            len(k_row),
            len(v_row),
            row,
        )
        if refusal == "gpu_kv_out_of_order_position":
            raise Error(
                "gpu_kv_out_of_order_position: layer "
                + String(layer)
                + " appends must be in order: expected position "
                + String(self.layers[layer].appended)
                + ", got "
                + String(position)
            )
        if refusal == "gpu_kv_window_wrapped":
            raise Error(
                "gpu_kv_window_wrapped: layer "
                + String(layer)
                + " position "
                + String(position)
                + " would wrap the window "
                + String(self.layers[layer].kind.window)
                + "; the Wave C device path refuses wraps (the CPU"
                + " cache remains the fallback beyond the window)"
            )
        if refusal == "gpu_kv_capacity_exceeded":
            raise Error(
                "gpu_kv_capacity_exceeded: layer "
                + String(layer)
                + " position "
                + String(position)
                + " is past the configured capacity "
                + String(self.layers[layer].capacity)
            )
        if refusal != "":
            raise Error(refusal + ": layer " + String(layer))
        var slot = kv_slot_of(self.layers[layer].kind, position)
        var offset = slot * row
        var k_stage = upload_f32(rt, k_row)
        rt.ctx.enqueue_function[kv_scatter_kernel](
            self.layers[layer].k_buf,
            k_stage,
            Int32(row),
            Int32(offset),
            grid_dim=1,
            block_dim=_KV_BLOCK,
        )
        var v_stage = upload_f32(rt, v_row)
        rt.ctx.enqueue_function[kv_scatter_kernel](
            self.layers[layer].v_buf,
            v_stage,
            Int32(row),
            Int32(offset),
            grid_dim=1,
            block_dim=_KV_BLOCK,
        )
        rt.ctx.synchronize()
        if not self.layers[layer].kind.is_global:
            self.layers[layer].slot_position[slot] = position
        self.layers[layer].appended += 1
        _ = log_debug(
            "gpu",
            "kv append layer="
            + String(layer)
            + " position="
            + String(position)
            + " slot="
            + String(slot),
        )

    # -- read-back (proof / debug path; attention consumes the
    #    raw buffers through key_buffer / value_buffer) --------------

    def _check_visible(self, layer: Int, position: Int) raises:
        self._check_layer(layer)
        var first = self.layers[layer].first_visible_position()
        if position < first:
            raise Error(
                "gpu_kv_position_evicted: layer "
                + String(layer)
                + " position "
                + String(position)
                + " was evicted; first visible position is "
                + String(first)
            )
        if position >= self.layers[layer].appended:
            raise Error(
                "gpu_kv_position_not_written: layer "
                + String(layer)
                + " position "
                + String(position)
                + " has not been appended yet; next position is "
                + String(self.layers[layer].appended)
            )
        if not self.layers[layer].kind.is_global:
            var slot = position % self.layers[layer].kind.window
            if self.layers[layer].slot_position[slot] != position:
                raise Error(
                    "gpu_kv_position_evicted: layer "
                    + String(layer)
                    + " position "
                    + String(position)
                    + " was evicted; slot "
                    + String(slot)
                    + " holds position "
                    + String(self.layers[layer].slot_position[slot])
                )

    def read_key_row(
        mut self, mut rt: GpuRuntime, layer: Int, position: Int
    ) raises -> List[Float32]:
        """One position's full K row, gathered on device into a
        one-row staging buffer and downloaded. Same visibility
        refusals as B1's key_row, gpu_kv_ prefixed."""
        return self._read_row(rt, layer, position, False)

    def read_value_row(
        mut self, mut rt: GpuRuntime, layer: Int, position: Int
    ) raises -> List[Float32]:
        """One position's full V row; see read_key_row."""
        return self._read_row(rt, layer, position, True)

    def _read_row(
        mut self,
        mut rt: GpuRuntime,
        layer: Int,
        position: Int,
        want_value: Bool,
    ) raises -> List[Float32]:
        self._check_visible(layer, position)
        var row = self.row_len()
        var slot = kv_slot_of(self.layers[layer].kind, position)
        var stage = create_zero_f32(rt, row)
        if want_value:
            rt.ctx.enqueue_function[kv_gather_kernel](
                self.layers[layer].v_buf,
                stage,
                Int32(row),
                Int32(slot * row),
                grid_dim=1,
                block_dim=_KV_BLOCK,
            )
        else:
            rt.ctx.enqueue_function[kv_gather_kernel](
                self.layers[layer].k_buf,
                stage,
                Int32(row),
                Int32(slot * row),
                grid_dim=1,
                block_dim=_KV_BLOCK,
            )
        rt.ctx.synchronize()
        return download_f32(rt, stage, row)
