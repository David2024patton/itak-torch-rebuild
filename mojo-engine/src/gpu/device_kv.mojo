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
# WAVE D (Lane D4): the Wave C wrap limit is RETIRED. B1's
# windowed append wraps the ring and evicts the oldest
# position; the device cache now does exactly the same
# (slot = position % window, slot_position tracks the holder,
# stored/first_visible carry B1's meanings past the wrap).
# The Wave C problem was the READER, not the writer: the
# attention kernel consumes visible rows in absolute-position
# order, which equals raw slot order only while
# first_visible % window == 0. Past a wrap, readers go
# through kv_visible_gather_kernel (below), which materializes
# the visible rows in absolute-position order into a scratch
# buffer the attention kernel consumes unchanged; see
# DeviceKvCache.gather_visible. The retirement is proven by
# tests/server_proof_gpu_kv_wrap.mojo (B1 wraparound sequences
# replayed element-exact on device, synthetic window + the
# real 2,048 window past 2,048) - the refusal name
# gpu_kv_window_wrapped is gone from this cache because that
# proof passes; it survives only in the legacy host-chained
# layer path (gpu_layer_step_hostchain), which still launches
# attention directly on the raw buffers. A global append past
# the configured context capacity refuses
# gpu_kv_capacity_exceeded (B1's global store grows; a device
# buffer cannot).

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
                "gpu_kv_invalid_window: sliding window must be positive, got "
                + String(pattern[i].window)
            )
    try:
        var acct = kv_size_accounting(pattern, kv_heads, head_dim, context_len)
        return acct.total_bytes
    except e:
        var msg = String(e)
        if msg.find("kv_size_overflow") != -1:
            raise Error(
                "gpu_kv_size_overflow: KV size overflows (from B1 accounting: "
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
    is VM-testable without a device. Check order mirrors B1:
    shape, then order, then the global capacity limit.
    Windowed appends NEVER refuse for wrapping (Wave D): the
    ring evicts exactly as B1's cache does."""
    if k_len != row_len or v_len != row_len:
        return "gpu_kv_shape_mismatch"
    if position != appended:
        return "gpu_kv_out_of_order_position"
    if kind.is_global and position >= capacity:
        return "gpu_kv_capacity_exceeded"
    return ""


def kv_visible_slot(
    kind: LayerKind, first_visible: Int, visible_index: Int
) -> Int:
    """Storage slot of the visible row at `visible_index`
    (0 = the oldest visible row, absolute position
    first_visible + visible_index). B1's presentation rule:
    global rows sit at their absolute position; windowed rows
    sit at (first_visible + visible_index) % window."""
    if kind.is_global:
        return first_visible + visible_index
    return (first_visible + visible_index) % kind.window


def kv_gather_needed(kind: LayerKind, first_visible: Int) -> Bool:
    """True when the raw buffer's slot order is NOT the
    visible (absolute-position) order, so a reader must go
    through the visible-order gather. Global layers never
    wrap (first_visible is always 0); a windowed layer's raw
    order matches visible order exactly when the first
    visible position lands on slot 0."""
    if kind.is_global:
        return False
    return first_visible % kind.window != 0


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


def kv_visible_gather_kernel(
    src: Pointer[Float32, MutAnyOrigin],
    dst: Pointer[Float32, MutAnyOrigin],
    row_len: Int32,
    first_slot: Int32,
    window: Int32,
):
    """Visible-order gather: one block per visible row.
    Block i copies storage row kv_visible_slot(first_visible,
    i) into output row i, so dst is the layer's visible rows
    in absolute-position order ([stored][row] flat), the exact
    layout attention_kernel consumes. window is the layer's
    ring size, or 0 for a global layer (slots are then the
    absolute positions themselves, first_slot = first_visible).
    Threads stride the row, as in the scatter/gather pair."""
    var vis = Int(block_idx.x)
    var slot: Int
    if Int(window) == 0:
        slot = Int(first_slot) + vis
    else:
        slot = (Int(first_slot) + vis) % Int(window)
    var src_off = slot * Int(row_len)
    var dst_off = vis * Int(row_len)
    var i = Int(thread_idx.x)
    while i < Int(row_len):
        dst[unsafe_offset=dst_off + i] = src[unsafe_offset=src_off + i]
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
    the retired Wave C wrap limit) B1 semantics."""

    var layers: List[DeviceKvLayer]
    var num_kv_heads: Int
    var head_dim: Int
    var context_len: Int
    var total_bytes: Int
    var reservation_released: Bool

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
        self.reservation_released = False

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
        var total = kv_total_bytes(pattern, num_kv_heads, head_dim, context_len)
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
        return DeviceKvCache(
            layers^, num_kv_heads, head_dim, context_len, total
        )

    # -- inventory ---------------------------------------------------

    def layer_count(self) -> Int:
        return len(self.layers)

    def row_len(self) -> Int:
        """Floats in one position's row: num_kv_heads * head_dim."""
        return self.num_kv_heads * self.head_dim

    def resident_bytes(self) -> Int:
        return self.total_bytes

    def release_reservation(mut self, mut rt: GpuRuntime) raises:
        """Return build()'s KV reservation to the runtime
        ledger (Wave E): call when this cache is being torn
        down, so a later cache build in the same runtime can
        reserve the same room instead of double-counting
        against the cap. Releasing twice refuses
        gpu_budget_exceeded by name (see DeviceWeights.
        release_reservation for the shape)."""
        if self.reservation_released:
            raise Error(
                "gpu_budget_exceeded: kv reservation of "
                + String(self.total_bytes)
                + " B was already released"
            )
        rt.unreserve_kv(self.total_bytes)
        self.reservation_released = True
        _ = log_debug(
            "gpu",
            "device kv reservation released bytes="
            + String(self.total_bytes),
        )

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

    def key_buffer(self, layer: Int) raises -> DeviceBuffer[DType.float32]:
        """The layer's raw K device buffer, for attention."""
        self._check_layer(layer)
        return self.layers[layer].k_buf.copy()

    def value_buffer(self, layer: Int) raises -> DeviceBuffer[DType.float32]:
        """The layer's raw V device buffer, for attention."""
        self._check_layer(layer)
        return self.layers[layer].v_buf.copy()

    # -- append --------------------------------------------------------

    def _append_slot(
        self, layer: Int, position: Int, k_len: Int, v_len: Int
    ) raises -> Int:
        """Validate one append (host or device-resident rows)
        and return its storage slot. Refuses by name on a wrong
        row length (gpu_kv_shape_mismatch), an out-of-range
        layer (gpu_kv_invalid_layer), an out-of-order position
        (gpu_kv_out_of_order_position), and a global append
        past capacity (gpu_kv_capacity_exceeded). Windowed
        appends wrap the ring, B1-exact (Wave D)."""
        var row = self.row_len()
        if k_len != row or v_len != row:
            raise Error(
                "gpu_kv_shape_mismatch: k/v rows have "
                + String(k_len)
                + "/"
                + String(v_len)
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
            k_len,
            v_len,
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
        return kv_slot_of(self.layers[layer].kind, position)

    def _append_commit(mut self, layer: Int, slot: Int, position: Int):
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

    def append(
        mut self,
        mut rt: GpuRuntime,
        layer: Int,
        k_row: List[Float32],
        v_row: List[Float32],
        position: Int,
    ) raises:
        """Append one position's K and V rows to a layer via
        one-row staging + scatter. Refusals per _append_slot;
        windowed appends wrap and evict, exactly as B1."""
        var row = self.row_len()
        var slot = self._append_slot(layer, position, len(k_row), len(v_row))
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
        self._append_commit(layer, slot, position)

    def append_device(
        mut self,
        mut rt: GpuRuntime,
        layer: Int,
        k_buf: DeviceBuffer[DType.float32],
        v_buf: DeviceBuffer[DType.float32],
        position: Int,
    ) raises:
        """Append one position's rows from RESIDENT device
        buffers (the residency path: the layer's K/V rows never
        cross to the host). The buffers' first row_len()
        elements are the row; the scatter is device-to-device.
        Same validation and refusals as append().

        NO SYNCHRONIZE (Wave F, Lane F1-A item 2): this append
        used to end in rt.ctx.synchronize(), which capped host
        lookahead at half a layer - the host could not enqueue
        layer N+1's work until layer N's scatters had finished
        on device. The sync bought nothing: the context's
        single queue executes in enqueue order, so the
        layer's attention launches (enqueued after these
        scatters) read the rows exactly as written, and the
        host-side commit below touches only host state. Every
        read-back path (download_f32, _read_row, the layer
        boundary download) synchronizes for itself. The host
        append() above KEEPS its sync: its callers are proofs
        and staging paths that read rows back immediately and
        are not the per-token hot path."""
        var row = self.row_len()
        var slot = self._append_slot(layer, position, row, row)
        var offset = slot * row
        rt.ctx.enqueue_function[kv_scatter_kernel](
            self.layers[layer].k_buf,
            k_buf,
            Int32(row),
            Int32(offset),
            grid_dim=1,
            block_dim=_KV_BLOCK,
        )
        rt.ctx.enqueue_function[kv_scatter_kernel](
            self.layers[layer].v_buf,
            v_buf,
            Int32(row),
            Int32(offset),
            grid_dim=1,
            block_dim=_KV_BLOCK,
        )
        # No synchronize: see the docstring (Wave F item 2).
        self._append_commit(layer, slot, position)

    # -- visible-order read for attention (Wave D) ----------------

    def gather_visible(
        mut self, mut rt: GpuRuntime, layer: Int, want_value: Bool
    ) raises -> DeviceBuffer[DType.float32]:
        """The layer's visible K (or V) rows in
        absolute-position order, as one device buffer of
        stored_len x row_len floats - the layout
        attention_kernel consumes. When the raw buffer is
        already in visible order (kv_gather_needed is False:
        global layers, and windowed layers whose first visible
        position lands on slot 0) the raw buffer itself is
        returned, no copy; otherwise kv_visible_gather_kernel
        materializes the order into a scratch buffer (at most
        window x row floats, inside the runtime's scratch
        envelope). Wave F item 3: the materialized buffer is
        drawn from the runtime's scratch POOL (the gather
        kernel writes every element of every visible row, so
        the pool's undefined contents are safe) and the caller
        owns it until release_f32 with stored_len x row_len
        elements - the attention launches do exactly that once
        their reads are enqueued; the batch path lets it die
        at scope exit, which is also correct (it just misses
        the reuse). The raw-buffer return is a VIEW of the
        cache's own memory and must NEVER be released."""
        self._check_layer(layer)
        var kind = self.layers[layer].kind.copy()
        var first = self.layers[layer].first_visible_position()
        var src = (
            self.layers[layer]
            .v_buf.copy() if want_value else self.layers[layer]
            .k_buf.copy()
        )
        if not kv_gather_needed(kind, first):
            return src^
        var stored = self.layers[layer].stored()
        var row = self.row_len()
        var out = rt.acquire_f32(stored * row)
        var window = 0 if kind.is_global else kind.window
        var first_slot = kv_visible_slot(kind, first, 0)
        rt.ctx.enqueue_function[kv_visible_gather_kernel](
            src,
            out,
            Int32(row),
            Int32(first_slot),
            Int32(window),
            grid_dim=stored,
            block_dim=_KV_BLOCK,
        )
        rt.ctx.synchronize()
        _ = log_debug(
            "gpu",
            "kv gather_visible layer="
            + String(layer)
            + " stored="
            + String(stored)
            + " first_visible="
            + String(first),
        )
        return out^

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
