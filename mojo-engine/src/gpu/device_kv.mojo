# device_kv.mojo - DeviceKvCache: the KV cache resident on
# device, B1's layout and semantics (Lane C1).
#
# =====================================================================
# PLACEHOLDER / STAND-IN (Lane C3, 2026-11-07). This file exists
# in the lane/mojo-c3-gen worktree ONLY so Lane C3's
# src/model/gpu_model.mojo compiles against the Wave C contract
# (mojo-engine/WAVE_C_INTERFACES.md) before Lane C1 lands.
# DELETE AT INTEGRATION: the coordinator replaces this whole
# file with Lane C1's real src/gpu/device_kv.mojo. Do not ship
# this stand-in as the C1 implementation: its append path does
# whole-row host staging through the context's copy helpers
# (the contract expects a small scatter copy kernel in C1's
# file), it has no server KV write/read proof, and its
# window-wrap refusal is enforced host-side only.
# =====================================================================
#
# Contract surface coded against (WAVE_C_INTERFACES.md, Lane C1):
#   Per layer, two float32 device buffers (K and V), each
#   capacity_positions x kv_heads x head_dim, where
#   capacity_positions is the layer's window (2,048) for
#   windowed layers and the configured context length for
#   global layers: B1's layout, byte for byte (row = one
#   position's [kv_heads][head_dim], slot = position % window
#   for windowed layers, slot = position for global layers).
#   Semantics mirror KvCache exactly: appends in order only,
#   reads by absolute position, the same refusal names as B1
#   for out-of-order append / evicted or unwritten read / bad
#   shape (prefix gpu_kv_ where a name would collide). Append
#   path: the caller passes the position's K and V rows as host
#   lists (already rotated for sliding layers). Read path for
#   attention: key_buffer(layer) / value_buffer(layer) expose
#   the raw device buffers, plus stored_len(layer) and
#   first_visible_position(layer) with B1's meanings. WAVE C
#   LIMIT: the GPU layer refuses by name (gpu_kv_window_wrapped)
#   when a windowed layer's append would wrap past its window.
#
# Stand-in shape notes: buffers are reached by parallel-list
# index (key_buffer_index / value_buffer_index) because
# DeviceBuffer handles are owned by this struct; the raw
# buffers themselves are the public fields key_buffers /
# value_buffers. C1's real key_buffer(layer) spelling wins at
# integration if it returns the buffer directly.

from max.gpu.host import DeviceBuffer

from diag import log_debug
from gpu.context import GpuRuntime, create_zero_f32
from kv.cache import LayerKind


struct DeviceKvCache(Movable):
    var key_buffers: List[DeviceBuffer[DType.float32]]
    var value_buffers: List[DeviceBuffer[DType.float32]]
    var kinds: List[LayerKind]
    var capacities: List[Int]
    var appended: List[Int]
    var kv_heads: Int
    var head_dim: Int
    var context_len: Int
    var total_bytes: Int

    def __init__(
        out self,
        mut rt: GpuRuntime,
        var kinds: List[LayerKind],
        kv_heads: Int,
        head_dim: Int,
        context_len: Int,
    ) raises:
        if len(kinds) == 0:
            raise Error("gpu_kv_zero_dimension: layer pattern must not be empty")
        if kv_heads <= 0 or head_dim <= 0:
            raise Error(
                "gpu_kv_zero_dimension: kv_heads and head_dim must be"
                " positive, got "
                + String(kv_heads)
                + " / "
                + String(head_dim)
            )
        if context_len <= 0:
            raise Error(
                "gpu_kv_zero_dimension: context length must be positive, got "
                + String(context_len)
            )
        self.kinds = kinds^
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.context_len = context_len
        self.key_buffers = List[DeviceBuffer[DType.float32]]()
        self.value_buffers = List[DeviceBuffer[DType.float32]]()
        self.capacities = List[Int]()
        self.appended = List[Int]()

        # Refuse first, allocate second: every capacity is
        # known from the pattern and the context length.
        var row = kv_heads * head_dim
        var total = 0
        for i in range(len(self.kinds)):
            var cap = self._capacity_of(self.kinds[i].copy())
            total += 2 * cap * row * 4
        rt.account(total, "device kv cache")
        self.total_bytes = total

        for i in range(len(self.kinds)):
            var cap = self._capacity_of(self.kinds[i].copy())
            self.capacities.append(cap)
            self.appended.append(0)
            self.key_buffers.append(create_zero_f32(rt, cap * row))
            self.value_buffers.append(create_zero_f32(rt, cap * row))
        _ = log_debug(
            "gpu",
            "device kv built layers="
            + String(len(self.kinds))
            + " resident_bytes="
            + String(self.total_bytes),
        )

    def _capacity_of(self, kind: LayerKind) -> Int:
        if kind.is_global:
            return self.context_len
        return kind.window

    def row_len(self) -> Int:
        """Floats in one position's row: kv_heads * head_dim."""
        return self.kv_heads * self.head_dim

    def layer_count(self) -> Int:
        return len(self.kinds)

    def _check_layer(self, layer: Int) raises:
        if layer < 0 or layer >= len(self.kinds):
            raise Error(
                "gpu_kv_invalid_layer: layer "
                + String(layer)
                + " out of range, cache has "
                + String(len(self.kinds))
                + " layers"
            )

    def stored_len(self, layer: Int) raises -> Int:
        """Positions currently stored (visible) in one layer,
        B1's meaning."""
        self._check_layer(layer)
        var kind = self.kinds[layer].copy()
        if kind.is_global:
            return self.appended[layer]
        if self.appended[layer] < kind.window:
            return self.appended[layer]
        return kind.window

    def first_visible_position(self, layer: Int) raises -> Int:
        """Absolute position of the oldest stored entry."""
        return self.appended[layer] - self.stored_len(layer)

    def next_position(self, layer: Int) raises -> Int:
        """The next position an append to this layer must carry."""
        self._check_layer(layer)
        return self.appended[layer]

    def key_buffer_index(self, layer: Int) raises -> Int:
        """Index into key_buffers for layer (stand-in spelling
        of the contract's key_buffer(layer) read path)."""
        self._check_layer(layer)
        return layer

    def value_buffer_index(self, layer: Int) raises -> Int:
        """Index into value_buffers for layer."""
        self._check_layer(layer)
        return layer

    def append(
        mut self,
        rt: GpuRuntime,
        layer: Int,
        k_row: List[Float32],
        v_row: List[Float32],
        position: Int,
    ) raises:
        """Append one position's K and V rows, in order only.

        Refusals (B1's names with the gpu_kv_ prefix where the
        contract calls for it): gpu_kv_append_out_of_order,
        gpu_kv_bad_shape, gpu_kv_window_wrapped (the Wave C
        limit: a windowed append that would wrap past the
        window refuses; the CPU path remains the fallback),
        gpu_kv_out_of_range (global position at or beyond the
        configured context length).
        """
        self._check_layer(layer)
        if position != self.appended[layer]:
            raise Error(
                "gpu_kv_append_out_of_order: layer "
                + String(layer)
                + " expects position "
                + String(self.appended[layer])
                + ", got "
                + String(position)
            )
        if len(k_row) != self.row_len() or len(v_row) != self.row_len():
            raise Error(
                "gpu_kv_bad_shape: K/V rows must have "
                + String(self.row_len())
                + " elements, got "
                + String(len(k_row))
                + " / "
                + String(len(v_row))
            )
        var kind = self.kinds[layer].copy()
        var slot: Int
        if kind.is_global:
            if position >= self.context_len:
                raise Error(
                    "gpu_kv_out_of_range: global position "
                    + String(position)
                    + " at or beyond context length "
                    + String(self.context_len)
                )
            slot = position
        else:
            if position >= kind.window:
                raise Error(
                    "gpu_kv_window_wrapped: windowed layer "
                    + String(layer)
                    + " append at position "
                    + String(position)
                    + " would wrap past window "
                    + String(kind.window)
                    + "; the CPU path is the fallback beyond the window"
                )
            slot = position % kind.window
        # STAND-IN LIMIT, stated plainly: this append validates
        # and counts but does NOT write the row into device
        # memory (no offset copy is attempted). No GPU result
        # may ever be claimed through this stand-in; C1's real
        # append (staging + scatter copy kernel into the slot)
        # replaces this whole file at integration.
        _ = rt
        _ = slot
        _ = k_row
        _ = v_row
        self.appended[layer] += 1
        _ = log_debug(
            "gpu",
            "device kv append layer="
            + String(layer)
            + " position="
            + String(position),
        )
