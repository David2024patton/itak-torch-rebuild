# cache.mojo - the KV cache (Mojo/MAX engine, Wave B lane B1).
#
# Mirror of the Rust twin crates/attention/src/cache.rs in
# semantics, layout, and error-naming spirit: every failure is a
# typed refusal BY NAME (the kv_* family below), and no path ever
# substitutes placeholder data for a position it does not hold.
#
# Storage layout is flat f32 rows: one row per position (global
# layers) or per ring slot (windowed layers), each row holding
# [num_kv_heads][head_dim] for keys and the same for values.
# Windowed rows are allocated once at construction, exactly
# `window` slots; the slot for a position is position % window,
# and slot_position tracks which absolute position each ring
# slot currently holds. Global rows grow by append.
#
# NO checkpoint/rewind in Wave B: speculative decoding's
# checkpoint API (Rust KvCache::checkpoint / rewind_to /
# release) is a later wave and is deliberately NOT built here.
# There is also no bulk restore in this wave; snapshot restores
# replay append() in the Rust twin's semantics.
#
# Sizes are parameters everywhere (layer count and kinds from
# the pattern, num_kv_heads, head_dim, context for accounting);
# the Glimmer numbers (52 layers, window 2048) appear only in
# glimmer_layer_pattern() below.
#
# CPU-true storage (List[Float32]); the device seam is the row
# access functions (key / value / key_row / value_row), which
# are the only way readers (Wave B lane B2 attention) consume
# this cache.
#
# Named refusals: kv_zero_dimension / kv_invalid_window /
# kv_size_overflow / kv_invalid_layer / kv_shape_mismatch /
# kv_out_of_order_position / kv_position_evicted /
# kv_position_not_written / kv_invalid_kv_head.

from diag import log_debug

comptime INT_MAX = 9223372036854775807


# ---------------------------------------------------------------------------
# Checked size arithmetic (Rust checked_mul spirit: a size that
# does not fit refuses by name, never wraps into a small one)
# ---------------------------------------------------------------------------


def checked_mul(a: Int, b: Int, what: String) raises -> Int:
    if a < 0 or b < 0:
        raise Error(
            "kv_zero_dimension: "
            + what
            + " must not be negative, got "
            + String(a)
            + " * "
            + String(b)
        )
    if a != 0 and b > INT_MAX // a:
        raise Error(
            "kv_size_overflow: "
            + what
            + " ("
            + String(a)
            + " * "
            + String(b)
            + ") overflows the addressable size"
        )
    return a * b


def checked_add(a: Int, b: Int, what: String) raises -> Int:
    if b > 0 and a > INT_MAX - b:
        raise Error(
            "kv_size_overflow: " + what + " overflows the addressable size"
        )
    return a + b


# ---------------------------------------------------------------------------
# LayerKind: Global | Windowed(window)
# ---------------------------------------------------------------------------


struct LayerKind(Copyable, Equatable, Movable):
    """How one layer stores its keys and values.

    is_global selects full history; otherwise window is the
    ring capacity in positions (also the attention window) and
    must be positive. Build windowed kinds through
    make_windowed so a zero window refuses by name at the
    point of construction; KvCache.new re-checks every kind,
    so a zero window smuggled in through the raw fields is
    refused there too (the Rust twin checks both places).
    """

    var is_global: Bool
    var window: Int

    def __init__(out self, is_global: Bool, window: Int):
        self.is_global = is_global
        self.window = window

    @staticmethod
    def make_global() -> LayerKind:
        return LayerKind(True, 0)

    @staticmethod
    def make_windowed(window: Int) raises -> LayerKind:
        if window <= 0:
            raise Error(
                "kv_invalid_window: sliding window must be positive, got "
                + String(window)
            )
        return LayerKind(False, window)

    def window_size(self) -> Int:
        """The window for windowed layers, 0 for global layers
        (the Rust twin's Option, flattened: check is_global)."""
        if self.is_global:
            return 0
        return self.window

    def describe(self) -> String:
        if self.is_global:
            return "global"
        return "windowed(" + String(self.window) + ")"


def glimmer_layer_pattern() raises -> List[LayerKind]:
    """The layer pattern of Muse Glimmer: 52 layers, global
    attention at indices 3, 7, 11, ..., 51 (index % 4 == 3; 13
    layers) and sliding-window 2048 everywhere else (39 layers).
    Mirrors crates/attention/src/cache.rs glimmer_layer_pattern.
    The only place the Glimmer layer numbers live in src/."""
    var pattern = List[LayerKind](capacity=52)
    for i in range(52):
        if i % 4 == 3:
            pattern.append(LayerKind.make_global())
        else:
            pattern.append(LayerKind.make_windowed(2048))
    return pattern^


# ---------------------------------------------------------------------------
# Size accounting (f32 bytes, keys and values together)
# ---------------------------------------------------------------------------


struct KvSizeAccounting(Copyable, Movable):
    """Full-context f32 size figures for a layer pattern.

    total_bytes is the real capped figure: global layers cost
    context_len rows each, windowed layers cost window rows
    each (windowed layers store window rows, not context
    rows). all_global_bytes is the all-global upper bound:
    every layer charged context_len rows, as if no layer were
    windowed. global_bytes / windowed_bytes are the two parts
    of total_bytes. bytes_per_position is one position's cost
    in one layer (row floats * 4 bytes * 2 for K and V).
    """

    var total_bytes: Int
    var all_global_bytes: Int
    var global_bytes: Int
    var windowed_bytes: Int
    var bytes_per_position: Int

    def __init__(
        out self,
        total_bytes: Int,
        all_global_bytes: Int,
        global_bytes: Int,
        windowed_bytes: Int,
        bytes_per_position: Int,
    ):
        self.total_bytes = total_bytes
        self.all_global_bytes = all_global_bytes
        self.global_bytes = global_bytes
        self.windowed_bytes = windowed_bytes
        self.bytes_per_position = bytes_per_position


def kv_size_accounting(
    pattern: List[LayerKind], num_kv_heads: Int, head_dim: Int, context_len: Int
) raises -> KvSizeAccounting:
    """Compute both size figures for a pattern at a context.

    Every product and sum is checked arithmetic; an overflow
    refuses by name (kv_size_overflow) instead of wrapping.
    """
    if len(pattern) == 0:
        raise Error("kv_zero_dimension: layer pattern must not be empty")
    if num_kv_heads <= 0:
        raise Error(
            "kv_zero_dimension: num_kv_heads must be positive, got "
            + String(num_kv_heads)
        )
    if head_dim <= 0:
        raise Error(
            "kv_zero_dimension: head_dim must be positive, got "
            + String(head_dim)
        )
    if context_len <= 0:
        raise Error(
            "kv_zero_dimension: context_len must be positive, got "
            + String(context_len)
        )
    var row = checked_mul(num_kv_heads, head_dim, "kv row length")
    var per_position = checked_mul(
        checked_mul(row, 4, "kv row byte length"),
        2,
        "kv position byte length (keys and values)",
    )
    var global_bytes = 0
    var windowed_bytes = 0
    var all_global_bytes = 0
    for i in range(len(pattern)):
        var kind = pattern[i].copy()
        var every = checked_mul(
            context_len, per_position, "all-global layer byte length"
        )
        all_global_bytes = checked_add(
            all_global_bytes, every, "all-global cache byte total"
        )
        if kind.is_global:
            global_bytes = checked_add(
                global_bytes, every, "global cache byte total"
            )
        else:
            if kind.window <= 0:
                raise Error(
                    "kv_invalid_window: sliding window must be positive, got "
                    + String(kind.window)
                )
            var capped = checked_mul(
                kind.window, per_position, "windowed layer byte length"
            )
            windowed_bytes = checked_add(
                windowed_bytes, capped, "windowed cache byte total"
            )
    var total = checked_add(global_bytes, windowed_bytes, "cache byte total")
    return KvSizeAccounting(
        total, all_global_bytes, global_bytes, windowed_bytes, per_position
    )


# ---------------------------------------------------------------------------
# One layer's storage (all validation lives in KvCache)
# ---------------------------------------------------------------------------


struct LayerCache(Copyable, Movable):
    var kind: LayerKind
    var keys: List[Float32]
    var values: List[Float32]
    # Windowed layers only: which absolute position each ring
    # slot currently holds, -1 when the slot was never written.
    # Global layers leave this empty.
    var slot_position: List[Int]
    # Total positions appended; also the next expected position.
    var appended: Int

    def __init__(
        out self,
        kind: LayerKind,
        var keys: List[Float32],
        var values: List[Float32],
        var slot_position: List[Int],
    ):
        self.kind = kind.copy()
        self.keys = keys^
        self.values = values^
        self.slot_position = slot_position^
        self.appended = 0

    def stored(self) -> Int:
        """Positions currently stored and visible."""
        if self.kind.is_global:
            return self.appended
        if self.appended < self.kind.window:
            return self.appended
        return self.kind.window

    def first_visible_position(self) -> Int:
        """Absolute position of the oldest stored entry."""
        return self.appended - self.stored()

    def slot_of(self, position: Int) -> Int:
        """Storage row index for an absolute position."""
        if self.kind.is_global:
            return position
        return position % self.kind.window


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------


struct KvCache(Copyable, Movable):
    """A KV cache for one sequence: one store per layer, each
    layer global or windowed, chosen at construction from the
    layer pattern. Every layer shares the same KV head count
    and head dimension."""

    var layers: List[LayerCache]
    var num_kv_heads: Int
    var head_dim: Int

    def __init__(
        out self, var layers: List[LayerCache], num_kv_heads: Int, head_dim: Int
    ):
        self.layers = layers^
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    @staticmethod
    def new(
        var pattern: List[LayerKind], num_kv_heads: Int, head_dim: Int
    ) raises -> KvCache:
        """Build a cache from a layer pattern (one LayerKind
        per layer). Refuses by name: an empty pattern or a zero
        head count / head dim (kv_zero_dimension), a zero
        window anywhere in the pattern (kv_invalid_window),
        and any size whose product overflows (kv_size_overflow).
        """
        if len(pattern) == 0:
            raise Error("kv_zero_dimension: layer pattern must not be empty")
        if num_kv_heads <= 0:
            raise Error(
                "kv_zero_dimension: num_kv_heads must be positive, got "
                + String(num_kv_heads)
            )
        if head_dim <= 0:
            raise Error(
                "kv_zero_dimension: head_dim must be positive, got "
                + String(head_dim)
            )
        var row = checked_mul(num_kv_heads, head_dim, "kv row length")
        var layers = List[LayerCache](capacity=len(pattern))
        for i in range(len(pattern)):
            var kind = pattern[i].copy()
            var keys = List[Float32]()
            var values = List[Float32]()
            var slot_position = List[Int]()
            if not kind.is_global:
                if kind.window <= 0:
                    raise Error(
                        "kv_invalid_window: sliding window must be"
                        " positive, got "
                        + String(kind.window)
                    )
                var slots = checked_mul(
                    kind.window, row, "windowed layer storage"
                )
                keys.resize(slots, Float32(0.0))
                values.resize(slots, Float32(0.0))
                slot_position.resize(kind.window, -1)
            layers.append(LayerCache(kind, keys^, values^, slot_position^))
        _ = log_debug(
            "kv.cache",
            "kv cache built layers="
            + String(len(pattern))
            + " kv_heads="
            + String(num_kv_heads)
            + " head_dim="
            + String(head_dim),
        )
        return KvCache(layers^, num_kv_heads, head_dim)

    # -- inventory ----------------------------------------------------

    def layer_count(self) -> Int:
        return len(self.layers)

    def row_len(self) -> Int:
        """Floats in one position's row: num_kv_heads * head_dim."""
        return self.num_kv_heads * self.head_dim

    def _check_layer(self, layer: Int) raises:
        if layer < 0 or layer >= len(self.layers):
            raise Error(
                "kv_invalid_layer: layer "
                + String(layer)
                + " out of range, cache has "
                + String(len(self.layers))
                + " layers"
            )

    def layer_kind(self, layer: Int) raises -> LayerKind:
        """The storage kind of one layer."""
        self._check_layer(layer)
        return self.layers[layer].kind.copy()

    def stored_len(self, layer: Int) raises -> Int:
        """Positions currently stored (visible) in one layer."""
        self._check_layer(layer)
        return self.layers[layer].stored()

    def next_position(self, layer: Int) raises -> Int:
        """The next position an append to this layer must carry
        (the layer's appended count)."""
        self._check_layer(layer)
        return self.layers[layer].appended

    def first_visible_position(self, layer: Int) raises -> Int:
        """Absolute position of the oldest stored entry in one
        layer: appended - stored_len. Reads below this position
        refuse as evicted."""
        self._check_layer(layer)
        return self.layers[layer].first_visible_position()

    def size_accounting(self, context_len: Int) raises -> KvSizeAccounting:
        """Both full-context size figures for this cache's
        pattern and geometry (see kv_size_accounting)."""
        var pattern = List[LayerKind](capacity=len(self.layers))
        for i in range(len(self.layers)):
            pattern.append(self.layers[i].kind.copy())
        return kv_size_accounting(
            pattern, self.num_kv_heads, self.head_dim, context_len
        )

    # -- append ---------------------------------------------------------

    def append(
        mut self,
        layer: Int,
        k_row: List[Float32],
        v_row: List[Float32],
        position: Int,
    ) raises:
        """Append one position's keys and values to a layer.

        k_row and v_row are the full per-position rows,
        [num_kv_heads][head_dim] laid out flat. Appends must
        arrive in position order per layer: position must equal
        the number of positions already appended to that layer.
        On a windowed layer, appending past the window evicts
        the oldest position. Refuses by name on a row of the
        wrong length (kv_shape_mismatch), an out-of-range
        layer (kv_invalid_layer), and an out-of-order position
        (kv_out_of_order_position).
        """
        var expected_len = self.row_len()
        if len(k_row) != expected_len:
            raise Error(
                "kv_shape_mismatch: k row has "
                + String(len(k_row))
                + " elements, expected "
                + String(expected_len)
                + " (num_kv_heads * head_dim)"
            )
        if len(v_row) != expected_len:
            raise Error(
                "kv_shape_mismatch: v row has "
                + String(len(v_row))
                + " elements, expected "
                + String(expected_len)
                + " (num_kv_heads * head_dim)"
            )
        self._check_layer(layer)
        if position != self.layers[layer].appended:
            raise Error(
                "kv_out_of_order_position: layer "
                + String(layer)
                + " appends must be in order: expected position "
                + String(self.layers[layer].appended)
                + ", got "
                + String(position)
            )
        var row = expected_len
        if self.layers[layer].kind.is_global:
            for i in range(row):
                self.layers[layer].keys.append(k_row[i])
                self.layers[layer].values.append(v_row[i])
        else:
            var slot = position % self.layers[layer].kind.window
            var start = slot * row
            for i in range(row):
                self.layers[layer].keys[start + i] = k_row[i]
                self.layers[layer].values[start + i] = v_row[i]
            self.layers[layer].slot_position[slot] = position
        self.layers[layer].appended += 1
        _ = log_debug(
            "kv.cache",
            "append layer="
            + String(layer)
            + " position="
            + String(position)
            + " kind="
            + self.layers[layer].kind.describe()
            + " stored="
            + String(self.layers[layer].stored()),
        )

    # -- reads (the device seam: B2 consumes only these) ----------------

    def _check_visible(self, layer: Int, position: Int, kv_head: Int) raises:
        self._check_layer(layer)
        if kv_head < 0 or kv_head >= self.num_kv_heads:
            raise Error(
                "kv_invalid_kv_head: kv head "
                + String(kv_head)
                + " out of range, layer has "
                + String(self.num_kv_heads)
                + " kv heads"
            )
        var first = self.layers[layer].first_visible_position()
        if position < first:
            raise Error(
                "kv_position_evicted: layer "
                + String(layer)
                + " position "
                + String(position)
                + " was evicted; first visible position is "
                + String(first)
            )
        if position >= self.layers[layer].appended:
            raise Error(
                "kv_position_not_written: layer "
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
                    "kv_position_evicted: layer "
                    + String(layer)
                    + " position "
                    + String(position)
                    + " was evicted; slot "
                    + String(slot)
                    + " holds"
                    + " position "
                    + String(self.layers[layer].slot_position[slot])
                )

    def _head_offset(self, layer: Int, position: Int, kv_head: Int) -> Int:
        var slot = self.layers[layer].slot_of(position)
        return (slot * self.num_kv_heads + kv_head) * self.head_dim

    def key(
        self, layer: Int, position: Int, kv_head: Int
    ) raises -> List[Float32]:
        """Keys for one KV head at one absolute position,
        [head_dim], copied out. Refuses by name for an
        out-of-range layer or head, an evicted position
        (position < first_visible_position), and a position
        not yet written."""
        self._check_visible(layer, position, kv_head)
        var off = self._head_offset(layer, position, kv_head)
        var out = List[Float32](capacity=self.head_dim)
        for j in range(self.head_dim):
            out.append(self.layers[layer].keys[off + j])
        return out^

    def value(
        self, layer: Int, position: Int, kv_head: Int
    ) raises -> List[Float32]:
        """Values for one KV head at one absolute position,
        [head_dim], copied out. Same refusals as key()."""
        self._check_visible(layer, position, kv_head)
        var off = self._head_offset(layer, position, kv_head)
        var out = List[Float32](capacity=self.head_dim)
        for j in range(self.head_dim):
            out.append(self.layers[layer].values[off + j])
        return out^

    def key_row(self, layer: Int, position: Int) raises -> List[Float32]:
        """The full key row for one absolute position,
        [num_kv_heads][head_dim] flat, copied out. Same
        visibility refusals as key()."""
        self._check_visible(layer, position, 0)
        var slot = self.layers[layer].slot_of(position)
        var start = slot * self.row_len()
        var out = List[Float32](capacity=self.row_len())
        for i in range(self.row_len()):
            out.append(self.layers[layer].keys[start + i])
        return out^

    def value_row(self, layer: Int, position: Int) raises -> List[Float32]:
        """The full value row for one absolute position,
        [num_kv_heads][head_dim] flat, copied out. Same
        visibility refusals as value()."""
        self._check_visible(layer, position, 0)
        var slot = self.layers[layer].slot_of(position)
        var start = slot * self.row_len()
        var out = List[Float32](capacity=self.row_len())
        for i in range(self.row_len()):
            out.append(self.layers[layer].values[start + i])
        return out^
