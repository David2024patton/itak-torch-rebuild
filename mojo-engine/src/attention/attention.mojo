# attention.mojo - single-position grouped-query attention,
# Lane B2 (Wave B). CPU twin + GPU kernel.
#
# Semantic references (Rust, branch main):
#   crates/attention/src/attention.rs::gqa_attention
#     The math core: GQA head sharing (kv_head = q_head //
#     (num_q_heads / num_kv_heads); Glimmer 32 Q / 2 KV, so
#     kv_head = q_head // 16), score scale 1/sqrt(head_dim),
#     causal + window masking over absolute positions, softmax
#     (max-subtracted), weighted sum of values. That core takes
#     PRE-ROTATED queries and keys: "RoPE is the caller's job."
#   crates/cpu-compute/src/cpu.rs::Compute::attention +
#     rope_row (the caller, and the RoPE finding this lane
#     mirrors exactly):
#     - The loop is one position at a time: rotate, append,
#       attend. Prefill and decode share this path; decode is
#       the rows == 1 case.
#     - RoPE runs on SLIDING-window layers only, at the SWA
#       base (500,000 for Glimmer). Global layers are NoPE:
#       Q and K pass through unrotated.
#     - rope_row rotates EVERY head of the position's Q row
#       (all 32 Q heads) and EVERY head of the position's K
#       row (both KV heads), each head at the SAME absolute
#       position (the position being appended), interleaved
#       pairing via torch-math rope_interleaved (Wave A
#       ops/rope.rope_cpu).
#     - The ROTATED K row is what gets appended: the Rust
#       cache holds rotated K (and raw V). Rotation happens
#       at WRITE time, once, when the position is appended;
#       keys already in the cache are never re-rotated.
#
# THE SEAM (stated plainly, per the Wave B brief): this module
# never takes a cache object. Per layer, the caller passes
# plain flat stores in VISIBLE ORDER plus range parameters:
#   keys   flat [stored][num_kv_heads][head_dim]
#   values flat [stored][num_kv_heads][head_dim]
#   first_visible: the absolute position of row 0
#   stored: the number of visible rows
# Visible order is exactly the Rust CacheView's presentation:
# row i is view index i, its absolute position is
# first_visible + i (CacheView::position_at), and the head
# slice is view.key(i, kv_head) / view.value(i, kv_head). The
# coordinator wires B1's accessors to this layout at
# integration; for a windowed (ring) layer that means rows
# gathered in position order starting at slot
# first_visible % window, NOT the raw ring buffer when it has
# wrapped.
#
# RoPE AT READ TIME (the one deliberate storage difference):
# attention_cpu takes RAW (unrotated) Q and K and applies the
# rotation at read time: Q at `position`, every visible K row
# at its own absolute position first_visible + i, both via
# Wave A's rope_cpu, both only when rope_base != ROPE_NONE.
# The outputs are the Rust path's outputs: rotation is a
# fixed per-position transform, so applying it to a K row at
# read time yields the same rotated row the Rust path stored
# at write time (rope_cpu is deterministic on the same raw
# values). The storage consequence for integration: a caller
# using attention_cpu must pass RAW keys, so B3 must append
# raw K to the B1 cache on this seam. If the coordinator
# instead keeps the Rust storage exactly (B1 holds ROTATED
# K), call attention_cpu_rotated_cache below, which rotates
# only Q and never touches the keys. The two functions share
# one implementation; the flag is explicit, never guessed.
#
# Sliding vs global differs ONLY through (a) the visible
# range the caller passes (the window is the cache's job,
# B1) and (b) the rope parameter: a base for sliding layers,
# ROPE_NONE for globals (NoPE). There is no layer-kind
# argument and no window arithmetic in the single-position
# function, exactly as the contract states.
#
# GPU kernel: attention_kernel mirrors the Rust math core
# (attention.rs): it consumes PRE-ROTATED Q/K device buffers
# (rotation is the separate Wave A rope op / rope_cpu step,
# as in Rust where gqa_attention takes pre-rotated inputs).
# It is a flash-style DECODING kernel: one block per Q head,
# a single streaming pass over positions, online softmax
# (running max, running sum, per-channel accumulators
# rescaled per position), the score row is never
# materialized, and each position's dot is a shared-memory
# tree reduction. That kernel is RETAINED unchanged as the
# Wave B reference shape (the A/B baseline in the proofs).
#
# WAVE D (Lane D2, occupancy): attention_tiled_kernel +
# attention_merge_kernel below are the production GPU path.
# The tiled kernel stages K/V tiles of TILE positions in
# shared memory (one tile buffer, reused for K then V), one
# block per (Q head, split): the KV sequence is SPLIT across
# blocks (split-KV), each block runs the same online-softmax
# over its range tile by tile and writes an UNNORMALIZED
# partial (acc, max, sum); the merge kernel combines the
# partials in split order with exact logsumexp rescaling
# (each partial rescaled by exp(its max - the combined max),
# no approximation). Scores are computed per tile from the
# staged K (one thread per position, sequential over the
# head dim in shared memory), so the per-position shared-
# memory tree reduction of the Wave B kernel (a barrier
# chain per position) is gone: barriers happen per tile.
# WAVE E (Lane E2): attention_tiled_direct_kernel is the
# splits == 1 form of the tiled kernel with the merge folded
# into the same launch (direct normalized write, no partial
# buffers, no merge launch); its CPU twin is
# attention_direct_cpu. The resident chain launches it
# exactly when its split chooser returns 1 (the short
# regime, stored <= one split span); splits >= 2 keep the
# tiled + merge pair unchanged.
#
# Prefill note: this engine has no batched prefill kernel -
# per the Wave C contract, prefill IS the step loop, so
# prefill positions run through this same kernel at growing
# stored counts; the split chooser keeps splits at 1 until
# the stored count passes one split span, so short prefill
# steps pay no merge cost beyond the trivial 1-split pass.
# The CPU twins of the split path (attention_split_partials_
# cpu / attention_merge_partials_cpu / attention_split_cpu)
# mirror the kernels' Float32 arithmetic tile by tile and
# are the VM-side proof surface. Trial record (tile sizes,
# split factors, kept structure): bench/mojo-engine-wave-d-
# 2026-11-07.md.
#
# Named refusals: attention_cpu_bad_shape,
# attention_cpu_nondivisible_heads,
# attention_cpu_no_visible_keys, attention_cpu_bad_position,
# attention_cpu_bad_rope_base, attention_tiled_bad_tile,
# attention_split_bad_shape, attention_split_bad_splits,
# attention_split_bad_config, attention_split_bad_range,
# attention_split_empty_partials.

from std.math import sqrt, exp, isnan, isinf
from max.gpu import thread_idx, block_idx, block_dim, barrier
from max.gpu.memory import AddressSpace
from std.memory import stack_allocation

from diag import log_debug
from ops.rope import rope_cpu
from ops.softmax import softmax_cpu

comptime ROPE_NONE = Float32(0.0)
comptime ATTN_THREADS = 128


def _validate(
    q_len: Int,
    keys_len: Int,
    values_len: Int,
    first_visible: Int,
    stored: Int,
    num_q_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    position: Int,
    rope_base: Float32,
) raises -> Int:
    """Shared shape/range checks; returns the causal row count."""
    if head_dim < 2 or head_dim % 2 != 0:
        raise Error(
            "attention_cpu_bad_shape: head_dim must be even and >= 2, got "
            + String(head_dim)
        )
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise Error(
            "attention_cpu_bad_shape: head counts must be > 0, got "
            + String(num_q_heads)
            + " q heads, "
            + String(num_kv_heads)
            + " kv heads"
        )
    if num_q_heads % num_kv_heads != 0:
        raise Error(
            "attention_cpu_nondivisible_heads: "
            + String(num_q_heads)
            + " q heads do not divide into "
            + String(num_kv_heads)
            + " kv heads"
        )
    if q_len != num_q_heads * head_dim:
        raise Error(
            "attention_cpu_bad_shape: q has "
            + String(q_len)
            + " elements, expected "
            + String(num_q_heads * head_dim)
            + " ([num_q_heads][head_dim])"
        )
    if stored <= 0:
        raise Error(
            "attention_cpu_no_visible_keys: stored visible range is empty"
            " (stored "
            + String(stored)
            + ") for query position "
            + String(position)
        )
    var row = num_kv_heads * head_dim
    if keys_len != stored * row or values_len != stored * row:
        raise Error(
            "attention_cpu_bad_shape: keys/values have "
            + String(keys_len)
            + "/"
            + String(values_len)
            + " elements, expected "
            + String(stored * row)
            + " each ([stored][num_kv_heads][head_dim])"
        )
    if first_visible < 0 or position < 0:
        raise Error(
            "attention_cpu_bad_position: first_visible and position must"
            " be >= 0, got "
            + String(first_visible)
            + " and "
            + String(position)
        )
    if position < first_visible:
        raise Error(
            "attention_cpu_no_visible_keys: query position "
            + String(position)
            + " precedes the visible range starting at "
            + String(first_visible)
        )
    if (
        isnan(rope_base)
        or isinf(rope_base)
        or rope_base < Float32(0.0)
        or (rope_base != ROPE_NONE and rope_base <= Float32(1.0))
    ):
        raise Error(
            "attention_cpu_bad_rope_base: rope_base must be 0 (NoPE) or"
            " finite and > 1, got "
            + String(rope_base)
        )
    # Causal by construction: rows past the query position are
    # never scored, even if a caller hands them over.
    var effective = stored
    if position - first_visible + 1 < effective:
        effective = position - first_visible + 1
    return effective


def _attention_impl(
    q: List[Float32],
    keys: List[Float32],
    values: List[Float32],
    first_visible: Int,
    stored: Int,
    num_q_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    position: Int,
    rope_base: Float32,
    rotate_keys: Bool,
) raises -> List[Float32]:
    var effective = _validate(
        len(q),
        len(keys),
        len(values),
        first_visible,
        stored,
        num_q_heads,
        num_kv_heads,
        head_dim,
        position,
        rope_base,
    )
    var group = num_q_heads // num_kv_heads
    var row = num_kv_heads * head_dim

    # RoPE (sliding layers only; ROPE_NONE leaves both raw).
    # Q: every Q head at the query position. K: every KV head
    # of every visible row at that row's absolute position.
    var q_rot: List[Float32]
    var k_rot: List[Float32]
    if rope_base != ROPE_NONE:
        var q_pos = List[Int](capacity=num_q_heads)
        q_pos.resize(num_q_heads, position)
        q_rot = rope_cpu(q, q_pos, head_dim, rope_base)
        if rotate_keys:
            var k_pos = List[Int](capacity=stored * num_kv_heads)
            k_pos.resize(stored * num_kv_heads, 0)
            for i in range(stored):
                for kv in range(num_kv_heads):
                    k_pos[i * num_kv_heads + kv] = first_visible + i
            k_rot = rope_cpu(keys, k_pos, head_dim, rope_base)
        else:
            k_rot = keys.copy()
    else:
        q_rot = q.copy()
        k_rot = keys.copy()

    var scale = Float64(1.0) / sqrt(Float64(head_dim))
    var out = List[Float32](capacity=num_q_heads * head_dim)
    out.resize(num_q_heads * head_dim, Float32(0.0))
    for h in range(num_q_heads):
        var kv_head = h // group
        var q_off = h * head_dim
        var scores = List[Float32](capacity=effective)
        scores.resize(effective, Float32(0.0))
        for i in range(effective):
            var k_off = i * row + kv_head * head_dim
            var dot = Float64(0.0)
            for d in range(head_dim):
                dot += Float64(q_rot[q_off + d]) * Float64(k_rot[k_off + d])
            scores[i] = Float32(dot * scale)
        var probs = softmax_cpu(scores, effective)
        var o_off = h * head_dim
        for d in range(head_dim):
            var acc = Float64(0.0)
            for i in range(effective):
                var v_off = i * row + kv_head * head_dim
                acc += Float64(probs[i]) * Float64(values[v_off + d])
            out[o_off + d] = Float32(acc)
    _ = log_debug(
        "attention.cpu",
        "q_heads "
        + String(num_q_heads)
        + " kv_heads "
        + String(num_kv_heads)
        + " head_dim "
        + String(head_dim)
        + " visible "
        + String(effective)
        + " position "
        + String(position)
        + " rope "
        + String(rope_base),
    )
    return out^


def attention_cpu(
    q: List[Float32],
    keys: List[Float32],
    values: List[Float32],
    first_visible: Int,
    stored: Int,
    num_q_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    position: Int,
    rope_base: Float32,
) raises -> List[Float32]:
    """Single-position GQA attention over RAW flat K/V stores.

    q is one position's raw query row [num_q_heads][head_dim];
    keys/values are the layer's visible rows in visible order
    (see the module header for the seam and the RoPE-at-read-
    time note). rope_base is the interleaved RoPE base for
    sliding layers, or ROPE_NONE (0.0) for global/NoPE layers.
    Returns [num_q_heads][head_dim].
    """
    return _attention_impl(
        q,
        keys,
        values,
        first_visible,
        stored,
        num_q_heads,
        num_kv_heads,
        head_dim,
        position,
        rope_base,
        True,
    )


def attention_cpu_rotated_cache(
    q: List[Float32],
    keys: List[Float32],
    values: List[Float32],
    first_visible: Int,
    stored: Int,
    num_q_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    position: Int,
    rope_base: Float32,
) raises -> List[Float32]:
    """Same as attention_cpu, for Rust-exact storage: keys are
    ALREADY rotated (the cache holds rotated K, as in the Rust
    CPU path), so only the raw Q row is rotated here. Never
    pass raw keys to this variant: K would stay unrotated."""
    return _attention_impl(
        q,
        keys,
        values,
        first_visible,
        stored,
        num_q_heads,
        num_kv_heads,
        head_dim,
        position,
        rope_base,
        False,
    )


def attention_prefill_cpu(
    q_rows: List[Float32],
    k_rows: List[Float32],
    v_rows: List[Float32],
    num_positions: Int,
    num_q_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    first_position: Int,
    window: Int,
    rope_base: Float32,
) raises -> List[Float32]:
    """Multi-position prefill: the Rust session's semantics.

    The Rust CPU path is append-then-attend per position
    (rotate, append, attend), so position p sees exactly the
    rows appended up to and including itself, window-truncated
    on windowed layers (a key at kp is visible to p when
    kp <= p and kp > p - window, crates/attention attends()).
    This helper mirrors that loop over raw per-position rows:
    q_rows/k_rows/v_rows are [num_positions][width] with the
    per-position layouts of attention_cpu; window is 0 for
    global layers (no truncation) or the sliding window size.
    Later rows are never passed to an earlier position's call,
    so causality holds by construction, not by masking.
    Returns [num_positions][num_q_heads][head_dim].
    """
    if num_positions <= 0:
        raise Error(
            "attention_cpu_bad_shape: num_positions must be > 0, got "
            + String(num_positions)
        )
    if window < 0:
        raise Error(
            "attention_cpu_bad_shape: window must be >= 0 (0 = global), got "
            + String(window)
        )
    if first_position < 0:
        raise Error(
            "attention_cpu_bad_position: first_position must be >= 0, got "
            + String(first_position)
        )
    var q_width = num_q_heads * head_dim
    var kv_width = num_kv_heads * head_dim
    if (
        len(q_rows) != num_positions * q_width
        or len(k_rows) != num_positions * kv_width
        or len(v_rows) != num_positions * kv_width
    ):
        raise Error(
            "attention_cpu_bad_shape: prefill rows have "
            + String(len(q_rows))
            + "/"
            + String(len(k_rows))
            + "/"
            + String(len(v_rows))
            + " elements for "
            + String(num_positions)
            + " positions"
        )
    var out = List[Float32](capacity=num_positions * q_width)
    out.resize(num_positions * q_width, Float32(0.0))
    for idx in range(num_positions):
        var start_idx = 0
        if window > 0 and idx - window + 1 > 0:
            start_idx = idx - window + 1
        var count = idx - start_idx + 1
        var q_row = List[Float32](capacity=q_width)
        q_row.resize(q_width, Float32(0.0))
        for i in range(q_width):
            q_row[i] = q_rows[idx * q_width + i]
        var k_sub = List[Float32](capacity=count * kv_width)
        k_sub.resize(count * kv_width, Float32(0.0))
        var v_sub = List[Float32](capacity=count * kv_width)
        v_sub.resize(count * kv_width, Float32(0.0))
        for r in range(count):
            for i in range(kv_width):
                k_sub[r * kv_width + i] = k_rows[(start_idx + r) * kv_width + i]
                v_sub[r * kv_width + i] = v_rows[(start_idx + r) * kv_width + i]
        var row_out = attention_cpu(
            q_row,
            k_sub,
            v_sub,
            first_position + start_idx,
            count,
            num_q_heads,
            num_kv_heads,
            head_dim,
            first_position + idx,
            rope_base,
        )
        for i in range(q_width):
            out[idx * q_width + i] = row_out[i]
    _ = log_debug(
        "attention.prefill",
        "positions "
        + String(num_positions)
        + " first_position "
        + String(first_position)
        + " window "
        + String(window),
    )
    return out^


def attention_kernel(
    q: Pointer[Float32, MutAnyOrigin],
    keys: Pointer[Float32, MutAnyOrigin],
    values: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    stored: Int32,
    num_kv_heads: Int32,
    head_dim: Int32,
    group: Int32,
    scale: Float32,
):
    """GPU single-position attention: one block per Q head.

    Buffers are PRE-ROTATED Q/K (the Rust gqa_attention core's
    contract; rotation is the caller's separate rope step) in
    the same visible-order flat layout as the CPU seam. Flash-
    style online softmax: a single streaming pass over the
    visible positions with a running max/sum and per-channel
    accumulators; the score row is never materialized. Host
    validates shapes before launch: block_dim must equal
    ATTN_THREADS and head_dim must be <= ATTN_THREADS (each
    thread owns one channel).
    """
    var head = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var hd = Int(head_dim)
    var kv_head = head // Int(group)
    var row = Int(num_kv_heads) * hd
    var shared = stack_allocation[
        ATTN_THREADS, Float32, address_space=AddressSpace.SHARED
    ]()
    var d = tid
    var q_d = q[unsafe_offset=head * hd + d] if d < hd else Float32(0.0)
    var acc = Float32(0.0)
    var run_max = Float32(-3.4028235e38)
    var run_sum = Float32(0.0)
    for vi in range(Int(stored)):
        var k_off = vi * row + kv_head * hd
        var prod = q_d * keys[unsafe_offset=k_off + d] if d < hd else Float32(
            0.0
        )
        shared[unsafe_offset=tid] = prod
        barrier()
        var stride = ATTN_THREADS // 2
        while stride > 0:
            if tid < stride:
                shared[unsafe_offset=tid] = (
                    shared[unsafe_offset=tid]
                    + shared[unsafe_offset=tid + stride]
                )
            barrier()
            stride //= 2
        var score = shared[unsafe_offset=0] * scale
        var new_max = run_max if run_max > score else score
        var alpha = exp(run_max - new_max)
        var w = exp(score - new_max)
        var v_d = values[unsafe_offset=k_off + d] if d < hd else Float32(0.0)
        acc = acc * alpha + w * v_d
        run_sum = run_sum * alpha + w
        run_max = new_max
        barrier()
    if d < hd:
        y[unsafe_offset=head * hd + d] = acc / run_sum


# ---------------------------------------------------------------------------
# Wave D (Lane D2): shared-memory K/V tiling + split-KV.
#
# Constants: the tiled kernel is compiled for tile sizes
# 8/16/32/64 positions (attn_validate_tile is the host-side
# gate) and head_dim up to ATTN_MAX_HEAD_DIM (= ATTN_THREADS,
# one thread per channel, the Wave B kernel's same limit).
# ATTN_NEG_INF is the online-softmax sentinel: the largest
# finite Float32 magnitude, negated, so sentinel arithmetic
# (exp of a huge negative difference) underflows to 0
# instead of producing NaN; the merge additionally guards
# empty partials (sum == 0) explicitly.
# ---------------------------------------------------------------------------

comptime ATTN_MAX_HEAD_DIM = 128
comptime ATTN_NEG_INF = Float32(-3.4028235e38)


def attn_validate_tile(tile: Int) raises:
    """Refuse by name a tile size with no compiled kernel
    instantiation (8/16/32/64 positions)."""
    if tile != 8 and tile != 16 and tile != 32 and tile != 64:
        raise Error(
            "attention_tiled_bad_tile: tile positions must be one of"
            " 8/16/32/64 (the compiled instantiations), got "
            + String(tile)
        )


def attn_split_count(stored: Int, span: Int, max_splits: Int) raises -> Int:
    """How many split-KV partials a stored count earns: 1
    while stored fits one span, else ceil(stored / span)
    capped at max_splits. Pure; the chooser gpu_layer uses."""
    if stored <= 0 or span <= 0 or max_splits <= 0:
        raise Error(
            "attention_split_bad_config: stored, span and max_splits"
            " must be positive, got stored "
            + String(stored)
            + " span "
            + String(span)
            + " max_splits "
            + String(max_splits)
        )
    if stored <= span:
        return 1
    var n = (stored + span - 1) // span
    if n > max_splits:
        n = max_splits
    return n


def attn_split_range(
    stored: Int, splits: Int, split: Int
) raises -> Tuple[Int, Int]:
    """(start, length) of one split's position range: ranges
    partition [0, stored) in order, lengths differ by at most
    one, earlier splits take the remainder. splits > stored
    is legal: trailing splits are empty (length 0) and their
    partials merge as no-ops. Pure; shared by the CPU twin,
    the host launcher, and the tests (the kernel recomputes
    the same arithmetic inline)."""
    if stored < 0 or splits <= 0 or split < 0 or split >= splits:
        raise Error(
            "attention_split_bad_range: stored "
            + String(stored)
            + " splits "
            + String(splits)
            + " split "
            + String(split)
            + " is not a valid split range request"
        )
    var base = stored // splits
    var rem = stored % splits
    if split < rem:
        return (split * (base + 1), base + 1)
    return (rem * (base + 1) + (split - rem) * base, base)


def _validate_split(
    q_len: Int,
    keys_len: Int,
    values_len: Int,
    stored: Int,
    num_q_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    splits: Int,
) raises:
    """Shape gate for the split path over PRE-ROTATED stores;
    refusal names are the attention_split_ family."""
    if head_dim < 2 or head_dim % 2 != 0 or head_dim > ATTN_MAX_HEAD_DIM:
        raise Error(
            "attention_split_bad_shape: head_dim must be even, >= 2 and <= "
            + String(ATTN_MAX_HEAD_DIM)
            + ", got "
            + String(head_dim)
        )
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise Error(
            "attention_split_bad_shape: head counts must be > 0, got "
            + String(num_q_heads)
            + " q heads, "
            + String(num_kv_heads)
            + " kv heads"
        )
    if num_q_heads % num_kv_heads != 0:
        raise Error(
            "attention_split_bad_shape: "
            + String(num_q_heads)
            + " q heads do not divide into "
            + String(num_kv_heads)
            + " kv heads"
        )
    if splits <= 0:
        raise Error(
            "attention_split_bad_splits: splits must be > 0, got "
            + String(splits)
        )
    if stored <= 0:
        raise Error(
            "attention_split_bad_shape: stored must be > 0, got "
            + String(stored)
        )
    if q_len != num_q_heads * head_dim:
        raise Error(
            "attention_split_bad_shape: q has "
            + String(q_len)
            + " elements, expected "
            + String(num_q_heads * head_dim)
        )
    var row = num_kv_heads * head_dim
    if keys_len != stored * row or values_len != stored * row:
        raise Error(
            "attention_split_bad_shape: keys/values have "
            + String(keys_len)
            + "/"
            + String(values_len)
            + " elements, expected "
            + String(stored * row)
            + " each ([stored][num_kv_heads][head_dim])"
        )


def attention_split_partials_cpu(
    q: List[Float32],
    keys: List[Float32],
    values: List[Float32],
    stored: Int,
    num_q_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    splits: Int,
    tile: Int,
) raises -> Tuple[List[Float32], List[Float32], List[Float32]]:
    """CPU twin of attention_tiled_kernel: the partials.

    q/keys are PRE-ROTATED (the GPU kernels' contract);
    values raw, all in the flat visible-order layout. Returns
    (partial_out, partial_max, partial_sum) with layouts
    [splits][num_q_heads][head_dim] and [splits][num_q_heads]:
    partial_out is the UNNORMALIZED online-softmax
    accumulator (divide by partial_sum after merging - never
    before). The Float32 arithmetic mirrors the kernel tile
    by tile: per tile, scores from ascending-dim dots, the
    tile max folded into the running max, one rescale by
    exp(old max - new max) (0 on the first tile), probs
    exp(score - new max), accumulate, sum. Mirroring the tile
    structure matters: the rescale points are part of the
    float path, so a different tile gives a different (still
    in-bar) rounding, exactly as on device.
    """
    _validate_split(
        len(q),
        len(keys),
        len(values),
        stored,
        num_q_heads,
        num_kv_heads,
        head_dim,
        splits,
    )
    attn_validate_tile(tile)
    var group = num_q_heads // num_kv_heads
    var row = num_kv_heads * head_dim
    var scale = Float32(1.0) / sqrt(Float32(head_dim))
    var p_out = List[Float32](capacity=splits * num_q_heads * head_dim)
    p_out.resize(splits * num_q_heads * head_dim, Float32(0.0))
    var p_max = List[Float32](capacity=splits * num_q_heads)
    p_max.resize(splits * num_q_heads, ATTN_NEG_INF)
    var p_sum = List[Float32](capacity=splits * num_q_heads)
    p_sum.resize(splits * num_q_heads, Float32(0.0))
    for h in range(num_q_heads):
        var kv_head = h // group
        for s in range(splits):
            var rng = attn_split_range(stored, splits, s)
            var start = rng[0]
            var end = start + rng[1]
            var acc = List[Float32](capacity=head_dim)
            acc.resize(head_dim, Float32(0.0))
            var run_max = ATTN_NEG_INF
            var run_sum = Float32(0.0)
            var ts = start
            while ts < end:
                var tcount = tile
                if end - ts < tcount:
                    tcount = end - ts
                var scores = List[Float32](capacity=tcount)
                scores.resize(tcount, Float32(0.0))
                for t in range(tcount):
                    var k_off = (ts + t) * row + kv_head * head_dim
                    var dot = Float32(0.0)
                    for d in range(head_dim):
                        dot += q[h * head_dim + d] * keys[k_off + d]
                    scores[t] = dot * scale
                var new_max = run_max
                for t in range(tcount):
                    if scores[t] > new_max:
                        new_max = scores[t]
                var alpha = Float32(0.0)
                if run_sum != Float32(0.0):
                    alpha = exp(run_max - new_max)
                var probs = List[Float32](capacity=tcount)
                probs.resize(tcount, Float32(0.0))
                var psum = Float32(0.0)
                for t in range(tcount):
                    probs[t] = exp(scores[t] - new_max)
                    psum += probs[t]
                for d in range(head_dim):
                    var a = acc[d] * alpha
                    for t in range(tcount):
                        var v_off = (ts + t) * row + kv_head * head_dim
                        a += probs[t] * values[v_off + d]
                    acc[d] = a
                run_sum = run_sum * alpha + psum
                run_max = new_max
                ts += tile
            var base = (s * num_q_heads + h) * head_dim
            for d in range(head_dim):
                p_out[base + d] = acc[d]
            p_max[s * num_q_heads + h] = run_max
            p_sum[s * num_q_heads + h] = run_sum
    _ = log_debug(
        "attention.split",
        "partials q_heads "
        + String(num_q_heads)
        + " stored "
        + String(stored)
        + " splits "
        + String(splits)
        + " tile "
        + String(tile),
    )
    return (p_out^, p_max^, p_sum^)


def attention_merge_partials_cpu(
    p_out: List[Float32],
    p_max: List[Float32],
    p_sum: List[Float32],
    splits: Int,
    num_q_heads: Int,
    head_dim: Int,
) raises -> List[Float32]:
    """CPU twin of attention_merge_kernel: combine partials
    in split order by exact logsumexp rescaling.

    For each head: walk the splits; a split with sum 0 is
    empty and skipped; the first non-empty split is adopted
    whole; every later split rescales BOTH accumulators by
    exp(max - combined max) before adding, so the merged
    (acc, sum) is the single-pass online-softmax state for
    the union of the ranges, and the output is acc / sum.
    No approximation: the rescale factors are the exact
    exponentials of the max differences.
    """
    if splits <= 0:
        raise Error(
            "attention_split_bad_splits: splits must be > 0, got "
            + String(splits)
        )
    if num_q_heads <= 0 or head_dim <= 0:
        raise Error(
            "attention_split_bad_shape: head counts must be > 0, got "
            + String(num_q_heads)
            + " q heads, head_dim "
            + String(head_dim)
        )
    if (
        len(p_out) != splits * num_q_heads * head_dim
        or len(p_max) != splits * num_q_heads
        or len(p_sum) != splits * num_q_heads
    ):
        raise Error(
            "attention_split_bad_shape: partial buffers have "
            + String(len(p_out))
            + "/"
            + String(len(p_max))
            + "/"
            + String(len(p_sum))
            + " elements for splits "
            + String(splits)
            + " q_heads "
            + String(num_q_heads)
            + " head_dim "
            + String(head_dim)
        )
    var out = List[Float32](capacity=num_q_heads * head_dim)
    out.resize(num_q_heads * head_dim, Float32(0.0))
    for h in range(num_q_heads):
        var acc = List[Float32](capacity=head_dim)
        acc.resize(head_dim, Float32(0.0))
        var run_max = ATTN_NEG_INF
        var run_sum = Float32(0.0)
        for s in range(splits):
            var ps = p_sum[s * num_q_heads + h]
            if ps == Float32(0.0):
                continue
            var pm = p_max[s * num_q_heads + h]
            var base = (s * num_q_heads + h) * head_dim
            if run_sum == Float32(0.0):
                run_max = pm
                run_sum = ps
                for d in range(head_dim):
                    acc[d] = p_out[base + d]
            else:
                var new_max = run_max if run_max > pm else pm
                var a1 = exp(run_max - new_max)
                var a2 = exp(pm - new_max)
                for d in range(head_dim):
                    acc[d] = acc[d] * a1 + p_out[base + d] * a2
                run_sum = run_sum * a1 + ps * a2
                run_max = new_max
        if run_sum == Float32(0.0):
            raise Error(
                "attention_split_empty_partials: every split partial for head "
                + String(h)
                + " is empty (all sums zero)"
            )
        for d in range(head_dim):
            out[h * head_dim + d] = acc[d] / run_sum
    return out^


def attention_split_cpu(
    q: List[Float32],
    keys: List[Float32],
    values: List[Float32],
    stored: Int,
    num_q_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    splits: Int,
    tile: Int,
) raises -> List[Float32]:
    """Single-position GQA attention over PRE-ROTATED flat
    stores, computed as tiled partials + exact merge: the
    full CPU twin of the GPU tiled + merge launch pair."""
    var parts = attention_split_partials_cpu(
        q,
        keys,
        values,
        stored,
        num_q_heads,
        num_kv_heads,
        head_dim,
        splits,
        tile,
    )
    return attention_merge_partials_cpu(
        parts[0], parts[1], parts[2], splits, num_q_heads, head_dim
    )


def attention_direct_cpu(
    q: List[Float32],
    keys: List[Float32],
    values: List[Float32],
    stored: Int,
    num_q_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    tile: Int,
) raises -> List[Float32]:
    """Single-position GQA attention over PRE-ROTATED flat
    stores, computed as ONE tiled pass with a direct
    normalized write: the CPU twin of
    attention_tiled_direct_kernel (Wave E, Lane E2).

    The splits==1 shape of the split path with the merge
    step folded in: the single partial's accumulator is
    divided by its own sum per head, which is exactly what
    attention_merge_partials_cpu does when it adopts the
    one non-empty partial whole - so this twin's float path
    is the partials path tile by tile plus that final
    divide, and its outputs match attention_split_cpu at
    splits == 1 to the last bit on the same fixtures (the
    VM suite asserts exactly that)."""
    var parts = attention_split_partials_cpu(
        q, keys, values, stored, num_q_heads, num_kv_heads, head_dim, 1, tile
    )
    var out = List[Float32](capacity=num_q_heads * head_dim)
    out.resize(num_q_heads * head_dim, Float32(0.0))
    for h in range(num_q_heads):
        var run_sum = parts[2][h]
        if run_sum == Float32(0.0):
            raise Error(
                "attention_split_empty_partials: the single direct"
                " partial for head "
                + String(h)
                + " is empty (sum zero)"
            )
        for d in range(head_dim):
            out[h * head_dim + d] = parts[0][h * head_dim + d] / run_sum
    return out^


def attention_tiled_kernel[
    TILE: Int
](
    q: Pointer[Float32, MutAnyOrigin],
    keys: Pointer[Float32, MutAnyOrigin],
    values: Pointer[Float32, MutAnyOrigin],
    p_out: Pointer[Float32, MutAnyOrigin],
    p_max: Pointer[Float32, MutAnyOrigin],
    p_sum: Pointer[Float32, MutAnyOrigin],
    stored: Int32,
    num_kv_heads: Int32,
    num_q_heads: Int32,
    head_dim: Int32,
    group: Int32,
    scale: Float32,
    splits: Int32,
):
    """GPU tiled split-KV attention: partials, one block per
    (Q head, split).

    Buffers are PRE-ROTATED Q/K in the visible-order flat
    layout (the Wave B kernel's contract). Block index x is
    head * splits + split; the block owns attn_split_range's
    range for its split and writes its unnormalized partial
    (acc, max, sum) for attention_merge_kernel to combine.
    Online softmax stays (no materialized score row; scores
    live in a TILE-sized shared array). Per tile: K is staged
    into the shared tile buffer cooperatively, scores are
    computed from shared (one thread per position, ascending
    dim), the tile max folds into the running max with one
    rescale, then V replaces K in the same buffer and the
    per-channel accumulators update from shared. Host
    validates before launch: block_dim == ATTN_THREADS,
    head_dim <= ATTN_MAX_HEAD_DIM, TILE a compiled size.
    """
    var n_splits = Int(splits)
    var head = Int(block_idx.x) // n_splits
    var split = Int(block_idx.x) % n_splits
    var tid = Int(thread_idx.x)
    var hd = Int(head_dim)
    var kv_head = head // Int(group)
    var row = Int(num_kv_heads) * hd
    var nstored = Int(stored)
    var base_chunk = nstored // n_splits
    var rem = nstored % n_splits
    var start: Int
    var length: Int
    if split < rem:
        start = split * (base_chunk + 1)
        length = base_chunk + 1
    else:
        start = rem * (base_chunk + 1) + (split - rem) * base_chunk
        length = base_chunk
    var end = start + length
    var tile_buf = stack_allocation[
        TILE * ATTN_MAX_HEAD_DIM, Float32, address_space=AddressSpace.SHARED
    ]()
    var scores = stack_allocation[
        TILE, Float32, address_space=AddressSpace.SHARED
    ]()
    var q_sh = stack_allocation[
        ATTN_MAX_HEAD_DIM, Float32, address_space=AddressSpace.SHARED
    ]()
    var red = stack_allocation[
        ATTN_THREADS, Float32, address_space=AddressSpace.SHARED
    ]()
    if tid < hd:
        q_sh[unsafe_offset=tid] = q[unsafe_offset=head * hd + tid]
    barrier()
    var d = tid
    var acc = Float32(0.0)
    var run_max = ATTN_NEG_INF
    var run_sum = Float32(0.0)
    var tile_start = start
    while tile_start < end:
        var tcount = TILE
        if end - tile_start < tcount:
            tcount = end - tile_start
        # Stage the K tile: element e is (t, dd) row-major.
        var e = tid
        while e < tcount * hd:
            var t = e // hd
            var dd = e % hd
            tile_buf[unsafe_offset=e] = keys[
                unsafe_offset=(tile_start + t) * row + kv_head * hd + dd
            ]
            e += ATTN_THREADS
        barrier()
        # Scores from shared: one thread per tile position.
        if tid < tcount:
            var s = Float32(0.0)
            for dd2 in range(hd):
                s += (
                    q_sh[unsafe_offset=dd2]
                    * tile_buf[unsafe_offset=tid * hd + dd2]
                )
            scores[unsafe_offset=tid] = s * scale
        barrier()
        if tid == 0:
            var m = run_max
            for t2 in range(tcount):
                if scores[unsafe_offset=t2] > m:
                    m = scores[unsafe_offset=t2]
            red[unsafe_offset=0] = m
        barrier()
        var new_max = red[unsafe_offset=0]
        var alpha = Float32(0.0)
        if run_sum != Float32(0.0):
            alpha = exp(run_max - new_max)
        if tid < tcount:
            scores[unsafe_offset=tid] = exp(scores[unsafe_offset=tid] - new_max)
        barrier()
        # Stage the V tile into the same shared buffer (the K
        # tile is dead once every score thread has read it,
        # which the barrier above guarantees).
        e = tid
        while e < tcount * hd:
            var t3 = e // hd
            var dd3 = e % hd
            tile_buf[unsafe_offset=e] = values[
                unsafe_offset=(tile_start + t3) * row + kv_head * hd + dd3
            ]
            e += ATTN_THREADS
        barrier()
        if tid == 0:
            var psum = Float32(0.0)
            for t4 in range(tcount):
                psum += scores[unsafe_offset=t4]
            red[unsafe_offset=1] = psum
        if d < hd:
            var a = acc * alpha
            for t5 in range(tcount):
                a += (
                    scores[unsafe_offset=t5]
                    * tile_buf[unsafe_offset=t5 * hd + d]
                )
            acc = a
        barrier()
        run_sum = run_sum * alpha + red[unsafe_offset=1]
        run_max = new_max
        tile_start += TILE
    var p_base = (split * Int(num_q_heads) + head) * hd
    if d < hd:
        p_out[unsafe_offset=p_base + d] = acc
    if tid == 0:
        p_max[unsafe_offset=split * Int(num_q_heads) + head] = run_max
        p_sum[unsafe_offset=split * Int(num_q_heads) + head] = run_sum


def attention_tiled_direct_kernel[
    TILE: Int
](
    q: Pointer[Float32, MutAnyOrigin],
    keys: Pointer[Float32, MutAnyOrigin],
    values: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    stored: Int32,
    num_kv_heads: Int32,
    num_q_heads: Int32,
    head_dim: Int32,
    group: Int32,
    scale: Float32,
):
    """GPU tiled attention, DIRECT-OUTPUT form (Wave E,
    Lane E2): the splits == 1 case of attention_tiled_kernel
    with the merge folded into the same launch.

    One block per Q head over the whole visible range; the
    tile loop, shared-memory staging, and online-softmax
    rescale points are the tiled kernel's, statement for
    statement, so the float path is the splits==1 partials
    path plus the merge's final divide by the head's own
    sum - no approximation, and no partial buffers or
    second (merge) launch. Buffers are PRE-ROTATED Q/K in
    the visible-order flat layout, as everywhere in this
    module. Host validates before launch: block_dim ==
    ATTN_THREADS, head_dim <= ATTN_MAX_HEAD_DIM, TILE a
    compiled size, and the caller launches this form only
    when its split chooser returned 1."""
    var head = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var hd = Int(head_dim)
    var kv_head = head // Int(group)
    var row = Int(num_kv_heads) * hd
    var nstored = Int(stored)
    var tile_buf = stack_allocation[
        TILE * ATTN_MAX_HEAD_DIM, Float32, address_space=AddressSpace.SHARED
    ]()
    var scores = stack_allocation[
        TILE, Float32, address_space=AddressSpace.SHARED
    ]()
    var q_sh = stack_allocation[
        ATTN_MAX_HEAD_DIM, Float32, address_space=AddressSpace.SHARED
    ]()
    var red = stack_allocation[
        ATTN_THREADS, Float32, address_space=AddressSpace.SHARED
    ]()
    if tid < hd:
        q_sh[unsafe_offset=tid] = q[unsafe_offset=head * hd + tid]
    barrier()
    var d = tid
    var acc = Float32(0.0)
    var run_max = ATTN_NEG_INF
    var run_sum = Float32(0.0)
    var tile_start = 0
    while tile_start < nstored:
        var tcount = TILE
        if nstored - tile_start < tcount:
            tcount = nstored - tile_start
        var e = tid
        while e < tcount * hd:
            var t = e // hd
            var dd = e % hd
            tile_buf[unsafe_offset=e] = keys[
                unsafe_offset=(tile_start + t) * row + kv_head * hd + dd
            ]
            e += ATTN_THREADS
        barrier()
        if tid < tcount:
            var sc = Float32(0.0)
            for dd2 in range(hd):
                sc += (
                    q_sh[unsafe_offset=dd2]
                    * tile_buf[unsafe_offset=tid * hd + dd2]
                )
            scores[unsafe_offset=tid] = sc * scale
        barrier()
        if tid == 0:
            var m = run_max
            for t2 in range(tcount):
                if scores[unsafe_offset=t2] > m:
                    m = scores[unsafe_offset=t2]
            red[unsafe_offset=0] = m
        barrier()
        var new_max = red[unsafe_offset=0]
        var alpha = Float32(0.0)
        if run_sum != Float32(0.0):
            alpha = exp(run_max - new_max)
        if tid < tcount:
            scores[unsafe_offset=tid] = exp(scores[unsafe_offset=tid] - new_max)
        barrier()
        e = tid
        while e < tcount * hd:
            var t3 = e // hd
            var dd3 = e % hd
            tile_buf[unsafe_offset=e] = values[
                unsafe_offset=(tile_start + t3) * row + kv_head * hd + dd3
            ]
            e += ATTN_THREADS
        barrier()
        if tid == 0:
            var psum = Float32(0.0)
            for t4 in range(tcount):
                psum += scores[unsafe_offset=t4]
            red[unsafe_offset=1] = psum
        if d < hd:
            var a = acc * alpha
            for t5 in range(tcount):
                a += (
                    scores[unsafe_offset=t5]
                    * tile_buf[unsafe_offset=t5 * hd + d]
                )
            acc = a
        barrier()
        run_sum = run_sum * alpha + red[unsafe_offset=1]
        run_max = new_max
        tile_start += TILE
    if d < hd:
        y[unsafe_offset=head * hd + d] = acc / run_sum


def attention_merge_kernel(
    p_out: Pointer[Float32, MutAnyOrigin],
    p_max: Pointer[Float32, MutAnyOrigin],
    p_sum: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    num_q_heads: Int32,
    head_dim: Int32,
    splits: Int32,
):
    """GPU split-KV merge: one block per Q head, one thread
    per channel, splits combined in order with exact
    logsumexp rescaling (exp of max differences), mirroring
    attention_merge_partials_cpu statement for statement,
    including the empty-partial (sum == 0) skip and the
    first-non-empty adoption. Output y[head][d] = acc / sum.
    Host guarantees at least one non-empty partial per head
    (stored >= 1 upstream)."""
    var head = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var hd = Int(head_dim)
    var qh = Int(num_q_heads)
    var d = tid
    var acc = Float32(0.0)
    var run_max = ATTN_NEG_INF
    var run_sum = Float32(0.0)
    for s in range(Int(splits)):
        var ps = p_sum[unsafe_offset=s * qh + head]
        if ps != Float32(0.0):
            var pm = p_max[unsafe_offset=s * qh + head]
            var base = (s * qh + head) * hd
            if run_sum == Float32(0.0):
                run_max = pm
                run_sum = ps
                if d < hd:
                    acc = p_out[unsafe_offset=base + d]
            else:
                var new_max = run_max if run_max > pm else pm
                var a1 = exp(run_max - new_max)
                var a2 = exp(pm - new_max)
                if d < hd:
                    acc = acc * a1 + p_out[unsafe_offset=base + d] * a2
                run_sum = run_sum * a1 + ps * a2
                run_max = new_max
    if d < hd:
        y[unsafe_offset=head * hd + d] = acc / run_sum
