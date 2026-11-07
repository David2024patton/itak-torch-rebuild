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
# tree reduction. What is still OWED, recorded plainly:
# shared-memory staging of K/V tiles and split-KV parallelism
# across position blocks (the block-per-head grid leaves the
# card underfilled at 32 blocks for very long contexts);
# both are tuning work after the correctness wave, and the
# current shape was chosen because it is the flash-decoding
# form whose online-softmax arithmetic is provable against
# the CPU twin at the 1e-5 class bar today.
#
# Named refusals: attention_cpu_bad_shape,
# attention_cpu_nondivisible_heads,
# attention_cpu_no_visible_keys, attention_cpu_bad_position,
# attention_cpu_bad_rope_base.

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
