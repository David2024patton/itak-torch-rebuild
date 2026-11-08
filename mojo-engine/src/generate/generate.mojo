# generate.mojo - the prefill/decode generation loop, Lane B4.
#
# Semantic references (Rust, branch main):
#   crates/itak-engine/src/engine.rs generate_inner: prefill
#   once over the whole prompt, then one forward per sampled
#   token (no full re-forward anywhere). The picked token is
#   checked against EOS BEFORE it is counted or emitted:
#   EOS ends the run with finish_reason "stop" and the EOS
#   token itself is NOT in the generated output and NOT in
#   the generated count. This loop mirrors that exactly:
#   EOS is EXCLUDED from the returned ids.
#   crates/itak-spec/src/session.rs begin/generate: the same
#   prefill-then-decode boundary stated for the speculative
#   session (its stop truncation is inclusive there by the
#   spec's own Section 4 step 6; the plain engine loop in
#   engine.rs is the twin this module follows, and it
#   excludes EOS).
#
# THE SEAM (read this before integrating B3): the brief's
# first shape, a pair of function parameters
#   step(token_id, position) -> hidden, logits(hidden) -> row
# does not compile in Mojo 1.2: function values carry a
# `thin` calling convention that mismatches a declared
# fn-type parameter (probed 2026-11-07, both `fn` and `def`
# spellings), there are no capturing closures, and there are
# no mutable module globals for a mock to record calls in.
# The seam as built is therefore a small trait the model
# adapter implements as a struct:
#
#   trait ForwardModel:
#       def step(mut self, token_id: Int, position: Int)
#           raises -> List[Float32]   # one position through
#           # the model; appends to the KV cache internally;
#           # returns the final hidden row for that position
#       def logits(mut self, hidden: List[Float32])
#           raises -> List[Float32]   # finished logits (after
#           # final norm, head, final_logit transform) for one
#           # hidden row
#
# and generate() is generic over it:
#   def generate[M: ForwardModel](mut model: M, ...)
# The mock in tests/test_generate.mojo implements the trait
# on a plain struct and records every (token, position) call
# in its own fields. At integration the coordinator wraps
# B3's forward/logits_last in one such struct; nothing else
# in this module changes. logits() takes self read-only: it
# is a pure read of one hidden row. An adapter holding
# mutable state behind a handle can still implement it; if
# B3's logits truly needs a mut self, flipping this one
# signature word is the whole change.
#
# Position accounting (provable from the loop below): a
# prompt of n ids is fed at positions 0..n-1. The first
# generated id is picked from the logits of position n-1's
# hidden row and, if generation continues, is fed back
# through step() at position n. Generated id k (0-based) is
# consumed at position n + k. No id is ever fed at a
# position the prompt already used.
#
# Named refusals: generate_empty_prompt,
# generate_bad_max_tokens. Shape discipline: greedy_argmax
# refuses an empty logits row by name, and the model's own
# logits() is the shape authority for hidden rows - if it
# reports a hidden-row mismatch it raises by name and this
# loop propagates that refusal untouched (proven in the
# tests with a mock whose logits refuses a short hidden row
# as generate_hidden_shape). This loop adds no shape guess
# of its own: hidden width is B3's business, vocab width is
# whatever row logits() returns.
#
# Sizes arrive as parameters: max_tokens and eos_id are the
# caller's (config wiring is integration work). No vocab or
# context literal appears in this module.

from diag import log_debug

from generate.sampler import greedy_argmax


trait ForwardModel:
    """One-position model forward plus the head, the B4 seam.

    See the module header for why this is a trait on a struct
    and not a pair of function parameters in Mojo 1.2.
    """

    def step(mut self, token_id: Int, position: Int) raises -> List[Float32]:
        """Run token_id at position, append to the cache
        internally, return that position's final hidden row."""
        ...

    def logits(mut self, hidden: List[Float32]) raises -> List[Float32]:
        """Finished logits for one hidden row."""
        ...


def generate[
    M: ForwardModel
](
    mut model: M, prompt_ids: List[Int], max_tokens: Int, eos_id: Int
) raises -> List[Int]:
    """Greedy generation: prefill the prompt, then decode.

    Returns the generated ids, EOS excluded (see header).
    Stops when the pick is eos_id or max_tokens ids have
    been generated, whichever comes first.
    """
    if len(prompt_ids) == 0:
        raise Error("generate_empty_prompt: prompt has no token ids")
    if max_tokens <= 0:
        raise Error(
            "generate_bad_max_tokens: max_tokens must be positive, got "
            + String(max_tokens)
        )
    var n = len(prompt_ids)
    # Prefill: feed the prompt at positions 0..n-1, discarding
    # every hidden row except the last.
    var hidden = model.step(prompt_ids[0], 0)
    for i in range(1, n):
        hidden = model.step(prompt_ids[i], i)
    _ = log_debug(
        "generate.prefill",
        "prompt ids " + String(n) + " positions 0.." + String(n - 1),
    )
    var out = List[Int]()
    while True:
        var row = model.logits(hidden)
        var next_id = greedy_argmax(row)
        if next_id == eos_id:
            _ = log_debug(
                "generate.decode",
                "eos id "
                + String(next_id)
                + " picked at position "
                + String(n + len(out))
                + "; stop, eos excluded from output",
            )
            break
        out.append(next_id)
        _ = log_debug(
            "generate.decode",
            "token id "
            + String(next_id)
            + " position "
            + String(n + len(out) - 1),
        )
        if len(out) >= max_tokens:
            break
        # Feed the just-generated id back at the next
        # position: generated id k is consumed at n + k, and
        # len(out) - 1 is k for the id just appended.
        hidden = model.step(next_id, n + len(out) - 1)
    return out^
