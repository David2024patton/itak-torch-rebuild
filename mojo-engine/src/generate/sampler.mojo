# sampler.mojo - greedy token selection, Lane B4.
#
# Semantic reference (Rust, branch main):
#   crates/itak-engine/src/sampler.rs: temperature at or below
#   zero is greedy, and the greedy rule is argmax with the
#   LOWEST id on ties ("The lowest id holding the maximum
#   logit (llama.cpp's greedy tie behavior...)"). The private
#   Rust argmax scans in id order and replaces the best only
#   on a STRICTLY greater logit, which is exactly first-max.
# This module is ONLY that greedy pick: no penalties, no
# temperature, no nucleus. Those belong to a later sampler
# lane; conflating them here would change the golden greedy
# stream the Wave B contract pins.
#
# Named refusal: greedy_argmax_empty_logits (an empty logits
# row has no argmax; the Rust sampler's select() returns 0
# for empty in its stochastic path, but a silent 0 in the
# greedy engine loop would mint a fake token, so this twin
# refuses by name instead - stated here plainly as a
# deliberate divergence in the failure case only).

from diag import log_debug


def greedy_argmax(logits: List[Float32]) raises -> Int:
    """The lowest index holding the maximum logit.

    First-max tie-break, exactly the Rust sampler's greedy
    rule: scan in index order, replace the best only when a
    logit is strictly greater than the best so far.
    """
    if len(logits) == 0:
        raise Error(
            "greedy_argmax_empty_logits: argmax over an empty logits row"
        )
    var best = 0
    var best_val = logits[0]
    for i in range(1, len(logits)):
        if logits[i] > best_val:
            best_val = logits[i]
            best = i
    _ = log_debug(
        "generate.sampler",
        "greedy_argmax chose id " + String(best) + " logit " + String(best_val),
    )
    return best


def golden_first_match(got: List[Int], expected: List[Int]) -> Bool:
    """The golden checker: True only when `got` covers the
    whole expected prefix, token for token. This is the one
    comparison every golden gate uses (the CLI's
    GOLDEN_FIRST8_MATCH line and the suite's poison gates),
    so a judge defect cannot hide in a private copy of the
    loop: tests/test_poison_gates.mojo feeds this function a
    flipped 8th token and asserts it answers False."""
    if len(got) < len(expected):
        return False
    for i in range(len(expected)):
        if got[i] != expected[i]:
            return False
    return True
