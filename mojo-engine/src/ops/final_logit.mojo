# final_logit.mojo - the final-logit transform, Lane A4.
#
# This is its OWN module, separate from softmax.mojo, on purpose:
# attention softmax normalizes scores inside attention; this
# transform shapes the LM head's output logits before sampling.
# Conflating them changes every probability the sampler sees.
#
# Semantic reference (Rust, branch main):
#   crates/muse-glimmer/src/model.rs::softcap_scale - multiply
#   by the logit scale FIRST, then softcap cap * tanh(z / cap)
#   (llama.cpp muse-glimmer graph: ggml_scale, then
#   build_softcap). A cap of 0 in that function means "no cap":
#   the scale still applies and the tanh step is skipped.
#   crates/torch-math/src/logits.rs holds the two steps as
#   separate named functions (logit_scale, logit_softcap).
#   crates/muse-glimmer/src/config.rs (measured Glimmer values):
#   logit scale 0.19611613 (f32; full decimal expansion of that
#   f32 is 0.1961161345243454), softcap 20.0, both read from the
#   file (muse-glimmer.logit_scale /
#   muse-glimmer.final_logit_softcapping).
#
# scale and cap are PARAMETERS sourced from config / GGUF
# metadata, never literals in the math.
# Wave E (Lane E3): the recorded Glimmer defaults that used
# to live here as GLIMMER_LOGIT_SCALE / GLIMMER_LOGIT_SOFTCAP
# now live in src/config.mojo as DEFAULT_LOGIT_SCALE /
# DEFAULT_LOGIT_SOFTCAP (EngineConfig.logit_scale /
# .logit_softcap, env MOJO_ENGINE_LOGIT_SCALE /
# MOJO_ENGINE_LOGIT_SOFTCAP).
#
# Named refusals: final_logit_cpu_bad_scale /
# final_logit_cpu_bad_cap.

from std.math import tanh, isnan, isinf
from max.gpu import thread_idx, block_idx, block_dim

from diag import log_debug

def final_logit_cpu(
    x: List[Float32], scale: Float32, cap: Float32
) raises -> List[Float32]:
    """Scale first, then softcap (cap == 0 means no cap).

    Matches Rust softcap_scale element for element, in f32.
    """
    if isnan(scale) or isinf(scale):
        raise Error(
            "final_logit_cpu_bad_scale: scale must be finite, got "
            + String(scale)
        )
    if isnan(cap) or isinf(cap) or cap < Float32(0.0):
        raise Error(
            "final_logit_cpu_bad_cap: cap must be finite and >= 0"
            " (0 = no cap), got "
            + String(cap)
        )
    var out = List[Float32](capacity=len(x))
    out.resize(len(x), Float32(0.0))
    for i in range(len(x)):
        var z = x[i] * scale
        if cap > Float32(0.0):
            z = cap * tanh(z / cap)
        out[i] = z
    _ = log_debug("ops.final_logit.cpu", "len " + String(len(x)))
    return out^


def final_logit_kernel(
    x: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    n: Int32,
    scale: Float32,
    cap: Float32,
):
    """GPU final-logit transform: one thread per logit."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        var z = x[unsafe_offset=i] * scale
        if cap > Float32(0.0):
            z = cap * tanh(z / cap)
        y[unsafe_offset=i] = z
