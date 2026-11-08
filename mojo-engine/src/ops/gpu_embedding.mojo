# gpu_embedding.mojo - the device embedding row (Lane C2).
#
# =====================================================================
# PLACEHOLDER / STAND-IN (Lane C3, 2026-11-07). This file exists
# in the lane/mojo-c3-gen worktree ONLY so Lane C3's
# src/model/gpu_model.mojo compiles against the Wave C contract
# (mojo-engine/WAVE_C_INTERFACES.md) before Lane C2 lands.
# DELETE AT INTEGRATION: the coordinator replaces this whole
# file with Lane C2's real src/ops/gpu_embedding.mojo. Do not
# ship this stand-in as the C2 implementation: the function
# below refuses by name (gpu_embed_stand_in) and launches NO
# device kernel, so no GPU result may ever be claimed through
# it.
# =====================================================================
#
# Contract surface coded against (WAVE_C_INTERFACES.md, Lane C2):
#   gpu_embed_token(rt, weights, token) -> List[Float32]:
#     a kernel that dequantizes one row of the embedding table
#     on device for the table's actual dtype (Q4_K on Glimmer;
#     Q6_K/Q5_K/F16/F32 handled or refused by name), proven
#     against GlimmerModel.embed_token (the CPU twin).

from diag import log_debug
from gpu.context import GpuRuntime
from gpu.weights import DeviceWeights


def gpu_embed_token(
    rt: GpuRuntime, weights: DeviceWeights, token: Int
) raises -> List[Float32]:
    """PLACEHOLDER: refuses by name; C2's real gpu_embed_token
    dequantizes the embedding row on device.
    """
    _ = rt
    _ = weights
    _ = token
    _ = log_debug("gpu_model", "gpu_embed_token stand-in called")
    raise Error(
        "gpu_embed_stand_in: Lane C2 gpu_embed_token is not integrated"
        " in this worktree (placeholder in src/ops/gpu_embedding.mojo)"
    )
