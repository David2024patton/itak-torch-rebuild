# gpu_layer.mojo - host-side GPU launch wrappers and the GPU
# layer step (Lane C2).
#
# =====================================================================
# PLACEHOLDER / STAND-IN (Lane C3, 2026-11-07). This file exists
# in the lane/mojo-c3-gen worktree ONLY so Lane C3's
# src/model/gpu_model.mojo compiles against the Wave C contract
# (mojo-engine/WAVE_C_INTERFACES.md) before Lane C2 lands.
# DELETE AT INTEGRATION: the coordinator replaces this whole
# file with Lane C2's real src/model/gpu_layer.mojo. Do not
# ship this stand-in as the C2 implementation: both functions
# below refuse by name (gpu_layer_stand_in / gpu_logits_stand_in)
# and launch NO device kernel, so no GPU result may ever be
# claimed through them.
# =====================================================================
#
# Contract surface coded against (WAVE_C_INTERFACES.md, Lane C2):
#   gpu_logits(rt, weights, hidden) -> List[Float32]:
#     final rms_norm (rms_norm_kernel, grid = 1 row) + the head
#     GEMV dispatched by the output slot's dtype (Q5_K on
#     Glimmer: q5k_gemv_kernel over 202,048 rows) +
#     final_logit_kernel (scale-then-softcap). Same math as
#     GlimmerModel.logits, device kernels.
#   gpu_layer_step(rt, weights, kv, cpu_model, x, layer,
#     position) -> List[Float32]:
#     statement-for-statement the CPU layer_step, with every
#     projection and op launched on device.
#
# Spelling choices where the contract leaves them open (the
# coordinator reconciles at integration): rt and weights are
# borrowed (the device handles inside them are used through
# borrowed references, as MAX's DeviceContext allows); kv is
# mut because the step appends to it; cpu_model is borrowed
# for config and the F32 norm weight values.

from diag import log_debug
from gpu.context import GpuRuntime
from gpu.device_kv import DeviceKvCache
from gpu.weights import DeviceWeights
from model.model import GlimmerModel


def gpu_logits(
    rt: GpuRuntime, weights: DeviceWeights, hidden: List[Float32]
) raises -> List[Float32]:
    """PLACEHOLDER: refuses by name; C2's real gpu_logits
    launches the final norm, head GEMV, and final_logit kernels.
    """
    _ = rt
    _ = weights
    _ = hidden
    _ = log_debug("gpu_model", "gpu_logits stand-in called")
    raise Error(
        "gpu_logits_stand_in: Lane C2 gpu_logits is not integrated in"
        " this worktree (placeholder in src/model/gpu_layer.mojo)"
    )


def gpu_layer_step(
    rt: GpuRuntime,
    weights: DeviceWeights,
    mut kv: DeviceKvCache,
    cpu_model: GlimmerModel,
    x: List[Float32],
    layer: Int,
    position: Int,
) raises -> List[Float32]:
    """PLACEHOLDER: refuses by name; C2's real gpu_layer_step
    launches the whole layer on device.
    """
    _ = rt
    _ = weights
    _ = kv
    _ = cpu_model
    _ = x
    _ = layer
    _ = position
    _ = log_debug("gpu_model", "gpu_layer_step stand-in called")
    raise Error(
        "gpu_layer_stand_in: Lane C2 gpu_layer_step is not integrated"
        " in this worktree (placeholder in src/model/gpu_layer.mojo)"
    )
