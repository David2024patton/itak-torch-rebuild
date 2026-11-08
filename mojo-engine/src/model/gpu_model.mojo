# gpu_model.mojo - the Glimmer transformer forward on device,
# Lane C3 (Wave C). The GPU twin of model.mojo's CPU path.
#
# Contract: mojo-engine/WAVE_C_INTERFACES.md, Lane C3.
# GpuGlimmerModel implements the SAME ForwardModel trait
# (generate/generate.mojo) as GlimmerForwardAdapter, so B4's
# generate() drives it unmodified: prefill is the step loop,
# there is no batched prefill, exactly as Wave B.
#
# Construction: the CPU GlimmerModel is opened first and stays
# the loader of record (its cached bytes are the upload
# source), the config source, the captures holder, and the
# documented fallback (the runner's CPU path). DeviceWeights
# uploads every slot ONCE from those bytes; DeviceKvCache is
# built at the configured context length (EngineConfig's
# context_len, handed in by the caller - this module reads no
# env vars, config.mojo's law). The budget cap comes from
# GpuRuntime's construction (Lane C1 owns the
# MOJO_ENGINE_GPU_BUDGET_BYTES config value; the runner hands
# the default cap constant in until C1's config field lands).
#
# step() mirrors GlimmerModel.step statement for statement:
#   1. embedding row on device (gpu_embed_token, C2);
#   2. the weightless embedding norm as a PLAIN rms_norm via
#      rms_norm_kernel (grid = 1 row, block = RMS_THREADS,
#      ones as the weight row - the kernel's weighted formula
#      with ones is the plain form, the same arithmetic-path
#      argument rms_norm_plain_cpu makes on the CPU side);
#   3. gpu_layer_step for all layers in order (C2), which
#      appends to the DeviceKvCache as it goes.
# Captures: the CPU model's captures remain the isolation tool
# (the contract says so for C2; the same holds here). What
# this file wires: when the CPU model's capture_enabled flag
# is set, step() records capture_embed (the post-norm
# embedding row, the CPU step's capture point) into the CPU
# model, and the per-layer in/out capture points live inside
# C2's gpu_layer_step against the same flags. A divergence
# hunt uses CPU captures plus C2's per-layer server proof;
# stated plainly so nobody expects device-side layer captures
# from this file.
#
# logits() is C2's gpu_logits over the resident head weights.
#
# Named refusals: gpu_model_bad_token (token outside vocab,
# checked before any device work, mirroring
# model_bad_token), gpu_model_bad_shape (hidden row width at
# logits time, mirroring model_bad_shape), plus whatever the
# C1/C2 surfaces refuse by name underneath (gpu_no_device,
# gpu_budget_exceeded, gpu_kv_window_wrapped, ...), propagated
# untouched. Diag seam: phase `model`, model.mojo's phase.

from diag import log_debug
from generate.generate import ForwardModel
from gpu.context import (
    DEFAULT_GPU_BUDGET_BYTES,
    GpuRuntime,
    download_f32,
    upload_f32,
)
from gpu.device_kv import DeviceKvCache
from gpu.weights import DeviceWeights
from model.gpu_layer import gpu_layer_step, gpu_logits
from model.model import GlimmerModel
from ops.gpu_embedding import gpu_embed_token
from ops.rms_norm import RMS_THREADS, rms_norm_kernel


struct GpuGlimmerModel(ForwardModel, Movable):
    """The device-resident Glimmer forward.

    Holds the CPU model (config, captures, fallback, loader of
    record) plus the device trio built from it at construction.
    """

    var cpu: GlimmerModel
    var rt: GpuRuntime
    var weights: DeviceWeights
    var kv: DeviceKvCache

    def __init__(
        out self,
        var cpu_model: GlimmerModel,
        context_len: Int,
        budget_bytes: Int,
    ) raises:
        # Build the device trio from the borrowed CPU model
        # BEFORE it moves into the struct: weights upload from
        # its cached bytes, the KV cache from its layer pattern
        # at the configured context length.
        var rt = GpuRuntime(budget_bytes)
        var weights = DeviceWeights(rt, cpu_model)
        var kv = DeviceKvCache(
            rt,
            cpu_model.cfg.kinds.copy(),
            cpu_model.cfg.kv_heads,
            cpu_model.cfg.head_dim,
            context_len,
        )
        self.cpu = cpu_model^
        self.rt = rt^
        self.weights = weights^
        self.kv = kv^
        _ = log_debug(
            "model",
            "gpu model built layers="
            + String(self.cpu.cfg.layers)
            + " hidden="
            + String(self.cpu.cfg.hidden)
            + " context_len="
            + String(context_len),
        )

    @staticmethod
    def open(path: String, context_len: Int) raises -> GpuGlimmerModel:
        """Open the CPU model at path (loader of record) and
        build the device model over it at the default budget
        cap. Mirrors GlimmerModel.open's shape."""
        var cpu_model = GlimmerModel.open(path)
        return GpuGlimmerModel(
            cpu_model^, context_len, DEFAULT_GPU_BUDGET_BYTES
        )

    def _plain_rms_norm(
        self, x: List[Float32], eps: Float32
    ) raises -> List[Float32]:
        """The CPU step's weightless embedding norm, on device:
        rms_norm_kernel with a ones weight row, grid = 1 row.
        """
        var n = len(x)
        if n == 0:
            raise Error(
                "gpu_model_bad_shape: embedding norm got an empty row"
            )
        var ones = List[Float32](capacity=n)
        ones.resize(n, Float32(1.0))
        var x_dev = upload_f32(self.rt, x)
        var w_dev = upload_f32(self.rt, ones)
        var y_dev = self.rt.ctx.enqueue_create_buffer[DType.float32](n)
        self.rt.ctx.enqueue_function[rms_norm_kernel](
            x_dev,
            w_dev,
            y_dev,
            Int32(n),
            eps,
            grid_dim=1,
            block_dim=RMS_THREADS,
        )
        self.rt.ctx.synchronize()
        return download_f32(self.rt, y_dev, n)

    def step(
        mut self, token_id: Int, position: Int
    ) raises -> List[Float32]:
        """One token at one absolute position through the whole
        stack on device; appends to the device KV cache (inside
        gpu_layer_step); returns the final hidden row. Mirrors
        GlimmerModel.step statement for statement."""
        var cfg = self.cpu.cfg.copy()
        if token_id < 0 or token_id >= cfg.vocab:
            raise Error(
                "gpu_model_bad_token: token "
                + String(token_id)
                + " outside vocab "
                + String(cfg.vocab)
            )
        if position < 0:
            raise Error(
                "gpu_model_bad_shape: negative position "
                + String(position)
            )
        var x = gpu_embed_token(self.rt, self.weights, token_id)
        x = self._plain_rms_norm(x, cfg.rms_eps)
        if self.cpu.capture_enabled:
            self.cpu.capture_embed = x.copy()
        for layer in range(cfg.layers):
            x = gpu_layer_step(
                self.rt,
                self.weights,
                self.kv,
                self.cpu,
                x,
                layer,
                position,
            )
        _ = log_debug(
            "model",
            "gpu step token="
            + String(token_id)
            + " position="
            + String(position),
        )
        return x^

    def logits(self, hidden: List[Float32]) raises -> List[Float32]:
        """Finished logits for one hidden row: C2's gpu_logits
        (final norm, head GEMV, scale-then-softcap on device).
        """
        var cfg = self.cpu.cfg.copy()
        if len(hidden) != cfg.hidden:
            raise Error(
                "gpu_model_bad_shape: logits got hidden row of "
                + String(len(hidden))
                + ", hidden is "
                + String(cfg.hidden)
            )
        return gpu_logits(self.rt, self.weights, hidden)

    def forward_rows(mut self, tokens: List[Int]) raises -> List[Float32]:
        """The step loop over a token list starting at layer
        0's next device-cache position; returns the LAST hidden
        row. The GPU twin of GlimmerModel.forward_rows, present
        so the runner's prefill-measurement construction keeps
        the identical shape on both paths."""
        var pos = self.kv.next_position(0)
        var hidden = List[Float32]()
        for i in range(len(tokens)):
            hidden = self.step(tokens[i], pos + i)
        return hidden^
