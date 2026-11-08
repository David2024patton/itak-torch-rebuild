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
# EngineConfig.gpu_budget_bytes in (config field: Lane C1).
#
# Wave D (Lane D4): gpu_layer_step is now the resident chain
# (activations stay on device inside each layer) and windowed
# layers run past the 2,048 wrap on device (ring appends +
# visible-order gather in DeviceKvCache); the CPU fallback is
# no longer the only path past the window.
#
# step() mirrors GlimmerModel.step statement for statement:
#   1. embedding row on device (gpu_embed_token, C2);
#   2. the weightless embedding norm as a PLAIN rms_norm via
#      rms_norm_kernel (grid = 1 row, block = RMS_THREADS,
#      ones as the weight row - the kernel's weighted formula
#      with ones is the plain form, the same arithmetic-path
#      argument rms_norm_plain_cpu makes on the CPU side);
#   3. gpu_layer_step_cfg for all layers in order (C2; Wave
#      D reconciliation threads the attention tile/split
#      knobs through, pointed at the resident step), which
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
# Wave D (Lane D1): prefill() / prefill_batch() add the
# batched GEMM prefill path (projections via
# src/kernels/gemm.mojo through gpu_layer_step_batch);
# step() and logits() - the decode path - are untouched, and
# prefill returns the same last hidden row step() would.
#
# Named refusals: gpu_model_bad_token (token outside vocab,
# checked before any device work, mirroring
# model_bad_token), gpu_model_bad_shape (hidden row width at
# logits time, mirroring model_bad_shape), plus whatever the
# C1/C2 surfaces refuse by name underneath (gpu_no_device,
# gpu_budget_exceeded, gpu_kv_capacity_exceeded, ...), propagated
# untouched. Diag seam: phase `model`, model.mojo's phase.

from config import (
    DEFAULT_ATTN_MAX_SPLITS,
    DEFAULT_ATTN_SPLIT_SPAN_POSITIONS,
    DEFAULT_ATTN_TILE_POSITIONS,
    DEFAULT_GEMM_TILED_MIN_BATCH,
    DEFAULT_GPU_BUDGET_BYTES,
    DEFAULT_PREFILL_GEMM_MIN_BATCH,
)
from diag import log_debug
from generate.generate import ForwardModel
from gpu.context import GpuRuntime, download_f32, upload_f32
from gpu.device_kv import DeviceKvCache
from gpu.weights import DeviceWeights
from model.gpu_layer import gpu_layer_step_cfg, gpu_logits
from model.gpu_layer_batch import gpu_layer_step_batch, batch_row_slice
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
    # Wave D attention knobs (config seam): tile positions,
    # split-KV cap, split span; threaded into every layer's
    # resident attention launch.
    var attn_tile_positions: Int
    var attn_max_splits: Int
    var attn_split_span_positions: Int
    # Wave D prefill GEMM knobs (config seam): the prompt
    # batch size that takes the batched GEMM prefill path and
    # the batch size that takes its register-tiled structure.
    var prefill_gemm_min_batch: Int
    var gemm_tiled_min_batch: Int

    def __init__(
        out self,
        var cpu_model: GlimmerModel,
        context_len: Int,
        budget_bytes: Int,
        attn_tile_positions: Int = DEFAULT_ATTN_TILE_POSITIONS,
        attn_max_splits: Int = DEFAULT_ATTN_MAX_SPLITS,
        attn_split_span_positions: Int = DEFAULT_ATTN_SPLIT_SPAN_POSITIONS,
        prefill_gemm_min_batch: Int = DEFAULT_PREFILL_GEMM_MIN_BATCH,
        gemm_tiled_min_batch: Int = DEFAULT_GEMM_TILED_MIN_BATCH,
    ) raises:
        if prefill_gemm_min_batch <= 0 or gemm_tiled_min_batch <= 0:
            raise Error(
                "gpu_model_bad_shape: prefill GEMM batch thresholds must"
                " be positive, got "
                + String(prefill_gemm_min_batch)
                + "/"
                + String(gemm_tiled_min_batch)
            )
        # Build the device trio from the borrowed CPU model
        # BEFORE it moves into the struct: weights upload from
        # its cached bytes, the KV cache from its layer pattern
        # at the configured context length.
        var rt = GpuRuntime.create(budget_bytes)
        var weights = DeviceWeights.build(rt, cpu_model)
        var kv = DeviceKvCache.build(
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
        self.attn_tile_positions = attn_tile_positions
        self.attn_max_splits = attn_max_splits
        self.attn_split_span_positions = attn_split_span_positions
        self.prefill_gemm_min_batch = prefill_gemm_min_batch
        self.gemm_tiled_min_batch = gemm_tiled_min_batch
        _ = log_debug(
            "model",
            "gpu model built layers="
            + String(self.cpu.cfg.layers)
            + " hidden="
            + String(self.cpu.cfg.hidden)
            + " context_len="
            + String(context_len)
            + " attn_tile="
            + String(attn_tile_positions)
            + " attn_max_splits="
            + String(attn_max_splits)
            + " attn_split_span="
            + String(attn_split_span_positions),
        )

    @staticmethod
    def open(
        path: String,
        context_len: Int,
        budget_bytes: Int,
        prefill_gemm_min_batch: Int = DEFAULT_PREFILL_GEMM_MIN_BATCH,
        gemm_tiled_min_batch: Int = DEFAULT_GEMM_TILED_MIN_BATCH,
    ) raises -> GpuGlimmerModel:
        """Open the CPU model at path (loader of record) and
        build the device model over it under the given budget
        cap (callers hand in EngineConfig.gpu_budget_bytes, so
        MOJO_ENGINE_GPU_BUDGET_BYTES is honored end to end).
        Mirrors GlimmerModel.open's shape. Attention knobs
        take their config defaults; open_attn is the explicit
        form. The GEMM batch thresholds default from
        config.mojo; the runner hands in EngineConfig's
        values so the MOJO_ENGINE_PREFILL_GEMM_MIN_BATCH /
        MOJO_ENGINE_GEMM_TILED_MIN_BATCH overrides are
        honored end to end."""
        var cpu_model = GlimmerModel.open(path)
        return GpuGlimmerModel(
            cpu_model^,
            context_len,
            budget_bytes,
            DEFAULT_ATTN_TILE_POSITIONS,
            DEFAULT_ATTN_MAX_SPLITS,
            DEFAULT_ATTN_SPLIT_SPAN_POSITIONS,
            prefill_gemm_min_batch,
            gemm_tiled_min_batch,
        )

    @staticmethod
    def open_attn(
        path: String,
        context_len: Int,
        budget_bytes: Int,
        attn_tile_positions: Int,
        attn_max_splits: Int,
        attn_split_span_positions: Int,
    ) raises -> GpuGlimmerModel:
        """open() with explicit Wave D attention knobs (the
        runner hands in EngineConfig's attn fields, so the
        MOJO_ENGINE_ATTN_* env vars are honored end to end)."""
        var cpu_model = GlimmerModel.open(path)
        return GpuGlimmerModel(
            cpu_model^,
            context_len,
            budget_bytes,
            attn_tile_positions,
            attn_max_splits,
            attn_split_span_positions,
        )

    def _plain_rms_norm(
        mut self, x: List[Float32], eps: Float32
    ) raises -> List[Float32]:
        """The CPU step's weightless embedding norm, on device:
        rms_norm_kernel with a ones weight row, grid = 1 row.
        """
        var n = len(x)
        if n == 0:
            raise Error("gpu_model_bad_shape: embedding norm got an empty row")
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

    def step(mut self, token_id: Int, position: Int) raises -> List[Float32]:
        """One token at one absolute position through the whole
        stack on device; appends to the device KV cache (inside
        gpu_layer_step_cfg); returns the final hidden row. Mirrors
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
                "gpu_model_bad_shape: negative position " + String(position)
            )
        var x = gpu_embed_token(self.rt, self.weights, token_id)
        x = self._plain_rms_norm(x, cfg.rms_eps)
        if self.cpu.capture_enabled:
            self.cpu.capture_embed = x.copy()
        for layer in range(cfg.layers):
            x = gpu_layer_step_cfg(
                self.rt,
                self.weights,
                self.kv,
                self.cpu,
                x,
                layer,
                position,
                self.attn_tile_positions,
                self.attn_max_splits,
                self.attn_split_span_positions,
            )
        _ = log_debug(
            "model",
            "gpu step token="
            + String(token_id)
            + " position="
            + String(position),
        )
        return x^

    def logits(mut self, hidden: List[Float32]) raises -> List[Float32]:
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
        return gpu_logits(self.rt, self.weights, self.cpu, hidden)

    def _plain_rms_norm_batch(
        mut self, x: List[Float32], batch: Int, eps: Float32
    ) raises -> List[Float32]:
        """The embedding norm over a flat batch: rms_norm_kernel
        with a ones weight row, grid = batch rows. The batched
        twin of _plain_rms_norm (same arithmetic per row)."""
        var cfg = self.cpu.cfg.copy()
        var n = cfg.hidden
        if len(x) != batch * n or batch <= 0:
            raise Error(
                "gpu_model_bad_shape: embedding norm batch has "
                + String(len(x))
                + " elements, expected "
                + String(batch * n)
            )
        var ones = List[Float32](capacity=n)
        ones.resize(n, Float32(1.0))
        var x_dev = upload_f32(self.rt, x)
        var w_dev = upload_f32(self.rt, ones)
        var y_dev = self.rt.ctx.enqueue_create_buffer[DType.float32](len(x))
        self.rt.ctx.enqueue_function[rms_norm_kernel](
            x_dev,
            w_dev,
            y_dev,
            Int32(n),
            eps,
            grid_dim=batch,
            block_dim=RMS_THREADS,
        )
        self.rt.ctx.synchronize()
        return download_f32(self.rt, y_dev, len(x))

    def prefill_batch(
        mut self, tokens: List[Int], start_position: Int
    ) raises -> List[Float32]:
        """Batched prefill (Wave D, Lane D1): the whole prompt
        through the stack with GEMM projections, returning the
        LAST hidden row. Per-position semantics are
        gpu_layer_step's, batched by gpu_layer_step_batch:
        embeddings gathered per token (the gather has no
        projection to batch), the plain embedding norm as one
        launch over the flat batch, then one batched layer
        step per layer. Token/position validation matches
        step()'s refusals, checked for the whole batch before
        any device work."""
        var cfg = self.cpu.cfg.copy()
        var batch = len(tokens)
        if batch == 0:
            raise Error("gpu_model_bad_shape: prefill batch is empty")
        if start_position < 0:
            raise Error(
                "gpu_model_bad_shape: negative start position "
                + String(start_position)
            )
        for i in range(batch):
            if tokens[i] < 0 or tokens[i] >= cfg.vocab:
                raise Error(
                    "gpu_model_bad_token: token id "
                    + String(tokens[i])
                    + " outside vocab "
                    + String(cfg.vocab)
                )
        var x = List[Float32](capacity=batch * cfg.hidden)
        x.resize(batch * cfg.hidden, Float32(0.0))
        for i in range(batch):
            var row = gpu_embed_token(self.rt, self.weights, tokens[i])
            for j in range(cfg.hidden):
                x[i * cfg.hidden + j] = row[j]
        x = self._plain_rms_norm_batch(x, batch, cfg.rms_eps)
        if self.cpu.capture_enabled:
            self.cpu.capture_embed = batch_row_slice(x, batch - 1, cfg.hidden)
        for layer in range(cfg.layers):
            x = gpu_layer_step_batch(
                self.rt,
                self.weights,
                self.kv,
                self.cpu,
                x,
                batch,
                layer,
                start_position,
                self.gemm_tiled_min_batch,
            )
        _ = log_debug(
            "model",
            "gpu prefill_batch tokens="
            + String(batch)
            + " start="
            + String(start_position),
        )
        return batch_row_slice(x, batch - 1, cfg.hidden)

    def prefill(mut self, prompt_ids: List[Int]) raises -> List[Float32]:
        """The ForwardModel prefill seam (Wave D): at or above
        the configured GEMM minimum batch, the batched GEMM
        path from the cache's next position; below it, the
        trait default's step loop from position 0 - which for
        a fresh model (generate()'s construction, the only
        caller that mixes prefill with decode steps) is the
        same start the batched path takes. Returns the last
        hidden row either way."""
        if len(prompt_ids) >= self.prefill_gemm_min_batch:
            var pos = self.kv.next_position(0)
            return self.prefill_batch(prompt_ids, pos)
        var hidden = self.step(prompt_ids[0], 0)
        for i in range(1, len(prompt_ids)):
            hidden = self.step(prompt_ids[i], i)
        return hidden^

    def forward_rows(mut self, tokens: List[Int]) raises -> List[Float32]:
        """The token list from layer 0's next device-cache
        position; returns the LAST hidden row. The GPU twin of
        GlimmerModel.forward_rows, present so the runner's
        prefill-measurement construction keeps the identical
        shape on both paths. Wave D: this IS the prefill seam
        (prefill() above) - the measurement now times the
        batched path, and the Wave C per-row path remains
        reachable by raising MOJO_ENGINE_PREFILL_GEMM_MIN_BATCH
        above the prompt length."""
        return self.prefill(tokens)
