# K4-K8 results: the wgpu Compute surface completed and proven

Date: 2026-10-30. Machine: ubuntu-server, AMD Radeon RX 7900 XTX
(RADV NAVI31, Vulkan). Command:
`cargo test -p wgpu-compute --release -- --nocapture`.
Full log: `~/k4-k8-server-run3.log` on the server.

## Verdict

Every `muse_glimmer::Compute` step now runs on the device.
Final run: all suites green (lib 24, cross-backend 1, K1
synthetic 3, K2/K3 synthetic 3, K4-K8 kernels 7, shader compile
2). The production Glimmer llama-server on the same machine was
health-checked before and after (`{"status":"ok"}`, pid 683675,
untouched throughout).

## Numbers (max abs diff vs the CPU reference, per test)

K4 norms and add:
- rms_norm, eps 1e-5 and eps 1e-8: 5.960e-8
- rms_norm_plain: 1.192e-7
- fused norm_residual vs the trait composition (norm, then
  add): 0.0 exactly
- residual add: exact (asserted equal)

K5 RoPE and KV append:
- rotate vs `torch_math::rope_interleaved` (rows at positions
  5..8, 32 heads x 128, base 500000): 3.055e-7
- append: windowed (window 4) rows landed in ring slots 2, 3;
  global rows in slots 6, 7; contents asserted equal

K6 attention vs `cpu-compute`'s attention step, same Q/K/V
stream through both backends (3-token prefill plus 4 decode
steps, 3 layers: sliding, global, sliding; window 4, so the
ring evicts during the run): all 15 chunk/layer outputs within
8.941e-8.

K7 activations:
- attn_gate through the trait: 2.980e-8
- swiglu vs `torch_math::swiglu`: 2.980e-8

K8 embedding gather vs `cpu-compute`'s embed, every dtype both
backends accept: F32, F16, BF16, Q8_0, Q4_K, Q5_K, Q6_K all
0.0 exactly (the in-kernel dequantizers restate gguf's
reference decoders operation for operation).

K8 head finish vs `muse_glimmer::softcap_scale` over the real
202,048-wide vocab (scale 0.19611613, cap 20.0): 2.384e-6.

## Cross-backend harness (tests/cross_backend.rs)

Synthetic 6-layer Glimmer (sliding x3, global, sliding,
global), real head geometry (32 Q / 2 KV, head dim 128),
hidden 512, FFN 1024, vocab 1024, window 4; all projections,
embedding, and head K-quantized (Q4_K; head Q5_K), norms F32;
byte-identical weights through both backends. Prompt of 4
tokens, then 5 greedy decode steps (9 positions through a
window of 4):

- steps 0-5 max abs logit diff: 2.3842e-7, 2.3842e-7,
  2.3842e-7, 3.5763e-7, 3.5763e-7, 2.3842e-7
- top-1 agreed at every step; worst diff 3.5763e-7 against the
  stated pass bound of 0.5

## Defects found by the proof (and fixed)

1. The K8 gather shaders widened f16 through a nested
   `pack2x16float` call, which is a WGSL type error
   (`pack2x16float` takes `vec2<f32>`). The first server run
   failed at pipeline creation with the validator's message.
   Fixed to the direct `unpack2x16float(bits).x` form in both
   K8 shaders.
2. The K4 fused-vs-composed test called `norm_residual` with
   its residual and branch arguments swapped (2.621e0 diff).
   The kernel was correct; the test call was fixed, and the
   fused form now matches the composition exactly (0.0).
3. An earlier draft of the K8 Q6_K decoder indexed the wrong
   ql bytes and qh bit pairs for streams 2 and 3; caught by
   restating gguf's reference dequantizer stream by stream
   before the server run. The shipped decoder matches
   cpu-compute exactly (0.0) on Q6_K gathers.

## Local gates (this workstation, no suitable adapter)

- `cargo test --workspace`: all suites pass; the GPU tests
  skip with printed reasons by design (llvmpipe does not count
  as an adapter, BACKEND.md Section 6).
- `cargo clippy --workspace --all-targets`: clean under the
  workspace's deny-all lints.

## Notes for the next step

- K8's `finish` kernel (scale-then-softcap) is proven but
  deliberately not wired into `lm_head`: the model applies
  `softcap_scale` on the host today, and applying it twice
  would be a bug. It is the epilogue the K3 head GEMV grows
  when the forward becomes device-resident.
- K4's fused `norm_residual` is likewise proven against the
  trait composition and held for the device-resident forward;
  the trait path composes `rms_norm` + `add`.
- Per-call host<->device transfers remain the trait seam's
  cost (weights are cached device-side by name; activations
  move per call), as documented in `compute.rs`.
