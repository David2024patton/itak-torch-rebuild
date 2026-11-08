# Wave C interfaces contract: the device seam

Status: coordinator contract, written before Lane C1/C2/C3
code, 2026-11-07 (the wave closes 2026-11-08 by the clock if
it runs long; the bench note carries the real date). Branch
`mojo-max-engine`, base ae0a053 (engine 1.3, Wave B golden
PASS on the CPU path). This file is the agreement the lanes
code against; the coordinator reconciles deviations at
integration and updates this file to match what shipped.

## The port shape (decided, not negotiable per lane)

The CPU twins stay as the parity reference and the fallback.
The GPU path is a device-resident forward with the CPU
model's EXACT per-position step semantics (Wave B shape:
step(token, position) through embedding, 52 layers, head;
prefill is the step loop, there is no batched prefill):

- Weights are uploaded ONCE at load and stay resident.
- The KV cache is resident on device with B1's exact layout
  and semantics (see DeviceKv below); its CPU semantics do
  not change.
- Activations are host `List[Float32]` rows at the layer
  boundary: each op uploads its input row(s), launches its
  device kernel against resident weights/KV, and downloads
  its output row. Activation traffic per step is tens of KB;
  weight and KV traffic is zero after load. A fully
  device-resident activation chain is a later wave's
  optimization, not this wave's correctness risk.
- Argmax parity is the hard gate at the model level; float32
  accumulation reordering on device is expected and reported
  inside the established tolerance class (cosine / max rel
  vs the CPU twin, bars as in Waves A and B).

## Memory budget (planned in numbers before the first run)

Card: RX 7900 XTX, 24 GiB = 25,769,803,776 B total.

- Weights resident: the sum of the file's tensor data bytes,
  computed exactly from the loader at upload time and printed
  by the C1 proof. Planning upper bound: the whole file,
  16,756,683,904 B.
- Device KV (f32, window-capped at the configured context
  length, B1 accounting): 3,653,238,784 B at context 131,072
  (13 global layers x 131,072 positions + 39 windowed layers
  x 2,048 positions, 2,048 B per position per layer for K+V).
- Scratch and activations envelope: 536,870,912 B (512 MiB).
  The largest single activation is the logits row,
  202,048 x 4 B = 808,192 B; everything else is hidden/FFN
  rows and one-row staging buffers.
- Planning total: 20,946,793,600 B, about 19.5 GiB, under
  the cap below by about 3.75 GB.

Budget cap: card total minus a 1 GiB reserve =
24,696,061,952 B by default, overridable through config
(`MOJO_ENGINE_GPU_BUDGET_BYTES`, parsed and refused by name
in config.mojo like the other values; Lane C1 owns that
config addition). If the computed resident total exceeds
the cap, construction refuses by name:
`gpu_budget_exceeded`. The observed resident figure (sysfs
VRAM used delta in the server window) is recorded in the
bench note next to this plan.

## Lane C1: src/gpu/ (device plumbing) - owns this package

New package `src/gpu/` with `__init__.mojo`, containing:

- `context.mojo`: `GpuRuntime` - owns the MAX
  `DeviceContext`, the budget accounting (weights bytes +
  KV bytes + scratch envelope vs cap, refusal above), and
  the copy helpers every other lane uses:
  `upload_u8(rt, List[UInt8]) -> DeviceBuffer[DType.uint8]`,
  `upload_f32(rt, List[Float32]) -> DeviceBuffer[DType.float32]`,
  `download_f32(rt, DeviceBuffer[DType.float32], n) -> List[Float32]`,
  plus zero-filled device buffer constructors by dtype and
  element count. Named refusals: `gpu_no_device` (context
  creation finds no device), `gpu_budget_exceeded`,
  `gpu_copy_shape` (length mismatches). Diag lines through
  the existing seam (`gpu` phase).
- `weights.mojo`: `DeviceWeights` - built from a
  `GlimmerModel`'s slots (the CPU model stays the loader of
  record; its cached bytes are the upload source). One
  device buffer per tensor slot: quantized and F16 slots as
  uint8 byte buffers over the slot's raw bytes, F32 slots as
  float32 buffers over the slot's values. The embedding
  slot, the output (head) slot, and the final-norm values
  get named accessors; per-layer slots are reached by
  (layer, suffix_index) exactly as `GlimmerModel.slot`.
  Accessors return the buffer AND the slot's type_id, rows,
  cols so launchers never guess. The exact resident byte
  total is accumulated during construction and checked
  against the budget before the last buffer is created
  (refuse first, allocate second, where the sizes are known
  up front - they are).
- `device_kv.mojo`: `DeviceKvCache` - per layer, two float32
  device buffers (K and V), each
  `capacity_positions x kv_heads x head_dim`, where
  capacity_positions is the layer's window (2,048) for
  windowed layers and the configured context length for
  global layers: B1's layout, byte for byte (row = one
  position's [kv_heads][head_dim], slot = position % window
  for windowed layers, slot = position for global layers).
  Semantics mirror `KvCache` exactly: appends in order only,
  reads by absolute position, the same refusal names as B1
  for out-of-order append / evicted or unwritten read / bad
  shape (prefix `gpu_kv_` where a name would collide).
  Append path: the caller passes the position's K and V rows
  as host lists (already rotated for sliding layers - the
  rotation is the model layer's job, exactly as in the CPU
  step); the implementation stages the row and scatters it
  into the slot (a small copy kernel in this file is the
  expected shape; whole-buffer copies are forbidden - they
  would move gigabytes per token).
  Read path for attention: `key_buffer(layer)` /
  `value_buffer(layer)` expose the raw device buffers, plus
  `stored_len(layer)` and `first_visible_position(layer)`
  with B1's meanings. WAVE C LIMIT, stated plainly: the
  attention kernel consumes visible rows in position order
  from offset 0, which equals slot order only before a
  windowed layer wraps. The GPU layer therefore refuses by
  name (`gpu_kv_window_wrapped`) when a windowed layer's
  append would wrap past its window; the CPU path remains
  the fallback for contexts beyond 2,048 positions and a
  device-side visible-order gather is owed work, recorded
  in the bench note. The golden run (22 positions) never
  approaches the wrap.
  Proof obligations (Lane C1): VM unit tests for the pure
  parts (budget arithmetic, slot/layout arithmetic, refusal
  names) wired by the coordinator into run_tests.sh; a
  server proof main (`tests/server_proof_gpu_plumbing.mojo`)
  that uploads the real model and hashes
  blk.0.ffn_gate.weight back from the device byte-exact
  against the loader's bytes (the Wave A FNV-1a64 /
  sha256 fixture check is the pattern), then runs a KV
  write/read sequence on device (including positions
  2047/2048/2049 on a small synthetic window config and a
  global sibling) compared element-exact against B1's CPU
  cache fed the same rows.

## Lane C2: GPU layer assembly - owns these files

- `src/model/gpu_layer.mojo`: host-side launch wrappers and
  the layer step. Public surface the other lanes code
  against:
  - `gpu_logits(rt, weights, hidden) -> List[Float32]`:
    final rms_norm (rms_norm_kernel, grid = 1 row) + the
    head GEMV dispatched by the output slot's dtype (Q5_K
    on Glimmer: q5k_gemv_kernel over 202,048 rows) +
    final_logit_kernel (scale-then-softcap). Same math as
    `GlimmerModel.logits`, device kernels.
  - `gpu_layer_step(rt, weights, kv, cpu_model, x, layer,
    position) -> List[Float32]`: statement-for-statement the
    CPU `layer_step`, with every projection and op launched
    on device: rms_norm_kernel for the norms (per-head Q/K
    norms launch rms_norm_kernel with grid = head count,
    width = head_dim, exactly as gpu_proof_ops proved the
    launcher); GEMV launches dispatched by slot dtype
    (k1 grid rows//ROWS_PER_WG block THREADS from
    k1_gemv.mojo; q6k/q5k/f32/f16 grid rows block 256);
    RoPE via rope_kernel (positions buffer filled with the
    step's absolute position; sliding layers rotate the K
    row BEFORE the DeviceKv append - write-time rotation,
    cache holds rotated K, the Wave B law - and rotate the
    Q row before the attention launch, because
    attention_kernel's contract is PRE-ROTATED Q and K);
    attention via attention_kernel launched directly on the
    DeviceKv buffers (grid = q_heads, block = ATTN_THREADS,
    stored = kv.stored_len, group = q_heads // kv_heads,
    scale = 1/sqrt(head_dim)); the sigmoid gate assembly
    (attn * sigmoid(gate)) and SwiGLU (swiglu_kernel exists;
    the gate assembly needs one small new elementwise kernel
    - put it in this file); residual adds via
    residual_add_kernel or host adds (say which, and why;
    host adds on downloaded rows are acceptable in this
    wave's shape).
    `cpu_model` is borrowed for config and the F32 norm
    weight values until DeviceWeights grows f32 accessors
    for every norm - prefer DeviceWeights accessors; if a
    norm weight must be uploaded per call, say so in the
    bench note (they are 26 KB; correctness first).
  - Captures: the same capture hooks as the CPU step
    (embed, named-layer in/out) or an explicit statement
    that the CPU model's captures are the isolation tool
    (they are; C3's divergence work uses CPU captures plus
    C2's per-layer server proof).
- `src/ops/gpu_embedding.mojo`: the device embedding row:
  a kernel that dequantizes one row of the embedding table
  on device for the table's actual dtype (Q4_K on Glimmer;
  Q6_K/Q5_K/F16/F32 handled or refused by name), proven
  against `GlimmerModel.embed_token` (the CPU twin) on the
  VM-independent server proof and in C2's layer proof by
  construction. Public surface:
  `gpu_embed_token(rt, weights, token) -> List[Float32]`.
- Proof obligations (Lane C2): a server proof main
  (`tests/server_proof_gpu_layer.mojo`) running layer 0
  (sliding) and layer 3 (global) at position 0 on the real
  weights with the rung-2 input construction (the model's
  own embedding + plain norm for token 13796, position 0),
  compared against the COUNTED rung-2 outputs (layer 0 sum
  65.22733633872122, first8 [0.08513339, 0.026624471,
  -0.42816293, 3.276528, -1.1669861, 2.0494282, -0.08531731,
  0.34014347]; layer 3 sum 157.77914571017027, first8
  [1.254974, -0.3695352, -0.49299628, 1.3146128, -0.27929068,
  1.1605207, 0.099737644, 0.68446195]). Bar: cosine >=
  0.999999 and max rel <= 1e-4 against those rows (the
  rung-1 bar class; the spot check counted the CPU rows at
  ~1e-6 against Float64, so this bar has two orders of
  headroom for device accumulation reordering), sums
  reported alongside. A fresh position-0 step on a FRESH
  DeviceKv per layer proof (each proof layer runs as the
  model's layer index so its kind and weights are real;
  state exactly how the proof drives a single layer).

## Lane C3: GPU generation - owns these files

- `src/model/gpu_model.mojo`: `GpuGlimmerModel` implementing
  the SAME `ForwardModel` trait (step/logits) over
  GpuRuntime + DeviceWeights + DeviceKvCache + C2's
  gpu_layer_step / gpu_logits / gpu_embed_token, with the
  CPU GlimmerModel held for config, captures, and fallback.
  step() mirrors the CPU step statement for statement
  (embedding via gpu_embed_token, plain rms_norm via
  rms_norm_kernel, layers via gpu_layer_step).
- `src/runner/generate_main.mojo` (edit, minimal): select
  the GPU model when the environment variable
  `MOJO_ENGINE_GPU` is set to a value other than "" and
  "0" (the diag seam's convention), default CPU exactly as
  Wave B shipped, so the Wave B proof still reproduces.
  The runner's printed lines (PROMPT_IDS, TOKENS,
  GOLDEN_FIRST8_MATCH, TEXT_STRICT_OK, timings) stay
  byte-compatible; add one line naming the path taken
  (MODEL_PATH_KIND gpu|cpu).
- `tests/server_proof_gpu_generate.mojo`: proof order per
  the wave brief: (1) teacher-forced check on device,
  positions 0..7 against the golden walk (prompt
  [200000, 954, 7963, 323, 11698, 373] teacher-forced must
  predict [13796, 26, 1573, 373, 290, 2167, 26899, 1212],
  8 of 8, printed per position); (2) the end-to-end runner
  run itself is the coordinator's integration window, not
  this file's job - this file prints its own verdict lines
  and exits nonzero on any mismatch.

## Lane C4: measurement - owns the bench note and timing

- `mojo-engine/tests/wave_c_timing.sh`: the timing harness
  for the final window: staged from the exact commit,
  runs the runner on the GPU path (and the C3 teacher-forced
  proof), captures prefill ms (the runner's own printed
  prefill measurement construction, same as Wave B's: a
  fresh model stepping the prompt ids) and decode s/step
  and tok/s from the generate run, each number read back
  from the actual log file in the foreground (the evidence
  law: backgrounded exec-completion output is not evidence).
- The bench note itself is written by the coordinator from
  the lane reports plus C4's timing output; C4's lane
  report supplies the protocol and any control runs it
  executed.

## Shared rules (all lanes)

- File ownership above is exclusive. `tests/run_tests.sh`
  is coordinator-only; lanes add test files and name them
  in their reports for wiring. `src/config.mojo` is C1-only
  this wave. Existing kernel/op modules are read-only for
  all lanes (their kernels are launched, never edited); a
  defect found in one is reported to the coordinator, not
  fixed inside a lane.
- Every module keeps the house law: diag seam wired, named
  refusals, limits from config with defaults.
- Server windows: phase lock
  `flock -w 14400 /tmp/itak-phase.lock`, one heavy job at a
  time, fresh staging (git archive of the exact commit,
  STAGED_COMMIT stamp, staged file hash), tenants per the
  CURRENT host state (see the wave brief: only
  cortex-embed.service holds the card; the two retired
  containers do not exist), VRAM measured via sysfs at both
  ends of every window, restore exactly what was stopped.
- GPU tests skip cleanly on the VM (no device) and RUN on
  the card; every tally in every report says which.
