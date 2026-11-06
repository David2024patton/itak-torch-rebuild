# Flash attention rework: K/V staged in workgroup memory, 2026-11-06

Status: BUILT and VM-gated. The staged flash form exists in
`crates/wgpu-compute`, env-gated off, with its correctness
tests written and its card protocol below. No card run has
happened yet; every number in this note that is not from
the 2026-11-05 profile is a prediction or a bound, and is
labelled as one.

## Why

The prefill profile (bench/prefill-profile-2026-11-05.md)
puts the shipped flash kernel at 1,013.1 ms of 6,659 ms
GPU time at 1,022 tokens (15.2 percent; 832 dispatches,
sliding layers 764.0 ms, global layers 249.2 ms), growing
from 3.2 ms in chunk 0 to 125.7 ms in chunk 15, the last
four chunks carrying 44 percent of it. The parity
extraction (bench/llama-parity-specs-2026-11-06.md, gap 5)
names the structural cause: the shipped kernel stages
only a 256-wide score tile in workgroup memory; its score
phase reads every key row from global memory once per
query row, and its reduce phase re-reads every value row
from global memory, per lane, inside the tile loop. The
profile's attack 4 prices halving flash at 507 ms at this
context and calls it plausibly the number one lever at
4k tokens.

## The spec taken from llama.cpp

All citations verified against the server tree
(/home/server/llama.cpp), which the parity note verified
identical to the VM mirror for the ggml-vulkan sources.

- Blocked form: `Bc = 32` KV positions per block,
  staged in shared memory; query rows blocked by `Br`
  (vulkan-shaders/flash_attn_base.glsl:4-21: WorkGroupSize
  128 default, Br 1, Bc 32, HSK/HSV specialized per head
  dim, D_split 16, row_split 1, SubGroupSize 32, and the
  SHMEM_STAGING flag as a first-class spec constant).
- AMD scalar tuning (ggml-vulkan.cpp:1200-1245,
  `get_fa_tuning_params_scalar`): on AMD non-GCN at head
  dim 128, row_split is 1 (hsk 128 meets the
  row_split_max_hsk of 128 for n_rows > 8) and block_cols
  is 32 ((hsk|hsv) = 128 has no factor 8). block_rows at
  row_split 1 is 8 for n_rows > 8 when n_kv >= 1024, and
  the workgroup size resolves to subgroup_size * 4 = 256
  at the device subgroup size of 64.
- Path choice (ggml-vulkan.cpp:1334-1375,
  `get_fa_tuning_params`): a single query row always
  takes the scalar path ("scalar is faster than coopmat
  when N==1"). This lane's routing mirrors that rule
  against our own forms: one-row calls keep the shipped
  kernel even with the switch on.

One nuance, recorded so it cannot be misread later:
llama's AMD scalar tuning sets SHMEM_STAGING to 0 (the
flag is enabled only for NVIDIA in the scalar tuner), and
its scalar path reads K from global memory with vec4
loads. The staged layout is the form its matrix paths
(coopmat1/coopmat2) run and the form the base spec
constants describe. This lane builds the staged form as
its charter orders, against OUR shipped kernel as the
baseline; the card A/B below is the judge, and no claim
of AMD-scalar parity is made for it.

## The design (FLASH_STAGED_SHADER in k6_attention.rs)

Same bindings (q, keys, values, out, params at 0..4),
same FlashParams struct, same output layout as the
shipped flash kernel, so the one resident call site swaps
forms without moving a byte. KV stays f32 end to end
(the f16 KV change is a separate gated step, per gap 5's
own scoping), and nothing here touches the coopmat path
(the bridge lane's).

- Grid: one workgroup of 256 threads per (block of
  Br = 8 query rows, Q head). At the prefill chunk shape
  (64 rows, 32 heads) that is 256 workgroups against the
  shipped form's 2,048.
- Thread map: `tid / 32` is the query row inside the
  block, `tid % 32` the lane. Score phase: lane j scores
  key j of the block (full head-dim dot from the staged
  tiles). Reduce phase: lane l owns channels l, l+32,
  l+64, l+96 of its row.
- Phases per 32-key block: stage K (coalesced global
  loads, slot-mapped exactly as the shipped kernels slot:
  ring modulus is the ring capacity, never the window),
  barrier, score into the score tile, barrier, stage V
  into the same tile, barrier, online-softmax reduce
  reading V from the tile, barrier. Four barriers per
  block. The query block is staged once before the walk.
- Bank behaviour: the K/V tile's row stride is padded by
  one float (head_dim + 1). The stride is odd for every
  even head dim, hence coprime with the 32 workgroup
  banks, so the score phase's reads across keys and the
  reduce phase's reads across channels are conflict-free
  at the head dims this engine serves (128, 64). The
  query tile is read broadcast (a warp shares its row).
- Math contract: the per-element float operation
  sequence is the shipped flash kernel's, exactly. Same
  ascending-channel dots over the same values, same
  scale multiply, same -1.0e30 mask sentinel and -1.0e29
  skip test, same per-key online-softmax rescale in
  ascending key order, state carried across blocks. Only
  the storage operands pass through differs. Outputs
  therefore agree with the shipped form to the bit in
  exact arithmetic, and to driver float-contraction
  noise in practice; the device tests hold them to the
  same 1e-6 the shipped form is held to against the
  two-pass form.
- Bounds: rows past the call's row count stage as zeros,
  compute, and never write out, so every barrier is in
  uniform control flow. Head dim is capped at 128 by the
  shared `check_geom` (the query tile is sized at that
  bound). Staging reads only slots the shipped form
  reads: every key in the stored range is visible to at
  least the block row at its own position.

## LDS arithmetic (per workgroup)

| Array | Shape | Floats | Bytes |
| --- | --- | --- | --- |
| kv_tile | 32 keys x 129 stride (128 + 1 pad) | 4,128 | 16,512 |
| q_tile | 8 rows x 128 (head-dim bound) | 1,024 | 4,096 |
| s_tile | 8 rows x 32 keys | 256 | 1,024 |
| **Total** | | **5,408** | **21,632** |

`FLASH_STAGED_LDS_BYTES` = 21,632 in the code, pinned
against the shader's declared arrays by a unit test. The
pipeline builds only when the device grants at least
that much workgroup storage (the K13 for_device terms):
the backend requests the adapter's own limits, RADV
grants 64 KiB (65,536 B), so on the server card the set
fits with the CU residency bound at 3 workgroups per CU
by LDS (3 x 21,632 = 64,896 <= 65,536). Under wgpu's
default limits (16,384 B) the pipeline is absent and the
absence is logged by name at construction; the shipped
form's own tile is 1,024 B, so LDS never binds for it.

## The switch and the routing

- `WGPU_COMPUTE_FLASH_STAGED` (crate::compute): the
  exact value `1` (trimmed) enables; unset or any other
  value is off, fail-closed, resolved once at backend
  construction, in the K22 switch's pattern. The backend
  construction log prints both the switch and whether
  the staged pipeline exists, so a run that cannot
  engage names it instead of timing the shipped form
  under the staged name.
- Routing (session.rs, the one prepare_flash call site):
  staged form iff switch on AND rows > 1 AND the staged
  pipeline exists on this device. Otherwise the shipped
  call, unchanged. With the switch unset the dispatch
  stream is the shipped stream exactly.
- Diag: the staged form stamps `k6_attn_flash_staged`
  from birth (the shipped form stamps `k6_attn_flash`),
  so a timing run attributes every dispatch to its form.

## VM gates (executed 2026-11-06, this lane)

- `cargo test -p wgpu-compute`: 482 passed, 0 failed
  across 25 test binaries. The new unit tests run on the
  VM for real: the resolver semantics
  (`flash_staged_switch_resolution`), the LDS pin
  (`staged_lds_arithmetic_matches_the_shader`), and full
  Naga parse + validation of the staged shader
  (`staged_shader_validates_under_naga`). The generated
  WGSL was also dumped and read end to end by the author.
- The device correctness tests
  (`k6_flash_staged_matches_shipped_at_glimmer_shapes`
  in tests/fused_kernels.rs) skip on the VM, which
  enumerates no adapter, exactly like every device test
  in this crate; they are step 1 of the card protocol.
  Their shapes are the real model's: head dim 128, 32 Q
  heads over 2 KV heads, the 2,048 window with ring
  2,111 at deep positions, block edges at stored 31/32/33
  and rows 9, a 64-row global chunk over 1,024 stored,
  and decode rows.
- `cargo clippy -p wgpu-compute --all-targets -- -D
  warnings`: clean. `cargo fmt --check`: clean.
  `cargo check --workspace --all-targets`: clean.

## Lose-conditions (named in advance)

1. **Occupancy.** 21,632 B of LDS per workgroup caps
   residency at 3 workgroups per CU (768 threads, 37.5
   percent of the 2,048-thread ceiling), where the
   shipped form is thread-bound at 8. If the score
   phase's serial 128-deep dots cannot hide
   workgroup-memory latency at that occupancy, the
   staged form loses despite strictly fewer global
   reads. This is the primary risk and the A/B exists
   to measure it.
2. **Small stored counts.** Four barriers per block
   plus staging loads are pure overhead when the stored
   set fits in one or two blocks (early chunks, short
   prompts). Flash is 3.2 ms in chunk 0; there is
   nothing to win there and a little to lose.
3. **The verify band.** Multi-row forwards of 2..7 rows
   run one row block with 6..1 of 8 rows idle in the
   score and reduce phases (staging is shared, so the
   waste is compute, not traffic), and launch only
   32 workgroups (one block x 32 heads), far under the
   card's 96 CUs. Spec verify may prefer the shipped
   form; the identity+golden leg runs it, the timing
   there is read from the same runs' diag tables.
4. **f32 traffic is unchanged in size.** Staging moves
   the same f32 bytes llama moves at f16. The win on
   the table is coalescing, shared reuse across 8
   rows, and no per-row global re-read of V, not halved
   traffic. If the shipped form's V reads were already
   landing in L2 at these shapes, the measured win will
   be smaller than the traffic arithmetic suggests.
5. **Shared-memory throughput.** The score phase
   issues 256 threads x head_dim shared loads per
   block. Conflicts are designed out, but raw shared
   bandwidth may bind before the saved global bandwidth
   pays.

## Card protocol (parent runs this, serialized)

Stage from the post-merge main, manifest-verified,
under `flock -w 14400 /tmp/itak-phase.lock`, VRAM guard
(33,271,808 B) before every measured arm. The Cortex
backends are stopped for the session and restored
after, per the standing card rule.

- **Step 0, autopsy.** Compile the staged shader under
  RADV_DEBUG=shaderstats (bench/shader-autopsy.sh
  pattern): LDS must read 21,632 B, VGPR at or under
  128, zero spills, zero scratch. A spill-heavy or
  LDS-mismatched compile stops the protocol here; the
  numbers are the finding.
- **Step 1, correctness.**
  `cargo test -p wgpu-compute --release --test
  fused_kernels k6_flash_staged` on the card: every
  case must run (no skip line for the staged pipeline)
  and pass at 1e-6 against the shipped flash and, where
  sized, the two-pass form. Then the resident suite
  with the switch on: `WGPU_COMPUTE_FLASH_STAGED=1
  cargo test -p wgpu-compute --release --test
  resident_session`. Then identity + golden through
  the spec harness with the switch on (both mode,
  depths 1 and 6): IDENTITY + GOLDEN PASS.
- **Step 2, prefill A/B at 1,022 tokens.**
  `prefill_profile` on the real GGUF, diag timing on,
  two arms in one session: switch unset (control) and
  `WGPU_COMPUTE_FLASH_STAGED=1`. Compare the
  `k6_attn_flash` total against the
  `k6_attn_flash_staged` total and the walls. Recorded
  marks: flash 1,013.1 ms of 6,659 ms GPU time; the
  current shipped wall is 6,330 ms (2026-11-06 host-gap
  confirmation). The same-session control arm is the
  operative baseline; the recorded marks are context.
- **Step 3, long-context leg.** Same harness, same two
  arms, prompt built by repetition to at least 8,192
  tokens (the harness prints `prompt_tokens=`; the leg
  is valid only if that line reads >= 8192). No recorded
  baseline exists at this length; the comparison is
  same-session, flash total against flash total and
  wall against wall. This is the leg the design exists
  for: flash's share grows quadratically, so a staging
  win must show here if it is real.
- **Step 4, no-leak.** Switch unset, diag timing on,
  one plain run: the diag table shows `k6_attn_flash`
  rows and zero `k6_attn_flash_staged` rows, and the
  construction log reads `enabled=false`. With the
  switch ON, a plain glimmer_generate decode run must
  still land in the recorded 35.5-35.7 tok/s band
  (single-row calls never route staged by construction;
  this measurement is the proof of that construction).

## Decision rule (written in advance)

The default does not move in this lane. A proposal to
flip `WGPU_COMPUTE_FLASH_STAGED`'s default (or to make
the staged form unconditional for multi-row calls) is
earned only if ALL of these hold:

1. Step 0: LDS reads 21,632 B, VGPR <= 128, zero
   spills.
2. Step 1: every correctness leg passes, including
   identity + golden with the switch on.
3. Step 2: the staged flash total at 1,022 tokens
   beats the recorded 1,013.1 ms by at least 25
   percent (<= 760 ms), with the wall improved
   against the same-session control.
4. Step 3: the staged flash total at >= 8k tokens
   beats the same-session control's flash total.
5. Step 4: no-leak clean, and decode with the switch
   on stays in the recorded band.

If step 2 fails but step 3 passes decisively, the
honest follow-up is a stored-count gate (staged only
above a context threshold), proposed with its own
numbers, not taken here. If both fail, the flag stays
off, the numbers are the finding, and gap 5's remaining
half is the f16 KV step, which this lane deliberately
did not touch.
