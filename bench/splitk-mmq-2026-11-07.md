# Split-K for the MMQ prompt path (K13 Q4_K, K14 Q6_K) - design, factor rule, drift analysis, and the card verdict protocol

Lane `lane/splitk-mmq`, built 2026-11-07 from main
`0ecdc8ff5efc3da0d00b5806c8f2796b65096132`. State: code
complete and VM-gated (counts below); the card verdict
protocol is written and NOT yet run (the parent
serializes card time in a consolidated session). The
routing config defaults OFF; nothing in the shipped
ladder changes with the env unset.

## The problem, from the measurements

The prefill profile (`bench/prefill-profile-2026-11-05.md`,
carried in `bench/baseline-session-2026-11-06.md`) put
83.2% of the 1,022-token prefill's GPU time in the MMQ
pair, K13 (Q4_K) and K14 (Q6_K) streaming at ~43.5 GB/s
combined against a card that streams a GEMV at ~650
GB/s in the same runs. The worst single shape is the
256-wide Q6_K KV projection: 416 dispatches at 3.2 GB/s,
about 186 ms of the prefill.

The shape math explains it. An MMQ workgroup owns a
whole output tile (64 rows at the shipped m64 shape)
and walks the entire reduction serially: for a 256-row
output the grid is 4 workgroups, each walking all 26
super-block columns of the 6656-wide input. Four
workgroups on a 96-CU card is about 4% occupancy of the
machine for that dispatch, and no register or staging
organization fixes a grid that small. llama.cpp's
`mul_mm` has a split path for exactly this reason; this
lane takes the design (partition the reduction, write
partials, reduce) and re-derives it in this tree's WGSL
and routing idiom, not the GLSL code.

## Design

**Split product.** The split shader is generated from
the same generator as the plain product
(`mmq_shader_split(shape, split)` in each kernel
module), so staging, the DP4A inner product, and the
per-sub-block fold are byte-for-byte the plain kernel's
code. Three things change:

- The `Params` uniform gains `splits: u32`. The factor
  is a runtime parameter, not a generated literal, so
  one compiled pipeline per family serves every factor.
- The super-block loop walks the workgroup's slice of
  the range: slice `s` (the grid's `wid.y`) covers
  `[s * per_split, min((s+1) * per_split, blocks))`
  with `per_split = ceil(blocks / splits)`. Slices
  partition the columns disjointly and completely; a
  slice past the end (factor greater than the block
  count) walks an empty range and stores its zero
  accumulators, so any factor in range is correct for
  any width.
- The stores index a partials buffer laid out
  `[splits][batch][out_rows]` instead of the output.

**Reduction.** `kernels/splitk_reduce.rs`: one thread
per output element sums the slices in ascending order
(`acc = p[0][i]; acc += p[1][i]; ...`), workgroup 256,
grid `ceil(total / 256)`. Pure bandwidth: at the KV
shape with factor 8 and a 64-row chunk it reads
8 x 64 x 256 x 4 B = 512 KB and writes 64 KB. The order
is fixed by construction (no atomics, no tree), so a
split run is deterministic run to run.

**Ownership.** The partials scratch is allocated by the
split prepare (`splits * batch * out_rows` f32) and
lives in the returned call's bind groups, mirroring how
the existing calls own their params buffers. The split
call type (`K13SplitCall` / `K14SplitCall`) encodes
product then reduction into the caller's open pass;
dispatches within one pass are ordered, the same
composition the K12 quantize + product chain relies on.
Both split pipelines and the reducer compile inside
`K13Pipeline::with_shape` / `K14Pipeline::with_shape`
regardless of the routing config, so a shader defect in
either form fails at backend construction, not at first
routed use.

**Routing seam.** The split is a property of the
standard product, not a new rung: inside
`mmq_product_q4k` / `mmq_product_q6k`, after the
coopmat / wide / K22 / K23 rungs have had their say,
the standard rung asks the backend for the projection's
factor and prepares the split form when it is 2 or
more, falling back to the unsplit prepare with the
refusal logged by name. It deliberately does NOT stack
on the opt-in prototypes (coopmat, K22, K23, wide):
those are unproven forms, and stacking two unproven
dispatch shapes would make any card result
unattributable. The session gains two `ProjCall`
variants (`MmqSplitQ4K`, `MmqSplitQ6K`) whose dispatch
accounting is product + reduction (+ quantize when
carried), and whose kernel names are the split stamps.

**Config.** `WGPU_COMPUTE_MMQ_SPLITK`, resolved once at
backend construction in the codebase's fail-closed
pattern:

- unset or `off` -> off (the default);
- `1`..=`8` -> a fixed factor;
- `auto` -> the per-projection rule below;
- anything else -> off, with the unrecognized value
  named in the construction log
  (`log_mmq_splitk_fallback`, the flash/K21F pattern).

The resolved mode is recorded as the `mmq_splitk` pair
(`off` / `fixed:N` / `auto`) in the diag manifest's
`resolved_config`, in both the resident and dflash
sessions, beside `k22_mmq` and `k21f_head`. Diag stamps
distinguish split dispatches: `k13_q4k_mmq_split` and
`k14_q6k_mmq_split` for the products (their shape field
carries `splits=N`), `splitk_reduce` for the reduction.

## The factor rule and its rationale

`mmq_splitk_factor(mode, tiles, blocks_per_row)` in
`compute.rs`, where `tiles` is the unsplit grid size
(output rows / tile) and `blocks_per_row` the
super-block columns:

- Off -> 1.
- Fixed(n) -> `n` clamped to `[1, min(8, blocks)]`:
  past the block count the extra slices are empty, so
  the cap is efficiency, not correctness.
- Auto -> `ceil(64 / tiles)` clamped to
  `[1, min(8, blocks)]`: target roughly 64 workgroups,
  about two-thirds of the card's CUs, leaving the
  scheduler room to overlap the tail.

Against the Glimmer shapes (tile 64, prefill chunks):

| projection | out | in | tiles | blocks | auto factor | split grid |
|---|---|---|---|---|---|---|
| K/V proj (Q6_K) | 256 | 6656 | 4 | 26 | 8 | 32 |
| Q proj (Q4_K) | 6656 | 6656 | 104 | 26 | 1 | 104 |
| O proj (Q4_K) | 6656 | 6656 | 104 | 26 | 1 | 104 |
| FFN gate/up (Q4_K) | 19968 | 6656 | 312 | 26 | 1 | 312 |
| FFN down (Q4_K) | 6656 | 19968 | 104 | 78 | 1 | 104 |
| any out <= 2048 | <=2048 | any | <=32 | any | >=2 | >=64 |

Under auto, exactly the starved shapes split and
everything already filling the card is untouched. At
the KV shape with factor 8 the slices walk
4/4/4/4/4/4/2/0 blocks (per_split = ceil(26/8) = 4):
the last slice is empty and the seventh walks 2, a
known imbalance the card A/B should weigh against
factor 4 (7/7/6/6, grid 16) and factor 2 (13/13,
grid 8). The fixed mode exists so the verdict can
sweep the factor directly instead of trusting the
rule.

**Costs, stated.** Total staged bytes are unchanged:
slices partition the columns, so every super-block
column is staged exactly once across the grid, by
exactly one slice (weights and activations alike).
The new traffic is the partials round trip:
`splits x batch x out_rows` f32 written and re-read,
plus one reduction dispatch per projection. At the KV
shape, factor 8, batch 64: about 1 MB of extra traffic
against 13.7 MB of weight bytes the dispatch streams -
worth paying only if the occupancy win is real, which
is the card's question, not this note's claim.

## Drift analysis

The split product is NOT bit-identical to the unsplit
product, and this lane does not claim it is:

- The integer dots are exact and identical: each
  slice's DP4A sums cover the same elements with the
  same operands as the unsplit kernel's.
- Within a slice, the f32 fold sequence (per
  sub-block, ascending super-block order) is the
  unsplit kernel's sequence restricted to that slice's
  columns: same terms, same order, same expressions.
- What changes is only the grouping of the f32
  accumulation: the unsplit kernel folds all 26
  super-blocks into one chain; the split form folds
  each slice into its own chain starting from 0.0 and
  adds the slice sums in ascending slice order. f32
  addition is not associative, so the results differ
  by reassociation noise - the same drift class the
  existing MMQ gate already bounds against the host
  reference running the same quantized math
  (`check_close`: `2e-3 + 5e-5 x set scale`).
- Empty slices contribute exact zeros: a stored 0.0
  partial added to a finite sum changes nothing (the
  only edge is the sign of a zero result when every
  partial is zero, which is +0.0 in both forms: both
  accumulation chains start at +0.0).

Bit-identity would require reproducing the unsplit
chain's exact left-to-right order across slices, which
serializes the accumulation and defeats the split. The
identity question for the card session is therefore
the standing one for reassociation-class changes: does
the token stream stay identical on the golden prompts
(the same evidence standard the shipped defaults
already carry - outputs token-identical with
probability deltas inside the f32 accumulation-noise
band).

The gate in `tests/splitk_mmq.rs` pins the split
product against the host quantized-math reference AND
against the unsplit GPU product, both at the
reassociation bound, at: the KV shape (6656 in, 256
out) at factors 2, 4 (Q6_K), and 8; a ragged width
(1280 in = 5 blocks, factor 4: ranges 2/2/1/empty);
and a factor past the block count (768 in = 3 blocks,
factor 8: five empty slices). On this VM the tests
skip GPU execution (no adapter) by the suite's
standing pattern; their GPU legs are owed on the card
(see below).

## Autopsy readiness (what the VM could prove)

Naga validation runs green here for every generated
variant (plain and split, m32 and m64, both families,
plus the reducer): the module unit tests
`plain_and_split_shaders_validate_under_naga` parse and
validate each generated shader with
`ValidationFlags::all()`. That proves the WGSL is
well-formed and well-typed; it cannot prove driver
acceptance or register behavior.

The register expectation, from the generated code: the
split shader is the plain shader's body plus three
index scalars (`per_split`, `blk_first`, `blk_last`)
and a multiply in the store index. The accumulator
shape is unchanged (16 f32 accumulators, the unrolled
16-wide i32 dot set per sub-block), and the workgroup
storage is unchanged (the same staging arrays:
41,216 B at m64 for Q4_K, 45,312 B for Q6_K). So the
split forms should land at the plain kernels' VGPR/LDS
picture (the plain Q4_K MMQ compiles at the VGPR 256
cap per the K22/K23 autopsies) with no new spill
pressure - but a compiler is free to surprise, and
spills are exactly what this VM cannot see. The
reduction kernel is one accumulator and streaming
reads; nothing about it can spill meaningfully.

One defect this lane caught in itself, recorded
because the gate pattern matters: the first Naga run
failed at end-of-file on BOTH plain and split shaders -
the split generator's loop header was a `format!`
VALUE carrying a doubled brace (`{{`), which survives
substitution literally and unbalances the shader. The
plain shader's identical header (moved into the same
value during the parameterization) broke with it. The
Naga tests now also assert no `{{` / `}}` appears in
any generated shader, so the escape-leak class is
pinned by name, not rediscovered by parse error.

## VM gates (this lane, 2026-11-07)

- `cargo test -p wgpu-compute`: lib 173 passed, 0
  failed, 1 ignored; every integration binary green,
  including `tests/splitk_mmq.rs` (3 passed; GPU legs
  skip on this adapter-less VM by design).
- `cargo test -p dflash`: 24 passed.
- `cargo test -p itak-spec`: 49 passed.
- `cargo clippy -p wgpu-compute --all-targets -- -D
  warnings`: clean.
- `cargo fmt --check`: clean.
- `cargo check --workspace`: clean.

New tests: resolver matrix and factor-rule pins
(`compute.rs`: `mmq_splitk_resolution`,
`mmq_splitk_value_recognition`, `mmq_splitk_factor_rule`);
shader text pins + Naga validation (K13, K14, reducer
modules); GPU correctness vs host reference and vs the
unsplit product (`tests/splitk_mmq.rs`).

## What the card session owes (exact list)

1. **Autopsy of the split shaders.** RADV/ACO VGPR,
   spills, and LDS for `k13_q4k_mmq_split` and
   `k14_q6k_mmq_split` at m64, against the plain
   kernels' known picture (VGPR 256, LDS 41,216 B for
   K13). Any spill the plain form does not have is a
   finding against the split form.
2. **Split GPU correctness legs.** Run
   `tests/splitk_mmq.rs` where an adapter exists: the
   KV-shape, ragged-width, and empty-slice cases at the
   reassociation bound, plus the printed split-vs-unsplit
   drift numbers (record them in the verdict note).
3. **Prefill A/B, interleaved** (the wave17/18
   standard: alternating arms, not sequential blocks).
   At 1,022 and 8,202 tokens: off vs `fixed:4` vs
   `fixed:8` vs `auto`. Read per-kernel time for
   `k14_q6k_mmq_split` + `splitk_reduce` on the KV
   projections and the prefill wall; the question is
   whether the 3.2 GB/s KV shape moves toward the
   pair's ~43.5 GB/s combined rate or better, and what
   the reduction costs.
4. **Identity + golden with the flag on.** Speculative
   identity and the 5 golden prompts at
   `WGPU_COMPUTE_MMQ_SPLITK=auto` and at `=8`: token
   identity must hold under the reassociation drift
   (the standard every prior routing flip met).
5. **Manifest and stamp check.** A capture with the
   flag on shows `mmq_splitk` in `resolved_config` and
   `k13_q4k_mmq_split` / `k14_q6k_mmq_split` /
   `splitk_reduce` stamps in the profile, with split
   stamps ONLY on the shapes the factor rule selects.

**Flip criterion (for the parent to judge, not this
lane):** the default moves only if (3) shows a real
prefill win at both lengths with no regression on the
unsplit shapes, (4) holds, and (1) shows no new spills.
Until then the shipped ladder with the env unset is
byte-for-byte the pre-lane ladder.

## Judgment calls in this lane

- Split applies at the standard rung only, never
  stacked on the coopmat/wide/K22/K23 prototypes:
  attribution of any card result stays clean.
- Contiguous ceil-division slice ranges (llama's
  design shape) over a balanced interleave: within a
  slice the fold order matches the unsplit kernel's,
  keeping the drift argument one sentence long; the
  tail imbalance at 26 blocks / factor 8 is accepted
  and named above for the A/B to weigh.
- Partials scratch is per-prepare (the K13 params
  buffer precedent), not a plan-owned arena: plans
  stay free of a new lifetime, at the cost of one
  allocation per split projection per plan build.
- The auto rule's 64-workgroup target is a heuristic
  from the shape table, not a measured optimum; the
  fixed mode is the verdict's sweep instrument.
- Q5_K has no MMQ product in this tree's prompt path,
  so there is no Q5_K split form; inventing one was
  out of scope by the tasking.

## Supersession (2026-11-07): the default moved to fixed factor 4

This note's history stands; its config section's
"default OFF" statement and the flip criterion's
closing position ("until then the shipped ladder
with the env unset is byte-for-byte the pre-lane
ladder") are superseded as follows.

The wave20 consolidated card session
(bench/wave20-session-2026-11-07.md, leg 1) ran this
note's verdict protocol and read against the flip
criterion: fixed factor 4 shows a real prefill win
at BOTH lengths (interleaved walls about -18% at
1,022 tokens, about -14% at 8,202) with the split
forms faster than plain on the wide shapes too (no
unsplit-shape regression in the kernel lines), the
split GPU legs executed with reassociation-class
drift, identity + golden held at `auto` and at
`=8`, and the family autopsy showed no new spills
(per-form attribution by name was not isolated;
the session scoped that leg partial). Fixed 8 is
disqualified by its own numbers (+50% / +88%
walls), and auto's win is marginal (about -3.5% /
-2%) with its factor-8 pick for the KV shape
implicated in the factor-8 collapse.

Under David's standing order (2026-11-07) that
anything the evidence shows makes the engine faster
is set on, the shipped default moved to fixed
factor 4 in lane `lane/splitk-default-flip`: unset
now resolves to `fixed:4` (`DEFAULT_MMQ_SPLITK`);
explicit values keep their meanings (`off` closes
it and is the rollback, `1`..=`8` fix the factor,
`auto` selects the per-projection rule, whose
arithmetic is unchanged); a present but
unrecognized value still fails closed to off
(`MMQ_SPLITK_FALLBACK`), never to the new default.
The flip is recorded in
bench/splitk-default-flip-2026-11-07.md, which
carries the full semantics matrix. Factor 8's
disqualification is recorded there so it is not
re-proposed as a default without new evidence.
