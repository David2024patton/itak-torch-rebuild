# Coopmat default flip: the rung and the K24 selector ship ON (2026-11-07)

Lane `lane/coopmat-default-flip`, from main
333f35407f38e9b99d40a412e86dcf41569bbad6. The Q4_K
coopmat routing switch (`WGPU_COMPUTE_COOPMAT`)
and the K24 v2 selector inside it
(`WGPU_COMPUTE_COOPMAT_V2`) flip their shipped
defaults from OFF to ON, together, in the shape of
the three earlier flips (MMV ceiling b342cdf,
staged flash 82017ad, K21F head d9825eb): named
`DEFAULT_` / `_FALLBACK` constants, unset resolves
to the shipped default, a present but unrecognized
value resolves to the closed fallback (never to
the default) and is logged by name at
construction. VM gates only on this lane; no
server GPU jobs. The card evidence this flip
stands on was measured by the wave20 session,
cited below; this lane changed the two resolvers,
the bridge implication the rung needs, their
docs, and their tests. The kernels, the routing
order, and the fallback chain are untouched.

Authority: David's standing order of 2026-11-07,
"anything that makes sense engine go faster
should be set to on", applied to the wave20
verdict.

## The evidence

1. **The build and its rule**
   (bench/coopmat-v2-2026-11-07.md). K24 is the
   register-resident rebuild of the v1 coopmat
   kernel (K23): fragments and folds live in
   registers instead of staging through workgroup
   memory, the defect the v1 autopsy convicted
   (v1 measured 4.0x SLOWER than K13 on the card
   and its switch stayed off). Its element mapping
   at wave64 could not be transcribed from llama
   (which forces wave32 on AMD), so it was derived
   from Mesa's lowering source and cross-validated
   against the wave32 case; the note's decision
   rule, written in advance: K24 earns a
   default-flip proposal only on a material
   kernel-line win over K13 with speculative
   identity intact, and the numeric gates run
   first because they are the mapping proof.
2. **The verdict** (bench/wave20-session-2026-11-07.md,
   leg 2, staged from main 515ce23, all arms
   interleaved, every run failure-bundled):
   - Numeric gates EXECUTED on the card:
     tests/k24_coopmat.rs 3 passed, 0 failed, no
     skips. K24 against the host reference at
     relative error at most 1.158e-6; against K13
     at most 2.9e-7. The wave64 mapping is PROVEN.
   - Kernel line at the 1,022-token prefill shape,
     2,912 dispatches per arm: k24_q4k_coopmat
     1,737.2 / 1,751.9 ms (112 GB/s) against
     k13_q4k_mmq 4,021.0 / 4,046.5 ms (48 GB/s),
     about 2.3x faster. Prefill walls 4,052 /
     4,085 ms against 6,307 / 6,401 ms, about 36
     percent lower. The K23 v1 arm in the same
     session: 16,900.4 / 16,963.0 ms, reproducing
     its standing verdict (correct, ~4x slower);
     K23 stays a fallback, never the default.
   - Identity + golden with COOPMAT=1 +
     COOPMAT_V2=1 (+ BRIDGE=1): IDENTITY PASS and
     GOLDEN PASS, the spec stream byte-identical
     to the golden stream, spec accounting
     (rounds=38, proposed=37, accepted=25)
     identical to the shipped-default control.
     The diag manifest carried
     "coopmat":"true", "coopmat_v2":"true".
   - Autopsy: K24 LDS 33,792 B (exactly the five
     staging arrays), VGPR 132 (not the 256 cap),
     zero spills, zero scratch.

   The decision rule is MET, on every clause.

## Why both switches together

The rung and its selector are one feature.
Defaulting only the rung would ship the coopmat
path serving K23, the kernel measured ~4x slower
than K13: a regression wearing the flip's name.
Defaulting only the selector would change nothing
at all: the selector is consulted only inside an
armed rung. So `DEFAULT_COOPMAT` and
`DEFAULT_COOPMAT_V2` move in the same commit, and
each env keeps its own independent rollback.

## The bridge question

The coopmat pipelines exist only on a device the
coopmat bridge opened (VK_KHR_cooperative_matrix
enabled through wgpu-hal interop), and the bridge
had its own exact-"1" switch,
`WGPU_COMPUTE_COOPMAT_BRIDGE`, read in
`GpuContext::open` before either routing resolver
exists. Every wave20 arm set BRIDGE=1 explicitly.
With the routing defaults on and the bridge left
exactly as it was, an unset run would open the
stock device, find no coopmat pipeline, refuse by
name at plan build, and serve K13: the flip would
be a log line, not a route. The smallest coherent
change, made in `coopmat_bridge.rs`:

- `coopmat_bridge_requested()` now reads: a
  PRESENT bridge value decides on its own, exactly
  as before (`1` opens the bridge; anything else
  refuses it, logged by name however the decision
  lands), so `WGPU_COMPUTE_COOPMAT_BRIDGE=0` stays
  a deliberate stock-device run even with the
  routing default on, and a malformed bridge value
  can never be widened by the routing default.
- With the bridge variable UNSET, the request
  defers to the routing switch's unset position
  (`compute::resolve_coopmat(None)`, which is
  `DEFAULT_COOPMAT`): routing default on means the
  device opens through the bridge so the rung can
  engage; `WGPU_COMPUTE_COOPMAT=0` means nothing
  asks for a bridged device and the stock open
  serves byte-for-byte as before the flip.

Nothing about the bridge itself changed: the
probes (extension exposed, feature bit, the
required s8 x s8 to s32 16x16x16 subgroup property
row) still gate every bridged open, every refusal
is still named on the diag channel with the stock
device serving instead, and the defaults-confirmation
session already proved the bridge numerically
transparent (bench/defaults-confirmation-2026-11-07.md,
leg 3: a bridge-only run's stream byte-identical
to the stock run's). Consequences of the new
default, stated plainly: every default run now
pays the bridge's device-open probes (read-only
driver queries at startup) and, on a card without
coopmat, logs the bridge refusal once and serves
the stock ladder exactly. On the target card the
probes pass and the rung engages.

## The semantic matrix (final, both envs)

| env value | rung (`WGPU_COMPUTE_COOPMAT`) | selector (`WGPU_COMPUTE_COOPMAT_V2`) |
|---|---|---|
| unset | ON (the shipped default) | SELECTED: K24 serves the rung |
| `1` (trimmed) | ON | SELECTED |
| `0` (trimmed) | OFF (explicit off; the rollback) | DESELECTED: K23 serves the rung (explicit off; the rollback) |
| anything else | OFF, fail-closed | DESELECTED, fail-closed |

A malformed value resolves to the closed fallback
(`COOPMAT_FALLBACK` / `COOPMAT_V2_FALLBACK`, both
`false`), NOT to the shipped default, and the
refusal is logged by name at construction:
`mmq coopmat env WGPU_COMPUTE_COOPMAT=<value> is
not a recognized value ("1" or unset enables, "0"
disables); failing closed, the coopmat rung is
OFF`, and the selector's twin line. The
construction log lines now read
`mmq coopmat config: enabled=<bool> (env
WGPU_COMPUTE_COOPMAT, "1" or unset enables, "0"
disables); k23_coopmat=<bool>` and
`mmq coopmat v2 config: selected=<bool> (env
WGPU_COMPUTE_COOPMAT_V2, "1" or unset selects
K24, "0" selects K23; consulted only when the
coopmat switch is on); k24_coopmat=<bool>`.
Resolution still happens once at backend
construction; the `coopmat` / `coopmat_v2`
accessors and the resolved pair in both sessions'
diag manifests are unchanged in shape and now
report the flipped defaults.

Rollbacks, each one variable, no rebuild:
`WGPU_COMPUTE_COOPMAT=0` closes the rung (the
pre-flip ladder exactly); `WGPU_COMPUTE_COOPMAT_V2=0`
keeps the rung and restores K23 as its form;
`WGPU_COMPUTE_COOPMAT_BRIDGE=0` forces the stock
device (the rung then refuses by name and the
standard ladder serves).

## What did not change

- The kernels (K23, K24, their shaders and
  artifacts), the routing order in
  `session::mmq_product_q4k` (coopmat rung first,
  K24 inside it, then wide, K22, K23-repack, K13),
  and the fallback chain: a K24 prepare refusal
  falls through to K23 by name, a K23 refusal to
  the standard ladder, and a missing pipeline
  refuses by name at plan build.
- The K23_MMQ repack switch (a different K23,
  measured 2.15x slower in its own verdict) stays
  exactly as it was: default off, untouched by
  this flip.
- Split-K, f16 KV, and every other default: this
  lane flips the coopmat pair only.

## Pins

- `compute::tests::coopmat_switch_resolution`
  re-pinned to the flipped matrix (unset on,
  `1` on, `0` off, malformed off), comments citing
  the wave20 kernel-line and wall numbers; the
  unset arm pins `DEFAULT_COOPMAT` and the
  malformed arms pin `COOPMAT_FALLBACK`.
- `compute::tests::coopmat_v2_selector_resolution`
  re-pinned the same way (unset selected, `0`
  deselects), citing the wave20 numeric-gate
  bounds (1.158e-6 against the reference, 2.9e-7
  against K13) and the 2.3x line; pins
  `DEFAULT_COOPMAT_V2` and `COOPMAT_V2_FALLBACK`.
- New `compute::tests::coopmat_value_recognition`
  and `compute::tests::coopmat_v2_value_recognition`,
  on the flash/K21F recognition-test pattern:
  the recognized set is exactly unset / `1` / `0`.
- `coopmat_bridge::tests::bridge_switch_is_fail_closed`
  unchanged: the explicit bridge resolver is
  untouched; the implication lives in
  `coopmat_bridge_requested`, whose env-reading
  arms are exercised on the card by the
  confirmation below (the VM has no bridged
  device to open).
- Doc updates where the old default was stated:
  the env, default, fallback, resolver, field,
  and accessor docs and the construction comments
  in compute.rs; the bridge gating docs in
  coopmat_bridge.rs and the open-site comment in
  device.rs; the ladder doc, both coopmat
  `ProjCall` variant docs in session.rs; the
  routing paragraphs in the K23 and K24 kernel
  module headers.
- bench/coopmat-v2-2026-11-07.md gains an appended
  supersession section (its history stands).

## Confirmation still owed (card, after merge)

One run at literally unset env (all three coopmat
envs unset: `WGPU_COMPUTE_COOPMAT`,
`WGPU_COMPUTE_COOPMAT_V2`,
`WGPU_COMPUTE_COOPMAT_BRIDGE`) on the real file,
staged from the merged commit with the staged
hash verified, tenants cleared under the standing
card rule and restored after, wrapped in
bench/failure-bundle.sh:

1. spec_generate_gpu, both mode, depth 1 (the
   wave20 identity arm's shape): IDENTITY PASS
   and GOLDEN PASS required, with spec accounting
   matching the shipped-default control
   (rounds=38, proposed=37, accepted=25).
2. The diag census must show the rung engaged by
   default: `k24_q4k_coopmat` stamps present,
   `k23_q4k_coopmat` stamps zero in the spec run,
   and the diag manifest carrying
   "coopmat":"true", "coopmat_v2":"true".
3. A 1,022-token prefill at unset env: the
   k24_q4k_coopmat stamp count is 2,912 (the full
   Q4_K dispatch count, equal to wave20's flagged
   K24 arm and to K13's count), and its kernel
   line lands in the wave20 band (1,737-1,752 ms,
   inside interleaving noise), against the K13
   control band (4,021-4,047 ms) it replaces.
4. The rollback arms in the same session, one
   run each on the same harness: COOPMAT=0 (zero
   coopmat stamps, K13 serves, identity intact),
   COOPMAT_V2=0 (k23_q4k_coopmat stamps present,
   k24 stamps zero), BRIDGE=0 (zero coopmat
   stamps, the plan-build refusal logged by name,
   identity intact).
