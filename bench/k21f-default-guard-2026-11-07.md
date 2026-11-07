# K21F default guard (2026-11-07)

Lane `lane/k21f-default-guard`, from main 7940bcf.
This note records a safety correction, not a kernel
fix: the K21F finish-carrying head route gets its
own fail-closed routing switch, default OFF. The
kernel (`k21_q5k_mmv_finish`) is untouched, and no
other MMV route changes.

## The card failure that forces this

The defaults-confirmation session
(bench/defaults-confirmation-2026-11-07.md, staged
from main 7940bcf70, real Glimmer Q4_K_M file plus
the DFlash drafter, card cleared, phase lock held)
ran its leg 1 against the exact configuration
shipped on main: MMV ceiling open to 16, staged
flash on, K21F head routing live behind the shared
MMV gate.

- Exposure PASSED: `k21_q5k_mmv_finish` fired 36
  times at depth 1, every one at the block batch
  with `finish=2`; the batch-1 head stayed on K3.
  The route was provably live.
- IDENTITY FAILED: plain and speculative streams
  agree through index 22 and diverge at index 23
  (plain 589, spec 1573). Deterministic: eight
  default-configuration runs in that session, at
  depths 1 and 6, all diverged at the same index.
- GOLDEN PASSED in every arm (the first 8 tokens
  precede the divergence).
- Control arm (the whole MMV gate closed,
  `WGPU_COMPUTE_MMV_MAX_ROWS=1`): identity and
  golden PASS, zero finish stamps. A rollback
  existed, at the price of every MMV route.
- Isolation: defaults plus `FLASH_STAGED=0` failed
  identically (staged flash is not a contributor);
  the baseline session's matrix supplies the other
  cells (MMV cap 16 without K21F passed; flash on
  without K21F passed). Between the baseline tree
  and 7940bcf70 the only spec-path delta that can
  move a verify argmax is the K21F chain. The
  session's reading: the K21F call as a unit
  (product core plus carried finish) produces head
  logits that flip a borderline verify argmax at a
  near-tie. The delta is below the fixture pins'
  cross-backend bar (max-abs 0.5): both fixture
  arms passed while the real file failed, so only
  a real-file near-tie exposes it.
- The leg-4 rates measured in that configuration
  (23.16 tok/s depth 1, 21.65 depth 6 medians) are
  NOT credited: the speed and the divergence are
  the same phenomenon until identity passes again.

Whether the delta sits in the product core or the
finish application is the fix lane's question.
This lane's question is narrower: a route proven
to break identity on the real file must not be
live by default while that fix is outstanding.

## Why the default closes

The route's only gate was the shared MMV ceiling
(`mmv_band_admits` over `WGPU_COMPUTE_MMV_MAX_ROWS`).
That ceiling is the plain products' gate (K19,
K20, K21), and its default opened to 16 on the
2026-11-06 baseline evidence, evidence gathered
for the plain products' shapes. The head route
opened with it as a side effect: nothing in the
flip's decision rule spoke for the head, whose
own verdict protocol (bench/k21-finish-2026-11-06.md)
had never run on the card. A shared gate lets one
route's proof silently arm another route. The
codebase's standing pattern for an unproven route
is a dedicated fail-closed switch (K22, K23,
coopmat: exact `1` enables, everything else off);
the head route now follows it.

## The change

- New env `WGPU_COMPUTE_K21F_HEAD`
  (`compute::K21F_HEAD_ENV`), resolved once at
  backend construction by `resolve_k21f_head` on
  the exact-`1` fail-closed pattern, stored as
  `WgpuCompute::k21f_head`, and logged by name at
  construction: `k21f head config: enabled=<bool>
  (env WGPU_COMPUTE_K21F_HEAD, only "1" enables)`.
  Semantic matrix:

  | value | resolves |
  |---|---|
  | unset | OFF (the shipped default) |
  | `1` (trimmed) | ON |
  | `0`, `true`, `11`, empty, any other text | OFF, fail-closed |

- `session::try_mmv_q5k_finish` now requires BOTH
  gates through the pure helper
  `k21f_head_admits(head_switch, max_rows, batch)`:
  the head switch on AND the shared band admitting
  the batch (the same `mmv_band_admits` arithmetic
  as before). Batch 1 remains K3 (the band floor
  refuses it at every switch position). The
  fallback on any refusal is the exact K3 fused
  call the site built before the route existed,
  asked for the identical finish; the finish math
  order is untouched.
- The resolved switch is recorded in both
  sessions' diag-capture manifests as the
  `k21f_head` resolved-config pair, beside
  `k22_mmq`, so a capture diff shows the route's
  position by name.
- Doc corrections where the old rule was stated:
  the `try_mmv_q5k_finish` and block-head site
  comments and the `mmv_band_admits` doc in
  session.rs; the construction comment, accessor
  docs, and env doc in compute.rs; the K21F
  paragraph in kernels/mod.rs; the routing
  paragraph in the K21F kernel module's header
  (docs only; the kernel's code and tests are
  untouched). bench/k21-finish-2026-11-06.md gains
  a supersession section pointing here; its
  earlier sections stand as the historical record
  of the design and the routing as landed.

## Pins

- `compute::k21f_head_switch_resolution`: the
  resolver matrix above, plus the env-name pin.
- `session::k21f_head_gate_requires_both_halves`:
  switch off admits nothing at ceiling 16 or even
  `u32::MAX` (the default-construction position:
  no K21F is prepared with the shared ceiling at
  its shipped 16); switch on admits exactly what
  the band admits and nothing when the ceiling is
  closed.
- The two pre-existing band pins stay, reframed as
  the gate's band half; the batch-1 pin now also
  asserts the combined gate refuses batch 1 with
  the switch on.
- `tests/k21f_head_routing.rs` is now three arms:
  switch-off (default construction; malformed
  switch values land in this arm by construction,
  which is the fail-closed pin), band-closed, and
  both-gates-open. The open arm requires
  `WGPU_COMPUTE_K21F_HEAD=1` in addition to the
  ceiling at 16. All arms keep the CPU-agreement
  half at the cross-backend bar. On this GPU-less
  VM all three arms self-skip; arm selection was
  verified with `--nocapture` in three env
  positions (unset: switch-off arm proceeds;
  `=1` with ceiling 16: only the open arm proceeds;
  `=true`: switch-off arm proceeds and the open arm
  skips naming the head switch).

## Re-enable recipe (for the fix lane)

```
WGPU_COMPUTE_K21F_HEAD=1
WGPU_COMPUTE_MMV_MAX_ROWS=16   # or unset: 16 is the shipped default
```

Both gates open, the route behaves exactly as it
did in the defaults-confirmation session, finish
stamps and all, so the fix lane reproduces the
failure and proves its fix in the same position
that failed.

## Rollback, and the reopen criterion

This guard IS the rollback from the broken
default: with it merged, shipped defaults serve
the block head on K3 while every plain MMV route
keeps its cap-16 opening. (On a tree without the
guard, the only rollback remains
`WGPU_COMPUTE_MMV_MAX_ROWS=1`, which closes all
MMV routes.) To close just the head route by hand
on a guarded tree: leave `WGPU_COMPUTE_K21F_HEAD`
unset or set it to anything but `1`.

The switch's default may move to on, or the guard
may come out, only when ALL of the following hold
on the card, on the real files, with the switch
explicitly on:

1. Speculative identity PASSES at depth 1 and at
   depth 6 (the two depths that diverged), in the
   same runs that show the exposure stamps.
2. GOLDEN PASSES in the same runs.
3. The exposure check shows `k21_q5k_mmv_finish`
   firing at the block batches (the pass must be
   the route's, not the control's).

Fixture-scale agreement does not count toward the
criterion: it passed while the real file failed.

## Gates on this VM

- `cargo test -p wgpu-compute`: 265 passed,
  0 failed, 28 test binaries (base 262; +1 resolver
  pin, +1 combined-gate pin, +1 integration arm).
- `cargo test -p dflash`: 24 passed.
- `cargo test -p itak-spec`: 49 passed.
- `cargo clippy -p wgpu-compute --all-targets
  -- -D warnings`: clean.
- `cargo fmt --check`: clean (one fmt pass over
  the new construction log line).
- `cargo check --workspace --all-targets`: clean.

Still owed to a card session (the fix lane's):
the three integration arms executing on the card,
and the reopen criterion above. Nothing in this
lane ran on the server.

## SUPERSESSION: the default flipped back to on (lane/k21f-default-flip, 2026-11-07)

The reopen criterion written above has been MET
and this guard's default is superseded. The chain:

1. The root cause was found by the read-only math
   audit and fixed in lane/k21f-mask-fix: a
   truncated dequant mask in the K21F shader
   (`0x0f0f0f` for `0x0f0f0f0f` on the qc line),
   dropping one nibble in sixteen elements per
   super-block. Record:
   bench/k21f-mask-fix-2026-11-07.md. The wave17
   diagnosis (bench/k21f-divergence-diagnosis-2026-11-07.md)
   measured the failure at the real geometry and
   carries the parent reconciliation of its
   attribution.
2. The wave18 card retest
   (bench/k21f-retest-2026-11-07.md) ran this
   note's reopen protocol against the fixed tree
   with the switch explicitly on: IDENTITY and
   GOLDEN PASS at depth 1 and depth 6 in the same
   runs that show the exposure stamps (criterion
   items 1-3 above), the kernel gates executing
   green on the card, and K21F's finished output
   bit-identical to K3's at the real head
   geometry. Verdict recorded there: REOPEN
   CRITERION MET.
3. Lane `lane/k21f-default-flip` therefore moved
   the switch's default: unset now resolves ON,
   in the MMV ceiling flip's shape. New semantic
   matrix (the matrix above is superseded):

   | value | resolves |
   |---|---|
   | unset | ON (the shipped default) |
   | `1` (trimmed) | ON |
   | `0` (trimmed) | OFF (the explicit off; the one-variable rollback) |
   | any other text | OFF, fail-closed, logged by name at construction |

   The switch itself, the combined gate
   (`k21f_head_admits`: switch AND shared band),
   the batch-1 floor, and the K3 fused fallback
   are unchanged; only the unset position moved.
   The flip is recorded in
   bench/k21f-default-flip-2026-11-07.md.

Everything above this section stands as the
historical record of why the guard existed and
what it demanded. The re-enable recipe section is
moot (the route is live at the shipped default);
the rollback now is `WGPU_COMPUTE_K21F_HEAD=0`.
