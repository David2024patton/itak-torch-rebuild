# K21F head switch default flip: the route earns its default back (2026-11-07)

Lane `lane/k21f-default-flip`, from main 7531139.
The K21F finish-carrying head route's own switch
(`WGPU_COMPUTE_K21F_HEAD`) flips its shipped
default from OFF to ON. The switch, the combined
gate, and the kernel are unchanged; only the
unset position moved, in the same shape as the
MMV ceiling flip (unset resolves to the shipped
default; a present but unrecognized value
resolves to the closed fallback, never to the
default).

## The evidence chain

1. **The guard and its criterion.**
   bench/k21f-default-guard-2026-11-07.md: after
   the defaults-confirmation session proved
   speculative identity FAILS on the real file
   with the head route live (deterministic
   divergence at stream index 23, depths 1 and
   6), the route got its own fail-closed switch,
   default OFF, with a written reopen criterion:
   on the card, on the real files, with the
   switch explicitly on, identity PASSES at
   depths 1 and 6 and GOLDEN PASSES in the same
   runs that show the exposure stamps.
   Fixture-scale agreement explicitly does not
   count.
2. **The root cause and the fix.**
   bench/k21f-mask-fix-2026-11-07.md: the
   divergence was a truncated dequant mask in
   the K21F shader (the qc line masked with
   `0x0f0f0f` instead of `0x0f0f0f0f`), zeroing
   the fourth byte so 16 of 256 elements per
   Q5_K super-block lost their entire low
   nibble. One-token fix; all four dequant
   lines pinned verbatim in both K21 modules'
   shader-text tests, with a negative pin
   against the truncated literal; a typo-class
   sweep found no other truncated mask in the
   workspace. The wave17 diagnosis
   (bench/k21f-divergence-diagnosis-2026-11-07.md)
   measured the failure at the real geometry
   (margin +0.382801 at the split; worst logit
   deviation 2.04) and its attribution is
   reconciled there.
3. **The retest.** bench/k21f-retest-2026-11-07.md
   (wave18, staged from the fixed tree): REOPEN
   CRITERION MET. With `WGPU_COMPUTE_K21F_HEAD=1`:
   identity and golden PASS at depths 1 and 6,
   with finish stamps in the same runs; the
   in-situ divergence at index 23 is gone
   (worst deviation 1.8e-5, was 2.04); K21F's
   finished output is bit-identical to K3's at
   the real head geometry; the finish and
   routing test files execute green on the
   card. The shipped default of that session
   (switch off) also passed identity and golden
   with zero finish stamps, so the guard held
   while it was the default.

Per the guard note's terms, the route has
earned its default back. This lane is that
flip, and nothing else.

## The change

- `compute::resolve_k21f_head` now resolves in
  the MMV ceiling flip's shape, over two named
  constants: `DEFAULT_K21F_HEAD` (`true`, the
  shipped default) and `K21F_HEAD_FALLBACK`
  (`false`, the closed fallback). New semantic
  matrix:

  | value | resolves |
  |---|---|
  | unset | ON (the shipped default) |
  | `1` (trimmed) | ON |
  | `0` (trimmed) | OFF (the explicit off; the one-variable rollback) |
  | any other text | OFF, fail-closed, logged by name at construction |

  A present-but-malformed value does NOT
  silently open: it resolves to the fallback,
  and `log_k21f_head_fallback` names it in the
  construction log, on the flash-staged
  fallback pattern. The construction log line
  now reads: `k21f head config: enabled=<bool>
  (env WGPU_COMPUTE_K21F_HEAD, "1" or unset
  enables, "0" disables)`. Resolution still
  happens once at backend construction; the
  `k21f_head` accessor and the `k21f_head`
  resolved-config pair in both sessions'
  diag-capture manifests are unchanged in
  shape and now report the flipped default.
- The combined gate is untouched:
  `session::k21f_head_admits` still requires
  the switch AND the shared band, batch 1
  still never routes, and the fallback on any
  refusal is still the exact K3 fused call.
  No other MMV route changes; the K21F kernel
  is untouched (its header's routing paragraph
  is updated as docs only).
- Doc updates where the old default was
  stated: the env, resolver, field, and
  accessor docs and the construction comment
  in compute.rs; the gate doc, the
  `try_mmv_q5k_finish` doc, and the block-head
  site comment in session.rs; the K21F
  paragraph in kernels/mod.rs; the routing
  paragraph in the K21F kernel module header.
  bench/k21f-default-guard-2026-11-07.md gains
  an appended supersession section (its
  history stands); bench/k21-finish-2026-11-06.md
  gains one supersession line in its guard
  section.

## Pins

- `compute::k21f_head_switch_resolution`
  re-pinned to the full new matrix (unset on,
  `1` on, `0` off, malformed off), with the
  retest cited as the justification; the two
  named constants are pinned transitively
  through the resolver arms.
- `compute::k21f_head_value_recognition` (new):
  the recognized positions are exactly unset,
  `1`, `0`.
- `session::k21f_head_gate_requires_both_halves`
  reframed: the off half is now the explicit
  `0` (or fail-closed) position and admits
  nothing at ceiling 16 or `u32::MAX`; the on
  half is the shipped default position and
  admits exactly what the band admits.
- `tests/k21f_head_routing.rs` is now four
  arms: switch-off (exercised explicitly with
  `WGPU_COMPUTE_K21F_HEAD=0`; unrecognized
  values land here too, the fail-closed pin),
  default construction (both envs unset; the
  block head must stamp the finish form: the
  flip's central pin), band-closed, and
  both-gates-open (explicit `1`). The
  closed-gate arm now selects on the ceiling
  being explicitly set to a closing value:
  before the flip it could proceed at an unset
  ceiling, which was only sound while unset
  resolved both gates closed; unset now
  resolves both open, so the arm names
  `WGPU_COMPUTE_MMV_MAX_ROWS=1` as its
  condition. Arm selection verified with
  `--nocapture` in five env positions on this
  VM (unset: default-construction arm proceeds;
  `=0`: switch-off arm proceeds; `=1` with
  ceiling 16: open arm proceeds; `=1` with
  ceiling 1: closed-gate arm proceeds; `=true`:
  switch-off arm proceeds). All four arms
  self-skip on this GPU-less VM; their on-card
  execution rides the next card session, the
  same position the guard lane's arms were in
  before wave18 executed them.

## Rates now in force

The shipped default (route live) now carries
the wave18 session's honest, identity-passing
rates, measured with the switch on under diag
timing on the real file: **22.07 tok/s at
depth 1, 16.80 tok/s at depth 6** (medians of
3). The defaults-confirmation session's
23.16 / 21.65 figures stay struck from the
record as speed claims: they were produced by
the corrupted head, whose depth-6 acceptance
was inflated by the corruption itself
(bench/k21f-retest-2026-11-07.md).

## Rollback

```
WGPU_COMPUTE_K21F_HEAD=0
```

One variable restores the guarded composition
(block head on the K3 fused call) exactly,
with every plain MMV route keeping its cap-16
opening: the position the guard shipped is now
the rollback instead of the default.

## Gates on this VM

- `cargo test -p wgpu-compute`: 267 passed,
  0 failed (guard-lane tree 265; +1 resolver
  recognition pin, +1 default-construction
  integration arm).
- `cargo test -p dflash`: 24 passed.
- `cargo test -p itak-spec`: 49 passed.
- `cargo clippy -p wgpu-compute --all-targets
  -- -D warnings`: clean.
- `cargo fmt --check`: clean (one fmt pass
  over the new closed-gate helper).
- `cargo check --workspace --all-targets`:
  clean.

No server jobs ran in this lane: the card
evidence for the flip is the wave18 retest,
already on record. Still owed to a future
card session: the four integration arms
executing on the card, and a defaults
confirmation rerun at the new shipped default
(the wave18 session's default arm ran under
the old default; its armed arm is the new
default's exact configuration, but the
confirmation ritual should be re-run as
shipped).
