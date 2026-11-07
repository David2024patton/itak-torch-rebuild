# Staged flash default flip: off to on (2026-11-07)

Lane: lane/flash-default-flip, based on main
910b800ea6d716aed18a708bb0cb47312cd5a206. Code commit
03fc780. VM gates only on this lane; no server GPU
jobs. The card evidence this flip stands on was
measured by two earlier sessions, cited below; this
lane changed one resolver, its docs, and its tests.

## What changed

`WGPU_COMPUTE_FLASH_STAGED` (crates/wgpu-compute/src/
compute.rs) shipped with an opt-in resolver: only the
exact value `1` enabled K6's staged flash form, and an
unset variable meant off. The resolver now ships the
staged form as the default for multi-row attention
calls. Rollback is one env var:
`WGPU_COMPUTE_FLASH_STAGED=0` restores the shipped
flash form exactly, no rebuild, no other setting.

### The semantic matrix (final)

| env value | resolved | notes |
|---|---|---|
| unset | ON | the new shipped default |
| `1` (trimmed) | ON | unchanged: a value that meant on before the flip means on after |
| `0` (trimmed) | OFF | the explicit off; the rollback |
| anything else | OFF | fail-closed; the refusal is logged by name at construction (`flash staged env ... is not a recognized value ... failing closed, the staged form is OFF`), following the MMV band fallback pattern |

The single-row exclusion is not part of the switch
and did not move: session.rs still routes staged only
when the switch is on AND the staged pipeline exists
on the device AND the call carries more than one row.
Single-row calls (decode) run the shipped flash form
in every switch position, as before.

## The evidence chain

1. **The kernel and its rule**
   (bench/flash-rework-2026-11-06.md). The staged form
   stages K/V blocks of 32 keys in workgroup memory,
   8 query rows per workgroup, same bindings, params,
   output layout, and per-element float sequence as
   the shipped flash. Its decision rule, written in
   advance, required for a default flip: autopsy clean
   (LDS 21,632 B, VGPR at most 128, zero spills);
   every correctness leg green including identity and
   golden with the switch on; the staged flash total
   at 1,022 tokens at most 760 ms (25 percent under
   the recorded 1,013.1 ms) WITH the prefill wall
   improved against the same-session control; the
   staged total ahead at 8k tokens or more; no-leak
   clean and decode unmoved.

2. **The split verdict**
   (bench/baseline-session-2026-11-06.md, leg (i)).
   Autopsy PASS (VGPR 60, zero spills, zero scratch).
   Correctness PASS (exactness test, resident suite,
   spec identity and golden at depths 1 and 6, all
   with the switch on). Kernel row at 1,022 tokens:
   605.9 ms against the control's 972.9, a 37.7
   percent improvement, clearing the 760 ms bar.
   Long context: flash row 47,360.9 to 22,820.3 ms
   (51.8 percent) and phase wall 93,509 to 66,704 ms
   (28.7 percent) at 8,202 tokens. But the 1,022 wall
   clause FAILED on that session's single pair of
   runs: 6,383 ms off against 6,558 ms on, read as
   roughly 540 ms of non-kernel time appearing with
   the switch on. The flip was correctly refused on
   that evidence.

3. **The phantom**
   (bench/flash-host-cost-2026-11-06.md). A dedicated
   lane hunted the 540 ms segment by segment under
   interleaved, cache-controlled, repeated
   measurement (arms interleaved off/on/on/off at
   1,022, fresh shader caches priced separately,
   every run failure-bundled). Findings: the
   wall-minus-GPU gap segment reads 201.5 to 226.9 ms
   in BOTH forms with no form-locked offset; the wall
   delta matches the GPU kernel delta to within
   scatter at every shape; session stats are
   identical between forms at every shape (same
   submissions, readbacks, dispatch counts), so no
   extra submit, sync, or readback exists to pay;
   pipeline compilation lands in backend_open before
   the measured window and is form-neutral; slice 0
   shows no deferred compile. The staged form wins
   the wall at every shape measured: 1,022 tokens
   about 6,150 ms on against 6,490 ms off (timing-off
   pair 6,025 against 6,342), 2,052 tokens 13,500
   against 15,227, 4,102 tokens 29,885 against
   36,945, 8,202 tokens 66,546 against 93,402 (the
   long-context win reproducing the baseline
   session's step 3 almost to the digit). The 540 ms
   was an artifact of the first verdict's evidence
   shape: one non-interleaved pair of wall samples,
   from a session whose own note records a host under
   parallel CPU collection with wide scatter; a
   +175 ms single-sample wall delta and a 367 ms
   kernel saving were combined into a host penalty
   that does not exist in the code.

With step 2's wall clause discharged by interleaved
evidence, every clause of the rework note's decision
rule holds, and the default flips. David approved
proceeding on that basis.

## Tests

- `flash_staged_switch_resolution` (compute.rs unit
  tests): re-pinned to the new matrix (unset on, `1`
  and trimmed `1` on, `0` and trimmed `0` off,
  `true`/empty/`11`/`on` off), with the justification
  cited in the test body
  (bench/flash-host-cost-2026-11-06.md, interleaved
  arms).
- `flash_staged_value_recognition` (new): pins the
  recognized set the construction log consults
  (unset, `1`, `0`, trimmed forms) against the
  unrecognized set that fails closed with a named
  refusal.
- The fused-kernel device tests that exercise the
  staged form directly
  (`k6_flash_staged_matches_shipped_at_glimmer_shapes`)
  do not consult the switch and are unchanged; the
  single-row routing exclusion is unmodified code.

## Gates (this lane, VM)

- `cargo test -p wgpu-compute`: 254 passed, 0 failed,
  1 ignored (the ignored one is the pre-existing diag
  hash-cost probe), across 27 test binaries; lib 146.
  Device tests skip on the VM, which enumerates no
  adapter, as throughout this crate.
- `cargo test -p dflash`: 24 passed, 0 failed.
- `cargo test -p itak-spec`: 49 passed, 0 failed.
- `cargo clippy -p wgpu-compute --all-targets -- -D
  warnings`: clean.
- `cargo fmt --check`: clean.
- `cargo check --workspace --all-targets`: clean.

## What a card run still owes

Nothing for the flip itself: correctness with the
staged form engaged was proven on the card by both
earlier sessions (exactness, resident suite,
identity and golden at depths 1 and 6, prefill argmax
identical between forms at every shape, bit-for-bit
on id and logit). The first post-merge session that
runs with the variable unset should see the
construction line read `flash staged config:
enabled=true`, and multi-row forwards should stamp
`k6_attn_flash_staged` where the shipped form used
to stamp `k6_attn_flash`; that is the exposure check
that the new default is what production runs get.
