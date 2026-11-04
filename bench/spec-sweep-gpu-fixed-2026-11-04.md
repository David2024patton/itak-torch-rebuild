# T-9.7 GPU sweep, fixed draft (server proof, 2026-11-04)

Stack: resident GPU target + GPU DFlash draft (spec_sweep_gpu),
prompt "The capital of France is", 128 tokens per depth, greedy.
Each depth ran in its own process invocation (a combined run was
OOM-killed in an earlier window on the RAM-tight server). This run
follows the GPU draft parity fix (BlockPlan attention runtime
values now reach every layer; see the T-9.6 verdict addendum in
bench/spec-gpu-port-2026-11-03.md). Emitted = 127 round tokens at
every depth.

| Depth | Rounds | Proposed | Accepted | Acceptance rate | Tokens/round | Emitted | Wall (s) | Effective tok/s |
|------:|-------:|---------:|---------:|----------------:|-------------:|--------:|---------:|----------------:|
| 1 | 79 | 78 | 48 | 0.615 | 1.61 | 127 | 7.5 | 16.86 |
| 2 | 68 | 136 | 59 | 0.434 | 1.87 | 127 | 8.0 | 15.91 |
| 3 | 64 | 191 | 63 | 0.330 | 1.98 | 127 | 9.1 | 14.03 |
| 4 | 65 | 258 | 62 | 0.240 | 1.95 | 127 | 10.6 | 12.04 |
| 6 | 64 | 377 | 63 | 0.167 | 1.98 | 127 | 13.1 | 9.69 |
| 8 | 64 | 497 | 63 | 0.127 | 1.98 | 127 | 15.9 | 7.98 |
| 12 | 64 | 730 | 63 | 0.086 | 1.98 | 127 | 20.6 | 6.17 |
| 16 | - | - | - | REFUSED | - | - | - | - |

Depth 16 refusal (verbatim): "session build for depth 16 failed:
draft depth 16 is out of range for draft block size 16". The
envelope maximum is block minus 1 = 15.

Winner by SPEC 9.4 (smallest depth within 2% of the best
effective rate): the best rate is 16.86 tok/s at depth 1, the band
floor is 16.52, and no other depth enters it. WINNER: depth 1,
measured acceptance 0.615. Recorded as the SpecConfig depth
default (crates/itak-spec/src/config.rs).

Honest context: every depth in this table sits below the same
night's plain decode rate (35.45 tok/s). Depth 1 is the measured
winner of a loop that does not yet beat plain decoding on this
stack; the remaining problem is acceptance economics and verify
cost, which is T-9.6's open speed half, not this item's proof.
