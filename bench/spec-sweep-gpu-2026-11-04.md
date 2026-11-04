# T-9.7 GPU sweep table (server proof, 2026-11-04)

Stack: resident GPU target + GPU DFlash draft (spec_sweep_gpu),
prompt "The capital of France is", 128 tokens per depth, greedy.
Run A (depths 1-4) was SIGKILLed (exit 137, OOM) during the depth 6
session build on the RAM-tight server; run B measured 6, 8, 12 and
was refused at 16. Emitted = 127 round tokens at every depth
(126 rounds, tokens/round 1.01).

| Depth | Rounds | Proposed | Accepted | Acceptance rate | Tokens/round | Emitted | Wall (s) | Effective tok/s |
|------:|-------:|---------:|---------:|----------------:|-------------:|--------:|---------:|----------------:|
| 1 | 126 | 125 | 1 | 0.008 | 1.01 | 127 | 12.4 | 10.24 |
| 2 | 126 | 249 | 1 | 0.004 | 1.01 | 127 | 14.7 | 8.62 |
| 3 | 126 | 372 | 1 | 0.003 | 1.01 | 127 | 19.3 | 6.59 |
| 4 | 126 | 494 | 1 | 0.002 | 1.01 | 127 | 28.4 | 4.48 |
| 6 | 126 | 735 | 1 | 0.001 | 1.01 | 127 | 25.8 | 4.93 |
| 8 | 126 | 972 | 1 | 0.001 | 1.01 | 127 | 31.0 | 4.10 |
| 12 | 126 | 1434 | 1 | 0.001 | 1.01 | 127 | 41.2 | 3.08 |
| 16 | - | - | - | REFUSED | - | - | - | - |

Depth 16 refusal (verbatim): "session build for depth 16 failed:
draft depth 16 is out of range for draft block size 16". This
stack's envelope is depth at most block minus 1 = 15, so the
candidate set's last entry is unmeasurable on the real draft.

Chosen default by SPEC 9.4 (smallest depth within 2% of best):
depth 1, 10.24 tok/s, band floor 10.04. DO NOT RECORD IT:
acceptance is about 0 at every depth because the GPU draft forward
diverges from the CPU reference (the dflash_session parity gate).
This table measures the broken-draft loop honestly; the winner is
an artifact of that failure, not an operating point. Re-run after
the draft parity fix before recording any depth as the SpecConfig
default. The sweep machinery itself (run_sweep widened over
TargetDriver, spec_sweep_gpu runner) is proven end to end on the
GPU loop by this run.
