# Model Suite — every model on the drive, one engine, one test set (2026-10-28)

Engine: llama.cpp master a7af149, **Vulkan** build, RX 7900 XTX (24 GB), one model at a
time with the GPU exclusive, full offload (`-ngl 999`), fixed 16K test context, temperature 0.
Speed: `llama-bench` pp512 / tg128. Time to first token: llama-server, streamed, warm run,
fixed ~600-token passage with a one-sentence summary request. Quality: 3 coding + 3 reasoning
prompts with a known answer key, 1024-token budget, each model's own chat template at default
settings (including default reasoning behavior). Raw per-model JSON + full answer text:
`/mnt/models/bench-2026-10-28/results/` on ubuntu-server; harness: `bench.py` alongside it.

## Results

| Model | File (GB) | Prompt t/s | Gen t/s | TTFT | Load | Score | Notes |
|---|---|---|---|---|---|---|---|
| DeepSeek-Coder-V2-Lite (Q4_K_M) | 10.36 | 5427 | 267.0 | 0.008 s | 4 s | 4/6 | Fastest raw speeds here; failed the bat-and-ball trap ($0.10) and botched 240/10 as 12 min |
| gpt-oss-20b (MXFP4) | 12.11 | 5159 | 228.2 | 0.027 s | 4 s | **6/6** | Perfect score at MoE speed; smallest VRAM footprint |
| Nemotron-Cascade-2-30B-A3B (IQ4_XS) | 18.17 | 4338 | 198.3 | 0.056 s | 4 s | **6/6** | Verbose (500-800 tok/answer) but always lands |
| North-Mini-Code-1.0 (UD-Q4_K_M) | 19.20 | 3411 | 187.7 | 0.067 s | 6 s | **6/6** | Code specialist matches its billing |
| Devstral-Small-2-24B (Q4_K_M) | 14.33 | 1153 | 42.9 | 0.036 s | 4 s | **6/6** | Concise, zero reasoning overhead, dense-model speeds |
| Qwen3.8-27B (UD-Q4_K_M) | 16.46 | 1015 | 41.7 | 0.159 s | 4 s | **6/6** | Disciplined reasoning (fast, short); see MTP below |
| Muse-Glimmer-30B (Q4_K_M) | 16.76 | 999 | 44.4 | 0.165 s | 4 s | **6/6** | Heavy default thinking (1.5-2.1K chars/answer, 9-22 s each); see DFlash below |
| Qwen3.6-27B (Q4_K_M) | 16.82 | 987 | 40.5 | 0.170 s | 4 s | 2/6 | Runaway thinking: 4 of 6 answers never emerged inside the 1024-token budget (all budget burned in reasoning, empty content) |
| Gemma-4-31B-it (Q4_K_M) | 18.32 | 869 | 38.5 | 0.477 s | 6 s | **6/6** | All correct, slowest TTFT; one answer consumed the full 1024 tokens |
| Qwen2.5-Coder-32B (Q4_K_M) | 19.85 | 811 | 38.4 | 0.035 s | 6 s | 4/6 | Coding perfect; reasoning sloppy: bat-and-ball $0.10, tank 20 min (same 240/10 slip) |
| DiffusionGemma-26B-A4B (Q4_K_M) | 16.81 | n/a | n/a | n/a | n/a | blocked | `unknown model architecture: 'diffusion-gemma'` in llama.cpp a7af149; file byte-verified good, waits on upstream |

Quality key: C1 write dedupe function + state its output on a sample; C2 find and fix an
off-by-one slice bug; C3 predict a sorted-reverse index print; R1 bat-and-ball ($0.05);
R2 syllogism (yes); R3 fill/drain tank (24 min). Every 6/6 model's C1 function was correct
and stated `[1, 2, 3, 5]`.

## Speculative decoding (measured, same C1 code prompt)

| Model | Drafter | Base gen | With drafting | Gain |
|---|---|---|---|---|
| Qwen3.8-27B | MTP head (Q4_0) | 41.7 t/s | **102.9 t/s** | 2.47x |
| Muse-Glimmer-30B | DFlash (Q4_K_M) | 44.4 t/s | **88.6 t/s** | 2.00x |

The pasted article's DFlash claim (75-110+ t/s) lands inside its range on this hardware:
88.6 t/s measured. The Glimmer server on ubuntu-server was restarted WITH the DFlash
drafter enabled after this suite.

## Caveats

- Quality set is 6 prompts: a smoke gate, not a benchmark like LiveCodeBench. Treat score
  differences of one prompt as noise; the structural findings (Qwen3.6's runaway thinking,
  the two 240/10 arithmetic slips) are the signal.
- Default template settings were used deliberately: reasoning models were NOT throttled
  (no reasoning_strength injection). That is why Qwen3.6 starved and Glimmer is slow per
  answer; both behave differently with reasoning controls set.
- VRAM delta readings from sysfs under-read on this driver and are not reported per model;
  generation speeds confirm full offload (a CPU-bound 30B dense model does not reach 40 t/s).
- HIP cross-check exists for Glimmer only (October + this session): HIP wins prefill ~17%,
  loses generation ~7% vs Vulkan. Suite ran on Vulkan, the serving pick.
