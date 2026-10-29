<!--
INSTRUCTIONS FOR THIS FILE (user and AI: read before editing. Keep this header.)

WHAT THIS FILE IS: the end state this project exists to reach, in plain
words. TODO.md and PLAN.md serve this file; work continues until the end
goal is reached. When David restates or extends the end goal, update this
file the same day and let PLAN/TODO follow.

EXACT STRUCTURE:
- "## The end state": the finished project described as the experience of
  the person using it, start to finish.
- "## How we will know it is reached": observable conditions, each one
  checkable, tied to the TODO item that proves it where one exists.
- "## Not in the end state": explicit exclusions David has set, so the
  goal cannot drift.

RULES:
- Sourced from David's stated vision, the repo README and features, and
  what he has talked about in the record. Never invented. Where the
  record is thin, say the end state at the level the record supports.
- No secrets, passwords, or tokens in this file, ever.
-->
# ENDGOAL: iTaK Torch Rebuild

## The end state

A developer or a small team points iTaK Torch at a model and serves it from hardware they already own: a single consumer GPU, a workstation, a small server. Models far larger than the card's memory still run, because experts stream from disk and state space layers hold constant-size state instead of a growing cache. Generation is steady, token after token, with no garbage collector pauses, because the hot path is Rust and allocates its memory once.

Everything the original Go engine reached for is present and working, in one system: the OpenAI-compatible API with scheduling, batching, and per-user fairness; speculative decoding that multiplies single-stream speed; grammar and JSON constrained output; model pulling, integrity checking, and versioning; a training stack that fine-tunes with LoRA or full trainers and feeds the results straight back into serving; and the hybrid state space architecture that was the author's research bet, including his own Mamba variant, running as a first-class model family rather than an experiment with the switch off.

The platform and the engine are separate programs with a clean seam. The engine can be restarted, upgraded, or replaced without the API going down. Every shared contract is a published, versioned module, so the single lost-folder failure that killed the original can never happen again. Every module carries its own diagnostics and lint gates from birth, so when something misbehaves the system itself says what it did and why it refused.

## How we will know it is reached

- A fresh clone builds end to end with one command, on a machine that has never seen the project (proof: T-1.1, T-1.2).
- The Rust engine serves a dense model at speeds benchmarked on real hardware against llama.cpp, with the comparison published in this repo (proof: T-2.6).
- A model larger than VRAM serves correctly via expert streaming, with cache behaviour measured (proof: T-5.4).
- The hybrid SSM model family runs, including the three-term recurrence and rotation on B and C, with state-tracking checks passing (proof: T-5.2, T-5.3).
- Trained Medusa heads load from versioned files and raise measured tokens per second over the non-speculative baseline (proof: T-3.3).
- A LoRA fine-tune runs through the training API and the adapter serves (proof: T-7.2).
- The Go platform from the original engine answers API calls against the new engine through the seam (proof: T-6.1).
- Every feature in the original inventory (PLAN.md S2 audit) is either implemented and proven, or explicitly retired by a DECISIONS.md entry.

## Not in the end state

- No dependency on a relative-path local module, ever. Shared code is published and versioned or it does not ship.
- No ROCm requirement. Vulkan and wgpu are the GPU path; ROCm support returns only if the platform install is remediated and proven.
- No wrapper around llama.cpp as the engine. It stays the reference to benchmark against, not the product.
- No experimental mode enabled by default before it is proven on hardware. Experiments ship behind config flags, off by default, the lesson of MAMBA_3_MODE.
