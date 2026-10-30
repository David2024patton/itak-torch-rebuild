<!--
INSTRUCTIONS FOR THIS FILE (user and AI: read before editing. Keep this header.)

WHAT THIS FILE IS: the settled-decision log for this project. IDEA.md is
the raw capture of what David said; this file is only the calls that are
FINAL, so no one re-asks a settled question or quietly reverses a call.

EXACT STRUCTURE: one entry per decision, newest first:
## <YYYY-MM-DD>: <decision in one line>
- **Decided:** <what was picked>
- **Why:** <the reason, in plain words>
- **Supersedes:** <the earlier call it replaces, or "none">
- **By:** David | <who, with David's approval noted>

RULES:
- A decision enters here the day it is made, in the same pass as any
  PLAN or TODO change it causes.
- Reversing a decision means a NEW entry that names what it supersedes;
  never edit an old entry into a different decision.
- No secrets, passwords, or tokens in this file, ever.
-->
# DECISIONS: iTaK Torch Rebuild
## 2026-10-27: Models live on a dedicated NVMe drive in the server
- **Decided:** Model weights get their own NVMe drive in the server (the Kingston NV3 1 TB David bought for this purpose), mounted as the models volume. Downloads, re-downloads, and the streaming experiments all target that drive.
- **Why:** Serving loads a model once and runs it from VRAM, but the MoE streaming work reads experts constantly, and NVMe reads run roughly ten times SATA speed. The array is bulk storage and the wrong tier for hot model data. David's words: models on the NVMe is exactly why he got it.
- **Supersedes:** the array placement in the "Model weights live on the large array" decision below. The never-on-the-root-disk rule in that decision stands unchanged.
- **By:** David

## 2026-10-27: This project lives in a public repo with the full standard file set
- **Decided:** The rebuild's plan, evidence, and record are published in this repo (David2024patton/itak-torch-rebuild), public, with TODO, IDEA, PLAN, ENDGOAL, and the jump-back-in files maintained per the repo standard.
- **Why:** The benchmark record and the design work existed only in a local folder. The original engine died of exactly that failure mode: valuable work in one unpublished place. David's words: "Yes make it public. Use the GitHub skill to make the to-do etc files."
- **Supersedes:** none
- **By:** David

## 2026-10-27: The rebuild splits Go and Rust by feature, two processes, one seam
- **Decided:** Rust owns the token loop (kernels, KV caches, SSM scan, MoE streaming, speculative decoding, training compute, GPU backends). Go owns the platform (API surface, scheduling, caches, cluster, tenancy, model operations, observability, Core). Engine and platform are separate processes: control over a local socket, tokens and tensors through shared memory. Mojo remains the research lab; proven ideas graduate into the Rust crates.
- **Why:** David's standing rule is each language for the features it is best at. Go's platform code in the original is architecturally sound and gets reattached, not rewritten. The GC dividing line is the language dividing line: platform code does not care about GC pauses, the token loop does. Two processes also fix the original's FFI failure shape: the engine can restart or be replaced without the API going down.
- **Supersedes:** the original single-binary Go design with llama.cpp FFI coupling
- **By:** David (the split was proposed by Lulu on 2026-10-27 and adopted through his follow-up questions and the repo order)

## 2026-10-27: Experimental modes ship behind config flags, off by default
- **Decided:** Mamba-3 mode, cache-aware MoE routing, linear-attention swaps, and anything else unproven ship behind config flags with the default off, and earn a default only with hardware proof recorded in this repo.
- **Why:** The original Mojo engine carried a fully implemented early-exit drafting system that was dead by one boolean, and a Mamba-3 mode that never ran as shipped. Experiments that are invisible or always-on both waste the work.
- **Supersedes:** none
- **By:** David (standing build law, applied to this project 2026-10-27)

## 2026-10-27: Model weights live on the large array, never the root disk
- **Decided:** All model weights on the home server live under /mnt/alphaflux-data. Nothing model-sized goes on the root NVMe.
- **Why:** On 2026-10-27 the root disk filled, Dokploy went down, and the Glimmer test weights were deleted on David's order to recover. The storage plan (NVMe for OS, the array for bulk) only works if the weights follow it.
- **Supersedes:** the original /home/server/models placement
- **By:** David

## 2026-10-26: Vulkan is the production GPU backend on the RX 7900 XTX; ROCm is parked
- **Decided:** llama.cpp serving and all benchmark claims run on the Vulkan build. ROCm/HIP remediation (clean reinstall, likely a reset window) is David's call and is not scheduled.
- **Why:** Proven on the machine: HIP bench and CLI both wedge after model load with the GPU idle, and a 20-line HIP program fails to link against the mixed ROCm install, while Vulkan runs flawlessly on the same driver. The GPU is healthy; the ROCm userland is not.
- **Supersedes:** none
- **By:** David

## 2026-10-26: Muse Glimmer 30B serving pick is the Meta Q4_K_M build on Vulkan
- **Decided:** llama-server Vulkan, Meta Q4_K_M, context 131072, 4 slots, --jinja, f16 KV, DFlash off for shared use and on (n-max 4) solo, with reasoning_strength low injected by default at the proxy layer.
- **Why:** Measured: Q4_K_M decodes at 44.07 tok/s solo and 37.14 tok/s per user at two users; Q3 is faster but weaker, Q5 slower for marginal quality, Q6 leaves no KV room. Default reasoning strength burns the whole token budget and returns empty content; low fixes it and the answers verify correct.
- **Supersedes:** none
- **By:** Lulu, from the benchmark evidence, reported to David 2026-10-26

## 2026-10-30: The recorded Glimmer speed anchors (44.31 plain, 88.6 DFlash) are harness-relative
- **Decided:** The 2026-10-28 anchors stay in the record with their harness named, and the Section 9 sweep harness (llama-server /completion, short raw prompts, fixed 256-token generations, temperature 0) is the comparison shape for depth/operating-point claims until T-9.7 re-runs it against our own stack. On that harness: plain 41.64 tok/s, DFlash depth 4 = 103.01, depth 6 = 107.39 (the operating point by SPEC 9.4's rule), depths 8+ collapse below or near plain. Full table: bench/gpu-proof-2026-10-30.md.
- **Why:** The sweep did not reproduce the anchors within a few percent (plain -6.0%, depth 4 +16.3%, ratio 2.47x vs the recorded 2.00x), which the RUNBOOK defines as a finding, not rounding. The recorded numbers came from different shapes (llama-bench tg128 after pp512; the model-suite run), so cross-harness comparisons silently move both the rates and the ratio.
- **Supersedes:** nothing; the 2026-10-26 serving pick and its measured basis stand, with the harness caveat attached
- **By:** Lulu, from the 2026-10-30 sweep evidence, during David's approved pause window

## 2026-10-27 (historical, recorded from the failure): Shared code is published and versioned, never a relative folder
- **Decided:** Any module another module depends on is a published, versioned artifact. Relative-path replaces in module files are forbidden in this project.
- **Why:** The original iTaKCore lived only as ../Core beside the Torch tree on one machine. It was never pushed, the folder is gone, and the engine cannot be built. The measured usage surface is small (six importing files); the loss was never about size, only about the module never existing anywhere durable.
- **Supersedes:** the original go.mod replace directive pattern
- **By:** David (build law, recorded here as this project's founding lesson)
