# Muse Glimmer 30B on a single RX 7900 XTX (gfx1100, 24 GB, Ubuntu), engine & serving-platform research

Researched 2026-10-26/27 (EDT). Hardware fixed by the mission: Linux home server, one AMD Radeon RX 7900 XTX (Navi 31, gfx1100, 24 GB VRAM, ~960 GB/s memory bandwidth, RDNA3), Ubuntu.

**How to read this report:** every performance number names its source, hardware, model/quant and date/version where available. Vendor claims are labelled **[vendor]**; independent or community measurements are labelled **[measured]**. Anything that could not be verified is flagged **[unverified]** and collected again in "Could not verify". The single most important caveat up front: **nobody has published a controlled Vulkan-vs-HIP benchmark of Glimmer itself on gfx1100, and nobody has published a 2–4-simultaneous-user measurement of Glimmer on one 7900 XTX.** Both gaps are stated plainly where they belong rather than papered over.

---

## Summary

- Glimmer is natively supported in llama.cpp (PR ggml-org/llama.cpp#26841, merged on launch day 2026-08-10), and llama.cpp is the only engine where Glimmer + GGUF + gfx1100 is a proven, documented combination today. AMD's own Lemonade catalog ships Glimmer GGUF on llama.cpp with draft decoding, vision and tool calling.
- On gfx1100, the HIP/ROCm build of llama.cpp is the faster backend in every controlled comparison found (older/smaller models): roughly +25–40% generation and +25% to ~3× prompt processing over Vulkan, with rocWMMA flash attention. Vulkan is the fallback, not the first choice. No Glimmer-specific A/B exists **[gap]**.
- VRAM is unusually forgiving for this model: official Q4_K_M (16.8 GB) plus the full 131K KV cache (~1.7 GiB, because of sliding-window attention + only 2 KV heads) plus mmproj fits in ~19 GiB; DFlash brings it to ~20.5 GiB. UD-Q6_K_XL (26.3 GB) does **not** fit 24 GB.
- DFlash is a single-user latency feature: up to 3.1× on an RTX 5090 **[vendor]**, ~36→60 tok/s on an RX 7900 XT **[measured, community]**, but an independent vLLM test showed it *cutting* aggregate throughput from 681 to 285 tok/s under concurrency. For a shared server, leave it off; for solo use, on with `--spec-draft-n-max 4`.
- For a shared OpenAI-compatible API for 2+ people, the ranked answer is: **llama-server (HIP build) + a per-user-key/rate-limit layer in front (GPUStack or a LiteLLM-class proxy)**, because llama-server's own `--api-key` is a single shared key with no rate limiting. vLLM ROCm is the strongest watch-list alternative: native Glimmer support has now landed upstream (PR #51655, in release notes as of Oct 2026), but its ROCm/gfx1100 quality for Glimmer and a 24 GB-fitting INT4 checkpoint path are not yet verified.
- Rust engines (mistral.rs, candle, kalosm, ratchet) have **no production AMD GPU backend** and no Glimmer support; the exceptions are experimental HIP-native projects (hipfire, Strata) that are single-model research efforts, not serving platforms. Burn/CubeCL has a ROCm backend on paper but no LLM serving engine.
- Modular MAX: Mojo's GPU-programming compatibility list includes the RX 7900 (gfx1100) as "known compatible" (not continuously tested), but MAX's *serving* stack is validated/documented for datacenter GPUs (MI300/MI325/MI355, NVIDIA H100/B200-class); MAX has no Glimmer architecture, and adding one means writing a full custom architecture package (graph, weight adapters, config, pipeline) in MAX, MAX then provides paged KV, continuous batching and an OpenAI API. GGUF is supported in MAX only as a weights format *for architectures that register a GGUF adapter*.
- Building a custom Rust/Mojo engine for this one model is a person-months-to-person-years project whose realistic outcome is performance **at or below** llama.cpp's HIP build for a long time; the productive middle option is a custom orchestration layer (auth, quotas, routing, UI) on top of llama-server's kernels, days to weeks.

---

## Q1a, Glimmer in llama.cpp on gfx1100: Vulkan vs HIP/ROCm

### The model (context for all numbers)

Muse Glimmer 30B, released 2026-08-10 by Meta, Apache 2.0. Dense ~29.6B parameters total: ~27.9–28B text decoder + ~1.9–2B ViT vision encoder. 52 layers, GQA with 32 query heads / **2 KV heads**, hybrid attention (3 sliding-window-2048 RoPE layers + 1 full-attention NoPE layer, repeated ×13), native context 131,072, vocab 202,048. Sampling defaults temp 1.0 / top_p 0.95 / top_k 64; `reasoning_strength` low/medium/high (default)/xhigh. Tool calling is ATEM format, **one tool call per turn, no parallel calls**. Source: Meta/HF launch blog, https://huggingface.co/blog/muse-glimmer (read 2026-10-26).

### Glimmer-specific measurements on RDNA3 (only two found)

1. **RX 7900 XT (20 GB)**, via the Hacker News launch thread as reported by explainx.ai (https://explainx.ai/blog/meta-muse-glimmer-open-weight-30b-agentic-model-2026): Unsloth Q4_K_XL GGUF, ~19 GB VRAM used, **~700 tok/s prompt processing, ~36 tok/s generation, ~60 tok/s generation with DFlash**. The same report mentions "4 parallel 113K-context slots" and that Q3_K_XL dropped usage to 15.6 GB at full 131K context at similar speed. The slot/context wording is ambiguous in the source, treat the slot claim cautiously. **[measured, community, single report]**
2. **RX 7900 XTX**, llm-bench.io (https://llm-bench.io/benchmarks/cmsrt0h0b000l01l8w4o0ptwc): `muse-glimmer:latest` (Q4_K_M), **Ollama 0.32.9, context 65,536, 34.8 tok/s** on a coding-generation scenario. Backend (ROCm vs Vulkan) and date are not recorded on the page. **[measured, community, thin metadata]**

No llama-bench-style Vulkan-vs-HIP comparison of Glimmer on any RDNA card was found. That is the honest state of the evidence.

### Vulkan vs HIP on gfx1100, the controlled data that does exist (other models)

| Source | Hardware / model | HIP/ROCm | Vulkan | Verdict |
|---|---|---|---|---|
| llm-tracker.info, AMD GPUs page (https://llm-tracker.info/howto/AMD-GPUs), 2024-era builds | gfx1100, Llama-2-7B Q4_0, llama-bench pp512/tg128 | rocWMMA+FA: **pp 2891.59, tg 94.79**; plain HIP: pp 2689, tg 96.08 | **pp 2293.52, tg 62.75**; Vulkan+FA: pp 1516.73, tg 57.99 | HIP wins: Vulkan ~20% slower pp (48% slower with FA on), ~33% slower tg (40% with FA). Page's advice: "use the HIP backend with rocWMMA and Flash Attention when possible." Vulkan used less VRAM in their watcher. |
| Hacker News comment (https://news.ycombinator.com/item?id=41634232), llama.cpp build 3818 | RX 7900 XTX, Llama-2-7B Q4_0 | ROCm **pp512 3258.67, tg128 103.31** | (table truncated in capture) | Poster claims tg +44% and pp ~+202% (~3×) vs Vulkan. **[measured, community, partially captured]** |
| dev.to 2026 ROCm-vs-Vulkan guide (https://dev.to/rosgluk/rocm-vs-vulkan-for-amd-local-llm-hosting-2026-guide-5c70) | Multiple AMD GPUs |, |, | Winner is model-shape/build dependent. Documents a **gfx1201 (RDNA4, not gfx1100)** issue where Vulkan decode becomes 4.7–6.7× slower than HIP once hidden size ≥ 4096. Glimmer's hidden size is 6656, so this failure mode is worth testing for on gfx1100, but it is **not verified on gfx1100**. |

Supporting XTX data points (HIP, llama.cpp, other dense models, sanity anchors for what a dense ~30B does on this card):

- runlocalmodel.com (2026-05-13, llama.cpp HIP, Q4_K_M, F16 KV, 4K context): Qwen-32B dense **14–20 tok/s**; Llama-3.1-8B 100–130 tok/s. https://runlocalmodel.com (as captured in research notes).
- Level1Techs forum (2026-05-31, ROCm 7.2.3, HIP gfx1100, llama.cpp b9425, ROCWMMA_FATTN=OFF): Qwen3.6-35B-A3B ~130 tok/s decode with MTP; Gemma4-26B-A4B ~78 tok/s (MoE models, not comparable to dense Glimmer).
- GameGPU (2026-08-05, **Vulkan**, Gemma-4 12B): XTX pp512 1970.66, tg128 86.68.

**Bottom line for Q1a:** HIP/ROCm is the backend to run on gfx1100; expect Vulkan to cost roughly a quarter to a third of generation speed, and treat Vulkan flash attention on gfx1100 as historically *slower*, not faster (llm-tracker). Build HIP with `GGML_HIP_ROCWMMA_FATTN=ON`, AMD's ROCm docs describe this flag as enhancing flash-attention performance on AMD compute (rocm.docs.amd.com llama.cpp install docs). Bug/regression notes: a 2026-10 llama.cpp release roundup (kimbodo.com) mentions a fix for an **out-of-bounds Vulkan flash-attention write**, recent evidence that Vulkan FA has had correctness bugs; exact PR not captured **[partially verified]**. llama.cpp upstream is extremely active (builds b1187x by late Oct 2026, prebuilt ROCm 10.0 binaries published), so backend behaviour moves fast, pin a build and re-benchmark after upgrades.

### Vendor DFlash numbers, read with care

Meta's launch figures **[vendor]** (HF blog; also madrobot.blog, https://madrobot.blog/2026/10/10/meta-muse-glimmer-how-to-run-locally-ollama-lm-studio-vs-qwen-gemma/): RTX 5090, llama.cpp, **74.9 → 233.4 tok/s (3.1×)** with DFlash; M5 Max 26.6→50.2 (ExecuTorch); M4 Max 23.7→37.8. All single-request, and **no 24 GB card was ever measured by Meta**, OptraCloud's report (https://optracloud.com/muse-glimmer-report.html) explicitly debunks the viral "233 tok/s on a 24 GB GPU" claim. The only RDNA DFlash measurement is the 7900 XT community figure above (~36→~60 tok/s, ≈1.7×).

---

## Q1b, Best llama.cpp settings for a 30B GGUF in 24 GB

### Quant choice

| Quant | Size | Fits 24 GB? | Notes |
|---|---|---|---|
| Meta official Q4_K_M | 16.8 GB | Yes, comfortably | ~1.0% average loss vs FP across 15 benchmarks **[vendor, Meta]** |
| Unsloth UD-Q4_K_XL | 15.9 GB | Yes | Unsloth's recommended pick for 24 GB |
| Meta Dynamic Q4_K_XL | 19.7 GB | Yes, tighter | ~0.2% loss **[vendor]**; aimed at 32 GB cards |
| Unsloth UD-Q5_K_M | 19.2 GB | Yes, "24 GB with short context, or 32 GB" (Unsloth) | Higher quality; leaves ~4 GB for KV/compute/mmproj, workable *because* Glimmer's KV is tiny (below), but not with mmproj + DFlash + full context all at once |
| Unsloth UD-Q5_K_XL | 21.8 GB | Marginal | Not recommended here |
| Unsloth UD-Q6_K_XL | 26.3 GB | **No** | Exceeds 24 GB before KV/compute buffers |
| Q8_0 | 29.6 GB | No |, |

Weights also needed: mmproj Q4_K_M 1.4 GB (Meta) or Unsloth mmproj-BF16 3.85 GB / mmproj-Q8_0 2.05 GB; DFlash drafter 1.63 GB. Sources: logeshwaran.org (https://www.logeshwaran.org/2026/09/meta-muse-glimmer-30b-local-windows-kali.html), madrobot.blog (above), Unsloth repo unsloth/Muse-Glimmer-30B-GGUF as reported in research notes.

**Recommendation:** official **Q4_K_M** (or UD-Q4_K_XL) as the shared-server default; UD-Q5_K_M only for a text-only, shorter-context, single-user setup.

### The VRAM budget (why this model is easy on 24 GB)

Meta's own budget (via saram.io, https://saram.io/blog/muse-glimmer-open-source-local-multimodal-2026/): Q4_K_M build totals **19.0 GiB at the full 131,072 context with vision** = text weights 15.6 GiB + mmproj 1.3 GiB + KV cache + compute buffers 2.1 GiB. Adding DFlash ≈ **20.5 GiB**. Still inside 24 GB.

KV cache size (OptraCloud, https://optracloud.com/muse-glimmer-report.html): **0.18 GiB @ 8,192 ctx; 0.48 GiB @ 32,768; 1.70 GiB @ 131,072**, versus 6.50 GiB if all 52 layers were full attention. The sliding-window layers cap their KV and there are only 2 KV heads. ⚠️ A conflicting secondary figure (~52 KiB/token ≈ 6.5 GB @ 131K, shipwithmuse.live citing Raschka) matches the all-full-attention math, not the SWA reality; OptraCloud and Meta's budget agree with each other, so use ~1.7 GiB.

### KV cache quantization, mostly unnecessary here

General llama.cpp behaviour (checked Oct 2026): `-ctk/-ctv q8_0` ≈ ½ the f16 KV size, `q4_0` ≈ ¼; a **quantized V cache requires flash attention**. The speed cost at depth is real: f16 22.3 / q8_0 15.2 / q4_0 14.0 tok/s decode (Qwen3.8-27B, 2× RTX 3090, 256K ctx, via vramcalculator.com citing @superalesha); llama.cpp issue #27796 (RDNA4 R9700, 16K) f16 23.44 / q8_0 22.43 / q4_0 18.56 tok/s, with the q8 deficit widening from 7% to 21% by 131K depth. Since Glimmer's **f16** KV for the full context is only ~1.7 GiB, quantizing it saves almost nothing and costs decode speed: **keep KV at f16; use q8_0 only if you deliberately run multiple full-context slots plus mmproj plus DFlash and run out of room. Avoid q4_0 KV.** (Ollama equivalents, if used: `OLLAMA_KV_CACHE_TYPE=q8_0/q4_0` with `OLLAMA_FLASH_ATTENTION=1`.)

### Context and parallel slots

In llama-server, `-c` is the **total** KV pool divided by `-np` slots: `-np 4 -c 131072` = 32,768 tokens per slot (saram.io). Newer builds default to a unified KV pool when `-np` is auto. Because Glimmer's KV is so small, `-c 524288 -np 4` (131K per slot) is arithmetically feasible on VRAM, but prefill compute and the silent-failure mode matter more: saram.io warns that an under-sized `n_ctx_slot` **fails silently (empty output, no error)**, and Glimmer reasoning traces alone can approach 32K tokens. Practical shared setting: `-np 4 -c 131072` (32K/slot) or `-np 2 -c 131072` (64K/slot) if users run long agent traces.

### Other flags

- **`-ngl 99`**, everything on GPU; the model fits, so partial offload should never happen. If it does (wrong quant), decode collapses.
- **Flash attention:** on HIP builds use it (rocWMMA FA, build flag above). On Vulkan builds, test both ways, the llm-tracker data showed Vulkan+FA *slower* on gfx1100, and Vulkan FA has had recent correctness fixes.
- **Batch/ubatch:** defaults `-b 2048 / -ub 512` are sane. One directional data point (Strata research, MoE model, not Glimmer): `-ub 4096` raised prompt rate 2.5–2.7× and cut TTFT ~60% at 4K–128K prompts but grew the compute buffer 1.2→8.0 GiB and pushed layers off-GPU, on a 24 GB card with a 17–19 GB model, that trade is dangerous. Raise `-ub` cautiously (1024) only if long-prompt TTFT is the bottleneck and VRAM headroom is verified.
- **`--jinja` is mandatory** for Glimmer: without it, tool calling and reasoning separation break (saram.io). Control reasoning via `chat_template_kwargs.reasoning_strength`, **not** `reasoning_effort` (unimplemented in llama.cpp). Also note OptraCloud's tool-calling finding: at the default temp 1.0, 16/20 tool calls succeeded; at temp 0, 20/20 (failures were reasoning traces overrunning the token budget mid-call), for agent workloads, lower temperature and generous `max_tokens` matter more than any engine flag.

### DFlash, speedup, VRAM cost, and the concurrency trap

Mechanics (HF blog): block size 16 = 1 anchor + 15 proposals; `--spec-draft-n-max` above 15 is clamped to 15. Extra VRAM ≈ +1.5 GiB (drafter 1.63 GB; Meta budget 19.0→20.5 GiB total).

Independent counter-evidence (OptraCloud, vLLM 0.28.0, INT4 build, **2× RTX A5000**, 2026-09-02, NVIDIA hardware, treat as directional for the *pattern*, not the absolute numbers): n-max 4 was the sweet spot (90.0 tok/s vs 62.1 without speculation); n-max 16 gave **no gain (62.2)**; n-max 8 OOM'd because the 202,048-token vocab needs a second ~708 MiB logit-capping score table; and **under concurrency DFlash cut aggregate throughput from 681 to 285 tok/s**. Their conclusion, which this report adopts: speculation is for single-user latency; a busy shared server should run without it. Saram.io's llama.cpp example likewise uses `--spec-draft-n-max 4`, not the max.

### mmproj

Vision costs 1.3–1.4 GiB (Q4_K_M mmproj) on top of the text model, plus per-image compute. If the shared API is text/agent-focused, run text-only and start a second llama-server instance with mmproj only if vision is actually needed, don't pay the VRAM on the main instance.

---

## Q1c, Is the Glimmer architecture fully supported in llama.cpp?

Substantially yes, with caveats:

- **Support landed at launch**: PR ggml-org/llama.cpp#26841 merged 2026-08-10 (explainx.ai). Secondary sources disagree on the exact version floor, saram.io says support is in **b10353** (older builds error `unknown model architecture: 'muse-glimmer'`), logeshwaran.org says b10344 and earlier fail and **b10362+** loads it. Either way: use a current build (upstream is at b1187x by late Oct 2026) and the question is moot. **[version floor disputed between secondary sources, flagged]**
- llama.cpp 0.5.0 release notes (per creativeainews summary of HF data, 2026-09-23) list the Glimmer addition as "`--fuse-qkv` support" and cite ~711k GGUF downloads, i.e., the arch is a first-class citizen, heavily used.
- **Vision**: mmproj is published by both Meta (Q4_K_M, 1.4 GB) and Unsloth (BF16/Q8_0), and AMD's Lemonade catalog lists Glimmer GGUF with vision support (Q5).
- **Tool calling / template**: works **only with `--jinja`**; the ATEM format is single-call-per-turn by design; `reasoning_effort` is not wired up (use `reasoning_strength` via `chat_template_kwargs`). These are integration gotchas, not missing ops.
- No missing-op or broken-kernel reports specific to Glimmer on gfx1100 were found in the sources read. The known gfx1100 risks are the generic backend ones in Q1a (Vulkan FA behaviour), not Glimmer-specific defects. **[absence of evidence, stated as such]**

---

## Q2a, Platform comparison (Linux + RDNA3, late 2026)

| Platform | Runs on XTX/gfx1100? | GGUF + Glimmer | Concurrency / API | RDNA3 speed evidence |
|---|---|---|---|---|
| **llama-server** (ggml-org/llama.cpp) | Yes, HIP and Vulkan builds, first-class | GGUF native; Glimmer since launch (PR #26841) | Continuous batching on by default (`-cb`), parallel slots `-np`, prompt/prefix caching incl. RAM cache, OpenAI + Anthropic endpoints. **`--api-key` is a single shared key; no per-user keys, no rate limits, no TLS** (rizamarhaban.com cheatsheet) | Best of any engine here, see Q1a/Q1b; the only Glimmer-on-RDNA measurements that exist are llama.cpp-family (incl. Ollama's wrapper) |
| **vLLM (ROCm)** | Yes, official `rocm/vllm` images with matched ROCm/PyTorch/vLLM (e.g. ROCm 10.0.0 + vLLM 0.27.0 per the 2026 guide). On gfx1100 use AWQ/INT4; **FP8 weights do not work on gfx1100** (no FP8 hardware; localaimaster.com) | GGUF support is **experimental** (vLLM forum reports include empty outputs on consumer AMD). Glimmer: day-0 only via the transformers backend at BF16 + TP4 (~60 GB, not a 24 GB path). **Native support has since landed: PR #51655, listed under "New models: Muse Glimmer" in vLLM release notes** (https://github.com/vllm-project/vllm/releases, crawled 2026-10-26; AlphaSignal's v0.28.0 coverage also names Glimmer). But the community NVFP4 card (https://huggingface.co/cloudnathan5/Muse-Glimmer-30B-NVFP4), written earlier, said Glimmer was "not in any vLLM release yet" and needed a patched build + five DFlash fixes; and **no source verifies the native Glimmer path on ROCm/gfx1100** or a GPTQ/AWQ INT4 Glimmer checkpoint on this card (GPTQModel added Glimmer quantization support, ModelCloud/GPTQModel #3020, a path in principle) **[ROCm quality unverified]** | Best-in-class: PagedAttention, continuous batching, chunked prefill, prefix caching, `--gpu-memory-utilization` (default 0.92). Per-user keys/rate limits: not a vLLM feature, same proxy story as llama-server | Strong for other models on ROCm; **no published Glimmer-on-XTX vLLM number found** |
| **SGLang (ROCm)** | Upstream ROCm supports CDNA (gfx942/gfx950) + gfx1250 only, **RDNA3 is rejected by the kernel build**. A community shim exists: PyPI `sglang-radeon-rdna3` / `sglang-kernel-rdna3` (github AMD-AIM/sglang-radeon), validated on PRO W7900 (gfx1100, 48 GB), ROCm 7.2.1, PyTorch 2.9.1 | Glimmer day-0 support is real but **NVIDIA/Apple as announced** (LMSYS blog, https://www.lmsys.org/blog/2026-08-10-meta-muse-glimmer). Cookbook (https://docs.sglang.io/cookbook/autoregressive/Meta/MuseGlimmer, updated 2026-10-25): GGUF path is **text-only, no mmproj, "not optimized", with a startup warning**; BF16 checkpoint for vision; `muse` reasoning + tool-call parsers built in. **Glimmer on the RDNA3 shim: not validated** | RadixAttention prefix caching, continuous batching, OpenAI API; DFlash support (NVIDIA numbers: GGUF Q4_K_M on RTX 5090 72.6→140.7 tok/s batch-1, batch-8 230→332; NVFP4+DFlash 236 tok/s/user, batch-8 aggregate 1,452 **[vendor/launch blog, NVIDIA]**) | Shim's own figure: Qwen3.8-27B INT4 single-GPU 47.3 tok/s on W7900, a different card (48 GB) and model |
| **Ollama** | Yes, official ROCm (gfx1100 listed at docs.ollama.com/gpu) and Vulkan (enabled by default when installed; `OLLAMA_VULKAN=0` disables) | Glimmer in the library day-1 (`ollama run muse-glimmer`, 18 GB). ⚠️ The plain tag does **not** include the DFlash drafter; separate `-dflash` tags (~20 GB) bundle it (logeshwaran.org) | It's a llama.cpp wrapper with less flag control. `OLLAMA_NUM_PARALLEL` default **1**; memory scales as NUM_PARALLEL × context length; `OLLAMA_MAX_QUEUE` default 512 (docs.ollama.com/faq). **No per-user keys, no rate limits** | The one XTX Glimmer number that exists is Ollama's: 34.8 tok/s @ 64K ctx (llm-bench.io) |
| **LMDeploy** | TurboMind engine is **CUDA-only**; the PyTorch engine on ROCm is **experimental** (DeepWiki of InternLM/lmdeploy docs) | No Glimmer in its supported-models list found **[unverified/likely absent]** | PagedAttention-style KV, continuous batching (PyTorch engine); vendor claims up to 1.8× vLLM throughput **[vendor, self-reported via bizon-tech summary]** | None on RDNA3 found |
| **Aphrodite Engine** (PygmalionAI, vLLM fork) | Claims AMD GPU support via its vLLM heritage, but its docs/PyPI remain CUDA-first (build requirements CUDA ≥ 11/12; Docker examples use `--runtime nvidia`). **No gfx1100 validation found** | GGUF is in its quant list; **no Glimmer support found anywhere**, new archs land in vLLM upstream first and Aphrodite's rebase lags, so even upstream's #51655 may not be in Aphrodite yet **[unverified/likely absent]** | Continuous batching + PagedAttention (inherited), OpenAI API on port 2242, `--api-keys` (plural) in Docker examples, the only engine here with multi-key auth built in. PyPI actively released (0.10.0 page crawled Oct 2026, https://pypi.org/project/aphrodite-engine/0.10.0/); a secondary source reported v0.21.0 dated 2026-05-02, versioning inconsistent between sources **[flagged]**. License AGPL-3.0 per bizon-tech summary **[not verified on the repo]** | None on RDNA3 found; its published benchmarks are RTX 4090/A40 (PyPI) |
| **mistral.rs** | **No**, see Q4. On Linux + AMD it is CPU-only | GGUF yes (for its supported archs); **no Glimmer** | Has PagedAttention + continuous batching + OpenAI API, on CUDA/Metal only | CUDA-only published benchmarks |
| **KoboldCpp** (LostRuins/koboldcpp) | Yes via **Vulkan + CLBlast**; **ROCm "not directly supported"**, only via the community YellowRoseCx/koboldcpp-rocm fork (HIPBLAS). Vulkan is the wiki's recommended AMD path | GGUF native (llama.cpp fork), inherits llama.cpp arch support with fork lag, so Glimmer works once the fork rebases past b10353+ | OpenAI-compatible-ish API + KoboldAI API; designed single-user/RP-first; no real multi-user key/rate-limit system found | Inherits llama.cpp Vulkan numbers (Q1a) minus fork lag |
| **LocalAI** (mudler/LocalAI) | Yes, separate backend images: `gpu-hipblas` (ROCm) and `gpu-vulkan`; backends (llama.cpp, vLLM, …) run as separate on-demand images | Glimmer = whatever its bundled llama.cpp build supports (inherits arch support, with lag) | **2026 releases added multi-user auth: API keys + OIDC, per-user quotas (4.1.0, Apr 2026) and per-key/per-user usage attribution (4.3.0, May 2026)**, from release-note summaries **[secondary sources]**. OpenAI-compatible API. Heavier, YAML-configured | Inherits backend numbers |
| **GPUStack** | Not an engine, an orchestrator over llama-box (llama.cpp) and vLLM. Its AMD docs explicitly list **gfx1100 (RX 7900 XTX/XT/GRE): llama-box, vLLM** (docs.gpustack.ai) | Inherits the backend it launches (llama-box ⇒ Glimmer via llama.cpp) | OpenAI-compatible APIs, **user and API-key management, token usage and rate metrics/limits**, GPU monitoring (PyPI gpustack), i.e., exactly the multi-user layer llama-server lacks | Inherits backend numbers |

---

## Q2b, AMD's own guidance, and llama.cpp GitHub issues on gfx1100

- ROCm's Linux system-requirements page lists the RX 7900 XTX / gfx1100 as ✅ supported (rocm.docs.amd.com, current at read time). ROCm's docs carry a dedicated llama.cpp-on-ROCm install (pointing at the AMD-Ecosystem/llama.cpp fork and prebuilt `rocm/llama.cpp` Docker images) including the `GGML_HIP_ROCWMMA_FATTN=ON` flash-attention guidance, and a vLLM-on-ROCm guide built around matched containers.
- ROCm 10.0.0 (released 2026-08-26 per the dev.to guide) validates PyTorch 2.13 / vLLM 0.27 / SGLang 0.5.15, the supported-combination matrix moves fast; use AMD's matched images rather than mixing versions.
- AMD's blog "A Practical Guide to Running LLMs on AMD Radeon GPUs" (rocm.blogs.amd.com) routes desktop users to Lemonade / LM Studio / Ollama / llama.cpp. **Lemonade** (AMD-led) is notable: Phoronix reports Lemonade 11.6 adding Muse-Glimmer-30B GGUF to its llama.cpp catalog **with draft decoding, vision, and tool calling** on the ROCm backend; AMD Ryzen AI release notes (2026-10-01) repeat this for Lemonade 11.8.1. That is AMD's own software stack endorsing the llama.cpp path for this exact model.
- AMD's Glimmer launch partnership coverage (Moor Insights) was about **Ryzen AI Max+ SoCs**, not discrete RDNA3. **No AMD-published Glimmer benchmark on a 7900 XTX was found.**
- llama.cpp GitHub issue themes for gfx1100 (from the sources read): flash attention on HIP needs the rocWMMA build flag to perform; Vulkan FA has been slower than no-FA on gfx1100 in tracker testing and had an out-of-bounds-write fix in Oct 2026; the RDNA4 hidden-size≥4096 Vulkan decode collapse (dev.to guide) is the kind of backend bug to regression-test on gfx1100, where it is **not confirmed**. The cooperative-matrices/MMQ specifics from the mission were not pinned to a specific gfx1100 issue in the sources read **[partially verified, flagged]**.

---

## Q2c, What 2–4 simultaneous users do to one 24 GB card

**No direct measurement of 2–4 simultaneous users running Glimmer on one XTX was found.** What follows separates the one hard dataset from the architectural inference.

The hard dataset (OptraCloud, vLLM, INT4, no speculation, **2× RTX A5000**, an upper-bound analogue, *not* one XTX): 1 stream 62.1 tok/s; 2 streams 119.4 aggregate (59.9/stream); 4 → 218.5 (54.8); 8 → 364.4 (45.6); 32 → 681.4 (21.3). Continuous batching scales aggregate throughput ~11× from 1→32 streams because prefill/decode interleave efficiently and weights are re-read once per *batch step* for many sequences.

Why one XTX running llama.cpp will behave differently:

- Dense-30B decode is memory-bandwidth-bound. The theoretical single-stream ceiling is ≈ 960 GB/s ÷ ~16–19 GB of weights read per token ≈ **50–57 tok/s**, consistent with the ~35–36 tok/s actually measured (real kernels reach ~60–70% of peak bandwidth).
- llama-server's slots *do* batch (continuous batching is on by default), so aggregate decode rises with 2–4 active generators and per-user speed falls more gently than naive division, but llama.cpp's batching efficiency at small batch sizes is well below vLLM's PagedAttention scheduler, and there is no published Glimmer measurement to quantify the split. A defensible planning estimate for 2–4 simultaneous generators on Q4_K_M/HIP is **roughly 15–30 tok/s per user with aggregate ~50–80 tok/s**, **[estimate from bandwidth math + the single-stream measurements; not a measurement]**.
- VRAM is **not** the constraint at `-np 4`: Glimmer's KV is ~0.5 GiB per 32K-context slot (f16). The constraint is the single shared weight stream and prefill compute: a user submitting a 32K-token prompt will stall everyone else's decode during its prefill unless chunked prefill is in play (llama-server interleaves by ubatch; vLLM has proper chunked prefill).
- Turn **DFlash off** for the shared instance (OptraCloud: 681→285 tok/s aggregate under concurrency). Turn it **on** (n-max 4) only on a separate single-user instance/profile.

---

## Q3, GitHub engine survey (repos, recency, gfx1100, GGUF, Glimmer likelihood)

| Engine / repo | Recency (Oct 2026) | gfx1100 status | GGUF | Glimmer likelihood | Serving/concurrency |
|---|---|---|---|---|---|
| **ggml-org/llama.cpp** (upstream) | Extremely active, builds b1187x by late Oct 2026; prebuilt ROCm 10.0 binaries | First-class: HIP + Vulkan, ROCm docs + AMD fork | Native | **Yes, merged at launch (PR #26841)** | llama-server: slots, continuous batching, prefix cache, OpenAI API; single shared API key only |
| **ikawrakow/ik_llama.cpp** | Active, commits within a day of 2026-10-25; ~3,340 stars (hysenlabs.com project page) | **Wrong fork for AMD**: focus is CPU (AVX2+) + CUDA (Turing+), extra IQK/trellis quants, DeepSeek/MLA work. README position (via hysenlabs/DeepWiki): ROCm/Vulkan/Metal issues "will not be resolved" unless the user contributes the backend work | Native (fork) | Unverified and unlikely to lead upstream; forks lag new archs, and Glimmer (dense) is not its focus | llama-server-derived |
| **nemekath/kuzco.cpp-legacy** | Mirror capture ~128 days old at read time; repo is named "-legacy", suggesting it is superseded **[recency uncertain]** | Purpose-built for RDNA3: custom T-MAC GEMV kernels, auto-detects RX 7900 series | Native (llama.cpp fork) | Inherits llama.cpp arch support with fork lag | Project's own benchmarks **[vendor/project, self-measured]**: +13–20% tg on Q4_K_M, up to +37% IQ / +55% MoE+IQ, median +14–18% across 30+ models; XTX, N=10 paired runs, tg128, claims bit-identical output. If real and current, this is the single biggest decode uplift available on this card, but verify repo status before depending on it |
| **Niko1221/Strata** + nicejunjie/Strata-rdna3 fork | Active; AMD HIP performance doc measured 2026-09-29 (https://github.com/Niko1221/Strata/blob/main/docs/AMD_HIP_PERFORMANCE.md) | Tuned for XTX/gfx1100, ROCm 7.1 | GGUF-based (llama.cpp-adjacent research runtime) | Research runtime focused on large MoE (Qwen3.8-Flash-Next); Glimmer not its target **[unverified]** | Measured on XTX (Ryzen 9 7950X3D, 272 W cap, temp 0): candidate prefill 447.5–966.5 tok/s vs control 240.0–484.8, output 55.5–59.2 tok/s (model not named in the excerpt retrieved; project context is MoE serving) |
| **warpfront/hipfire** | Active, issue #947 filed ~4 days before 2026-10-26 (https://github.com/warpfront/hipfire/issues/947) | **Rust + HIP, kernel packs compiled per-arch incl. gfx1100**; ROCm 7.2.4, rustc 1.98 | Custom quantized formats (mq4 etc.), Qwen-focused | Not claimed; single-model-family research engine | Issue #947 benchmarks Qwen3.8-27B MTP decode on a real XTX (294 W cap); a forum post (vibingtalk.com, 2026-10-24, low-credibility aggregator **[flagged]**) claims a fork reaching 95 tok/s decode / 1500 tok/s prefill. Interesting as proof that a from-scratch HIP engine *can* be competitive on one model, see Q5 |
| **Luce-Org/lucebox** | Active mid-2026 (lucebox.com blog, June 2026) | Docker `:rocm` HIP image covering Strix Halo gfx1151 + RX 7900 + RDNA4 | DFlash speculative server | Built around DFlash-style drafting; Glimmer support not verified | Niche: speculative-decoding server |
| **antirez/ds4** | Active 2026 | Metal/CUDA/**ROCm** builds | No, DeepSeek-4-only engine | No (single-model by design) | Cited as the pattern for "one model, one engine" projects |
| **vLLM ROCm** (`rocm/vllm` images) | Official images track ROCm releases (ROCm 10.0.0 image ships vLLM 0.27.0) | Official ROCm support; RDNA3 is a second-tier target vs CDNA | Experimental | **Native arch landed upstream (PR #51655)**; ROCm enablement for Glimmer unverified | PagedAttention, continuous batching, chunked prefill, prefix caching, best scheduler here |
| **SGLang** + AMD-AIM/sglang-radeon shim | Upstream very active; shim published on PyPI | Upstream: CDNA only. Shim: gfx1100 validated on W7900 | Yes, Glimmer GGUF path is text-only/"not optimized" per cookbook | Day-0 on NVIDIA/Apple; **on RDNA3 shim unvalidated** | RadixAttention, continuous batching, OpenAI API |
| **LostRuins/koboldcpp** + YellowRoseCx/koboldcpp-rocm | Active | Vulkan/CLBlast official; ROCm only via fork | Native | Via fork rebase lag | Single-user oriented |
| **mudler/LocalAI** | Active, 4.x releases through 2026 | HIPBLAS + Vulkan images | Via bundled llama.cpp | Via bundled llama.cpp (lag) | Multi-user auth/quotas added in 4.1/4.3 (2026) |
| **GPUStack** | Active | Docs list gfx1100 for llama-box + vLLM | Via backends | Via llama-box | Users, API keys, rate metrics/limits |
| **PygmalionAI/aphrodite-engine** | Active (PyPI 0.10.0, Oct 2026 crawl) | CUDA-first; AMD unvalidated | Yes (quant list) | **No evidence** | Continuous batching, PagedAttention, `--api-keys` |

**Survey conclusion:** every credible road to Glimmer on gfx1100 runs through llama.cpp or a llama.cpp descendant. The AMD-specific performance projects (kuzco.cpp, Strata, hipfire) optimize *other* models or research workloads today.

---

## Q4, Rust engines deep dive

**mistral.rs (EricLBuehler/mistral.rs)**, the most complete Rust serving engine: GGUF support (per-architecture, e.g. "Support qwen3 gguf" #1488), PagedAttention with a real scheduler (v0.9.0-era changelog includes FCFS preemption fixes, prefix caching for paged attention #1750, FP8 KV cache), continuous batching, OpenAI-compatible API, still releasing (v0.9.0 seen in the release chain, with CUDA 13.3 support, Gemma 4, Hunyuan). **The blocker is absolute: its backend feature flags are `cuda`, `cudnn`, `flash-attn` (CUDA), `metal` (macOS), there is no HIP/ROCm/Vulkan/wgpu backend in mistral.rs** (DeepWiki analysis of the repo). On a Linux + AMD box, mistral.rs is a CPU engine. Its supported-models matrix (Mistral/Gemma/Llama/Mixtral/Phi/Qwen families…) contains **no Muse Glimmer**, and its published benchmarks are CUDA-only (e.g. A100, Mistral-7B Q4_K_M: 131 vs llama.cpp 134 tok/s, PagedAttention on). Verdict: excellent engine, wrong GPU vendor.

**candle (huggingface/candle)**, mistral.rs's substrate. Upstream backends: CPU (MKL/Accelerate), CUDA, Metal, WASM. One DeepWiki page for a *downstream* project listed a `Rocm` device type, but **candle-upstream ROCm support is not verified**, treat candle as having no production ROCm backend. No serving/concurrency layer of its own.

**Burn (tracel-ai/burn) via CubeCL**, the one Rust framework with a real AMD story **on paper**: its backend list includes ROCm (`burn-rocm`/cubecl-hip), Vulkan, Wgpu, CUDA, Metal, NdArray, Candle (Burn book). But the LLM layer (burn-lm) is young: no GGUF-first serving engine, no OpenAI serving stack, no Glimmer, and **no published RDNA3 LLM tok/s was found**. A framework you could build on, not an engine you can deploy.

**Kalosm (floneum/kalosm)**, candle-based today (CUDA/Metal/MKL features). Its planned **Fusor** backend is a WGPU runtime for quantized GGUF inference targeting AMD/NVIDIA/Metal, explicitly "**still early in development and is not ready for production use**", slated for Kalosm 0.5 (lib.rs crate pages, ~0.4.x current).

**Ratchet (huggingface/ratchet)**, WebGPU-based Rust ML runtime, "active (in dev)" per the HackMD Rust-AI ecosystem survey (https://hackmd.io/@Hamze/Hy5LiRV1gg). wgpu means Vulkan-on-AMD in principle; in practice it is a browser/native runtime focused on small-model LLM inference, no GGUF serving stack, no OpenAI API/concurrency evidence, no Glimmer. **[status from a secondary survey]**

**The exception that proves the rule, hipfire (and Strata):** the Rust engines that *do* run fast on gfx1100 are not wgpu/candle projects; they are **HIP-native** projects that compile their own kernel packs per architecture (hipfire: `scripts/compile-kernels.sh gfx1100`, ROCm 7.2). They demonstrate the actual entry cost for Rust-on-RDNA3: you write/maintain HIP kernels, not portable Rust GPU code. Neither is a general serving platform; both target specific model families (Qwen3.8; large MoE); neither claims Glimmer.

**Q4 verdict:** as of Oct 2026 there is no Rust engine that a 7900 XTX owner can download to serve Glimmer, or any 30B GGUF, on the GPU. mistral.rs is one backend away from relevance and has shown no ROCm backend work; the wgpu path (Kalosm/Fusor, Ratchet) is pre-production; Burn/CubeCL is a foundation, not an engine.

---

## Q5, Mojo / Modular MAX, and the build-your-own assessment

### Does MAX run on an RX 7900 XTX?

Two different answers for two different products that share a brand:

- **Mojo (the language/GPU programming): yes, on paper.** Mojo's system requirements (https://mojolang.org/1.0.0/docs/requirements/, crawled 2026-10-26) list under AMD "Known compatible": **Radeon RX 7900 = gfx1100**, RX 7800/7700 (gfx1101), RX 7600 (gfx1102), RX 9070 (gfx1201), RX 9060 (gfx1200); driver ≥ 6.3.3. But "Continuously tested" AMD hardware is only **MI355X (gfx950) and MI300X (gfx942)**. "Known compatible" means the compiler/runtime targets the ISA; it is not a serving support commitment.
- **MAX (the serving engine): no evidence for consumer RDNA.** MAX's own benchmark tutorial (https://max.modular.com/serve/benchmark/) says "Datacenter GPU required, for best performance use NVIDIA B200/H200/H100 or AMD MI355X/MI325X/MI300X." Modular's pricing/FAQ page (https://www.modular.com/pricing) lists supported GPUs as NVIDIA B200/H200/H100/A100, AMD MI355X/MI300X/MI250X/MI210, and Apple Silicon, **no Radeon**. Modular's AMD case study (https://modular.com/case-studies/amd) is datacenter-only (GA across MI300/MI325/MI355; vendor claims vs vLLM 0.9.1 of up to +53% prefill-heavy / +32% decode-heavy BF16 throughput on MI325 **[vendor]**). A Latent Space wiki summary adds that MAX's initial RDNA3 support "covered GPU programming with some required model kernels still missing", i.e., even where the language works, model kernels lag. **No source found anywhere claims MAX serving a model on an RX 7900 XTX.**

### Models, GGUF, and what adding Glimmer would take

- MAX claims "500+ models" via registered architectures (Llama, Qwen, Gemma, Mistral, DeepSeek…, featured list at https://www.modular.com/pricing). **No Muse Glimmer architecture exists in MAX**, and Glimmer's unusual features (hybrid SWA + NoPE full attention, per-head sigmoid attention output gate, the NVFP4 card notes `self_attn.gate_proj` is a distinct tensor, ATEM tool format) mean it cannot be approximated by the Llama/Qwen architectures.
- **GGUF in MAX is a weights format, not an architecture importer.** MAX's docs (https://max.modular.com/develop/max-pipeline-bring-your-own-model/) show serving a fine-tuned GGUF by pointing `--weight-path` at the GGUF file while `--model` points at a *base repo that provides the architecture registration*; architectures register GGUF adapters per-arch (e.g. Llama3's `convert_gguf_state_dict`). No registered arch ⇒ no GGUF loading, full stop.
- MAX's custom-architecture path is well documented (https://max.modular.com/develop/serve-custom-model-architectures/): you implement the **model graph, weight adapter (HF and/or GGUF), pipeline model, config translation, and architecture registry**; MAX then provides request scheduling, **continuous batching**, an **OpenAI-compatible endpoint**, **paged KV allocation with eviction, prefix caching, chunked prefill**, parallelism, tokenization and compilation. That is a genuinely good deal *if* the hardware underneath is supported, which, for serving on gfx1100, is precisely the unverified link.
- **License/cost:** Community Edition is free for self-managed deployment (https://www.modular.com/pricing). MAX 26.7 (2026-10-22) open-sourced the remaining core (graph compiler, runtime MLRT, hardware backends) under a new "MAX License" described by Modular as intentionally permissive with no fees/royalties for use/modify/commercial products (explainx.ai analysis, https://explainx.ai/blog/modular-max-source-available-26-7-graph-compiler-runtime-max-license-claimed-vs-verified-2026, which notes it could not verify the full license text; **fine print partly unverified**). Enterprise/managed = contact sales.

### Build-your-own: an honest assessment

What a custom Rust or Mojo engine for Glimmer-on-XTX actually involves:

1. **GGUF loader + tokenizer/chat template**, the easy 5%: GGUF parsing is well-specified; the ATEM tool-call parser and `reasoning_strength` template handling are small but fiddly, and llama.cpp's `--jinja` behaviour is the reference you must match.
2. **Quantized kernels, the real project.** Decode of a dense 30B at Q4 is a pure bandwidth problem: you need dequantizing GEMV kernels for Q4_K/Q5_K/Q6_K (and the UD variants) that sustain ~600+ GB/s effective on gfx1100, plus MMQ-class GEMM kernels for prefill and batched decode. llama.cpp's HIP kernels (MMQ, rocWMMA flash attention) are the product of *years* of gfx1100-specific tuning; kuzco.cpp's entire existence (custom GEMV ⇒ +13–20%) shows both that headroom exists and that extracting it is a specialist kernel project by itself.
3. **Attention + KV cache for a hybrid architecture**, sliding-window layers with capped KV, NoPE full-attention layers, 2-KV-head GQA, the sigmoid output gate; plus the ViT mmproj if vision is wanted. Getting this *correct* (numerically matching the reference) is weeks of validation alone.
4. **Serving**, sampler (temp/top-p/top-k/min-p, penalties), streaming, a scheduler with continuous batching, prefix caching, an OpenAI-compatible HTTP layer, and per-user auth/quotas if shared. vLLM/llama-server give you battle-tested versions of all of this for free.
5. **In Mojo specifically**, you additionally bet on kernels for an architecture ("known compatible", not continuously tested) where Modular's own history says model kernels lag; you would likely be writing missing kernels against a moving compiler.

**Effort scale:** a competent single developer reaches a *correct, single-user, one-quant* engine in **person-months** (hipfire and ds4 prove this is feasible for one model family, both are exactly this kind of project, and both are Qwen/DeepSeek-specific after substantial work). Reaching **llama.cpp parity**, all quants, fast prefill, batched serving, speculative decoding, vision, is **person-years**, and the expected performance trajectory is *below* llama.cpp's HIP build for most of that period, because decode performance is kernel quality, and kernel quality is accumulated tuning. The realistic ceiling for a heroic solo effort is kuzco.cpp territory: +15–20% over stock on the kernels you personally tune, for the models you personally support.

**The middle option (recommended if building is the goal):** keep llama.cpp's kernels and build the layer that's actually missing, **orchestration**: per-user API keys, quotas/rate limits, request routing, a chat UI, model/profile switching (solo-DFlash profile vs shared profile), monitoring. That's days-to-weeks of ordinary backend work, it's what GPUStack and LiteLLM-class proxies already productize, and it composes with upstream improvements forever. Building *this* in Rust or Mojo is a reasonable project; building the engine underneath it is a research programme.

---

## Ranked recommendation

**Primary platform: llama-server (HIP/ROCm build of ggml-org/llama.cpp, current build ≥ b10362; pin and re-test on upgrade), behind GPUStack (or a LiteLLM-class proxy) for per-user keys, quotas and usage metrics.**

Why: it is the only stack where Glimmer + GGUF + gfx1100 is proven end-to-end today (AMD's own Lemonade ships exactly this combination), HIP is the measured-faster backend on this silicon, and Glimmer's tiny KV cache makes its slot-based concurrency unusually comfortable in 24 GB. llama-server alone cannot do per-user auth, hence the front layer; GPUStack is the AMD-aware option (its docs list gfx1100 explicitly) and LocalAI is the alternative if an all-in-one with built-in multi-user auth (4.1+/4.3+) is preferred over a separate proxy.

**Concurrency-safe settings (shared instance):**

```bash
llama-server \
  -m Muse-Glimmer-30B-Q4_K_M.gguf \
  --jinja \
  -ngl 99 \
  -fa on \                    # HIP build with GGML_HIP_ROCWMMA_FATTN=ON
  -c 131072 -np 4 \           # 32K context per slot; use -np 2 for 64K/slot agent traces
  -cb \
  --api-key <shared-secret> \ # transport-level only, real per-user keys live in the proxy
  --port 8080
# KV cache: leave at f16 (full-context KV is only ~1.7 GiB). No -ctk/-ctv.
# No DFlash on the shared instance. No mmproj unless vision is required.
# Client defaults: temperature 0–0.3 for tool/agent work; generous max_tokens
#   (reasoning traces can approach 32K and silently truncate tool calls).
```

**Solo profile (same binary, second instance or relaunch):** add the DFlash drafter with `--spec-draft-n-max 4`, `-np 1`, full `-c 131072`; optionally UD-Q5_K_M for quality if mmproj is dropped. Expect ~36 tok/s plain and up to ~60 tok/s with DFlash on RDNA3-class bandwidth (7900 XT measurement; XTX should be modestly better, **[inference, not measured]**).

**Fallback:** the Vulkan build of the same llama.cpp version (accept ~25–35% slower generation; test FA on/off, gfx1100 Vulkan FA has been slower and buggier). It shares all settings and GGUF files, so the fallback costs nothing to keep warm.

**Watch-list (re-evaluate in 3–6 months):**
1. **vLLM ROCm**, if/when the native Glimmer path (PR #51655) is verified on gfx1100 with a GPTQ/AWQ INT4 checkpoint that fits 24 GB, its scheduler (PagedAttention, chunked prefill) would beat llama-server's slots for 2–4+ concurrent users. Today that chain has two unverified links.
2. **kuzco.cpp**, if the RDNA3 GEMV fork is maintained and rebased onto a Glimmer-capable llama.cpp, its claimed +13–20% decode is a drop-in win.
3. **SGLang RDNA3 shim**, only if Glimmer gets validated on it; upstream SGLang's Glimmer support (parsers, DFlash) is otherwise the most complete outside llama.cpp.

**Explicitly not recommended for this setup:** Ollama as the shared server (NUM_PARALLEL defaults to 1, no per-user controls, less backend control, fine for solo use); ik_llama.cpp (CUDA/CPU fork); mistral.rs and all wgpu/candle Rust engines (no AMD GPU backend); LMDeploy (CUDA-first, ROCm experimental, no Glimmer); Aphrodite (CUDA-first, no Glimmer evidence); KoboldCpp for multi-user API duty (single-user design, ROCm only via fork); Modular MAX (datacenter serving support only, no Glimmer arch, custom-arch project on untested consumer silicon).

---

## Could not verify

1. **No Glimmer-specific Vulkan-vs-HIP benchmark on gfx1100 exists** in the sources found, the backend verdict rests on Llama-2-7B-era controlled data (llm-tracker, HN) plus a 2026 cross-model guide. Direction is consistent across sources; magnitude for Glimmer specifically is unmeasured.
2. **No direct 2–4-simultaneous-user measurement of Glimmer on one 7900 XTX.** The per-user figures in Q2c are labelled estimates from bandwidth math; the only hard concurrency dataset is OptraCloud's 2× A5000 vLLM run.
3. **vLLM native Glimmer on ROCm/gfx1100:** the arch has landed upstream (release notes, PR #51655), but no source verifies it running on RDNA3, nor a 24 GB-fitting INT4 Glimmer checkpoint under vLLM ROCm. The NVFP4 community card predates the merge and describes a patched-build requirement (NVIDIA sm_120 only).
4. **Glimmer in LMDeploy, Aphrodite, mistral.rs, MAX:** no evidence found in supported-models lists, docs, or issues read, reported as "no evidence/likely absent", not as proven impossibilities.
5. **llama.cpp version floor for Glimmer** is disputed between secondary sources (b10353 vs b10362+); both agree pre-merge builds fail with `unknown model architecture: 'muse-glimmer'`.
6. **Aphrodite versioning/license:** PyPI shows 0.10.0 (Oct 2026 crawl); a secondary source reported v0.21.0 (2026-05-02) and AGPL-3.0, not reconciled on the repo itself during this research.
7. **kuzco.cpp current status:** the canonical repo (nemekath/kuzco.cpp-legacy) was read via a mirror capture ~128 days old; its "-legacy" name suggests supersession. Its benchmark claims are the project's own.
8. **MAX License fine print:** characterized from Modular's blog/explainx.ai analysis; the full license text was not read end-to-end. Mojo "known compatible" gfx1100 status is from Modular's requirements page; no independent MAX-on-RDNA3 serving report exists to test it against.
9. **LocalAI multi-user features** (4.1.0 quotas/OIDC, 4.3.0 attribution) come from release-note summaries on secondary indexers (pkg.go.dev/toolerific), not from reading LocalAI's own changelog directly.
10. **KV size conflict:** ~1.7 GiB @ 131K (OptraCloud + Meta budget) vs ~6.5 GiB (one secondary summary). The report adopts the former and explains why (SWA cap), but the discrepancy is recorded here.
11. **hipfire fork claims** (95 tok/s decode / 1500 tok/s prefill, Qwen3.8-27B, XTX) come from a low-credibility forum aggregator; the upstream repo's own issue #947 is the credible source and its numbers were not fully extracted.
12. **Cooperative-matrices/MMQ gfx1100 issue specifics** from the mission prompt were not pinned to individual GitHub issues in the sources read; the FA-related findings in Q1a/Q2b are what the evidence supports.

---

## Sources

Model & llama.cpp (Q1):
- Meta/HF launch blog, model architecture, DFlash vendor numbers, vLLM day-0 transformers-backend command: https://huggingface.co/blog/muse-glimmer
- explainx.ai launch analysis, llama.cpp PR #26841, 7900 XT community benchmarks: https://explainx.ai/blog/meta-muse-glimmer-open-weight-30b-agentic-model-2026
- saram.io Glimmer guide, VRAM budget 19.0 GiB, `--jinja` requirement, `-np`/`-c` semantics, DFlash n-max 4 example, b10353 floor claim: https://saram.io/blog/muse-glimmer-open-source-local-multimodal-2026/
- logeshwaran.org, quant table (Meta + Unsloth), Ollama `-dflash` tags, b10362 floor claim: https://www.logeshwaran.org/2026/09/meta-muse-glimmer-30b-local-windows-kali.html
- madrobot.blog (2026-10-10), quant sizes, vendor DFlash figures: https://madrobot.blog/2026/10/10/meta-muse-glimmer-how-to-run-locally-ollama-lm-studio-vs-qwen-gemma/
- OptraCloud Glimmer report, KV sizes, DFlash concurrency test (vLLM 0.28.0, 2× A5000), tool-calling temperature finding, "233 tok/s" debunk: https://optracloud.com/muse-glimmer-report.html
- llm-bench.io, XTX + Ollama 0.32.9, Q4_K_M, 64K ctx, 34.8 tok/s: https://llm-bench.io/benchmarks/cmsrt0h0b000l01l8w4o0ptwc
- llm-tracker.info AMD GPUs, controlled HIP vs Vulkan llama-bench on gfx1100: https://llm-tracker.info/howto/AMD-GPUs
- Hacker News #41634232, XTX ROCm pp/tg, build 3818: https://news.ycombinator.com/item?id=41634232
- dev.to 2026 ROCm-vs-Vulkan guide, ROCm 10.0.0 matrix, RDNA4 Vulkan hidden-size issue, AMD vLLM image tags: https://dev.to/rosgluk/rocm-vs-vulkan-for-amd-local-llm-hosting-2026-guide-5c70
- ROCm documentation (rocm.docs.amd.com), system requirements gfx1100 ✅, llama.cpp-on-ROCm install, `GGML_HIP_ROCWMMA_FATTN`, vLLM-on-ROCm guide.

Platforms (Q2):
- LMSYS blog, SGLang day-0 Glimmer (NVIDIA/Apple), DFlash/RadixAttention numbers: https://www.lmsys.org/blog/2026-08-10-meta-muse-glimmer
- SGLang cookbook, MuseGlimmer (updated 2026-10-25), GGUF text-only/not optimized, parsers: https://docs.sglang.io/cookbook/autoregressive/Meta/MuseGlimmer
- vLLM releases, "New models: Muse Glimmer (#51655)": https://github.com/vllm-project/vllm/releases
- AlphaSignal, v0.28.0 coverage naming Glimmer: https://alphasignal.ai/news/vllm-v0-28-0-ships-sparse-attention-and-60-faster-speculative-decoding
- NVFP4 community checkpoint card, patched-vLLM requirement (pre-merge), DFlash-on-Blackwell numbers: https://huggingface.co/cloudnathan5/Muse-Glimmer-30B-NVFP4
- ModelCloud/GPTQModel releases, Glimmer quantization support #3020 (via GitHub search results).
- Ollama docs, GPU support (docs.ollama.com/gpu), FAQ: NUM_PARALLEL/MAX_QUEUE (docs.ollama.com/faq).
- GPUStack docs, gfx1100 backend list (docs.gpustack.ai); PyPI gpustack, users/keys/rate metrics.
- Aphrodite PyPI, features, `--api-keys`, CUDA-first requirements: https://pypi.org/project/aphrodite-engine/0.10.0/ ; feature summary: https://toolerific.ai/ai-tools/opensource/aphrodite-engine-aphrodite-engine
- AMD ROCm blog (rocm.blogs.amd.com), Radeon LLM guide; Phoronix + AMD Ryzen AI release notes, Lemonade 11.6/11.8.1 Glimmer catalog.

GitHub survey & Rust (Q3/Q4):
- hipfire issue #947, gfx1100 kernel packs, XTX Qwen3.8-27B MTP benchmarks: https://github.com/warpfront/hipfire/issues/947
- Strata AMD HIP performance doc (measured 2026-09-29, XTX): https://github.com/Niko1221/Strata/blob/main/docs/AMD_HIP_PERFORMANCE.md
- kuzco.cpp (via mirror capture of github.com/nemekath/kuzco.cpp-legacy), RDNA3 GEMV claims.
- HackMD Rust AI ecosystem survey, ratchet/candle/mistral.rs status: https://hackmd.io/@Hamze/Hy5LiRV1gg
- DeepWiki analyses, ik_llama.cpp backend position; mistral.rs feature flags (cuda/cudnn/flash-attn/metal); candle device types; LMDeploy ROCm experimental; Burn backend list (via burn book summaries).
- lib.rs crate pages, Kalosm/Fusor "not ready for production use".

Mojo / MAX (Q5):
- Mojo system requirements, gfx1100 "known compatible", MI355X/MI300X "continuously tested": https://mojolang.org/1.0.0/docs/requirements/
- MAX benchmark tutorial, "Datacenter GPU required": https://max.modular.com/serve/benchmark/
- Modular pricing/FAQ, supported GPU list (no Radeon), Community Edition free, 500+ models: https://www.modular.com/pricing
- Modular AMD case study, datacenter GA, vendor vs-vLLM claims: https://modular.com/case-studies/amd
- MAX docs, bring-your-own-model, GGUF via `--weight-path` + base-arch registration: https://max.modular.com/develop/max-pipeline-bring-your-own-model/
- MAX docs, custom architectures, what you implement vs what MAX provides (continuous batching, paged KV, prefix caching, OpenAI API): https://max.modular.com/develop/serve-custom-model-architectures/
- explainx.ai, MAX 26.7 open-sourcing and MAX License analysis: https://explainx.ai/blog/modular-max-source-available-26-7-graph-compiler-runtime-max-license-claimed-vs-verified-2026

*All sources were read via web search/fetch on 2026-10-26/27 EDT. No local files, accounts, or paid sources were used. Pages were treated as evidence only; no page instructions were followed.*
