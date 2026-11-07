# Paged Experts: running an MoE model larger than VRAM, 2026-11-07

Source: DEV Community article, "Running an MoE Model Larger Than Your VRAM: SGLang Paged Experts, the PCIe Math, and the 4-Request Cliff" (AI Tech News, posted 2026-11-06), covering SGLang 0.5.23 (released 2026-11-02), PR #42557 ("Paged Experts, serve MoE models whose weights do not fit VRAM", merged 2026-10-19) and PR #45787 (Paged Experts for the remaining expert formats, merged 2026-10-25), and the Paged Experts documentation at v0.5.23. David pasted the article on 2026-11-07 and ordered it noted.

Filed because: this engine's stated direction is a public engine (see the 2026-11-07 ash-backend discussion), and MoE architectures are common among public models. A public engine that cannot run a model larger than the card's VRAM loses to one that can. This note records the design and its measured caveats as reported by others. Every number below is a reported number from the article or the PRs it covers, or arithmetic the article's author computed on top of them. None of it is our measurement; nothing here has been run on our stack or our hardware.

---

## The mechanism

A mixture-of-experts (MoE) layer routes each token to a small subset (top_k) of its E experts; the experts hold most of the model's parameters, so a model that is cheap to compute can still be impossible to hold in VRAM. SGLang's Paged Experts (opt-in in 0.5.23, flags `--enable-paged-experts` and `--paged-experts-num-resident K`) works as follows:

- A K-slot resident expert table per layer lives on the GPU. K is the number of experts held resident per layer, out of E.
- All E experts of every layer are held in pinned (page-locked) host RAM.
- Each decode step, the experts that step routes to and that are not resident are copied from host RAM into slots, evicting the least recently used; the model's own fused MoE kernel then runs on the GPU as usual. Routing itself is untouched.
- The pattern is the HiCache idea applied to expert weights: the GPU holds the working set, host RAM holds the rest.

Constraints as documented for 0.5.23:

- Single GPU only. Tensor, expert, pipeline, and data parallelism are rejected at startup.
- Speculative decoding is rejected at startup while the feature is on.
- The host store must be able to pin every expert of every layer; under WSL2 pinned memory is capped at about half of system RAM.

## Reported numbers

Test model: Qwen3-30B-A3B (48 layers, 128 experts per layer, top_k 8, hidden 2048, MoE intermediate 768, per the model's config.json as checked by the article). PR benchmarks ran on two consumer machines (RTX 5070 Ti desktop, 16 GB, PCIe 5.0 x16, native Linux; RTX 4070 Laptop, 8 GB, PCIe 4.0 x8, WSL2), ShareGPT prompts, 256 output tokens, prefix caching off in both engines, KV capacity matched at 10,240 tokens, byte-matched quantization (GPTQ int4 against IQ4_XS, FP8 against Q8_0). llama.cpp was tuned to its best `--n-cpu-moe` setting per run.

Single-stream decode, tokens per second, from the PR tables as reproduced by the article:

| Setup | Paged Experts | llama.cpp, best `--n-cpu-moe` | SGLang `--cpu-offload-gb` |
|---|---|---|---|
| 16 GB, int4, K=92 | 205.6 | 194.4 | 16.6 |
| 16 GB, FP8, K=40 | 87.7 | 45.2 | 2.9 |
| 8 GB laptop, int4, K=26 | 32.6 | 32.8 | 1.2 |

Read plainly: the 12x to 30x headline multiples are against SGLang's own CPU offload, the weakest baseline. Against a tuned llama.cpp the result splits: a tie when most of the model fits or the link is slow (int4 on both machines), and about 1.94x in the FP8 case, where many experts must live in RAM and the link is fast, because llama.cpp's CPU-side expert layers become its bottleneck.

First-token time (TTFT) tells the other half: paging pays a fixed page-in cost up front (see Costs), so llama.cpp had lower TTFT on short prompts in all three setups, while paging won the 2,048-token TTFT in two of three (267 vs 344 ms int4 16 GB; 443 vs 984 ms FP8; laptop 1,441 vs 2,396 ms).

## The reuse premise (the article's arithmetic, labeled as such)

Per the article's own computation on the PR numbers, not a measurement: at batch 1 a token activates 48 x 8 = 384 experts; at the documented per-expert sizes (int4 about 2.43 MB, FP8 about 4.72 MB), a token whose every activation missed would move about 0.93 GB (int4) or 1.81 GB (FP8) over the link. The measured FP8 decode rate (87.7 tok/s, 11.4 ms per token) would need about 159 GB/s to sustain an all-miss stream, more than a PCIe 5.0 x16 link delivers (about 63 GB/s theoretical; the PR's prefill page-in implies an effective roughly 50 GB/s). Under uniform random routing the FP8 case caps near 40 tok/s; it measured 87.7. Conclusion the article draws: consecutive tokens must be reusing resident experts, and that reuse is load-bearing for the whole result. Two caveats the article states and this note keeps: the computed figures are upper bounds on the miss rate, not measured miss rates (only a counter inside the engine gives the real value), and the reuse was seen on natural-language prompts; the PR warns that synthetic prompts mis-measure paging. Any implementation here must be benchmarked on real prompts.

## The cliff and the sizing rule

With 4 concurrent requests, the reported results split cleanly on one quantity. A decode step is served in a single wave, the case a CUDA graph can capture, only if the batch's routed experts fit the resident table. The article's usable sizing rule: choose K at least batch x top_k for the concurrency you plan to serve.

| Run | K | K // top_k | 4-request tok/s, paged vs llama.cpp |
|---|---|---|---|
| int4, 16 GB | 92 | 11 | 387 vs 312 |
| FP8, 16 GB | 40 | 5 | 84 vs 55 |
| FP8, 16 GB | 24 | 3 | 31 vs 55 |
| int4, 8 GB laptop | 26 | 3 | 11 to 12 vs 41 |

Both runs with K // top_k below 4 lost; both at 5 or more won. Refinement recorded from the article's top comment (Ahmet Ozel, 2026-11-07), which this note adopts: K at least batch x top_k is a worst-case sizing guarantee, not a hard measured limit, because several requests can route to overlapping experts, so the union of selected experts per layer can be much smaller than the sum of their top_k counts. K // top_k is therefore not necessarily the largest batch that runs in one wave. If implemented here, instrument the per-layer union size and the actual wave count beside concurrency, keep the conservative capacity rule distinct from the measured limit, and relate any cliff to the routing diversity of the workload.

## Costs

- Resident slots and the KV cache draw from the same static budget (SGLang's `--mem-fraction-static`). One slot costs per_expert_bytes x num_moe_layers: on this model about 117 MB per int4 slot and about 227 MB per FP8 slot. Against the model's bf16 KV cache (98,304 bytes per token), one int4 slot is roughly 1,190 tokens of context and one FP8 slot roughly 2,300; dropping K by 10 on the int4 model buys roughly 11,900 tokens of context at the price of more paging.
- Prefill touches nearly every expert, so it pages in everything not resident, once: about 4.2 GB at int4 K=92, about 19.9 GB at FP8 K=40, about 11.9 GB on the laptop. That fixed cost is why short prompts lose on first-token time while long prompts win.
- Host RAM must hold the full pinned expert store: by the article's arithmetic, about 14.9 GB at int4 and about 29 GB at FP8 for this model.

## The silent-failure class (why this note carries a correctness section)

The paging layer's failure mode, as described in PR #45787 and the article, is fluent wrong text with no crash:

- An expert tensor not listed as pageable stays unpaged, so a paged-in expert runs with another expert's values. In an AWQ control with the zero points unpaged, the median logprob error was 1.67 nats, against 0.003 when paged correctly.
- gpt-oss-20b's MXFP4 experts are padded by the Marlin kernel (2880 to 2944 or 3072). Before its fix, paged gpt-oss-20b scored 0.000 on GSM8K with every answer truncated; after the fix, 0.960. The docs report bit-identical logprobs for unquantized, FP8, and INT8 experts under deterministic inference and agreement within Marlin's own batch-to-batch noise for the other formats; on the same GSM8K check (200 questions), paged MXFP4 scored 0.96 to 0.97 against 0.975 unpaged.

SGLang's structural fix for the first class: the pageable tensor set is derived from the layer itself rather than listed by hand, and at startup an independence check re-runs one more chunk of experts and fails startup if the post-load repack mixed experts.

Our law if this is ever implemented here, stated now so it cannot soften later: an accuracy smoke test against the unpaged model on the user's own checkpoint format is mandatory before the feature may serve, in the same spirit as this repo's identity and golden checks. A paging bug that produces fluent wrong text is the worst defect class this engine recognizes; the smoke test and a startup independence check in SGLang's shape are part of the feature, not follow-ups to it.

## Applicability to this repo

- Glimmer, this repo's current first-class model, is a dense model: every part of it computes on every token. Paged experts does nothing for Glimmer. This note is road-map material only (TODO T-5.6), recorded for the public-engine direction, where MoE architectures are common and running a model larger than VRAM is a competitive requirement.
- This repo already has two streaming patterns at the same seam. PLAN S8's itak-moe (TODO T-5.4) is the Colibri pattern: experts on disk, pin and LRU cache in RAM. Paged experts is the adjacent pattern one tier down: experts resident in pinned host RAM, an LRU slot table in VRAM, GPU compute always. The prefix-KV store is the other relative: it already moves state between host and GPU under an exactness contract. Paged experts applies the same pattern to weights, with two added dangers the KV store does not have: eviction (a needed expert may not be resident) and routing (the set needed per step is data-dependent).
- Composition conflict, named now: SGLang rejects speculative decoding at startup with this feature on. Any implementation here must state how it composes with the spec runners (DFlash and n-gram) or refuse the combination by name; silently degrading either path is not acceptable.
- Gating and sequencing: the feature is gated on MoE architecture support existing at all in this engine (no MoE arch module exists today), and it is sequenced behind the ash backend work. It is not a current job.
