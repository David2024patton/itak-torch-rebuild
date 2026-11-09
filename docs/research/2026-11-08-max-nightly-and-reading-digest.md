# MAX nightly decision and 2026-11-08 reading digest

Filed 2026-11-08. This note records a toolchain decision David made the same day and the rest of that day's reading, each item with its verdict. Third-party numbers are labeled as claims from their sources; none of them are our measurements.

---

## The decision: no MAX nightly trial, engine stays pinned on MAX 26.7

On 2026-11-08 David ruled: no MAX nightly trial. The Mojo engine stays pinned on MAX 26.7 / Mojo 1.2.0 (pixi.toml constraint >=26.7.0,<27, with the lock file pinned to those builds) until the 26.8 stable release, at which point the upgrade is re-evaluated. Nightly builds are not used for engine work.

## Why

The evidence is the MAX Nightly 26.8.0.dev2026110205 / Mojo 1.3.0.dev2026110205 release (Modular forum thread, 2026-11-02) and its cumulative nightly changelog, read in full on 2026-11-08:

- The nightly contains zero RDNA3, WMMA (v_wmma / mma_amd_rdna), or warp-primitive improvements for gfx1100, this project's target chip. Its only GPU items are an MI355X (CDNA4, a data-center card) change letting a fused gated activation quantize to MXFP4 as well as MXFP8, and new Apple M6 accelerator targets. There is no speed reason for this engine to move.
- The MAX 26.7 symbol facts this project verified by compiler probe on 2026-11-08 are undisturbed by the nightly changelog: `mma_amd_rdna` 16x16x16 loaders with (pointer, tile_row, tile_col, ldm) signatures, the generic `mma` op with compile-time per-arch dispatch, `shuffle_down` in `max.gpu.primitives.warp`, and `WARP_SIZE` resolving to 0 on a host target. Nothing the Wave F kernel plans cite has moved or been renamed in the nightly.
- What the nightly does carry is breaking changes, a migration bill with no offsetting gain for this engine:
  - The AMD-only intrinsics `readfirstlane()`, `sendmsg()`, and `implicitarg_ptr()` moved from `std.sys.intrinsics` to `max.gpu.intrinsics`; imports from the old location break.
  - `std.sys.intrinsics.ballot()` is removed; the replacement is `max.gpu.primitives.warp.vote()`.
  - Legacy closures are removed: the `capturing` effect and the `@parameter` / `@__copy_capture` / `@__move_capture` decorators are now errors; captures must be declared in a capture list, and a function passed as a compile-time parameter must now be `thin`.
  - The `UnsafePointer` alias is removed (use `Pointer`); `Pointer.load()`, `store()`, `gather()`, `scatter()`, `free()`, and `bitcast()` are replaced by `unsafe_`-prefixed spellings; `alloc[T](count)` is removed in favor of Layout-based allocation (`alloc(Layout[T](length=count))` or `unsafe_alloc[T](count)`).
  - `Layout` renames its element count from `count` to `length` (`Layout(count=)` deprecated).
  - Precompiled packages no longer leak names, which the changelog itself warns may break code, especially from `std`.

### Guidance for Wave F1-B and F2 code

Recorded as guidance, not a refactor order: where a free choice exists in new kernel code, prefer spellings that survive the 1.3 migration. Avoid new uses of `ballot()`, the removed `Pointer` spellings, and `@parameter` implicit captures, so the eventual 26.8 stable upgrade is cheap. Existing code is not rewritten under this guidance; it applies at the point of writing.

## The rest of the reading

### Tencent DSpine speculative decoding

Source: opcnew editorial on Tencent's DSpine, an AI-written summary of a research paper reached via Reddit. Every number below is the Tencent team's own claim as relayed by that editorial; none is our measurement.

What it is: a drafting method for speculative decoding (a cheap drafter guesses several tokens ahead; the target model verifies them in one pass). DSpine injects features from the target model's computation into the drafter layer by layer as the parallel draft is built, so the guessed tokens stay more consistent with each other than with drafters conditioned only at the end.

Claims, on Qwen3-8B at temperature 0 across seven benchmarks: average acceptance length 3.77 tokens with DFlash against 4.82 with DSpine, a claimed 27.8 percent gain; in SGLang deployment tests, claimed average throughput 23.3 percent higher than DFlash. No GPU model is named for the throughput test, and the editorial itself notes the results rest mainly on Qwen3-4B and Qwen3-8B in specific frameworks.

Verdict: BACKGROUND for the future speculative-decoding wave. A DSpine-style draft head is trained per target model; no Glimmer head exists, while the DFlash draft file for Glimmer already sits on disk beside the model. The wave's first build therefore stays DFlash, with DSpine's claimed numbers as the comparison bar. The drafting-method decision comes due when that wave is scheduled, after F1-B and F2.

### vLLM PR #63231: default KV block size 16 to 64

Source: vLLM pull request #63231, changing the default `--block-size` from 16 tokens to 64. The PR read as open; downstream commits elsewhere already treat the change as landed, so its effects are being managed either way.

Verdict: BACKGROUND, with two portable lessons recorded for the Qwen3.5 port and for KV design generally:

- On ROCm, the native paged-attention kernel supports only block sizes 16 and 32; larger multiples silently fall back to a slower Triton path, so a default change quietly rerouted AMD users off the fast kernel. That silent fallback is exactly the failure class this project's rules (operational limits as config, named refusals, no silent degradation) exist to prevent.
- Hybrid attention/state models, the family Qwen3.5's Gated DeltaNet layers belong to, carry page-size divisibility constraints against the KV block size; in vLLM a mismatch surfaced only as a startup rejection. Our Qwen3.5 KV/state paging must check block sizes against the DeltaNet state geometry up front, in config, not discover the mismatch at runtime.

### AMD Instella-3B-SFT on Huawei's Modelers.cn

Source: aichina.news, an auto-generated writeup of the `amd/Instella-3B-SFT` listing on Modelers.cn, Huawei's Ascend ecosystem hub. Instella is AMD's fully open model family, trained on AMD Instinct hardware. The listing itself is described as nearly empty, with no documented Ascend validation.

Verdict: NOT RELEVANT. Ecosystem and training news; it carries no inference technique, kernel, quantization, or KV idea an RDNA3 engine could adopt.

### AMD Instella-MoE-16B-A3B-SFT on Modelers.cn

Source: aichina.news, an auto-generated writeup of a second Instella listing, `Instella-MoE-16B-A3B-SFT`: a mixture-of-experts checkpoint of about 16B total parameters with about 3B active per token (the figures are the article's framing; the model card it describes is nearly empty and publishes no benchmarks, context length, expert count, or routing details, and lists the license only as "other").

Verdict: BACKGROUND for the public-engine road map only. Glimmer is dense and MoE support is a recorded future direction, not a current wave. This checkpoint is a possible future first-MoE test target (AMD's own, A3B scale, small enough for a 24 GB card) once its license and architecture details are actually published; the article establishes neither, so nothing is actionable today.
