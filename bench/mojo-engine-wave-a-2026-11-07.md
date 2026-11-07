# Mojo engine Wave A: loader, tuned K1, GEMV family, ops, tokenizer

Date: 2026-11-07. Branch `mojo-max-engine`. Engine version
1.1 -> 1.2. Built on David's order: "launch all the sub agents
you can and let's get this built fast." Five lanes ran in
parallel in their own worktrees on disjoint file sets; the
coordinator integrated them one at a time in dependency order
(A1, A3, A4, A2, A5), ran the full Mojo suite after each
integration, pushed the branch after each, and closed with an
independent verification pass over the integrated tree.
Remote main never moved off b416987.

Suite at wave end (VM, `tests/run_tests.sh`): 78 tests across
16 TestSuite files plus the 3-test seam-on diag rerun, config
probes, and the engine hello: PASS, 0 failed, 0 skipped. An
independent rerun from a scratch copy reproduced 81/81 test
executions passing.

## Lane A1: GGUF loader (src/gguf/loader.mojo)

Staged file read, not mmap (Mojo 1.2's stdlib has no mmap
binding; MAX host buffers would force a device context on a
pure host loader). Header/metadata/directory parse through a
buffered sequential reader; tensor bytes served by seek +
chunked read. Same observable semantics as the Rust mmap
loader. Version 3 only, alignment from `general.alignment`
else 32 with power-of-two validation, mainline ggml type
table, supported set F32/F16/Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/
BF16, parse-time element-count and tensor-range validation,
all 13 metadata value types. Named refusals: gguf_bad_magic,
gguf_unsupported_version, gguf_truncated,
gguf_unknown_dtype, gguf_unsupported_tensor_type,
gguf_tensor_not_found, plus nine more (missing metadata,
type mismatch, invalid utf8, length overflow, bad element
count, out of bounds, invalid format, io error).

Proofs: VM synthetic round-trip suite 14/14 in both seam
modes (17 metadata keys covering every value type incl. 4
array kinds; F32, F32-2D, Q4_K tensors; 11 crafted bad files
each asserting their named refusal). Server proof against
the real Glimmer file (sha256 4cc57c0f...f60e): exactly 731
tensors, 32 metadata KV, version 3, alignment 32, data_start
13,116,544 (matches an independent Python parse). Spot
values exact: general.architecture muse-glimmer, block_count
52, embedding_length 6656, head_count 32, head_count_kv 2,
context_length 131072, tokenizer.ggml.tokens length 202048.
Fixture tensor blk.0.ffn_gate.weight (6656x19968, Q4_K,
74,760,192 bytes): FNV-1a64 over loader-served bytes
0x88b58419387d763c, identical to an independent Python FNV
over the raw file range; sha256 of the loader-served dump
6f877bd6...7142 also identical to an independent extraction.

## Lane A2: tuned K1 (src/kernels/k1_gemv.mojo)

The shipped wgpu r2sg shape expressed in Mojo/MAX idioms:
128 threads/workgroup as 8 groups of 16, 2 rows per
workgroup, one 16-thread group per Q4_K super-block, packed
scale extraction from header words loaded as one vec4,
activation quads loaded once per super-block and shared by
both rows, per-sub-block dots in d*sc*dot(q,x) - dmin*m*sum(x)
form, vector nibble unpack, bit-trick f16 decode, segmented
shuffle_xor butterfly plus a shared-memory finish.

The ladder (RX 7900 XTX, cleared card, fixture
blk.0.ffn_gate.weight, 74,760,192 B/iter, 200 iters + 20
warmup, correctness gate first every window; gate: rel no
worse than 1.267e-07, cosine 1.000000000 class):

- Straight port baseline: 39.8 GB/s recorded median
  (in-window arm 40.3).
- Iter 1, r2sg restructure: 633.5 median. 15.7x baseline.
- Iter 2, workgroup sweep: WG64 617.9, WG256 605.4, both
  worse and rel drifted up; WG128 kept.
- Iter 3, header vec4 load: 637.6 median (vs 626.2 same
  window). Kept.
- Iter 4, helper/SIMD-accumulator refactor: 580.2 (-9%),
  reverted; rows=4 in that form 610.5.
- Iter 5, rows=4 scalar-unrolled: 624.5 vs 631.2 same window.
  No win.
- Iter 6, super-block loop unroll-by-2: 647.5 median but rel
  rose to 1.490e-07, over the gate. REJECTED; the gate was
  not traded for speed.

Final committed kernel, fresh staged window: correctness rel
1.267e-07, cosine 1.000000000 (vs the wgpu output rel
1.788e-07), exactly the shipped wgpu kernel's recorded
figures. Timing 620.5 / 626.2 / 624.1, median 624.1 GB/s =
74.4% of the recorded wgpu median 838.4 GB/s.

VERDICT, plainly: BELOW the within-10% line (754.6 GB/s) by
~130 GB/s. A 15.7x speed-up over the straight port, not
parity. Remaining-gap analysis (the lane's, recorded as
analysis): the binding constraint measured is the
correctness gate itself; every accumulation reorder that
bought speed moved rel to 1.34-1.49e-07 (all would pass a
1.267e-6 bar; whether the bar should move is a decision for
David, not a lane). Beyond that: Mojo/MAX codegen issues
scalar u32 qs loads per lane where the WGSL path gets better
scheduling, reduce_add on SIMD4 lowers to scalar chains vs
WGSL dot(), all 16 lanes redundantly load/decode the header,
and shuffle_xor's RDNA lowering cost is unmeasured. Future
levers: ISA/occupancy inspection, MAX block-reduction
primitives, finer launch-config sweep.

## Lane A3: dequant + GEMV family (src/kernels/)

Census (crates/muse-glimmer weights.rs/config.rs, verified
live on the server file; full-file census: Q4_K 365, F32 313,
Q6_K 52, Q5_K 1, F16 0): Q6_K = blk.*.ffn_down [19968x6656]
and blk.*.attn_v [6656x1024]; Q5_K = output.weight only
[6656x202048]; F32 = all norms; F16 = no weight tensors in
this model (built as the loader-served storage variant).

Files: dequant.mojo (shared: exact f16->f32, Q6_K and Q5_K
per-element dequant mirroring crates/gguf/src/dequant.rs term
for term), q6k_gemv.mojo, q5k_gemv.mojo, f32_gemv.mojo (F32
and F16-storage kernels). Each: GPU kernel + CPU twin, diag
wired, named refusals. Straight ports per the wave charter;
tuning them is later work.

VM proofs (synthetic fixtures, named as such, in-test
independent Float64 references; bars cosine >= 0.999999999,
rel <= 1e-5): Q6_K cosine 1.0 rel 0.0; Q5_K cosine 1.0 rel
0.0; F32 pass; F16 cosine 0.9999999999999934 rel 2.94e-07.
One real bug found and fixed en route (Q6_K stream nibble
selection).

GPU proofs: Q6_K by the lane on the real blk.0.ffn_down rows
0..63 (k2_check LCG seed 42, independent Float64 numpy
reference): cosine 1.000000000, rel 9.710e-08, at the
recorded k2_check bars. Q5_K, F32, F16 by the independent
verification pass on the integrated tree, same window
discipline, rows 0..63 x 6656, k3-class bars (rel < 1e-4,
cosine > 0.999999):

- q5k_gemv: cosine 0.999999999999996, rel 9.425e-08. PASS.
- f32_gemv (f32): cosine 0.999999999999790, rel 8.426e-07.
  PASS. Fixture: the first 64 real F32 [6656] tensors in file
  order (the file's only F32 tensors are norm vectors).
- f32_gemv (f16): cosine 0.999999999999710, rel 1.033e-06.
  PASS. Fixture: the F16 encoding of those same real rows;
  the file carries zero native F16 tensors, so this proves
  the kernel against real model values in F16 storage, not
  against a native F16 tensor. Stated plainly.

## Lane A4: elementwise ops (src/ops/)

Files: rms_norm.mojo (weighted + plain forms; Glimmer's
embedding norm is a distinct step in the Rust code),
rope.mojo, swiglu.mojo, residual.mojo, softmax.mojo,
final_logit.mojo (its own module: scale first, then
cap * tanh(z/cap), exactly muse-glimmer model.rs
softcap_scale), embedding.mojo (F32 row gather; quantized
dtypes refuse by name, dequant belongs to the kernels
family). Each: CPU twin + GPU kernel, diag wired, named
refusals.

Values with Rust citations: RMS epsilon 1e-5
(crates/muse-glimmer/src/config.rs census + config test);
formula from crates/torch-math/src/norm.rs (Float64
mean-of-squares). RoPE base 500,000 (same census); pairing is
the interleaved NORM (adjacent) pairing from
crates/torch-math/src/rope_interleaved.rs (the half-split
NEOX in rope.rs is qwen2's); head dim 128. RoPE supports
NoPE global layers by the caller simply not calling it; no
skip flag exists. Final logit scale 0.1961161345243454,
softcap 20.0, scale-then-softcap order per model.rs.

CPU proofs (synthetic fixtures, independent in-test Float64
references; 16/16 pass): rms_norm max_rel 5.66e-08 (plain
4.04e-08); against Rust's recorded known_vector [1,2,3,4]
max_abs 1.75e-05 inside Rust's own 1e-4 bar (no recorded RMS
fixture file exists; the Rust unit-test vectors are the
closest recorded fixture and were used). RoPE max err
5.94e-08. SwiGLU max_rel 1.05e-07. Residual exact 0.0.
Softmax max err 1.39e-08 (incl. a 1000-magnitude stability
row). Final logit max err 1.04e-06; the Rust recorded case
[0,20,-20,1000] at scale 0.5/cap 20 matched within 5.1e-08
(this case distinguishes scale-first from cap-first).
Embedding exact 0.0; Q8_0 refuses by name.

GPU proof by the lane (server card): rms_norm row max_abs
1.19e-07, softmax 2x128 max_abs 3.73e-09, GPU_PROOF PASS.

## Lane A5: tokenizer (src/tokenizer/)

Mirror of crates/tokenizer semantics. The file declares
tokenizer.ggml.model "gpt2", pre "llama4"; 202,048 tokens,
439,802 merges, bos 200000, eos 200001, pad 200018, no
unknown token id. Both families implemented: byte-level BPE
(GPT-2 byte alphabet, GPT-4o pre-tokenizer pattern as
llama.cpp assigns to pre "llama4", merges lowest rank first,
specials matched literally longest-first, no BOS/EOS added)
and SentencePiece BPE (space escape, score-ordered merges).
Named refusals across the module family (missing vocab
metadata, invalid token id, invalid merge, empty vocab,
duplicates, length mismatches, unsupported model,
unencodable symbol, malformed byte token, invalid UTF-8).

Parity: the recorded reference (crates/tokenizer/
conformance/glimmer.tokref, generated 2026-10-29 from the
production llama-server /tokenize) encodes 16/16 prompts
token-identical with 16/16 decode round-trips exact, run by
the lane on the server against the real file, and RE-RUN by
the independent verification pass against the integrated
tree after the seam swap below: 16/16 + 16/16, exit 0.
VM: test_tokenizer 22/22, test_tokenizer_gguf 5/5, tiny
fixture parity 8/8.

Integration seam swap (coordinator): A5's temporary
self-contained GGUF metadata reader (435 lines) was replaced
by a mapping over Lane A1's loader (parse_gguf + metadata
getters -> TokenizerData), the interface
load_tokenizer_data(path) preserved. Two seam tests that
asserted the temporary reader's private refusal names now
assert the loader's canonical gguf_bad_magic /
gguf_truncated. Post-swap suite: seam 5/5, full suite PASS,
and the server parity re-run above covers the swapped seam
against the real vocabulary.

## Integration and verification record

Integration order as executed: A1 (7badbfb + contract fix
3e653c5), A3 (ba73b79), A4 (02d1bf1), suite wiring (ccd5fbc),
A2 (7f57f76 + wiring ae604d5), A5 (a6a7348 + 24a65e0 +
6826f99), seam swap (0388f14), wave close (version 1.2,
DESIGN.md status, this note). The branch was pushed after
every integration step via the replay push; every commit
SHA-verified MATCH; remote main verified at b416987 before
each push and never moved.

Server sessions: lanes A2, A3, A4 each ran their own
cleared-card windows under the phase lock with tenant stop/
restore and exact VRAM bookends (drained 33,271,808 B,
restored 8,932,147,200 B). A1 and A5 used CPU/file IO only.
The final verification window (integrated tree, Q5_K/F32/
F16 proofs): tenants restored healthy (cortex-embed.service
active, both containers healthy), port 8080 clear
throughout, VRAM restored measured at exactly 8,932,147,200
B. One honest gap: that window's in-tenant-stopped drained
VRAM figure was not directly measured (the verifier's smi
tooling was not on the non-interactive PATH and the sysfs
source was found only afterward; a second window was not
opened for one number, per the one-window discipline).
Indirect evidence: the sanctioned tenants were confirmed
stopped before the kernel runs and the restored figure
matches the expected bookend to the byte. Glimmer serving
stayed down throughout the wave; no server reboot.

## What Wave B still needs

In DESIGN.md build order: step 3 attention (scores, causal
masking, the sliding-window 2048 / global NoPE split,
flash-style staging), step 4 one full layer then all layers
(the ops and GEMVs from this wave are the parts), step 5 the
generation loop over a KV cache with the golden first-8
token check [13796, 26, 1573, 373, 290, 2167, 26899, 1212],
step 6 the sampler (greedy first), step 7 the serving seam
decision. Open tuning items: K1 sits at 74.4% of the wgpu
median with the correctness-bar question recorded above;
Q6_K/Q5_K/F32 kernels are correct straight ports, untuned.
Config follow-up: ops currently take eps/rope base/logit
scale/cap as parameters with GLIMMER_* comptime defaults
exposed; folding them into EngineConfig proper belongs to
the layer lane that first wires them together.
