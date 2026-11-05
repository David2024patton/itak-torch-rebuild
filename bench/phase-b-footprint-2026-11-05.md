# Phase B host footprint audit, 2026-11-05

Why the speculative harness's phase B setup peaks at about 34.6 GB of
host RAM and dies to the OOM killer on a loaded box, what the queued
`--kv-capacity=2048` workaround would actually do (nothing, on the
host side), and what would shrink the peak. Code reading only, at
main `3b2afbcc`; no server runs were made for this note.

## Verdict

Phase B's host peak is two full copies of the 16.76 GB target model
plus the draft file's image, all anonymous memory, all resident at
the same moment:

1. the target GGUF file image, read whole into a `Vec<u8>` and held
   for the entire process (15.61 GiB), and
2. a second, per-tensor copy of every target tensor's bytes, built
   by `load_weights` (15.59 GiB), alive for the whole of verify
   session construction,
3. plus the draft GGUF file image, also read whole and held for the
   entire process (about 1.5 GiB).

Sum: about 32.7 GiB = 35.1 GB, against the kernel log's kill figure
of 34,678,624 KiB anon-rss = 33.07 GiB = 35.51 GB. The remaining
roughly 0.4 GB is bounded small residue (Section 3); there is no
missing giant. Nothing on the path is a KV cache, a workspace, or a
quantization staging buffer: the target session's KV is device-only
with no host mirror, and the draft's KV is a device buffer too,
which is why `--kv-capacity` cannot move this number (Section 5).

## 1. The phase B setup path, in order

All line references are to `crates/itak-spec/examples/spec_generate_gpu.rs`
unless another file is named. `run()`:

1. Opens the draft file and the target file, in that order, before
   any phase split (lines 228 and 239). Both `GgufFile`s live in
   `run()`'s scope until the process exits, in every mode, split or
   combined. `GgufFile::open` is `std::fs::read` into an owned
   `Vec<u8>` (`crates/gguf/src/file.rs:100-104`); the struct doc
   says it plainly: "The whole file is held in memory"
   (`crates/gguf/src/file.rs:83-93`).
2. Builds the tokenizer from the target file's metadata (token
   strings cloned into `TokenizerData`, line 257 and
   `data_from_info`): megabytes, counted in the residue.
3. Phase A (combined mode only): builds a resident session through
   the same `resident_session` helper, runs, and drops it at the
   end of its block. Its weight copies are gone before phase B
   starts; only the two file images carry over.
4. Phase B, GPU draft arm (the default, and the arm that OOM'd):
   - `DflashRawWeights::load` (line 357) copies every draft tensor
     out of the draft image into `Arc<[u8]>` views: about 1.4 GiB,
     transient.
   - `token_embd.weight` and `output.weight` are fetched fresh from
     the target image (lines 358-362): another 1.56 to 1.89 GiB of
     copies, transient (arithmetic in Section 4).
   - `DflashSession::with_diag` (lines 363-374) uploads everything
     to device buffers and retains no host weight bytes; its KV
     and scratch are `wgpu::Buffer`s. The two transient copy sets
     drop when the match arm's block ends.
   - `resident_session_on` (called at line 377; the helper spans
     lines 190-205) prints
     `target coverage: expected=731 ...` (line 197), then:
   - `itak_loader::load_weights` (line 203) builds the second full
     copy of the target weights (Section 2, row 2), then
   - `ResidentSession::new` (line 204;
     `crates/wgpu-compute/src/session.rs:3278`) uploads tensor by
     tensor to the device. The host `Weights` stay alive for the
     whole construction and drop only when `resident_session_on`
     returns.

Every kill landed "right after target coverage: 731", which is
exactly step 4's copy build and upload window: the only place on
the path where the resident total crosses 30 GiB.

## 2. The ranked allocation table

Sizes in GiB (2^30) with GB (10^9) alongside. "Resident at kill"
means alive at the step-4 window where the kills landed.

| # | Allocation | Size | Where | Resident at kill |
|---|---|---|---|---|
| 1 | Target GGUF file image (`GgufFile.bytes`) | 15.61 GiB / 16.76 GB | `crates/gguf/src/file.rs:91,102` | yes, whole process |
| 2 | Target `Weights`: one `Arc` copy per tensor, copied out of the image | 15.59 GiB / 16.74 GB | `crates/itak-loader/src/adapter.rs:149-163,194`; `crates/muse-glimmer/src/weights.rs:76` | yes, during session construction |
| 3 | Draft GGUF file image | about 1.5 GiB / 1.6 GB | `crates/gguf/src/file.rs:102` via example line 228 | yes, whole process |
| 4 | Draft raw weight copies (`DflashRawWeights`) | about 1.4 GiB | `crates/wgpu-compute/src/dflash_session.rs:254-360` | no, dropped before step 4 |
| 5 | Borrowed-table copies for the draft session (target embedding + head, fetched fresh) | 1.56-1.89 GiB | example lines 339-344 | no, dropped before step 4 |
| 6 | Everything else (Section 3) | at most about 0.4 GiB | various | partly |

Rows 1+2+3 = 32.70 GiB = 35.11 GB at the kill window.

Arithmetic for row 2: the file is 16,756,683,904 B = 15.606 GiB.
Tensor data is the file minus its header (metadata including the
tokenizer arrays, plus the 731-entry tensor directory, on the
order of 17 MB), and the muse-glimmer crate's own doc pins the
weight set at "the real model is 15.59 GiB"
(`crates/muse-glimmer/src/weights.rs:76`). The adapter copies
every tensor: `tensor_bytes` returns a borrowed slice
(`crates/gguf/src/file.rs:325`), and the adapter wraps it in
`Arc::from(bytes)` (adapter.rs:194), a memcpy per tensor, per its
own doc comment: "Tensor bytes are copied out of the file image
into an Arc" (adapter.rs:149).

Arithmetic for rows 3-4: the real draft config (measured values,
`crates/dflash/src/testkit.rs:16-32`) is 5 layers, hidden 6656,
FFN 19968, 32 Q heads over 8 KV heads at head dim 128 (so
q width 4096, k/v width 1024), vocab 202,048. Per layer:
q 6656x4096 + k 6656x1024 + v 6656x1024 + o 4096x6656 +
gate/up/down 3 x 6656x19968 = 466,880,768 elements; times 5 =
2,334,403,840. The `fc` conditioning projection is
(5 x 6656) x 6656 = 221,511,680. Total about 2.556 x 10^9
elements, which at the Q4_K_M blend (0.5625 B/elem for Q4_K,
0.8203 for Q6_K) is 1.44 to 1.61 GB of tensor bytes, inside a file
the testkit calls "a 1.5 GiB fixture" (testkit.rs:15).

## 3. What is NOT in the number

- **Target KV cache: 0 bytes of host RAM.** The resident session's
  `DeviceKv` is documented as the device half "without the host
  mirror" (`crates/wgpu-compute/src/kernels/k6_attention.rs:367`),
  allocates its buffers lazily on first append (same file,
  `DeviceKv::new` starts every layer as `None`), and is built
  empty by `ResidentSession::new` (`session.rs:3361`). The
  1.83 GB full-context hybrid KV figure is VRAM, reached only as
  prefill appends, and never has a host-side counterpart on this
  path. (The older trait path's `GpuKvState` does mirror on host;
  the spec harness does not use it.)
- **Workspaces and activations** in both sessions are device
  buffers via `GpuContext::scratch_buffer`
  (`crates/wgpu-compute/src/device.rs:192-201`, STORAGE | COPY_DST
  | COPY_SRC, no host mapping).
- **Upload staging is transient, not cumulative.** K1/K3 upload
  through `queue.write_buffer` straight from the host slice;
  K2 (Q6_K) first builds one padded host copy per tensor
  (`repack_q6k`, `crates/wgpu-compute/src/kernels/k2_q6k_gemv.rs:692`),
  dropped after that tensor's write. The largest single Q6_K
  tensor (an FFN down, 132.9M elements) is about 110 MB padded.
- **The residue** (the roughly 0.4 GiB between the table and the
  kill figure): tokenizer tables cloned from metadata, two leaked
  `WgpuCompute` backends' pipeline-compilation host memory, wgpu
  `write_buffer` staging chunks, the K2 repack transient if the
  fault landed inside `ResidentSession::new`, and malloc arena
  retention from rows 4-5's dropped copies. Each is tens of MB at
  most; the OOM killer also fires on a failed allocation, not at
  an exact total, so the process need not have reached a precise
  sum when it died.

Consistency checks. Phase A alone peaks at rows 1+2 = 31.20 GiB =
33.50 GB and frees row 2 right after construction: it fits the
same nights phase B died, and the margin between phase A
surviving and phase B dying is, to first order, row 3, the draft
file image that phase B's process also holds. On 2026-11-04,
with about 54 GB available, the same 35.5 GB peak completed
repeatedly. Both observations match the table.

## 4. Duplicate-copy waste

The same bytes resident more than once, worst first:

- **D1. The target model, twice, for the whole construction:**
  the file image (row 1) and the `Weights` copy (row 2) are the
  same 15.6 GiB in two anonymous allocations. The image must
  outlive the copy because the harness holds `target_file` in
  `run()` scope and the loader borrows it; nothing frees either
  until the session is built.
- **D2. The draft model, twice, transiently:** the draft image
  (row 3) plus the `DflashRawWeights` copies (row 4) for the
  duration of draft session construction.
- **D3. The borrowed tables, copied a second time:** the draft
  session's embedding and head are fetched fresh from the target
  image (example lines 358-362) instead of sharing the `Arc`s
  the target `Weights` will hold minutes later. Head (`output.weight`,
  Q5_K per `crates/muse-glimmer/src/weights.rs:155`):
  202,048 x 6656 = 1,344,831,488 elements x 176/256 B =
  924,572,274 B = 0.861 GiB. Embedding: same element count,
  0.704 GiB stored as Q4_K or 1.027 GiB as Q6_K.
- **D4. CPU draft arm only (not the measured path):** with
  `--draft=cpu`, the speculator owns dequantized f32 copies for
  the whole of phase B (example lines 340-355, served by
  `DflashTensorSource`'s `load_f32`, adapter.rs:359-377): draft
  weights 2.556 x 10^9 elements x 4 B = 9.52 GiB, plus f32
  embedding and head at 5.01 GiB each. That arm's construction
  peak computes to about 52 GiB, worse than the GPU arm's
  35.5 GB, and also resident rather than transient.

## 5. The `--kv-capacity` verdict

The flag is parsed at example lines 115-132 and flows to exactly
one place: `DflashSessionConfig { kv_capacity }` for
`DflashSession::with_diag` (line 369). The config default is
8192 (`crates/wgpu-compute/src/dflash_session.rs:389`, pinned by
the test at line 1721). Inside the session it sizes only device
buffers (dflash_session.rs:915-916):

- per layer, K buffer = capacity x k_width x 4 B and V likewise:
  at 8192, 8192 x 1024 x 4 = 33,554,432 B each; K+V across
  5 layers = 335,544,320 B = 320 MiB of VRAM. At 2048:
  80 MiB of VRAM.
- the only host structure it bounds is the committed-position
  bookkeeping (`held: BTreeSet<usize>`, at most `capacity`
  entries, under a megabyte).

Expected phase B host peak at `--kv-capacity=2048`: **32.70 GiB,
unchanged; the delta is 0 bytes of host RAM** (and minus 240 MiB
of VRAM). The queued workaround cannot unblock the OOM kills,
because the footprint it was aimed at is not a KV footprint.
Split-phase mode is the same story: it changes nothing about the
peak, since both file images open at the top of `run()` before
any mode dispatch (lines 228, 239) and phase A's copies already
drop in combined mode. The only no-code lever that works is
scheduling: run phase B when the box has at least about 40 GB
free. That is a window, not a fix.

## 6. Ranked reductions

Config-only first, then small code changes, then structural.
Savings are against the 32.70 GiB kill-window residency.

### Config-only

1. **Nothing reduces the peak.** `--kv-capacity=2048` saves
   0 B host (Section 5); split-phase mode saves 0 B (both images
   open before the split). Scheduling phase B into a window
   with about 40 GB or more free RAM works tonight but leaves
   the footprint exactly as measured. Risk: none; it is an
   operational posture, not a repair.

### Small code changes

2. **Memory-map the GGUF image** (gguf crate). Replace
   `std::fs::read` with an mmap backing for `GgufFile.bytes`.
   Saving: rows 1 and 3 leave the anonymous tally, 17.11 GiB;
   the kill-window anon peak falls to about 15.6 GiB (row 2
   plus residue), inside even the 32-38 GB nights. Risk: low
   to medium. The crate's own doc already names this as the
   planned step with an unchanged API: "memory-mapped backing
   is the planned next step for the loader module (PLAN T-2.1)
   and does not change this API" (file.rs:83-88). Mapped pages
   still count in RSS, but they are file-backed and evictable,
   and the kernel log's kill figure is anon-rss.
3. **Zero-copy tensor views** (loader/muse-glimmer seam).
   Serve `TensorView`s as ranges over one shared image buffer
   instead of `Arc::from` copying every tensor. Saving:
   row 2, 15.59 GiB (and row 4's 1.4 GiB on the draft load);
   kill-window peak falls to about 17.1 GiB with the plain
   `fs::read` image kept. Risk: medium. `TensorView.data` is
   `Arc<[u8]>` "owns its bytes by contract"
   (weights.rs:82-96), so the type becomes a view (shared
   buffer plus offset/length); every consumer already reads it
   as `&[u8]`, so the blast radius is the type, not the math.
   Items 2 and 3 attack the same seam from opposite sides;
   either one alone is sufficient, both together leave only
   one mapped, shared image.
4. **Harness: open the draft file inside phase B** (or drop it
   before the target load). Saving: row 3, about 1.5 GiB.
   Risk: trivial. Not sufficient alone (32.70 to 31.2 GiB is
   still at the observed kill line); worth doing only as a
   companion to 2 or 3.

### Structural

5. **Fuse load and upload.** Stream the resident session's
   construction: fetch one tensor, upload it, drop its host
   copy, so the full `Weights` set is never materialized.
   Saving: up to 15.59 GiB minus the largest single tensor
   (the 0.861 GiB head), about 14.7 GiB, and it stacks with
   item 2. Risk: medium-high. `Weights::load` plus the
   coverage gate is a shared seam (CPU path, engine, tests);
   the gate must still fail by name on a missing tensor
   before any upload starts, which argues for a two-pass form
   (inventory first, then streamed upload).
6. **Share the borrowed tables with the draft session.** Hand
   `DflashSession` the target `Weights`' own embedding/head
   `Arc`s instead of re-fetching them from the image. Saving:
   rows 5's 1.56-1.89 GiB transient on the GPU arm; on the CPU
   arm, serving the draft's tables from the shared raw views
   instead of separate f32 dequantizations removes about
   10 GiB of residency (D4). Risk: low on the GPU arm (the
   views are the same bytes by construction); medium on the
   CPU arm, whose model consumes f32 `TensorData` by contract.

The smallest single change that unblocks the spec verdicts on a
loaded box is item 2 (mmap), with item 3 as its equal-cost
alternative; either takes the phase B host peak from about
35.5 GB to about 17 to 19 GB.
