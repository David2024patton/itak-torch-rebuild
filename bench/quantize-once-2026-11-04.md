# Quantize-once (profile candidate C3): one K12 quantize of `h` per layer stage

Branch `lane/quantize-once` over main 72a4b9d. All counts below
are counted from the plan builder in
`crates/wgpu-compute/src/session.rs` (the same code the host
unit tests now pin); nothing here is measured on a GPU yet.
The server run that executes the GPU legs and the prefill pair
is specified at the end and is still owed.

## The change

In an MMQ-served chunk plan, every projection over a layer's
`h` prepared its own K12 plain quantize of that buffer. The
plan builder now prepares ONE shared quantize per layer
stage into a dedicated scratch pair (`q8h_qs` / `q8h_ds`,
kept apart from the general pair so no single-consumer
quantize can clobber it mid-stage), and each member product
binds that pair and carries no quantize of its own
(`ProjCall`'s MMQ variants now hold `Option<K12QuantCall>`;
the shared form is the `None` case). The layer plan encodes
the shared quantize ahead of every member: the attention
stage's right after the attention norm, the FFN stage's
right after the post-attention chain.

Byte-identity is by construction: the quantize is a pure
function of (input buffer, batch, width), every member of a
stage shares all three, and the product kernels bind the
same buffers in both forms, so a member product reads
exactly the bytes its own quantize dispatch would have
written. Membership is guarded: a projection whose input
width differs from the share's prepared width never joins
(it keeps its own quantize over the general pair), and a
member whose product prepare refuses leaves the share
untouched and falls back to the GEMM form, the standing
fallback ladder unchanged. Every refusal keeps its name.

## A correction to the profile's premise, counted from the code

The profile (section 1.4 and candidate C3) derived 78
redundant quantizes per chunk by counting three consumers
of "the identical `h`": the QKV batch, the separate V, and
the FFN pair. The encode order says otherwise. The batch
and V read the `h` the attention norm produced; the
post-attention chain (L8) then OVERWRITES `h` with the FFN
norm's output, and the pair reads that second tensor.
Batch and pair quantize different contents that happen to
share a scratch buffer, so no single quantize can serve
both, and sharing across the chain would change the math.

The true unit is the stage: one `h` content, several
readers. Two shares per layer, never one. The savings
below are counted on that basis.

## Dispatch arithmetic, counted from the plan builder

K12 legs per layer before the change (MMQ serving; every
h projection carried its own quantize):

| Layer class | Attention stage | FFN stage | h legs |
|---|---|---|---|
| Full batch, paired (26 of the real layers) | batch = 1 | pair = 1 | 2 |
| Partial batch, paired (26 of the real layers) | batch + V = 2 | pair = 1 | 3 |
| Unbatched, unpaired (no real layer; code path) | Q + K + V + gate = 4 | gate + up = 2 | 6 |

After the change each stage holds ONE shared quantize
regardless of member count, so the removed legs per layer
are: full batch 0, partial batch 1 (the V leg), unbatched
up to 4 (three in the attention stage, one in the FFN
stage). A stage with a single member is neutral by
construction: its one quantize moves from the product to
the layer, and the count does not move (pinned in the host
tests).

Per 64-row chunk on the real file (52 layers, 26 full and
26 partial, the profile's V split): K12 legs in total go
from 234 to 208, and total dispatches drop by the same 26,
from the flash era's derived 679 to 653 (both totals
derived under MMQ serving, not measured). The removed legs
also remove their re-reads of `h`: 26 legs x 1.70 MB
(64 rows x 6656 x 4 B) is about 44 MB per chunk. (The
profile's ~133 MB figure scales with its 78 count and
falls with it.)

At fixture scale the same arithmetic is pinned twice over:
host unit tests pin `layer_dispatch_count` per class
(fixture layers 9 and 8; full batch 13 before and after;
partial 15 to 14; the mixed class 18 to 16; unbatched 23
to 19), and the new mixed-fixture resident test pins the
assembled chunk: 93 dispatches with the shares, 105
before, 69 on a backend with no MMQ set (no quantize legs
exist there at all).

## Why the resident pin (51 per chunk) does not move

The resident fixture's asserted plans are 1-row and 4-row
plans. Below `GEMM_MIN_ROWS` (16) every projection takes
its GEMV form, which consumes f32 activations and quantizes
nothing: those plans contain no K12 legs before the change
and none after. The pin stands at 51, untouched, and the
mixed-fixture test carries the new pins where the MMQ
forms actually engage.

## Decode

Decode plans are 1-row plans: GEMV forms throughout, no
quantize legs, no duplication to remove. Quantize-once
does not apply to decode and decode is untouched; the
mixed test pins the decode delta at exactly 3 x 69
dispatches for three decode chunks, the count the mixed
layers record with no shares anywhere.

## Expected effect (expectation, not a measurement)

26 of 679 dispatches per chunk is 3.8 percent of the
dispatch count, and the legs themselves are small
(1.70 MB read, about 0.81 MB written per leg, single-digit
microseconds at the profile's measured stream rate), so
the traffic saving alone is worth well under a tenth of a
percent of the chunk's chain time. Any real gain has to
come from the 26 removed dependent boundaries; the
profile's decode-era pricing of a boundary (~12-15 us)
would bound that near 0.3-0.4 ms per chunk, about 0.2
percent of the ~194 ms chunk. Expect a small effect,
likely inside run-to-run noise at 1,024 tokens. The
change's certain value is structural: fewer dispatches,
44 MB per chunk of re-reads gone, and an accounting that
now proves the count instead of deriving it per run. The
profile's M6 discipline is the gate: dispatch count down
by the derived amount, outputs token-identical, marginal
recorded honestly even if it does not move.

## Server proof (ubuntu-server, GPU free, one heavy job)

Stage this branch's tree, verify the staged hash equals
the branch tip, then:

```sh
export PATH="$HOME/.cargo/bin:$PATH"
# 1. The mixed-fixture proof must EXECUTE (not skip) and
#    print the mmq serving line with dispatches 93:
cargo test -p wgpu-compute --test resident_session -- --nocapture
# 2. The fused and flash legs execute:
cargo test -p wgpu-compute --test fused_kernels -- --nocapture
# 3. The spec path stays green:
cargo test -p wgpu-compute -p itak-spec
```

Then the prefill pair on the real file, same discipline as
the earlier notes (medians of 3, same window, this branch
against merged main as the control):

```sh
cargo build --release -p itak-loader --example glimmer_generate
GEN=./target/release/examples/glimmer_generate
MODEL=/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
# The repeated-sentence prompts from bench/mmq-wide-bench.sh
# (about 254 and about 1,024 tokens), greedy, resident path:
$GEN "$MODEL" "<254-token prompt>" 32 --gpu
$GEN "$MODEL" "<1024-token prompt>" 32 --gpu
```

Acceptance:

1. The mixed chunk reads 93 dispatches under MMQ serving
   (SessionStats from the fixture test), and the real-file
   chunk's SessionStats dispatches drop by 26 against the
   control build at 64-row chunks.
2. Exactness: first-token probability 0.403462, the 8/8
   Stage 1 ids, continuations token-identical to the
   control at 254 and 1,024 tokens.
3. Gross and marginal at 1,024 tokens recorded against the
   flash era's readings (gross 134.7 tok/s, marginal
   157.1 tok/s) and against the same-window control,
   reported as measured, whichever way they land.

## VM gate record (this lane, before push)

- cargo build --release -p wgpu-compute: ok
- cargo test -p wgpu-compute: 122 passed, 0 failed (lib
  suite 64, including the 5 new dispatch-accounting pins,
  which execute on the host; the GPU legs of
  resident_session, including the new mixed-fixture test,
  compile and skip on adapter absence, the standing policy
  for this VM)
- cargo fmt --all -- --check: clean
- cargo clippy -p wgpu-compute --all-targets: no warnings
- cargo deny check: advisories ok, bans ok, licenses ok,
  sources ok
