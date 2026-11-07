# Record rewrite, 2026-11-07 (critic panel Category I)

Lane: lane/record-rewrite, base 34358e33f6b0fe8f0ef5977675ed10e04228501a
(remote main verified equal before branching). Docs only: no code
behavior changed anywhere; the single code-file edit is doc comments
in crates/wgpu-compute/src/compute.rs.

Source of truth for every change: the panel merge report
(CRITIC-ENGINE-MERGE.md, Category I and defaults table), checked
against the tree at the base (git log 333f354..34358e3, file reads,
greps, and counts run at the base). Every claim below names its
evidence.

## Item 1 [HIGH]: STATUS.md said Wave 3 was "still ahead"

- STATUS.md header: "## As of 2026-10-31" -> "## As of 2026-11-07".
- STATUS.md In flight, before: "Wave 3 (operability: the engine diag
  module, the gguf diag emission fix, the speculator guard test, the
  sampler named refusals, the worker catch_unwind) and the
  environment-blocked reruns from the panel ... are still ahead, then
  a cold panel re-run ... Until that re-run lands, the honest verdict
  on this tree is the panel's: DO NOT SHIP."
  After: the 2026-11-07 panel's remaining categories (B, C, J, K) and
  Category G riders, the owed card confirmation run, and this record
  pass itself.
  Evidence: TODO.md's Wave 3 list is checked, with landing commits
  b092782, db38d99, 590b6ce, 94babd2, a928eb2 (git log, 2026-10-31);
  the cold re-run happened as the 2026-11-07 panel (merge report).
- STATUS.md Proven state: a Wave 3 bullet added (same evidence), and
  the Wave 1/2 bullet's "manifest descriptions corrected to the real
  six socket ops and K1 through K14" became "...and the K1 through
  K14 kernel set of that date", because the correction was true on
  2026-10-31 and the set has since grown.
- STATUS.md Blocked, before: the T-9.6/T-9.7 GPU speculative port
  "waits on the fix wave finishing" and wide-tile MMQ "wait[s] behind
  the GPU speculative work". Both are stale: T-9.6/T-9.7 landed
  (merges 3e785cf, fd78479, 2026-11-03; 72a4b9d, 2026-11-04) and wide
  MMQ landed and was convicted (dff467d). After: the CPU serving
  backend decision (defaults table row 23, NEEDS DECISION), the
  session.rs split's GPU verification window, production serving on
  the owner's word, DiffusionGemma.
- STATUS.md Next moves, before: land Wave 3, GPU speculative port,
  wide MMQ. After: (1) the card confirmation at the shipped defaults,
  (2) the panel's remaining fix categories B, C, J, K, (3) wire the
  prefix-KV store into a production caller, then reconsider its
  default.
- README.md Status paragraph, before: "...fix wave Waves 1 and 2
  have landed ..., and Wave 3 plus a cold panel re-run are still
  ahead." After: all three waves landed; the 2026-11-07 panel graded
  the grown tree DO NOT SHIP again (83 findings, no criticals, the
  numeric core verified clean); its categories A, D, E, F, H fixes
  have landed (commits 7a71727, ca73795, 07bce8a, 0578b44, 07fe5aa,
  ce555fd, a47dcf8, from git log 333f354..34358e3); the remaining
  categories and the card confirmation are still ahead. The same
  paragraph's "the speculative loop is proven on CPU" became "the
  GPU-resident forward, the GPU speculative loop, and the prefix-KV
  store are proven on the card, not yet wired into the serving
  engine" (merges 3e785cf and fd78479; defaults table row 23).

## Item 2 [MEDIUM]: a week of shipped work absent from STATUS and CHANGELOG

Evidence: git log 333f354..34358e3 (20 commits) for the post-freeze
work, and the dated git log --since=2026-10-30 (271 commits) for
the week before it. CHANGELOG's newest entry was 2026-10-30;
entries added, newest first, in the file's format:

- 2026-11-07: second cold panel and its first fix categories.
- 2026-11-07: coopmat v2 (K24) and split-K ship on by default.
- 2026-11-07: build-all wave (itak-kv store 9d6d434 and seam
  e94f30c; F16 KV 4a7343e; split-K 40e5c70; K24 7df5d58; n-gram
  debdc73 and d95234b; calibration profiles f9612cc).
- 2026-11-07: default flips (MMV ceiling b342cdf; staged flash
  82017ad; K21F switch 5f3ef6d, mask fix f7bc24d, flip d9825eb).
- 2026-11-06: MMV band routing, staged flash, K23 repack (a1de267)
  and coopmat v1 (e143b61, bb9af9f), diag capture (1325b9d).
- 2026-11-05: overnight verification wave (T-9.3 closed, 9ce57f3;
  Q2_K/Q3_K bit-exact, b3a1c6c), the MMV family (40c11c3, 09a03c9,
  433077a, 5eeab9b), prefill host gap (2682e02), GGUF mmap (b24b5d9),
  K22 prototype (9d07b13).
- 2026-11-04: T-9.7 closed with depth 1 the winner (72a4b9d); flash
  chunk plans (merge d94b852); quantize-once (merge 504d46f); K18
  landed and verdicted correct-but-slower (merge ef6ca5d, bda7ff5);
  wide MMQ verdict (2797923, dff467d).
- 2026-11-03: GPU speculative loop (merge 3e785cf), GPU-resident
  DFlash draft (merge fd78479), sweep machinery (merge 2ed7389),
  wide-tile MMQ (merge 8b77bf6).
- 2026-10-31: panel fix waves 1 to 3 (merge 19f2cba; 258911d
  through 7eacc30; Wave 3 commits above) and the T-9.17 live proof
  record (e1d791c).

STATUS's shipped list gained matching proven-state bullets: the
grown kernel set, the wave20 session's separate arms (K24 wall -36
percent at 1,022 tokens; split-K fixed 4 -18/-14 percent; prefix
restore about 0.8 s against 29.4 s cold; bench/
wave20-session-2026-11-07.md), the 2026-11-05 overnight wave, the
build waves, the three flips, the second panel and its landed
categories, and the shipped-defaults inventory (resolvers in
compute.rs, verified below).

## Item 3 [MEDIUM]: kernel set described as "K1 through K14"

Evidence: ls crates/wgpu-compute/src/kernels/ at the base lists
k1..k24 (k21 plus its finish variant; k23 in MMQ and coopmat forms;
k24 coopmat) plus splitk_reduce.rs and the coopmat/ directory.

- README.md grid: "kernels K1 through K14, including the
  device-resident forward session" -> "kernels K1 through K24 plus
  the split-K reducer and the cooperative-matrix variants,
  including the device-resident forward session".
- docs/MAP.md grid: the same sentence corrected the same way.
- crates/wgpu-compute/Cargo.toml description: "the full kernel set
  K1 through K14 (Q4_K, Q6_K and Q5_K dequant GEMVs, ..., and the
  packed Q4_K and Q6_K MMQ kernels)" -> "the full kernel set K1
  through K24 plus the split-K reducer and the cooperative-matrix
  variants (..., GQA attention with staged flash, ..., the packed
  Q4_K and Q6_K MMQ kernels and their split-K forms, the
  register-streaming MMV products, and the K23 and K24
  cooperative-matrix MMQ kernels)".

## Item 4 [MEDIUM]: RULES.md asserted gates and fuzz targets that do not exist

Evidence: no .github directory, no gate script, and no fuzz target
anywhere in the tree at the base (find for "*fuzz*" returns nothing;
the only gitleaks mention in the repo was RULES.md itself and TODO
T-1.5). TODO.md:39 (T-1.5, commit gates incl. gitleaks) and
TODO.md:41 (T-1.7, language gates and CI) are both still open
(checkboxes unchecked at the base).

- docs/RULES.md, before: "Secret scanning (gitleaks or betterleaks)
  runs in the gates." After: the target-state wording naming TODO
  T-1.5, stating no gate script or CI exists in the tree today, and
  that until it lands the no-secrets rule is enforced by review.
- docs/RULES.md, before: "Untrusted input gets a bounded parser and
  a fuzz target, not trust." After: the bounded parsers exist and
  are tested (present tense, per the panel's own evidence); a fuzz
  target for each is target state naming TODO T-1.7's gate work,
  with the plain statement that no fuzz target exists yet.

## Item 5 [MEDIUM]: PLAN.md Phase 9 milestone without the harness qualifier

Evidence: DECISIONS.md 2026-10-30 entry (anchors harness-relative;
the sweep harness is the designated comparison shape) and
bench/gpu-proof-2026-10-30.md:154-159 (44.31 from llama-bench tg128
after a 512-token prefill; 88.6 from the 2026-10-28 model-suite run;
the sweep reads llama-server /completion: plain 41.64, depth 4 =
103.01, depth 6 = 107.39).

- PLAN.md S13, before: "...at or above the recorded llama.cpp
  baselines on the 7900 XTX (44.31 tok/s plain Vulkan, 88.6
  effective with DFlash)." After: the milestone names the sweep
  harness as the comparison shape with its numbers (41.64 plain;
  107.39 at depth 6; depth 4: 103.01) and keeps the older anchors
  standing with their harnesses named (44.31 llama-bench tg128
  after pp512; 88.6 the 2026-10-28 model-suite run, depth 4).
- BACKEND.md Section 1 table: the two anchor rows gained their
  harness names, a sweep-harness row was added with the 2026-10-30
  numbers and its proof note, and a lead-in sentence states the
  anchors are harness-relative per DECISIONS.md (2026-10-30).

## Item 6 [MEDIUM]: TODO.md session.rs split size

Evidence: wc -l crates/wgpu-compute/src/session.rs at the base =
6,564 (at the panel freeze it was 6,460, matching the panel);
git show --stat e94f30c shows the prefix-KV session wiring added
778 lines to session.rs in one commit; split-K (40e5c70), coopmat
v2 (7df5d58), and F16 KV (4a7343e) added 90, 53, and 33 more.

- TODO.md, before: "Split crates/wgpu-compute/src/session.rs
  (2,644 lines)". After: "(6,564 lines, measured 2026-11-07; the
  count recorded here when this item was scheduled was 2,644, and
  the file has since grown by about 3,920 lines while the split
  waited, the largest single growth being the prefix-KV session
  wiring at 778 lines in e94f30c, with the split-K, coopmat v2,
  and F16 KV session seams of the 2026-11-07 wave adding more)".

## Item 7 [NIT]: crate counts off by one

Evidence: the workspace Cargo.toml members list has 13 entries,
including crates/itak-kv (added 2026-11-07, commit 9d6d434);
itak-engine is the binary and the other 12 are library crates.

- README.md, before: "Twelve crates: the GGUF loader, ..."
  After: "Thirteen crates", with "the persistent prefix-KV store"
  added to the enumeration; a grid row for itak-kv was added.
- docs/RUNBOOK.md, before: "# Rust workspace: the engine binary
  plus the eleven library crates". After: "the twelve library
  crates".

## Item 8 [NIT]: compute.rs module docs described an older backend

Evidence: the trait impl in the same file at the base. attention
(compute.rs fn attention) validates and runs K5 rotate and append
into the backend's own GpuKvState resident buffers, then K6 scores
and reduce; it refuses only contract violations, by name. proj_impl
dispatches Q4_K/Q6_K/Q5_K to K1/K2/K3; the wider ladder lives in
session.rs. Resolvers at the base: DEFAULT_MMV_MAX_ROWS = 16
(compute.rs:179), resolve_flash_staged unset = true,
DEFAULT_K21F_HEAD = true, DEFAULT_COOPMAT = true,
DEFAULT_COOPMAT_V2 = true, DEFAULT_MMQ_SPLITK = Fixed(4),
resolve_kv_f16 unset = false; every other default claim in the
file's doc comments was checked against its resolver and is
current (K18 and chunked GEMV "shipped default (1) routes
nothing" is accurate: both constants are 1).

- Module doc, before: "The one step that still refuses, by name,
  is `attention`: the K5 ... and K6 ... pipelines are built, but
  their composition over this backend ... is not wired yet."
  After: attention described end to end as wired (K5 rotate and
  append into the backend's resident KV buffers, K6 scores and
  reduce, the host mirror holding the same rows), plus a paragraph
  naming the resident session half of the crate and the shipped
  defaults at this writing (MMV ceiling 16, staged flash on, K21F
  head route on, coopmat rung on with K24 selected, split-K fixed
  factor 4, F16 KV storage off), each with its env override and
  fail-closed fallback.
- MmvBands doc, before: "so the shipped ceiling default (1) keeps
  every product out of the ladder whatever the table says, and
  opening the ceiling admits exactly the measured-win cells".
  After: "so the shipped ceiling default (16) admits exactly the
  measured-win cells instead of the whole band".

## Same-class finds beyond the panel's list

Fixed in this lane (same defect class, verified against the tree):

- docs/MAP.md had no itak-kv node at all (grid, links, diagram),
  although the crate is a workspace member and wgpu-compute depends
  on it (crates/wgpu-compute/Cargo.toml: itak-kv path dependency).
  The MAP file's own header law says every module appears in the
  grid, the links, and the diagram in the same commit; all three
  gained the node, with the wgpu-compute -> itak-kv and itak-kv ->
  attention links traced both ways.
- README.md "Proof so far": "llama.cpp decodes Glimmer Q4_K_M at
  44.31 tok/s (Vulkan) on this card, and the DFlash depth sweep
  found its operating point at depth 6, 107.39 tok/s" put a
  llama-bench number and a sweep-harness number in one sentence
  with no harness named, the item 5 defect in miniature. The
  harnesses are now named per number, with the DECISIONS
  2026-10-30 designation stated.

Recorded as follow-ups, not fixed (outside this lane's scope):

- The spec runners' depth argument still defaults to 6
  (crates/itak-spec/examples/spec_generate_gpu.rs:103) against the
  library's measured default of 1: defaults table row 22, a code
  change, owned by the Category G follow-ups.
- Category J's skip-count gate and the remaining categories B, C,
  K are code and gate work, tracked in STATUS's In flight list.

## Verification pass (after the edits)

- grep "K1 through K14" / "K1-K14" over *.md and *.toml outside
  bench/: one hit remains, STATUS.md's Wave 2 bullet, now
  explicitly "the K1 through K14 kernel set of that date" (a true
  historical claim about the 2026-10-31 correction, kept
  deliberately).
- grep "still ahead": no Wave 3 hit remains in STATUS.md,
  README.md, or docs/.
- grep "Twelve crates" / "eleven library crates": no hits outside
  bench/.
- grep "shipped ceiling default (1)" in crates/: no hits.
- git diff added lines contain no em dashes (repo writing law).

## Gates (lane tree, as observed)

- cargo test -p wgpu-compute: 324 passed, 0 failed, 1 ignored,
  32 suites. The only code-file touch is doc comments, so no
  behavioral delta is attributable to this lane.
- cargo check --workspace: clean.
- cargo fmt --check: clean (exit 0, no diff).
- go test ./... from platform/: all 9 packages ok, 123 tests
  (counted from --- PASS lines), matching the pre-lane count.

No server jobs were run; nothing was merged or pushed.
