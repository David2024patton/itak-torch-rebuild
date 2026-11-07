# N-gram GPU runner wiring, 2026-11-07

Lane `lane/ngram-gpu-runner`, based on main `0ecdc8ff` with
the n-gram lane cherry-picked on top (originals `4e86e5ad`
code and `76aa1265` note on `lane/ngram-spec`; landed here as
`78fa7d7` and `ebc5b98`). Question: can the GPU runner
`spec_generate_gpu` run the n-gram arm the CPU runner already
runs, so the resident stack gets a real-file n-gram
measurement? Answer: yes, it is a flag, exactly as the n-gram
note predicted; this lane wires it and pins the construction
path. No card run here (VM only); the protocol below is what
the card session now executes.

## What was wired

One file changed in the runner, `crates/itak-spec/examples/
spec_generate_gpu.rs`:

- New flag `--drafter=<dflash|ngram>`, spelled and parsed
  exactly as the CPU runner's: `SpecMode::parse` (the shared
  parser the n-gram lane added as `DraftBackend::parse`'s
  twin; no second parser exists), admitted values exactly
  `dflash` and `ngram`, default `dflash`. The GPU runner's
  pre-existing `--draft=<gpu|cpu>` (the DFlash device path)
  is a different flag and is untouched; `--drafter=...` does
  not match its prefix scan and vice versa.
- Refusal before any model load: `--drafter` with an unknown
  value, or with a value that parses but is not a runner
  draft source (`off`, `medusa`), prints `FAIL: --drafter
  must be dflash or ngram, got ...` and exits 2, before the
  target or draft file is touched. Proven on the built binary
  on this VM: `--drafter=bogus` and `--drafter=off` both exit
  2 with that line and no load attempt.
- N-gram mode never opens the draft file. Following the CPU
  runner's precedent exactly: the positional draft path is
  still required by the usage line (pass any placeholder),
  its contents are ignored, and the run says so out loud:
  `drafter: ngram prompt-lookup (draft file <path> not
  loaded; --draft and --kv-capacity configure the dflash arm
  only)`. Proven on this VM: with a nonexistent placeholder
  in the draft slot and a nonexistent target, the n-gram arm
  prints that line and fails at the TARGET open; the dflash
  arm in the same setup fails at the DRAFT open. No stricter
  rule was invented for the draft-path argument.
- `--draft` and `--kv-capacity` in n-gram mode: still parsed
  and validated by name (an unknown `--draft` value refuses
  as before), and not used; the line above states that they
  configure the dflash arm only, so nothing is silently
  ignored.
- Phase B construction: in n-gram mode the runner builds
  `NgramProposer::new(NgramConfig::default(), vocab)` and
  the session through `SpecSession::new_ngram` over
  `ResidentTarget::new(resident_session)` — the same
  resident session, backend, and leak discipline as the
  DFlash arm — with `SpecConfig { mode: Ngram, depth,
  eos_tokens, .. }`. No `DflashSpeculator`, no draft session,
  no draft weights in that arm. The prefill line distinguishes
  the arm the CPU runner's way: `spec prefill (resident,
  plain + proposer commit)` versus the DFlash arm's unchanged
  `spec prefill (resident, capturing)`.
- Everything downstream is the shared code: the same
  `generate`, the same `spec stats: rounds= proposed=
  accepted= acceptance_rate= ... tokens_per_round=` line,
  the same resident counters, the same effective / all-in
  rate lines, and the same IDENTITY, GOLDEN, SPEED, and
  T-9.6 verdict block, so a card run in n-gram mode is
  directly comparable to the DFlash arms line for line. The
  exit code still follows correctness only (identity and
  golden); the SPEED line prints either way against the same
  88.6 tok/s default bar.

## Tests

The GPU runner had no argument tests of its own, so per the
lane terms the parse-level coverage went beside
`SpecMode::parse`'s round-trip test instead of into a new
harness: `config::tests::runner_drafter_vocabulary_accepts_
only_dflash_and_ngram` pins the parse half of the runners'
guard (admits `dflash` / `ngram`; refuses `off`, `medusa`,
`gpu`, `cpu`, the empty string, and case variants).

The construction path is pinned by a new integration test,
`crates/itak-spec/tests/ngram_gpu_runner.rs`: it builds the
wgpu-compute resident suite's synthetic model (hidden 512,
vocab 1024, six layers), constructs a `ResidentSession`,
attaches a default-config proposer, and builds the session
through `new_ngram` over `ResidentTarget`, then runs
`begin` and an 8-token generation and asserts the stream and
the round accounting (`accepted <= proposed`, rounds >= 1).
No draft file, speculator, or draft config exists anywhere
in the test. It follows the suite's adapterless skip pattern
(`WgpuCompute::new` returning `NoSuitableAdapter` prints the
reason and returns); on this VM it took that path, verified
with `--nocapture`: `skipping n-gram GPU runner construction
test: no suitable adapter (enumerated: [])`. Its live-backend
assertions therefore remain card-owed, like every GPU test
authored on this VM.

## Gates (this lane)

- `cargo test -p itak-spec`: 69 lib tests passed (68 from the
  cherry-picked lane plus the new parse test), 1 integration
  test passed (adapterless skip path), 0 failed.
- `cargo test -p dflash`: 24 passed, 0 failed.
- `cargo test -p wgpu-compute`: 277 passed, 1 ignored,
  0 failed.
- `cargo clippy -p itak-spec --all-targets -- -D warnings`:
  clean (one `if_same_then_else` in the new test's fixture
  builder was caught by this gate and fixed by merging the
  K/V and gate/up branches, the resident fixture's own form).
- `cargo fmt --check`: clean.
- `cargo check --workspace`: clean.
- `cargo check -p itak-spec --example spec_generate_gpu`:
  clean; the refusal and not-loaded behavior was additionally
  exercised on the built debug binary (see above).

## The card protocol now runnable

Server, card free, release build, every run wrapped in
`bench/failure-bundle.sh` per the standing convention (the
SPEED line will trip its FAIL marker on the n-gram prose
arms; that is the target line, not a leg verdict — IDENTITY
and GOLDEN inside the log are the leg verdicts, as in the
baseline session's bundles). Paths: target
`/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf`,
DFlash draft
`/mnt/models/Muse-Glimmer-30B/dflash-Muse-Glimmer-30B-Q4_K_M.gguf`.

Build: `cargo build --release -p itak-spec --example
spec_generate_gpu --example spec_generate` (plus
`spec_generate` for the CPU arm).

1. N-gram GPU arms, default prompt, `both` mode, 64 tokens,
   at depths 1 and 6:
   `spec_generate_gpu <target> unused.gguf 64 <depth> "The
   capital of France is" both <stream-file> --drafter=ngram`
   Record: IDENTITY, GOLDEN, the spec stats line (rounds /
   proposed / accepted / acceptance_rate / tokens_per_round),
   spec rates (effective and all-in), spec resident counters,
   and phase walls. Compare arm to arm against the standing
   DFlash GPU numbers at the same depths (wave18 retest,
   diag timing: depth 1 effective 22.07 tok/s, depth 6
   effective 16.80 tok/s, both identity-passing).
2. N-gram GPU arm on a repetitive prompt (a repeated
   instruction block as the prompt text, same token count
   and depths): this is the arm the fixture record says the
   proposer exists for; acceptance and effective rate against
   arm 1's prose numbers is the contrast that matters.
   IDENTITY still must pass (the plain phase A stream is
   generated in the same run); GOLDEN prints only for the
   default prompt.
3. DFlash control arms at the same depths if the session
   wants same-day, same-card comparisons rather than the
   wave18 record: `--draft=gpu` (the runner default) with
   the real draft path and no `--drafter` flag.
4. CPU-driver real-file acceptance (the n-gram note's owed
   item 1, independent of the card):
   `spec_generate <target> <any-placeholder> 64 <depth> "The
   capital of France is" both <stream-file> --drafter=ngram`
   at depths 6 and 1. Minutes-to-hours class; it measures
   acceptance on the real tokenizer's stream, which neither
   fixture could.

## Honest expectation

From the n-gram lane's fixture record
(`bench/ngram-spec-2026-11-07.md`): on a recorded repetitive
stream the proposer accepted everything proposed (rate
1.0000, 4.23 stream tokens per round at depth 6); on prose
it proposed almost nothing and accepted nothing (1.008
tokens/round, the bonus-token floor — plain speed, never
worse). The fixture target sessions read lower (repetitive
0.45 at depth 1 falling to 0.17 at depth 6 by rate, with the
accepted COUNT nearly flat in depth; prose 0.29 at depth 1,
0.07 at depth 6) because a random-weight model's stream is
not text. Expect the real file to land between those records
and to split by content: chat boilerplate, code, lists, and
quoted or templated passages should accept heavily; novel
prose should sit near the floor, where the n-gram arm's win
over DFlash is only the draft forward it never pays (about
40.6 ms per round in the round-economics record) minus its
own wider verify. Whether that trade nets out on the real
chat/code mix is exactly the open measurement arms 1 and 2
take; nothing in this lane's fixtures settles it.

Files: `crates/itak-spec/examples/spec_generate_gpu.rs`
(`--drafter` wiring), `crates/itak-spec/src/config.rs`
(parse-guard test), `crates/itak-spec/tests/ngram_gpu_
runner.rs` (new; resident construction test and fixture).
