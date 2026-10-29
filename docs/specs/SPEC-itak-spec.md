<!--
WHAT THIS FILE IS: the design spec for the itak-spec module (speculative
decoding) in the iTaK Torch rebuild. Design only: no engine code is
specified into existence here, and nothing in this file is a claim that
code exists. PLAN.md S6 says what itak-spec must contain and ARCHITECTURE.md
Section 3.1 fixes its grid slot; this file fixes the algorithms, the
interfaces, the rollback semantics, and the bench protocol its TODO items
(T-3.1 to T-3.4, T-9.6, T-9.7, parked T-P.1) are executed against.

Grounding rule: a claim in this file is either (a) measured, cited to the
record it came from, (b) read from code that exists today, cited to the
file, or (c) labelled ASSUMED, with the step that settles it named in
Section 10 (open questions).
-->
# SPEC: itak-spec, speculative decoding

Status: design, written 2026-10-29. Module slot: ARCHITECTURE.md
Section 3.1 (`itak-spec`: speculative execution; `draft(session) ->
candidates`, `verify(...)`; every mode config-gated, every gate with a
behavior test; depends on itak-forward and itak-arch). Plan authority:
PLAN.md S6 (the module in full) and S13 (the Glimmer fast path). This
spec covers the DFlash block loop in full, the shared verify and rollback
core every mode uses, the T-P.1 acceptance and depth sweep protocol, and
the Medusa salvage verdict. N-gram lookup is specified to the depth its
TODO item (T-3.1) needs, because it shares the verify and rollback core.
MTP, the EKF speculative update, early-exit drafting, and PowerInfer-style
activation prediction are named in PLAN S6 and are out of scope here;
they get their own specs when their phases open.

## 1. Measured ground and assumed ground

Measured, from this repo's own record:

| Fact | Value | Source |
| --- | --- | --- |
| Glimmer plain decode, llama.cpp Vulkan, 7900 XTX | 44.31 tok/s | GLIMMER-DISSECTION.md; bench record 2026-10-28 |
| Glimmer with DFlash, llama.cpp, same card | 88.6 tok/s (2.00x) | bench/model-suite-2026-10-28.md |
| Conditions of the 88.6 | Solo profile: `-md <dflash gguf> --spec-type draft-dflash --spec-draft-n-max 4`, `-np 1` | docs/RUNBOOK.md serving profile; DECISIONS.md serving decision |
| DFlash draft file | Own arch string `dflash`, 5 layers, 1.51 GiB, 8 KV heads, `block_size` 16 | GLIMMER-DISSECTION.md (GGUF census 2026-10-29) |
| Target KV shape | 52 layers: 39 windowed rings of exactly 2048 slots, 13 global; 2 KV heads, head dim 128 | crates/attention/src/cache.rs `glimmer_layer_pattern()`; GLIMMER-DISSECTION.md |
| The pasted article's DFlash claim | Depth 16, 75 to 110+ tok/s | REFERENCES.md: graded unverified; its depth conflicts with our measured profile at depth 4. Settling it is T-P.1 |

Assumed, flagged wherever used below:

- llama.cpp's internal DFlash wiring (how the draft is conditioned, how
  it verifies) is llama.cpp's business. This spec designs our loop from
  the speculative sampling literature's standard form and from the
  interfaces in our own crates. Nothing in the design depends on
  llama.cpp internals; the 88.6 is a performance anchor, not a design
  input.
- Whether the `dflash` arch consumes target hidden states as a
  conditioning input is NOT settled by our dissection (the census gives
  shapes, not wiring). See open question Q1; the trait design in
  Section 6 carries an optional conditioning handle so the answer does
  not change the trait when it lands.

## 2. The interfaces this design is built against (code that exists)

Read from the tree on 2026-10-29:

- `GlimmerModel::forward(&self, tokens: &[u32], cache: &mut dyn Cache)
  -> Result<Vec<f32>>` (crates/muse-glimmer/src/model.rs). Contract that
  matters: input is validated before any compute; on success the cache
  advances by `tokens.len()`; on any refusal the cache is untouched;
  the return is the FINAL position's logits only, already finished with
  the file's softcap and scale (`softcap_scale`, applied in the model,
  not the kernel). Two consequences drive this whole spec:
  1. Verify needs per-position distributions, so verify cannot use
     `forward` as it stands. Section 3 names the extension.
  2. The distribution the correction math runs on is the finished
     distribution (post softcap, post scale), bit-for-bit the same
     distribution the plain path samples from. Speculation that
     corrected against pre-softcap logits would not be the same model.
- `Cache` trait (crates/muse-glimmer/src/cache.rs): `position()` and
  `advance(n)` only. It is the position contract; real KV storage sits
  behind `Compute::attention` today and behind itak-kv when T-2.4 lands.
  The spec therefore places checkpoint and rewind on the KV store
  itself (the attention crate's `KvCache`), not on this trait.
- `KvCache` (crates/attention/src/cache.rs): one `LayerCache` per layer.
  Windowed layers are rings: slot is `position % window`,
  `slot_position[slot]` records which absolute position a slot holds,
  `appended` counts total appends and is the next expected position.
  `append` refuses out-of-order positions (`AttentionError::
  OutOfOrderPosition`). Global layers append rows at the tail of flat
  `Vec<f32>` key and value stores. `CacheView` reads visible positions
  oldest first, masks are defined over absolute positions, and
  `KvCache::stats()` exposes a per-layer snapshot for diagnostics. The
  rollback design in Section 5 is written against exactly these fields.
- `Compute::attention` applies RoPE to Q and K at the cache's current
  position (crates/muse-glimmer/src/compute.rs). Consequence: rows in
  the KV store are post-RoPE at absolute positions. A restored row is
  already correct for its position and must never be re-appended under
  a different one; the append order check enforces this.
- Error style (crates/muse-glimmer/src/error.rs, crates/attention/
  src/error.rs): hand-written enums, every variant named, `Display`
  states what failed and with what values, no fallback values anywhere.
  `SpecError` in Section 6 follows the same shape.
- Diag style (crates/muse-glimmer/src/diag.rs, model.rs): a `Diag`
  trait, `event(name, detail)` with dotted names (`forward.begin`,
  `forward.end`) and `key=value` detail strings. Section 6.4 fixes the
  itak-spec event catalogue in the same style. When itak-diag lands,
  these names become its stable codes unchanged.
- Salvage verdicts (SALVAGE.md, itak-spec table): the old FFI
  `spec_decode.go` is DROP (wired only into the retired FFI engine and
  never actually run in the serve loop; its lessons live in the bench
  record). The old native v1 `speculative_decoder.go` is RUST, the
  starting point for the verify design (it is a working draft and
  verify design, exercised only by its own test). The old Medusa
  `speculative_v2.go` is DROP as code; Section 8 is the concept verdict.

## 3. Required extensions (dependencies this spec names)

itak-spec cannot be built against the tree exactly as it stands. Three
extensions are required; each is proposed as a TODO item in Section 9.

E1. **Block forward with per-position logits.** A forward variant that
takes a block of tokens against a cache and returns the finished logits
for EVERY position in the block (row-major `[block][vocab]`, the same
softcap and scale finishing as `GlimmerModel::forward`). Teacher
forcing a block of length L yields L distributions; the round consumes
L of them (Section 4). Owned by the forward layer (itak-forward and the
arch forward), not by itak-spec: speculation is a caller of the forward,
never a special case inside it (ARCHITECTURE.md Section 3.1 gives
itak-spec `itak-forward` as a dependency, not the reverse).

E2. **KvCache checkpoint and rewind.** A transactional API on the
attention crate's `KvCache`: `checkpoint()`, `rewind_to(checkpoint,
position)`, `release(checkpoint)`. Semantics in Section 5. Owned by the
attention crate (later itak-kv), used by itak-spec and by n-gram
rollback (T-3.1 names rollback in its proof for the same reason).

E3. **A dflash arch module.** The draft is a model: it loads through the
same loader and arch seam as the target (ARCHITECTURE.md Section 6),
with its config read from its own file's metadata under the same
no-silent-default rule as `GlimmerConfig`. itak-spec defines the
contract it needs from that module (Section 6.2, the `Speculator`
implementation for DFlash); the module itself is a Phase 9 item.

## 4. The DFlash verify round

One round of speculative decode, for one sequence. Definitions:

- `base`: the target cache position at round start. The cache holds K/V
  for exactly the committed sequence prefix, positions `0..base`.
- `pending`: the most recently emitted token, sampled in an earlier
  round (or by prefill for the first round), not yet forwarded. The
  loop invariant of the whole design: the emitted sequence is the
  committed prefix plus `pending`; the target cache lags the emitted
  sequence by exactly the pending token.
- `n`: the verify depth for this round, `min(config.n_max, draft block
  size, remaining token budget)`. The draft file's block size is 16;
  `n_max` is config (Section 6.3), so `n <= 16` for Glimmer's draft.
- Speculation applies to decode only. Prefill is a plain forward over
  the prompt; its final logits sample the first `pending`.

The round, step by step:

1. **Checkpoint.** `cp = kv.checkpoint()` before anything speculative
   touches the cache. Draft state is not checkpointed (step 7).
2. **Propose.** The DFlash speculator drafts `t[1..=n]` as continuations
   after `pending`, returning the tokens and the draft's per-position
   distributions `q[i]` (the draft's finished distribution for the
   token following position `i-1`, in the notation of step 4). The
   draft may return fewer than `n` (its own EOS or its block
   boundary); the round proceeds with what it returned.
3. **Verify forward.** One block forward (E1) over
   `[pending, t[1], ..., t[n]]`, `n+1` tokens, appending their K/V.
   It returns finished logits for all `n+1` positions. Read them as
   distributions `p[0..=n]`, where `p[i]` is the target distribution
   for the token that follows the prefix extended by
   `[pending, t[1..=i]]`. Note `p[0]` judges `t[1]`, and `p[n]` is the
   distribution a fully accepted block draws its bonus token from.
   Because attention is causal, each position's K/V and distribution
   are exact: nothing later in the block influences them. This is what
   makes step 5's keep-the-prefix rule exact rather than approximate.
4. **Correct.** Walk `i = 1..=n`, drawing `r` uniform in `[0, 1)` from
   the session RNG (Section 6.3):
   - Accept `t[i]` if `r < min(1, p[i-1](t[i]) / q[i](t[i]))`.
   - On the first rejection at position `j`: sample the replacement
     token from the residual distribution
     `norm(max(0, p[j-1] - q[j]))` (elementwise, clipped at zero,
     renormalized over the full vocabulary). The walk stops.
   - If all `n` accept: sample the bonus token from `p[n]`.
   Greedy decoding is the special case where both distributions are
   point masses: accept exactly when the draft token equals the
   target's argmax, residual is the target argmax. At temperature 0
   the emitted sequence must be identical to non-speculative
   generation; that identity is the standing proof obligation
   (T-3.1, T-9.6) and a behavior test, not a hope.
   The sampler parameters of the session (temperature, top-p, top-k,
   penalties, per T-2.5) are applied to the target logits BEFORE the
   correction walk, so `p` is the distribution the plain path would
   sample from. The draft distributions `q` are taken under the
   draft's configured sampler (default: same temperature as the
   target; config, Section 6.3).
5. **Commit or rewind the target cache.** Let `a` be the number of
   accepted draft tokens (`a = j-1` on a rejection at `j`, `a = n` on
   full acceptance). The verify forward appended K/V for `pending`
   plus all `n` drafts; the committed prefix grows by `pending` plus
   the `a` accepted drafts, so `kv.rewind_to(cp, base + 1 + a)`
   (Section 5), then `kv.release(cp)`. The replacement or bonus token
   becomes the new `pending`: emitted, not yet forwarded, exactly the
   loop invariant. Emitted this round: `a + 1` tokens (accepted drafts
   plus the replacement or bonus). `pending` itself was emitted in an
   earlier round and is not re-emitted.
6. **Stop conditions inside a round.** If any emitted token (an
   accepted draft, the replacement, or the bonus) is EOS or matches a
   stop rule, the emitted stream truncates at that token and the
   session ends. Cache state after a terminal round is irrelevant
   (the session is torn down), but the round's diag events still fire
   (Section 6.4) so the acceptance record is complete.
7. **Draft state sync.** After the round, the speculator is told
   exactly the newly committed tokens, `[pending_old, t[1..=a]]`, via
   `commit()` (Section 6.2), and nothing else. The contract: after any
   sequence of `propose` and `commit` calls, the draft's internal
   state equals the state of having processed exactly the committed
   sequence, no more. How the draft holds that state (its own KV over
   the committed prefix, recomputed windows, hidden-state carry) is
   the draft module's business; speculative state the draft built
   while proposing is discarded by the draft itself and never leaks
   into the target cache. If a draft cannot honor this contract, it
   does not ship as a speculator.

Failure inside a round: if the verify forward refuses (any
`GlimmerError` or `AttentionError`), the spec layer rewinds to `base`
(discarding the pending token's rows too; `pending` stays pending),
releases the checkpoint, and returns the refusal wrapped in
`SpecError::VerifyRefused`. No partial commit exists. What happens
next is policy, fixed in Section 6.3: the default is that the request
fails with the named error. A latched fallback exists as config and is
never silent.

## 5. Cache rollback against windowed rings (the hard part, specified exactly)

### 5.1 Why naive append-and-truncate corrupts a ring

A windowed layer's ring holds the most recent `window` (2048)
positions, slot `position % window`. Appending position `p` overwrites
the slot that held position `p - window`. That evicted position is
still visible to the committed prefix: a query at the last committed
position `base - 1` attends over `[base - window, base - 1]`, and
`p - window >= base - window` for every speculative `p >= base`. So
the moment a speculative append lands in a ring, it destroys K/V that
committed tokens still attend to. Truncating `appended` afterward does
not bring the evicted rows back. Any rollback design that appends
first and restores only counters is corrupt by construction; this is
the trap the old tree's never-wired sliding window code never had to
face, and the reason T-3.1 names rollback in its proof.

Global layers have no such problem: rows append at the tail and a
rewind is a truncation. The design below treats the two kinds
accordingly.

### 5.2 Checkpoint contents

`KvCache::checkpoint()` captures, before the round's first speculative
append:

- For every layer: the `appended` count at checkpoint time (one
  number; all layers share the round's `base` in this design, but the
  checkpoint records it per layer because the cache API is per layer
  and the restore must not assume).
- For every WINDOWED layer: for each slot the round may touch, the
  slot's current key row, value row, and `slot_position` entry. The
  round forwards at most `max_round_tokens` positions (pending plus
  `n_max` drafts; a config bound, Section 6.3), so the touched slots
  are the `max_round_tokens` consecutive slots starting at
  `base % window`, wrapping. With Glimmer's numbers (17 tokens worst
  case, 39 windowed layers, rows of 2 heads x 128 dims x 4 bytes =
  1 KiB per K or V row) a checkpoint is at most about 1.4 MB,
  allocated once per session and reused every round (RULES.md: no
  per-token allocation; a per-round reused buffer is the same
  discipline).
- Global layers need no row copies: their restore is truncation.

One checkpoint may be outstanding per cache at a time. A second
`checkpoint()` while one is live is refused with
`SpecError`-adjacent cache error `CheckpointBusy` (named in the
attention crate's error set when E2 lands; listed here so the refusal
has a name from birth). Stack discipline matches the caller: rounds
are sequential per sequence, and batching (Section 10, Q5) does not
nest checkpoints inside one sequence's cache.

### 5.3 Rewind semantics

`rewind_to(cp, target)` where `base <= target <= base + n + 1`:

- Every layer: `appended` becomes `target`.
- Global layers: key and value stores truncate to `target` rows. Rows
  for positions `base..target` (the kept speculative rows) stay; they
  are exact by the causality argument in Section 4 step 3.
- Windowed layers: slots holding kept positions (`base..target`) keep
  the speculative rows. For every other touched slot, restore the
  checkpoint's saved key row, value row, and `slot_position`. Slots
  the round never touched are untouched throughout.
- Postcondition, checkable and checked in tests: for every layer,
  `view()` after rewind returns exactly the rows that a cache fed only
  the committed prefix would hold, and the `slot_position` bookkeeping
  agrees with every visible row (the invariant `LayerCache::key`
  already debug-asserts). `stats()` after rewind shows
  `appended == target` on all layers.

Full-round abort is `rewind_to(cp, base)`: every touched windowed slot
restored, global stores truncated to `base`, draft told nothing.

### 5.4 What rollback never does

- It never re-appends a restored row under a new position (RoPE is
  baked in at append; Section 2). Restored rows return to their slots
  with their original absolute positions, and `slot_position` says so.
- It never runs on the muse-glimmer `Cache` position trait. Position
  and storage move together or the design has two truths; the position
  the arch forward sees is derived from the KV store's `appended` when
  itak-kv lands, and until then the engine session owns both and the
  round updates them in the same step.
- It is not a snapshot or persistence format. Checkpoints are
  in-memory, per-session, and die with the round. Versioned KV save
  and load is itak-kv's separate contract (ARCHITECTURE.md Section
  5.3: state save and load either round-trip verified or the verb
  does not exist).

## 6. Interfaces

### 6.1 Errors: named refusals only

`SpecError`, hand-written in the house style of `GlimmerError` and
`AttentionError`. Every variant names the thing that failed; there is
no variant for "fell back", because fallback is a state (Section 6.3),
not an error, and it is never entered silently.

| Variant | Meaning |
| --- | --- |
| `ModeUnavailable { mode }` | Config names a speculation mode with no implementation loaded in this build |
| `DraftNotLoaded` | Mode requires a draft and none is attached to the session |
| `DraftArchMismatch { found, expected }` | The draft file's arch string is not the one the speculator serves (mirrors `GlimmerError::WrongArchitecture`) |
| `DraftLoadFailed { detail }` | The loader's coverage report for the draft named a gap; carries the report's wording |
| `EnvelopeMismatch { field, expected, actual }` | A versioned artifact (Medusa heads, Section 8) disagrees with the runtime context envelope on `field` |
| `ArtifactVersionUnsupported { found, supported }` | Artifact format version outside the supported set |
| `ZeroInitializedHeads { artifact }` | Medusa artifact loaded but its heads are zero-initialized; a reported load state, never served as if trained |
| `DepthOutOfRange { n_max, block_size }` | Configured `n_max` outside `1..=block_size` |
| `CheckpointBusy` | A cache checkpoint is already outstanding (raised by E2, surfaced unchanged) |
| `CacheOutOfSync { expected, actual }` | Post-round cache position is not the committed length; the round's bookkeeping and the cache disagree, which is a defect, reported as one |
| `GrammarConflict` | Speculation requested for a session with an active grammar, JSON mode, or schema constraint (v1 refusal; Q4) |
| `VerifyRefused { source }` | The verify forward refused; wraps the `GlimmerError` or `AttentionError` that caused it |

Session creation with speculation enabled validates everything
validatable up front (mode available, draft attached and loaded,
`n_max` in range, no grammar conflict) and refuses by name at creation.
A session that starts speculative never discovers a load-time problem
mid-stream.

### 6.2 The Speculator trait and the round driver

```rust
/// What a proposer believes about one position: the draft's finished
/// distribution, borrowed for the round. Row-major over the vocab.
pub struct DraftDist<'a> {
    /// Probabilities after the draft's own finishing and sampler,
    /// length = target vocab.
    pub probs: &'a [f32],
}

/// One round of drafting: tokens plus the distribution each was
/// drawn from, position-aligned.
pub struct BlockProposal<'a> {
    /// Drafted tokens, continuations after the pending token.
    pub tokens: &'a [u32],
    /// `dists[i]` is the distribution `tokens[i]` was drawn from.
    pub dists: &'a [DraftDist<'a>],
}

/// Everything a proposer may condition on. The hidden-state handle is
/// optional because the DFlash conditioning question is open (Q1);
/// a proposer that requires it and finds `None` refuses with
/// `SpecError::ModeUnavailable` naming the missing input rather than
/// drafting unconditioned garbage.
pub struct ProposeContext<'a> {
    /// The committed sequence prefix (at minimum its tail; the exact
    /// window a proposer may see is its own documented contract).
    pub committed: &'a [u32],
    /// The pending token the proposal continues from.
    pub pending: u32,
    /// Target final hidden states for the committed prefix, when the
    /// deployment provides them.
    pub target_hidden: Option<&'a [f32]>,
    /// Maximum tokens to propose this round (the round's `n`).
    pub max_tokens: usize,
}

/// A drafting mode. Implementations: DFlash (Section 4), n-gram
/// lookup (Section 7), Medusa (Section 8, later).
pub trait Speculator {
    /// Which mode this is (config and diagnostics name modes by this).
    fn mode(&self) -> SpecMode;
    /// The largest block this proposer can emit (16 for the shipped
    /// DFlash file; the n-gram proposer's configured match cap).
    fn max_depth(&self) -> usize;
    /// Draft the next block. Builds only proposer-internal state;
    /// touches neither the target cache nor the emitted sequence.
    fn propose(&mut self, ctx: &ProposeContext<'_>) -> Result<BlockProposal<'_>, SpecError>;
    /// Advance the proposer's state by exactly these committed
    /// tokens, in order. After any propose/commit sequence the
    /// proposer's state equals having processed exactly the committed
    /// sequence (Section 4 step 7).
    fn commit(&mut self, tokens: &[u32]) -> Result<(), SpecError>;
}
```

The round driver (owned by itak-spec, called by the engine's
generation loop; the loop calling it is the wiring ARCHITECTURE.md
Section 5.4 requires, so a configured-but-uncalled decoder is
impossible by construction):

```rust
/// The outcome of one round, also the source of the round's diag
/// events and the session's cumulative spec stats.
pub struct RoundOutcome {
    /// Tokens emitted this round (accepted drafts plus the
    /// replacement or bonus), in order. Length is `accepted + 1`
    /// unless a stop condition truncated it.
    pub emitted: Vec<u32>,
    /// Draft tokens proposed this round.
    pub proposed: usize,
    /// Draft tokens accepted this round.
    pub accepted: usize,
    /// Index of the first rejected draft token, if any.
    pub first_rejection: Option<usize>,
    /// Target cache position after the round's commit or rewind.
    pub cache_position: usize,
    /// True when a stop condition ended the session inside this round.
    pub stopped: bool,
}

/// Run one speculative round (Section 4). `forward_block` is E1,
/// `kv` is the session's target cache, `sampler` applies the
/// session's sampling parameters to raw finished logits.
pub fn run_round(
    spec: &mut dyn Speculator,
    session: &mut SpecSessionState,
    // ... forward_block, kv, sampler, rng, diag handles
) -> Result<RoundOutcome, SpecError>;
```

(Signature shown at design level: exact handle types land with the
crate, and the shape, not the spelling, is what this spec fixes.)

The correction walk itself (Section 4 step 4) is a pure function over
`(target_probs, proposal) -> (accepted, replacement_or_bonus)`, so
T-3.2's distribution tests exercise it without a model.

### 6.3 Config: every limit a knob, every knob behavior-tested

All under a `spec` config block; defaults in parentheses. Every entry
has a behavior test proving the knob changes behavior (ARCHITECTURE.md
Section 5.1: a flag read only by a log line is a build failure).

| Knob | Default | Meaning |
| --- | --- | --- |
| `spec.mode` | `off` | `off`, `dflash`, `ngram`; `medusa` becomes valid when Section 8's path lands |
| `spec.dflash.n_max` | 4 | Verify depth for DFlash. 4 is the measured profile's value; the sweep (Section 9 in this spec, TODO T-9.7) may re-default it with its table filed |
| `spec.dflash.draft_path` | none | The draft GGUF; required when mode is `dflash` |
| `spec.dflash.draft_temperature` | follow target | Draft sampler temperature |
| `spec.ngram.max_depth` | 8 | Longest n-gram proposal |
| `spec.ngram.min_match` | 2 | Shortest history match that may propose |
| `spec.max_round_tokens` | 17 | Bound on positions one round may append (pending + max depth); sizes the checkpoint (Section 5.2) |
| `spec.on_verify_error` | `refuse` | `refuse`: a verify refusal fails the request by name. `latch_off`: the first verify refusal latches speculation off for the session, emits `spec.fallback.latched`, and the session continues non-speculatively in that explicit, queryable state |

RNG: the correction walk draws from a per-session seeded RNG injected
into the driver, never a global. Temperature 0 needs no draws (the
greedy special case), and seeded draws make the sampled arm of the
sweep reproducible. Which crate owns the RNG implementation is Q6.

### 6.4 Diagnostic events, per verify round

Through the module's diag channel, house style (`Diag::event`, dotted
names, `key=value` detail). A session's cumulative counters
(`spec_stats()`: rounds, proposed, accepted, rejections, rollbacks,
fallback state) are pullable the way `KvCache::stats()` is, because a
number only in a log stream is a number nobody can gate on.

| Event | When | Detail fields |
| --- | --- | --- |
| `spec.session.begin` | Session created with speculation on | `mode`, `draft` (file + arch), `n_max`, `target` arch |
| `spec.session.end` | Session teardown | `rounds`, `proposed`, `accepted`, `accept_rate` |
| `spec.disabled` | Session created with mode `off` | `reason=config` (emitted once, so "off" is a queryable state, not an absence) |
| `spec.round.begin` | Round start | `round`, `base`, `pending` |
| `spec.propose.end` | Proposal returned | `proposed`, `draft_ms` |
| `spec.verify.end` | Verify forward returned | `verify_ms` |
| `spec.round.accept` | Correction walk done | `proposed`, `accepted`, `first_rejection` (or `none`), `emitted` |
| `spec.cache.rewind` | Any rewind, including commit-path | `from`, `to`, per-kind layer counts |
| `spec.cache.commit` | Round committed | `position` after commit |
| `spec.fallback.latched` | `on_verify_error=latch_off` fires | `reason` (the refusal's wording), `round` |
| `spec.refused` | Session creation refusal | the `SpecError` variant and its fields |
| `spec.heads.loaded` | Medusa artifact accepted (Section 8) | `artifact`, `version`, envelope fingerprint |
| `spec.sweep.row` | Bench mode only, one per grid cell | the full sweep row of Section 9, so bench tables are built from diag output, not scraped prose |

## 7. N-gram prompt lookup (T-3.1), to the depth it shares this core

The n-gram proposer needs no weights: it searches the committed
sequence (prompt plus generated so far) for the longest recent suffix
match and proposes the tokens that followed the earlier occurrence
(the algorithm is working today in the Mojo engine; PLAN S6 says port
the algorithm, and Mojo lab informs itak-spec per docs/MAP.md). It
implements the same `Speculator` trait: `propose` reads
`ProposeContext.committed`, `commit` is bookkeeping only, and verify,
correction, checkpoint, and rewind are literally the same code path as
DFlash. That sharing is why T-3.1's proof names KV state rollback:
SSM state in the hybrid families rolls back through the same
checkpoint discipline, with the SSM state block copied into the
checkpoint beside the ring rows (state size is a config-visible
quantity of the arch module, and a checkpoint that omits it is the
same corruption as Section 5.1 in a different store).

## 8. Medusa verdict: what salvaging the self-drafting heads requires, and why it sequences after DFlash

Verdict: the concept is sound and stays in the plan (PLAN S6, TODO
T-3.3, ENDGOAL.md names it a finished-system property). The old code is
not salvage: SALVAGE.md drops `speculative_v2.go` (zero references, no
test, and no train, load, or serialize path exists for its heads, so
they could never hold trained weights), and the 2026-10-26 research
note records the heads as random-init with acceptance near zero
expected until trained. Salvaging the IDEA requires building four
things that do not exist today, in this order:

1. **A training path (TODO T-7.3, Phase 7).** A job type of its own in
   the training stack: harvest final hidden states from the frozen
   target over a real corpus, train the K heads (the v2 design: K MLPs
   on the final hidden state predicting offsets +1..+K) against the
   target's own next-token labels. This needs a hidden-state export
   from the forward, which does not exist yet; Section 9 proposes it
   as its own item so it is scheduled rather than discovered.
2. **A versioned artifact format.** Heads ship as files with a version
   field (project law: data formats carry version fields), loaded
   through the same loader discipline as models: coverage checked,
   shapes checked, gaps named. Zero-initialized heads are a load state
   the engine reports (`SpecError::ZeroInitializedHeads`), never a
   silent default (ARCHITECTURE.md Section 5.4).
3. **The runtime-context envelope.** The artifact carries the envelope
   it was trained under, and the loader verifies it field by field
   before the heads may propose: target architecture string, hidden
   size, vocabulary size, layer count, the target file's identity
   (content fingerprint as registered by model operations), the
   tokenizer identity, and the artifact format version. Heads read
   the target's final hidden states, so any change to the backbone
   (a requant, a different checkpoint, a tokenizer swap) shifts the
   distribution the heads were trained on; an envelope mismatch is
   `SpecError::EnvelopeMismatch` naming the field, at load time,
   never a degraded acceptance rate discovered in production.
4. **Tree verify.** Medusa proposes a TREE (per-head top-k candidates
   branched), not a block. Verifying it needs E1 extended with tree
   attention masks (candidates attend only to their ancestors), which
   is a forward extension beyond per-position logits, plus the same
   correction math generalized per path. Rollback is unchanged:
   checkpoint and rewind do not care what shape was verified.

Why after DFlash, stated once and plainly:

- DFlash's weights exist. They were trained against this exact
  backbone by the model's authors, the file is on the server and
  byte-verified, and the pair is measured at 2.00x on David's card
  today. Medusa's weights exist nowhere; every gram of Medusa value
  waits on item 1.
- Medusa's critical path runs through the whole training stack
  (Phase 7) plus a forward export hook. DFlash's path needs only the
  loader, E1, and E2. Sequencing DFlash first puts working 2x
  speculation in the engine a full phase earlier.
- DFlash exercises the entire shared core (verify, correction,
  checkpoint, rewind, diag, config gates) with a single-path proposal,
  the simplest shape that can prove it. Medusa then adds exactly two
  new things (artifacts with envelopes, tree verify) on proven
  machinery, instead of debugging all of it at once.
- The plan already encodes this order (Phase 9 fast path carries
  DFlash, T-9.6 and T-9.7; T-3.3 consumes Phase 7 output, T-7.3).
  This spec ratifies it with reasons rather than re-deciding it.

## 9. The acceptance and depth sweep protocol (parked T-P.1; executed as T-9.7)

T-P.1 as parked asks a narrow question: draft depth 4 (our measured
profile) versus 16 (the article's claim). This protocol answers that
question and, in the same runs, produces the operating point the new
stack ships as config. It runs twice by design: first against
llama.cpp (that is T-P.1 itself, unblocking nothing but settling the
claim on the reference stack), then against itak-engine once T-9.6
lands (that run is T-9.7, and its output is the config default).

### 9.1 Grid

- Depth (`n_max`): 1, 2, 3, 4, 6, 8, 12, 16. Depth 1 is the control
  (speculation machinery on, one token verified per round). The plain
  non-speculative run is recorded alongside as the 44.31-class anchor.
  Depths above the draft's block size (16) do not exist for this file.
- Sampler arms: (a) temperature 0; (b) the deployment's recorded
  default sampler settings. Acceptance is sampler-dependent and the
  two arms are reported separately, never averaged together. Whether
  the proxy-layer reasoning_strength injection belongs in the bench
  prompts is frozen in the bench note before the run (Q7).
- Shape: solo, one slot (the conditions of the 88.6 measurement).
  Multi-slot speculation is Q5 and is not swept here.

### 9.2 Prompts and run discipline

- Prompts: the bench prompt set (bench/accuracy-prompts.md) plus a
  fixed long-form generation prompt and a code-shaped prompt, named
  in the bench note; every cell runs the identical set.
- Fixed generation length per prompt (256 tokens or the prompt set's
  own stop, whichever first), warm-up round discarded, at least 3
  repetitions per cell, median reported. Same binary, same commit,
  same model files for the whole grid; the commit hash is recorded in
  the bench note (the T-9.4 discipline).
- Harness anchor first: before any grid cell counts, the harness must
  reproduce the recorded profile on the reference stack (44.31 plain,
  88.6 at depth 4, each within a few percent; docs/RUNBOOK.md: a
  result more than a few percent off the recorded numbers is a
  finding, not a rounding error). An unanchored sweep is not run.

### 9.3 Measurements per cell

- Wall tok/s (end to end, the number that decides).
- Tokens accepted per verify round (mean), and overall acceptance
  rate (accepted / proposed).
- Per-position acceptance: for each draft position `i`, the fraction
  of rounds that reached position `i` and accepted it. This decay
  curve is the real output of the sweep: it shows where the draft
  stops paying, and it is what makes the operating point a reading
  rather than a guess.
- Draft ms and verify ms per round (from the diag events of
  Section 6.4 on our stack; from llama-server's own counters on the
  reference stack), so a depth that wins on acceptance but loses on
  draft cost is visible as such.

### 9.4 Picking the operating point

- The winner is the SMALLEST depth whose median wall tok/s is within
  2 percent of the grid's best median in the temperature 0 arm and not
  worse in the sampled arm. Ties and near-ties resolve shallower on
  purpose: shallower depth wastes less verify compute per rejection,
  rewinds less cache per round, and has better tail latency, and the
  sweep exists to find where depth stops paying, not to maximize
  depth.
- If per-position acceptance past some position `k` is under 5
  percent in both arms, depths above `k` are recorded as dead weight
  in the bench note regardless of the wall-clock outcome.
- Output: the sweep table filed in bench/ (per T-9.7's proof), the
  chosen depth written as the `spec.dflash.n_max` default, and, if
  the reference-stack run moves the recorded 88.6 anchor materially,
  a DECISIONS.md note in the same pass. The result is config, never
  code (ARCHITECTURE.md Section 6).

## 10. Open questions (explicit, with owners of the answer)

- Q1. DFlash conditioning. Does the `dflash` arch consume target
  hidden states, token ids only, or both? Our census gives shapes,
  not wiring. Settled by: the draft file's full tensor census plus
  the llama.cpp draft-dflash source, read before the E3 arch module
  is written. The `ProposeContext.target_hidden` handle exists so
  either answer fits the trait.
- Q2. DFlash cache geometry. The draft's own layer pattern, window,
  and KV geometry under its `dflash.*` metadata keys are unmeasured.
  Settled by: reading the draft file's metadata through the loader
  when E3 lands, under the same no-silent-default rule as the target.
- Q3. Where E1 lives in the phase order. T-2.4 scopes the forward as
  incremental single-token steps; per-position block logits are an
  addition nobody's TODO item currently owns. Proposed as T-2.10 in
  Section 11. Until it lands, T-3.x and T-9.6 are blocked on it in
  fact, whatever the phase numbers say.
- Q4. Grammar and speculation. v1 refuses the combination
  (`GrammarConflict`). The alternative, applying the grammar mask to
  both distributions before the correction walk, is mathematically
  available but unproven against T-3.5's conformance suite. Decided
  when T-3.5 integrates, not here.
- Q5. Batched rounds. This spec is one sequence per session (the solo
  profile every baseline was measured under). Continuous batching
  with ragged per-slot acceptance is undesigned; the scheduler phase
  (T-6.2) owns the question, and nothing here pretends otherwise.
- Q6. RNG ownership. The correction walk needs a per-session seeded
  RNG; no crate in the current tree owns one. Landing spot (sampler
  code beside T-2.5 is the natural home) is proposed, not settled.
- Q7. Sweep prompt freeze. Whether the proxy layer's default
  reasoning_strength injection applies to the sweep's prompt set must
  be frozen in the bench note before T-P.1 runs, so the anchor and
  the grid are measured under one stated configuration.

## 11. Proposed TODO items

Proposed in this spec for adoption into TODO.md under its rules (new
items at the end of their phase, IDs stable once published). They
decompose existing items; they do not renumber or replace any.

Phase 2 (PLAN S5):

- [ ] T-2.10 Block forward with per-position finished logits, plus
  KvCache checkpoint and rewind (SPEC-itak-spec E1 and E2) (proof: a
  unit test rewinds a windowed ring to a mid-block position and the
  cache view is byte-identical to a cache fed only the kept prefix;
  per-position logits match sequential single-token forwards on the
  same prompt at temperature 0) (PLAN S5)

Phase 3 (PLAN S6):

- [ ] T-3.6 itak-spec crate skeleton: Speculator trait, SpecError set,
  spec config block, diag event catalogue, session validation
  (proof: cargo tests green; a session configured mode=dflash with no
  draft attached refuses creation by name; every config knob has a
  behavior test) (PLAN S6)
- [ ] T-3.7 Correction sampler as a pure function: accept rule and
  residual distribution per SPEC-itak-spec Section 4 (proof: the
  distribution test cases T-3.2 names pass against it; greedy walk
  equals argmax comparison on recorded distribution pairs) (PLAN S6)

Phase 7 (PLAN S10):

- [ ] T-7.6 Hidden-state export hook in the target forward for Medusa
  head training (proof: exported final hidden states for a fixed
  prompt match the forward's own final hidden states within
  tolerance, asserted by a test) (PLAN S10)

Phase 9 (PLAN S13):

- [ ] T-9.11 dflash arch module: config from the draft file's own
  metadata, tensor map from its census, conditioning question
  (SPEC-itak-spec Q1) answered from the census and the llama.cpp
  source before the module is written (proof: the draft GGUF loads
  through the same loader with a complete coverage report; a
  non-dflash file is refused by name) (PLAN S13)
