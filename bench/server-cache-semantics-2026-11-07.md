# Server cache semantics: key distinctness, sampled bypass, template fetch errors (2026-11-07)

Lane `lane/server-cache-semantics`, from main 012396a.
Three findings from the cold critic panel
(`~/workspace/reports/critic-engine-2026-11-07/CRITIC-ENGINE-MERGE.md`,
Category E), all in `platform/internal/server/server.go`: one HIGH
(the cache key's distinctness was untested, seat 5's executed
mutation) and two MEDIUMs (seat 3, code inspection). The two
chat-template HIGHs of Category E landed in the previous lane
(0578b44); this lane closes what remains of the category in the
server package.

## Finding 1 (HIGH, Operability seat): the cache key's distinctness was untested

`chatCacheKey` hashes the full marshaled request, and the comment
beside it shows the collision risk was understood — but the only
cache test repeated one identical request and asserted a hit.
Seat 5 mutated the key to hash only `req.Model` and
`go test ./internal/server/` stayed fully green: a key that stops
covering messages, temperature, or seed (a JSON-tag refactor
would do it silently) serves one caller's completion to every
other caller of the model, green suite throughout.

### What the key actually covers

Read before fixing: `chatCacheKey` marshals the whole
`chatRequest`. Its real inputs are every field of `chatRequest`
(model, messages, tools, stream, max_tokens, temperature, top_p,
stop, seed, top_k, presence_penalty, frequency_penalty) and, one
level down, every field of `chatMessage` (role, content, name,
tool_call_id, tool_calls, reasoning_content, recipient, end_turn),
`chatTool`/`toolFunction` (type, name, description, parameters),
and `chatToolCall`/`toolFunction` (id, type, name, arguments).
Pointer fields marshal distinctly from absent ones.

### Fix: tests, no code change

The key was correct; the proof was missing. Two test layers now
pin it:

- `platform/internal/server/cachekey_internal_test.go` (new,
  in-package because the key is unexported):
  `TestChatCacheKeyFieldByFieldDistinct` moves exactly one input
  at a time from a base request that sets all of them (28
  variants) and requires each variant's key to differ from the
  base key AND from every other variant's key.
  `TestChatCacheKeyStableForIdenticalRequests` pins the other
  half: identical requests key identically, map fields included.
- `platform/internal/server/server_cache_test.go` (new):
  `TestChatCacheDistinctRequestsGetDistinctEntries` posts a base
  request and 11 single-field variants over HTTP (message
  content, role, appended message, max_tokens, present-zero
  temperature, seed, seeded temperature, top_p, top_k, stop,
  model) and asserts for each: `X-Torch-Cache: miss`, the engine
  call count up by exactly one, and the entry count in `/debug`
  cache stats up by exactly one. A final repeat of the base
  request still hits its own entry. All variants stay cacheable
  under the finding-2 semantics, so a miss can only mean the key
  told them apart.

### Mutation-kill proof

Applied seat 5's exact mutation (`json.Marshal(req.Model)` in
place of `json.Marshal(req)`), with the new tests in the tree:

- `TestChatCacheKeyFieldByFieldDistinct`: FAIL — 28 variant arms
  report "key equals the base key" (every non-model field), plus
  pairwise collisions.
- `TestChatCacheDistinctRequestsGetDistinctEntries`: FAIL — the
  first same-model variant is served as a hit.
- `TestChatCompletionAndCache` (the pre-existing test): PASS
  under the mutation — the original hole, reproduced verbatim.

Mutation reverted; all three pass. The reverted tree's diff for
`chatCacheKey` is comment-only (a pointer to these tests).

## Finding 2 (MEDIUM, Failure seat): the cache stored sampled generations

`cache.Set` ran after any successful non-streaming completion
with no temperature/seed guard. Two identical unseeded
temperature > 0 requests share a key, so the second caller got
the first caller's sampled output byte-identical for the TTL —
the variation the temperature was set to obtain, frozen.

### Fix: cache semantics, stated where the cache is defined

New `chatCacheable(req)`: a request is cacheable iff it carries
a seed, or its temperature is absent, or its temperature is
<= 0. In `handleChatCompletions` the gate applies to BOTH sides:
uncacheable requests skip the `cache.Get` and the `cache.Set`
and answer `X-Torch-Cache: skip`; cacheable requests answer
`hit`/`miss` exactly as before. The semantics are documented in
the package header and on `chatCacheable` itself: seeded =
deterministic (same seed, same sample), temperature 0 = greedy,
unseeded temperature > 0 = variation requested, never frozen.

### Evidence

Fails before: against the unmodified handler,
`TestChatUnseededSampledRequestsBypassCache` failed at the first
assertion (first request answered `X-Torch-Cache: miss`, i.e.
the sample was being stored for replay).

Passes after, with a new `sampledEngine` fake whose completion
differs on every call so a replay is visible in the body:
two identical unseeded temperature-0.7 requests return two
distinct completions ("take 1", "take 2"), both `skip`, two
engine calls, zero cache entries. The widening guards pass
before and after, as regression pins:
`TestChatSeededSampledRequestsStillCache` (miss then hit, one
engine call) and `TestChatZeroTemperatureRequestsStillCache`
(same).

## Finding 3 (MEDIUM, Failure seat): a template-fetch error silently flattened, and the flattening was cached

`buildPrompt` returned `flattenMessages(msgs), nil` whenever
`resolveTemplate` returned nil — and `resolveTemplate` returned
nil WITHOUT setting `tplDone` on any fetch error other than
`ErrNoChatTemplate`. So a transient engine hiccup during the
template fetch produced a role-flattened prompt for that request
with no error anywhere, and the handler's unconditional
`cache.Set` then replayed the wrongly-prompted generation for
the TTL — contradicting the handler's own comment that a bad
prompt build is "never a silent fallback". Secondary defect in
the same function: `resolveTemplate` held `tplMu` across the
5-second network fetch, serializing every concurrent chat
request behind one slow fetch.

### Fix: split the outcomes, unlock the fetch

- `resolveTemplate` now returns `(*engine.ChatTemplate, error)`.
  Definitive outcomes are unchanged and still cached: config
  template wins, engine template fetched once,
  `ErrNoChatTemplate` settles as (nil, nil) = flatten. Any other
  fetch error returns a named wrapped error
  ("chat template fetch failed: %w"), is logged, and is NOT
  cached, so the next request retries. `buildPrompt` propagates
  it, so the request takes the handler's existing 502
  `chat_template_error` path: nothing generates, nothing is
  cached.
- `tplMu` now guards the cached outcome only, never the fetch:
  the outcome is read under a short lock, the fetch runs
  unlocked, and the new `cacheTemplateOutcome` records the
  definitive result first-writer-wins, so concurrent fetches
  agree on one outcome. (Concurrent requests may fetch twice;
  that is the stated trade for not serializing chat behind a
  5-second fetch.)

### Evidence

Fails before: `TestChatTemplateFetchFailureIsNamedNotFlattened`
— against the unmodified code the failing-fetch request
answered 200 with the flattened generation (the test wants 502
`chat_template_error`); `TestChatTemplateFetchDoesNotSerializeRequests`
— with one fetch parked inside a blocking fake, a second
request never reached its own fetch within 3 s.

Passes after: the failing-fetch request answers 502
`chat_template_error`, the engine's Generate is never called,
and the cache holds zero entries; once the fake recovers, the
next request renders through the real template (prompt asserted
byte-equal to the ChatML expectation), and a repeat is a cache
hit on that templated generation with the engine called once in
total. The concurrency test's second request reaches its own
fetch while the first is still parked, and both complete 200.

## Gates (lane tree, after all fixes)

From `platform/`:

- `go build ./...` — clean.
- `go test ./...` — all 9 packages pass; `--- PASS` count 123,
  counted as `go test ./... -v | grep -c -- "--- PASS"`
  (baseline at 012396a by the same count: 115; +8 new tests =
  2 key pins + 6 HTTP tests in this lane, no test removed).
- `go test -race ./internal/server/` — clean (the concurrency
  test above runs under it).
- `go vet ./...` — clean.
- `gofmt -l .` — empty.
- `golangci-lint run ./...` (v2.15.0, built with go1.26.9) —
  0 issues. One annotation added on the way: gosec G101
  pattern-matches the literal text `max_tokens` inside the HTTP
  test's request bodies, the same false-positive shape the
  pre-existing `TestChatMaxTokensCapRefusedByName` bodies would
  trip if they sat in a map literal; the variants map carries a
  `//nolint:gosec` with the reason, in the repo's existing
  nolint-with-reason style.
- `cargo check --workspace` from the repo root — clean (the
  tree is shared; no Rust file changed).

## Left alone, deliberately

- The embeddings NIT (server.go handleEmbeddings has no
  server-side generation deadline or input-count cap of its
  own): outside this lane's three findings; bounds exist one
  layer down, per the panel.
- Cache eviction/TTL internals in `internal/cache`: the panel
  checked that lifecycle clean (seat 3); untouched.
- The chat-template MEDIUM in `render.go` (dict.items() sorted
  order) belongs to the chattemplate package, not this cluster;
  untouched.
