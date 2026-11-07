# Chat template tool conversations (2026-11-07)

Lane `lane/chat-template-tools`, from main 2a2030e.
Two HIGH findings from the cold critic panel
(`~/workspace/reports/critic-engine-2026-11-07/CRITIC-ENGINE-MERGE.md`,
Category E), both executed by the Mop-up seat against the model's own
ATEM template (`platform/internal/chattemplate/testdata/glimmer-atem.jinja`).

## Finding 1 (HIGH): tool messages silently flattened to an empty name

`chattemplate.Message` carried only `Role` and `Content`, and `Render`
built each message dict with only those two keys. The fixture's tool
branch looks up `message.get('name')`, falls back to a `tool_call_id`
search over `tool_calls` — neither could exist — and ended with an
empty name.

### Before evidence

Executed probe on the base tree (a temporary test through the real
`Render`, since removed): a bare tool message rendered, with no error,

```
<|start|>tool <|message|><tool_output name="">
result-42
</tool_output></tool_output><|eot|>
```

Assistant messages lost the same way in the other direction: with no
`tool_calls` key representable, a tool-calling turn rendered as a
plain text turn. This contradicted the package's own contract
("never a silently flattened prompt").

### Fix

`Message` now carries every field the fixture's message branches
read, and only those:

| Message field | Template reads it as | Where the fixture reads it |
|---|---|---|
| `Name` | `name` | tool branch, `message.get('name')` |
| `ToolCallID` | `tool_call_id` | tool branch fallback lookup |
| `ToolCalls []ToolCall` | `tool_calls` | tool branch scan; assistant branch |
| `ReasoningContent` | `reasoning_content` | assistant branch |
| `Recipient` | `recipient` | assistant branch |
| `EndTurn *bool` | `end_turn` | assistant branch |

`ToolCall` carries `ID`, `Type`, and `Function` (`FunctionCall`:
`Name`, `Arguments map[string]any`). The template refuses a call
whose arguments are not a mapping, so arguments are structured data,
never a JSON string.

Message dicts bind keys in a fixed authored order (role, content,
then the optional fields in the order above), and optional fields
are ABSENT when unset rather than present-but-empty, so
`message.get('name')` answers none for a plain message exactly as
the reference environment does.

Name resolution: `Render` resolves every tool message's name before
rendering — the explicit `Name`, else the function name of the tool
call whose id equals the message's `ToolCallID` (the same lookup the
template itself performs) — and refuses the render with the new
`ErrUnresolvableTool` when neither yields a name. The refusal is
deliberately template-independent: a name the ATEM protocol prompt
would print empty is a malformed conversation for this platform,
whatever template is configured, and the server maps it to the same
named `chat_template_error` (502) as every other render refusal.
That is the panel's prescribed shape ("refuse an unresolvable tool
message by name") taken at the package boundary rather than the API
boundary, so no caller of `Render` can flatten one either.

Two existing tests met the new contract and were adjusted with the
reason recorded in place: `TestRoleConditionalsWithElifAndElse`'s
tool message now carries a name (its branch does not read it), and
`TestUntakenBranchIsNotEvaluated`'s taken-branch tool message
carries a name so the render reaches the branch it tests (a nameless
one now refuses earlier, with `ErrUnresolvableTool`).

## Finding 2 (HIGH): the template's `tojson` refused; tool paths unreachable

The fixture calls `| tojson` at L43 and throughout the tool
definitions macro (L68, L73). The renderer implemented four filters
and refused `tojson` by name (pinned by a test); `Render` bound no
`tools` value, so `{% if tools %}` was always falsy and the tool
paths never evaluated. The golden test that "renders the real
template whole" covered only tool-less conversations.

### Before evidence

Executed probe on the base tree: `{{ 'x' | tojson }}` returned
`chattemplate: unsupported construct: filter "tojson"`.

### Fix

- `evalFilter` gains `tojson` (`evalToJSON` in render.go), with
  Jinja's semantics for the shapes involved:
  - Default form is Python `json.dumps` compact output: separators
    `,` and `:` with no spaces, mapping keys SORTED (Jinja's
    `sort_keys` defaults to true — the ATEM schema dump's key order
    depends on it), integral floats without a fraction, `true` /
    `false` / `null`, NaN and infinities as Python's bare tokens.
  - String escaping follows the plain `json.dumps` form for the
    quote (`\"`), backslash, control characters (short escapes for
    `\n \r \t \b \f`, else `\u00XX`), non-ASCII as `\uXXXX` with
    UTF-16 surrogate pairs above the BMP (ensure_ascii), and the
    HTML-safe escapes for `<`, `>`, `&` (`\u003c` etc.).
    Named divergence: Jinja's own `tojson` additionally unicode-
    escapes `"` and `'` in its HTML-safe pass. This renderer
    follows the plain form for the quote because the fixture's
    schema dumps are produced, in the reference stack, by
    llama.cpp's minja runtime, whose `tojson` is a plain
    nlohmann::json dump; the golden tests byte-compare against
    that form. The `'` escape is not applied by either form the
    fixture can reach (plain dumps never escape it; minja does
    not), so it is not applied.
  - Keywords: `indent` (non-negative integer) switches to the
    pretty form (`, ` after colons... precisely: `": "` after keys,
    one entry per line, padded by level) and `sort_keys=false`
    keeps recorded insertion order. The parser previously refused
    ALL filter keyword arguments; it now carries them for `tojson`
    only, and every other filter still refuses them by name.
    Positional arguments, a boolean `indent`, unknown keywords,
    and `tojson` over an undefined or builtin value refuse by name.
  - Nesting is depth-capped (512) with a named refusal, and the
    encoded output charges the render's output budget, like
    `replace` and `join` already do.
- `Params.Tools []Tool` binds the `tools` value
  (`Tool{Type, Function ToolFunction{Name, Description,
  Parameters map[string]any}}`). A definition with an empty `Type`
  binds as the function object itself, matching how the template
  unwraps either shape. When no tools are given the name stays
  UNDEFINED (not an empty list), so `tools is defined` and
  `{% if tools %}` answer exactly as before for tool-less requests.
- The test that pinned the `tojson` refusal is replaced by
  `TestToJSONFilter` (eleven pinned encodings) and
  `TestToJSONFilterRefusals` (five named refusals).
- The coverage claim is corrected in both places it was made (the
  extended-subset section header and the user-only golden's
  comment), and the golden coverage is extended, below.

### Two latent bugs the tool paths exposed once they could run

Both were invisible while the paths were unreachable; both are in
this lane because the golden below cannot render without them.

1. **String + filtered value.** The fixture's recipients line is
   `'# Valid recipients: ' + rns.recipients | join(', ') + '.'`.
   Jinja binds the filter tighter than `+`; this parser's `add`
   operand is a whole conditional expression, so the filter node
   landed as the tree top and the left add evaluated to undefined,
   refusing. The `add` evaluation now completes exactly that shape:
   when the right operand's evaluated value is undefined AND its
   tree root is a bare filter application over a name/attribute/
   index chain, that filter node is evaluated. This is sound
   because filters are pure. The deeper parse difference stays:
   a filter whose BASE is itself the concatenation still refuses
   by name, and no shipped shape uses it. Pinned indirectly by
   both pre-existing goldens (the user-only and system goldens
   carry the same recipients line and pass unchanged).
2. **Empty mapping literal.** `{'a': {}}` cannot tokenize: the
   template tokenizer cuts a tag at the first `}}`, which is the
   literal's own close followed by the tag's. Real Jinja lexes
   strings and nesting; this bounded tokenizer does not, and the
   gap is pre-existing for every construct, not specific to this
   lane. The lane's nested-literal test case was rewritten to a
   non-empty inner mapping rather than reworking the tokenizer.

## The dict.items() ordering MEDIUM, and the decision

The panel's MEDIUM (render.go, Category E): `dict.items()` iterated
in sorted-key order while Jinja iterates insertion order, and the
ATEM renderer prints one parameter block per pair, so the order IS
prompt text. The panel notes it "lands the moment that gap is
fixed" — this lane is that moment.

Decision: CARRY INSERTION ORDER; the change was contained.

- `value` gains a `keys []string` field: the iteration order of a
  dict's keys, nil meaning "no recorded order, iterate sorted"
  (dict literals, `namespace()` state — orders authored in the
  template or accumulated as state, which keep their historical
  sorted iteration; templates like the fixture's keep their named
  state in lists anyway).
- A new `orderedDict` constructor records order; message dicts,
  tool-call dicts, tool definitions, and JSON-converted data use
  it. `items()` iterates `orderedKeys()`; `{% set ns.attr %}` on
  an ordered dict appends a genuinely new key to its order (with a
  scope write-back for the grown slice), matching Python dict
  assignment.
- `tojson` with `sort_keys=false` likewise emits recorded order;
  the default still sorts, so ATEM schema dumps are unaffected.

The one order this lane does NOT restore is the HTTP wire order of
`arguments` / `parameters` objects: the API layer decodes them into
`map[string]any`, and `encoding/json` has already discarded key
order before conversion runs. The converter therefore records
sorted order for decoded maps. Consequences, stated plainly:

- Tool DEFINITIONS are unaffected: the fixture serializes them
  through `tojson`, whose default sorts keys, exactly Jinja's
  default. Byte-identical either way.
- Tool CALL arguments print through `items()`, so over the HTTP
  path they print in sorted key order, where a reference Jinja
  render of the same request JSON would print wire order. The
  ATEM protocol parses parameter blocks by name, so this is a
  prompt-text difference, not a protocol difference, but it is a
  real one and it is named here rather than papered over: the
  golden test's argument expectations (`city` before `days`) are
  the sorted order the shipped path produces, and they would hold
  in Jinja only for a request whose wire order was also sorted.
- Restoring wire order end to end requires decoding chat requests
  with an order-preserving JSON decoder and carrying that order
  through the API types. That is a server-layer change with its
  own blast radius (the cache key marshals the request struct);
  it is the immediate follow-up this lane makes reachable, and it
  is deliberately NOT half-done here.

A caller building `Arguments`/`Parameters` in Go code (the
`chattemplate` package used directly) gets true insertion order
only in the sense above: a Go map literal has no order to give, so
direct callers who care must also accept sorted as the recorded
order today. The value model itself is no longer the blocker.

## Field flow, HTTP layer to template

The API types had no place for these fields; they do now.

- `chatRequest` gains `Tools []chatTool` (`"tools"`).
  `chatTool{type, function{name, description, parameters}}`;
  `parameters` decodes into `map[string]any`.
- `chatMessage` gains `name`, `tool_call_id`, `tool_calls`
  (`[]chatToolCall{id, type, function{name, arguments}}`),
  `reasoning_content`, `recipient`, `end_turn` (`*bool`).
- `buildPrompt` now takes the whole `*chatRequest` and converts:
  `convertMessage` maps each wire message field for field into
  `chattemplate.Message` (including each call's id/type/name/
  arguments), and tools map into `chattemplate.Tool` for
  `Params.Tools`.
- `validateChat` relaxes one rule for the new shape: an assistant
  message carrying `tool_calls` may have empty `content` (a pure
  tool-calling turn); content is still required for every other
  message.
- Nothing else in the request path changed: the response cache
  key still marshals the whole request struct, so requests that
  differ in tools or tool calls key differently with no extra
  work (the cache findings themselves belong to a separate lane).

End-to-end pin: `TestChatWithToolsRendersToolNames` posts a full
tool request over HTTP through a configured ATEM-shaped template
and asserts the engine receives the definitions block (through
`tojson`), the call, and the tool output under the name resolved
from `tool_call_id`; `TestChatUnresolvableToolMessageIsNamed`
asserts the orphan tool message returns 502 `chat_template_error`.

## After evidence

New and extended tests (all passing; the failing-before captures
are the executed probes quoted above):

- `TestToJSONFilter` — 11 encodings, including the HTML-safe
  escapes, sorted mapping keys, nesting, and both indent forms.
- `TestToJSONFilterRefusals` — positional indent, boolean indent,
  unknown kwarg, builtin, undefined: all `ErrUnsupported`.
- `TestGlimmerToolMessageWithName` — explicit name renders.
- `TestGlimmerToolMessageResolvedByCallID` — id-resolved name
  renders, and the assistant call renders its arguments.
- `TestGlimmerToolMessageUnresolvableRefuses` — orphan tool
  message returns `ErrUnresolvableTool`.
- `TestGlimmerATEMTemplateWithTools` — the end-to-end golden:
  one tool definition (metadata line through `tojson`, schema
  line with sorted nested keys), user turn, tool call (ATEM form,
  argument blocks in sorted order as recorded above), tool result
  under its resolved name, plain assistant reply, generation
  prompt. Byte-exact.
- The two pre-existing Glimmer goldens and the full pre-existing
  suite pass unchanged apart from the two test adjustments
  recorded under Finding 1 and the removed `tojson` refusal pin.

## Gates (lane tree, as observed)

- `go build ./...` — clean.
- `go test ./...` — all 9 packages ok; 115 `--- PASS` lines
  (107 at base; +6 chattemplate tests, +2 server tests).
- `go test -race ./internal/chattemplate/ ./internal/server/` —
  both ok.
- `go vet ./...` — clean. `gofmt -l .` — empty (gofumpt via
  golangci-lint also clean).
- `golangci-lint run ./...` — 0 issues (gosec enabled; no new
  dependency added, so no new govulncheck surface).
- `cargo check --workspace` from the repo root — clean; the Rust
  tree is untouched by this lane.

## Deliberately left

- Wire-order preservation for `arguments`/`parameters` through
  the HTTP decode (named above as the follow-up this lane makes
  reachable).
- Content-part lists: the fixture's `render_content` also accepts
  a LIST of typed parts (`text`/`image`/`video`) in place of a
  string `content`. `Message.Content` remains a string in this
  lane; a request carrying part lists still cannot express them,
  and llm servers that send them get the string path only. The
  panel did not flag it; it is recorded here because the fixture
  reads it and a future lane widening `Content` should know the
  template side already handles the shape.
- The server.go template-fetch and cache MEDIUMs of Category E
  (a separate lane's scope, untouched here).
- The tokenizer's `}}`-inside-a-tag limit (named above); a
  string- and nesting-aware tag lexer is its own change.
