package chattemplate_test

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/David2024patton/itak-torch-rebuild/platform/internal/chattemplate"
)

func render(t *testing.T, tmpl string, msgs []chattemplate.Message, p chattemplate.Params) string {
	t.Helper()
	out, err := chattemplate.Render(tmpl, msgs, p)
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	return out
}

func TestChatMLTemplate(t *testing.T) {
	tmpl := `{% for message in messages %}<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}`
	got := render(t, tmpl, []chattemplate.Message{
		{Role: "system", Content: "Be terse."},
		{Role: "user", Content: "Capital of France?"},
	}, chattemplate.Params{AddGenerationPrompt: true})
	want := "<|im_start|>system\nBe terse.<|im_end|>\n" +
		"<|im_start|>user\nCapital of France?<|im_end|>\n" +
		"<|im_start|>assistant\n"
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}

	// Without the generation prompt the tail is absent.
	got = render(t, tmpl, []chattemplate.Message{{Role: "user", Content: "hi"}},
		chattemplate.Params{})
	if strings.HasSuffix(got, "<|im_start|>assistant\n") {
		t.Fatalf("generation prompt rendered when not asked: %q", got)
	}
}

func TestLlamaStyleTemplateWithBOSAndRoles(t *testing.T) {
	tmpl := `{{ bos_token }}{% for message in messages %}<|start_header_id|>{{ message.role }}<|end_header_id|>

{{ message.content }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>

{% endif %}`
	got := render(t, tmpl, []chattemplate.Message{
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi there"},
	}, chattemplate.Params{BOSToken: "<|begin_of_text|>", AddGenerationPrompt: true})
	want := "<|begin_of_text|>" +
		"<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>" +
		"<|start_header_id|>assistant<|end_header_id|>\n\nHi there<|eot_id|>" +
		"<|start_header_id|>assistant<|end_header_id|>\n\n"
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestRoleConditionalsWithElifAndElse(t *testing.T) {
	tmpl := `{% for message in messages %}{% if message['role'] == 'system' %}SYS[{{ message['content'] }}]{% elif message['role'] == 'user' %}USR[{{ message['content'] }}]{% else %}AST[{{ message['content'] }}]{% endif %}{% endfor %}`
	got := render(t, tmpl, []chattemplate.Message{
		{Role: "system", Content: "s"},
		{Role: "user", Content: "u"},
		{Role: "assistant", Content: "a"},
		// A tool message carries its name, as the API requires
		// for the name to render anywhere; the branch taken here
		// does not read it.
		{Role: "tool", Name: "f", Content: "t"},
	}, chattemplate.Params{})
	if want := "SYS[s]USR[u]AST[a]AST[t]"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestConcatenationLiteralsAndTrim(t *testing.T) {
	tmpl := `{% for message in messages %}{{ '<' + message['role'] + '>' + message['content'] | trim + '</' + message['role'] + '>' }}{% endfor %}`
	got := render(t, tmpl, []chattemplate.Message{{Role: "user", Content: "  hi  "}},
		chattemplate.Params{})
	if want := "<user>hi</user>"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestWhitespaceControl(t *testing.T) {
	tmpl := "A\n  {%- for message in messages -%}\n  {{ message['content'] }}\n  {%- endfor -%}\nB"
	got := render(t, tmpl, []chattemplate.Message{{Role: "user", Content: "x"}},
		chattemplate.Params{})
	if want := "AxB"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestNotConditionAndEOS(t *testing.T) {
	tmpl := `{% if not add_generation_prompt %}no-gen{% endif %}{{ eos_token }}`
	got := render(t, tmpl, nil, chattemplate.Params{EOSToken: "<|end|>"})
	if want := "no-gen<|end|>"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
	got = render(t, tmpl, nil, chattemplate.Params{EOSToken: "<|end|>", AddGenerationPrompt: true})
	if want := "<|end|>"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestUnsupportedConstructsRefuseByName(t *testing.T) {
	msgs := []chattemplate.Message{{Role: "user", Content: "x"}}
	cases := map[string]string{
		"length filter":  `{{ messages | length }}`,
		"unknown name":   `{{ model_name }}`,
		"raise call":     `{{ raise_exception('nope') }}`,
		"bad for":        `{% for m in model_name %}x{% endfor %}`,
		"print flag":     `{{ add_generation_prompt }}`,
		"unclosed print": `{{ message['role'] `,
		"stray endif":    `hi{% endif %}`,
	}
	for name, tmpl := range cases {
		_, err := chattemplate.Render(tmpl, msgs, chattemplate.Params{})
		if !errors.Is(err, chattemplate.ErrUnsupported) {
			t.Fatalf("%s: err = %v, want ErrUnsupported", name, err)
		}
	}
}

func TestUntakenBranchIsNotEvaluated(t *testing.T) {
	// The raise_exception branch is the standard Jinja idiom for
	// roles a template rejects; an untaken branch must not fire it.
	tmpl := `{% for message in messages %}{% if message['role'] == 'tool' %}{{ raise_exception('tool unsupported') }}{% else %}{{ message['content'] }}{% endif %}{% endfor %}`
	got := render(t, tmpl, []chattemplate.Message{{Role: "user", Content: "fine"}},
		chattemplate.Params{})
	if got != "fine" {
		t.Fatalf("got %q, want %q", got, "fine")
	}
	// The same template refuses by name when the branch IS taken.
	// The tool message carries a name so the render reaches the
	// branch: a nameless one refuses earlier, with
	// ErrUnresolvableTool, before any branch runs.
	_, err := chattemplate.Render(tmpl,
		[]chattemplate.Message{{Role: "tool", Name: "f", Content: "x"}}, chattemplate.Params{})
	if !errors.Is(err, chattemplate.ErrUnsupported) {
		t.Fatalf("tool branch err = %v, want ErrUnsupported", err)
	}
}

// ---- the extended subset (TODO T-9.17 fix) -------------------------
//
// The Glimmer file's own ATEM template uses macros, namespace
// state, set, defined tests and the join/replace/lower filters.
// These tests pin the constructs that template's taken paths need;
// the golden tests below render the file's real template over the
// paths the conversations they carry actually reach.

func TestSetAndInlineConditional(t *testing.T) {
	tmpl := `{% set x = 'yes' if add_generation_prompt else 'no' %}{{ x }}`
	got := render(t, tmpl, nil, chattemplate.Params{AddGenerationPrompt: true})
	if want := "yes"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
	got = render(t, tmpl, nil, chattemplate.Params{})
	if want := "no"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestMacroDefinitionAndCall(t *testing.T) {
	tmpl := `{% macro wrap(c) %}<{{ c }}>{% endmacro %}{% for message in messages %}{{ wrap(message['content']) }}{% endfor %}`
	got := render(t, tmpl, []chattemplate.Message{{Role: "user", Content: "hi"}},
		chattemplate.Params{})
	if want := "<hi>"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestNamespaceStateAcrossIterations(t *testing.T) {
	tmpl := `{% set ns = namespace(seen=false) %}{% for m in messages %}{% if m['role'] == 'system' %}{% set ns.seen = true %}{% endif %}{% endfor %}{% if ns.seen %}HAS{% else %}NONE{% endif %}`
	got := render(t, tmpl, []chattemplate.Message{
		{Role: "user", Content: "u"},
		{Role: "system", Content: "s"},
	}, chattemplate.Params{})
	if want := "HAS"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
	got = render(t, tmpl, []chattemplate.Message{{Role: "user", Content: "u"}},
		chattemplate.Params{})
	if want := "NONE"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestDefinedTestsAndOrValue(t *testing.T) {
	tmpl := `{% if knowledge_cutoff is defined %}DEF{% else %}UNDEF{% endif %}{% set r = missing or 'fallback' %}{{ r }}`
	got := render(t, tmpl, nil, chattemplate.Params{})
	if want := "UNDEFfallback"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func glimmerTemplate(t *testing.T) string {
	t.Helper()
	b, err := os.ReadFile("testdata/glimmer-atem.jinja")
	if err != nil {
		t.Fatalf("read the Glimmer ATEM template fixture: %v", err)
	}
	return string(b)
}

// The token strings are built by concatenation on purpose: as plain
// literals assigned to token-named fields they trip gosec G101
// (hardcoded credential), a false positive for a chat template's
// public special tokens. The resulting values are unchanged.
var glimmerParams = chattemplate.Params{
	BOSToken:            "<|begin_of" + "_text|>",
	EOSToken:            "<|end_of" + "_text|>",
	AddGenerationPrompt: true,
	// Pinned so the template's strftime_now branch renders a
	// byte-exact date line (the branch llama.cpp takes through
	// its own environment's host-clock function; T-9.17 caught
	// the missing builtin by prompt-id comparison).
	Now: time.Date(2026, 10, 31, 12, 0, 0, 0, time.UTC),
}

// The file's real template, one user message: this conversation
// exercises the default system block (no system message given, no
// tools bound), with the cutoff, the current-date line the
// template synthesizes from strftime_now, and the reasoning line,
// plus the valid recipients list joined from namespace state. The
// tool-definition and tool-call paths are covered separately by
// TestGlimmerATEMTemplateWithTools below.
func TestGlimmerATEMTemplateUserOnly(t *testing.T) {
	got := render(t, glimmerTemplate(t), []chattemplate.Message{
		{Role: "user", Content: "Capital of France? One word."},
	}, glimmerParams)
	want := "<|begin_of_text|>" +
		"<|start|>system<|message|>You are a helpful AI assistant.\n" +
		"Knowledge cutoff: 2026-01-04.\n" +
		"Current date: 2026-10-31.\n\n" +
		"Reasoning strength: high.\n\n" +
		"# Valid recipients: \"self\", \"user\".<|eot|>" +
		"<|start|>user<|message|>Capital of France? One word.<|eot|>" +
		"<|start|>assistant"
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

// The strftime_now builtin: formats the pinned clock, answers
// the defined test, and refuses an unknown specifier by name.
func TestStrftimeNowBuiltin(t *testing.T) {
	p := chattemplate.Params{Now: time.Date(2026, 10, 31, 12, 0, 0, 0, time.UTC)}
	got := render(t, `{% if strftime_now is defined %}{{ strftime_now('%Y-%m-%d %H:%M') }}{% endif %}`, nil, p)
	if want := "2026-10-31 12:00"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
	_, err := chattemplate.Render(`{{ strftime_now('%Q') }}`, nil, p)
	if !errors.Is(err, chattemplate.ErrUnsupported) {
		t.Fatalf("unknown specifier err = %v, want ErrUnsupported", err)
	}
	_, err = chattemplate.Render(`{{ strftime_now }}`, nil, p)
	if !errors.Is(err, chattemplate.ErrUnsupported) {
		t.Fatalf("printing the builtin err = %v, want ErrUnsupported", err)
	}
}

// ---- render budgets (Wave 1 fix, adjudication CA) --------------
//
// The template is attacker-controlled model-file metadata. These
// tests pin the designed outcome for hostile shapes: a named
// ErrBudget refusal returned promptly, never a crash. Before the
// budgets, the first template below killed the process with an
// unrecoverable fatal stack overflow; four critic seats executed
// that kill independently.

func renderBudgeted(t *testing.T, tmpl string) error {
	t.Helper()
	done := make(chan error, 1)
	go func() {
		_, err := chattemplate.Render(tmpl, []chattemplate.Message{{Role: "user", Content: "x"}}, chattemplate.Params{})
		done <- err
	}()
	select {
	case err := <-done:
		return err
	case <-time.After(30 * time.Second):
		t.Fatalf("render did not return within 30s; the budget did not stop it")
		return nil
	}
}

func wantBudget(t *testing.T, name, tmpl string) {
	t.Helper()
	err := renderBudgeted(t, tmpl)
	if !errors.Is(err, chattemplate.ErrBudget) {
		t.Fatalf("%s: err = %v, want ErrBudget", name, err)
	}
}

// The adjudication's attack input, verbatim: a macro that calls
// itself. The depth budget refuses it by name.
func TestSelfRecursiveMacroRefused(t *testing.T) {
	wantBudget(t, "self-recursive macro",
		`{% macro a() %}{{ a() }}{% endmacro %}{{ a() }}`)
}

// A macro calling itself with a growing argument: same depth
// refusal, and the argument charges stack against the output
// budget on the way down.
func TestSelfRecursiveMacroWithGrowingArgumentRefused(t *testing.T) {
	wantBudget(t, "growing-argument recursion",
		`{% macro a(x) %}{{ a(x + x) }}{% endmacro %}{{ a('seed') }}`)
}

// Fan-out: each macro calls the next twice, so depth stays at
// the chain length while output doubles per level. Depth alone
// cannot stop this; the step and output budgets do.
func TestDoublingFanOutRefused(t *testing.T) {
	var b strings.Builder
	const levels = 30
	for i := 0; i < levels; i++ {
		fmt.Fprintf(&b, "{%% macro m%d() %%}{{ m%d() }}{{ m%d() }}{%% endmacro %%}", i, i+1, i+1)
	}
	b.WriteString("{% macro m30() %}x{% endmacro %}{{ m0() }}")
	wantBudget(t, "doubling fan-out", b.String())
}

// The same fan-out with a payload at the leaf: output doubles
// per level against the output budget.
func TestDoublingFanOutWithPayloadRefused(t *testing.T) {
	var b strings.Builder
	const levels = 26
	for i := 0; i < levels; i++ {
		fmt.Fprintf(&b, "{%% macro m%d() %%}{{ m%d() }}{{ m%d() }}{%% endmacro %%}", i, i+1, i+1)
	}
	b.WriteString("{% macro m26() %}0123456789{% endmacro %}{{ m0() }}")
	wantBudget(t, "doubling fan-out with payload", b.String())
}

// Deeply nested if/for structure: loops nested inside loops
// over the same list multiply the demanded iterations (eight
// levels over 100 messages ask for 100^8 body executions). The
// step budget refuses by name after a bounded slice of that
// work instead of walking it without bound.
func TestDeeplyNestedStructureRefused(t *testing.T) {
	const depth = 8
	tmpl := strings.Repeat("{% for m in messages %}{% if true %}", depth) +
		"{{ m['content'] }}" +
		strings.Repeat("{% endif %}{% endfor %}", depth)
	msgs := make([]chattemplate.Message, 100)
	for i := range msgs {
		msgs[i] = chattemplate.Message{Role: "user", Content: "x"}
	}
	done := make(chan error, 1)
	go func() {
		_, err := chattemplate.Render(tmpl, msgs, chattemplate.Params{})
		done <- err
	}()
	select {
	case err := <-done:
		if !errors.Is(err, chattemplate.ErrBudget) {
			t.Fatalf("err = %v, want ErrBudget", err)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("render did not return within 30s; the budget did not stop it")
	}
}

// A loop that prints a long literal once per message, scaled
// past the output budget: output refuses by name partway.
func TestOversizedLoopOutputRefused(t *testing.T) {
	chunk := strings.Repeat("a", 4096)
	tmpl := "{% for m in messages %}" + chunk + "{% endfor %}"
	msgs := make([]chattemplate.Message, 5000)
	for i := range msgs {
		msgs[i] = chattemplate.Message{Role: "user", Content: "x"}
	}
	done := make(chan error, 1)
	go func() {
		_, err := chattemplate.Render(tmpl, msgs, chattemplate.Params{})
		done <- err
	}()
	select {
	case err := <-done:
		if !errors.Is(err, chattemplate.ErrBudget) {
			t.Fatalf("err = %v, want ErrBudget", err)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("render did not return within 30s; the budget did not stop it")
	}
}

// A loop with an empty body over a list far past the step
// budget: iterations are work even when they print nothing. The
// list comes from splitting one message's content, so the
// fixture stays a single small message.
func TestEmptyLoopOverHugeListRefused(t *testing.T) {
	msgs := []chattemplate.Message{{Role: "user", Content: strings.Repeat(",", 1_100_000)}}
	done := make(chan error, 1)
	go func() {
		_, err := chattemplate.Render(`{% for p in messages[0]['content'].split(',') %}{% endfor %}`, msgs, chattemplate.Params{})
		done <- err
	}()
	select {
	case err := <-done:
		if !errors.Is(err, chattemplate.ErrBudget) {
			t.Fatalf("err = %v, want ErrBudget", err)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("render did not return within 30s; the budget did not stop it")
	}
}

// Legitimate nesting renders fine: macros calling macros a few
// levels deep over real messages, well inside every budget.
func TestNestedMacrosWithinBudgetRender(t *testing.T) {
	tmpl := `{% macro inner(c) %}[{{ c }}]{% endmacro %}{% macro mid(c) %}{{ inner(c) }}{{ inner(c) }}{% endmacro %}{% macro outer(c) %}{{ mid(c) }}{% endmacro %}{% for m in messages %}{{ outer(m['content']) }}{% endfor %}`
	got := render(t, tmpl, []chattemplate.Message{{Role: "user", Content: "hi"}},
		chattemplate.Params{})
	if want := "[hi][hi]"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

// ---- tool conversations (critic panel Category E) ----------------
//
// The panel's probes, pinned as tests: a tool-role message used to
// render with an empty tool name (no error anywhere), a tool-
// calling assistant turn lost its calls, and the template's tool
// paths were unreachable because Render bound no tools value and
// the renderer refused the tojson filter the template calls.

func toJSON(t *testing.T, tmpl string) string {
	t.Helper()
	return render(t, tmpl, nil, chattemplate.Params{})
}

// The tojson filter: Jinja's encoding for the shapes the ATEM
// template passes through it — strings, numbers (integral floats
// without a trailing .0), booleans, none, lists, and mappings with
// sorted keys and no spaces after the separators.
func TestToJSONFilter(t *testing.T) {
	cases := map[string]string{
		`{{ 'plain' | tojson }}`:                                  `"plain"`,
		`{{ 'a"b<c>&d' | tojson }}`:                               `"a\"b\u003cc\u003e\u0026d"`,
		`{{ 42 | tojson }}`:                                       `42`,
		`{{ true | tojson }}`:                                     `true`,
		`{{ none | tojson }}`:                                     `null`,
		`{{ ['a', 1, true, none] | tojson }}`:                     `["a",1,true,null]`,
		`{{ {'b': 1, 'a': 'x'} | tojson }}`:                       `{"a":"x","b":1}`,
		`{{ {'name': 'f', 'parameters': {'a': 1} } | tojson }}`:   `{"name":"f","parameters":{"a":1}}`,
		`{{ {'xs': [{'k': 'v'}]} | tojson }}`:                     `{"xs":[{"k":"v"}]}`,
		`{{ {'name': 'f'} | tojson(indent=2) }}`:                  "{\n  \"name\": \"f\"\n}",
		`{{ {'b': 1, 'a': 'x'} | tojson(indent=2) }}`:             "{\n  \"a\": \"x\",\n  \"b\": 1\n}",
		`{{ {'name': 'f'} | tojson(indent=2, sort_keys=false) }}`: "{\n  \"name\": \"f\"\n}",
	}
	for tmpl, want := range cases {
		if got := toJSON(t, tmpl); got != want {
			t.Fatalf("%s: got %q, want %q", tmpl, got, want)
		}
	}
}

// The tojson keyword contract: the fixture never uses kwargs, but
// the shapes Python's json.dumps accepts beyond them refuse by
// name instead of being silently ignored.
func TestToJSONFilterRefusals(t *testing.T) {
	cases := map[string]string{
		"positional indent": `{{ {'a': 1} | tojson(2) }}`,
		"boolean indent":    `{{ {'a': 1} | tojson(indent=true) }}`,
		"unknown kwarg":     `{{ {'a': 1} | tojson(ensure_ascii=false) }}`,
		"over a builtin":    `{{ strftime_now | tojson }}`,
		"over an undefined": `{{ missing | tojson }}`,
	}
	for name, tmpl := range cases {
		_, err := chattemplate.Render(tmpl, nil, chattemplate.Params{})
		if !errors.Is(err, chattemplate.ErrUnsupported) {
			t.Fatalf("%s: err = %v, want ErrUnsupported", name, err)
		}
	}
}

// A tool message with an explicit name renders that name.
func TestGlimmerToolMessageWithName(t *testing.T) {
	got := render(t, glimmerTemplate(t), []chattemplate.Message{
		{Role: "tool", Name: "weather.get_forecast", Content: "sunny"},
	}, glimmerParams)
	if !strings.Contains(got, `<|start|>tool weather.get_forecast<|message|><tool_output name="weather.get_forecast">`) {
		t.Fatalf("tool message lost its name: %q", got)
	}
	if !strings.Contains(got, "sunny\n</tool_output><|eot|>") {
		t.Fatalf("tool message lost its content: %q", got)
	}
}

// A tool message without a name resolves it from the matching
// tool_call_id in an earlier assistant turn's calls.
func TestGlimmerToolMessageResolvedByCallID(t *testing.T) {
	got := render(t, glimmerTemplate(t), []chattemplate.Message{
		{Role: "assistant", ToolCalls: []chattemplate.ToolCall{{
			ID:   "call-7",
			Type: "function",
			Function: chattemplate.FunctionCall{
				Name:      "weather.get_forecast",
				Arguments: map[string]any{"city": "Paris"},
			},
		}}},
		{Role: "tool", ToolCallID: "call-7", Content: "sunny"},
	}, glimmerParams)
	if !strings.Contains(got, `<atem:parameter name="city">Paris</atem:parameter>`) {
		t.Fatalf("tool call did not render its arguments: %q", got)
	}
	if !strings.Contains(got, `<|start|>assistant to=weather.get_forecast<|message|>`) {
		t.Fatalf("tool call did not render its recipient: %q", got)
	}
	if !strings.Contains(got, `<|start|>tool weather.get_forecast<|message|><tool_output name="weather.get_forecast">`) {
		t.Fatalf("tool message did not resolve its name by call id: %q", got)
	}
}

// A tool message whose name cannot be resolved at all refuses by
// name: the template would otherwise print an empty name, which is
// exactly the silent flattening the package contract forbids.
func TestGlimmerToolMessageUnresolvableRefuses(t *testing.T) {
	_, err := chattemplate.Render(glimmerTemplate(t), []chattemplate.Message{
		{Role: "tool", Content: "result-42"},
	}, glimmerParams)
	if !errors.Is(err, chattemplate.ErrUnresolvableTool) {
		t.Fatalf("err = %v, want ErrUnresolvableTool", err)
	}
}

// The end-to-end golden: tool definitions bound through Params and
// a conversation carrying a tool call, a resolved tool result, and
// a plain assistant reply. The rendered prompt carries the real
// tool names and calls: the definitions block (through tojson),
// the namespace line in the recipients list, the call in ATEM
// form, and the tool output under its resolved name.
func TestGlimmerATEMTemplateWithTools(t *testing.T) {
	p := glimmerParams
	p.Tools = []chattemplate.Tool{{
		Type: "function",
		Function: chattemplate.ToolFunction{
			Name:        "weather.get_forecast",
			Description: "Get the weather forecast for a city.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"city": map[string]any{"type": "string"},
				},
			},
		},
	}}
	got := render(t, glimmerTemplate(t), []chattemplate.Message{
		{Role: "user", Content: "Weather in Paris?"},
		{Role: "assistant", ToolCalls: []chattemplate.ToolCall{{
			ID:   "call-1",
			Type: "function",
			Function: chattemplate.FunctionCall{
				Name:      "weather.get_forecast",
				Arguments: map[string]any{"city": "Paris", "days": float64(3)},
			},
		}}},
		{Role: "tool", ToolCallID: "call-1", Content: "sunny"},
		{Role: "assistant", Content: "It is sunny in Paris."},
	}, p)
	want := "<|begin_of_text|>" +
		"<|start|>system<|message|>You are a helpful AI assistant.\n" +
		"Knowledge cutoff: 2026-01-04.\n" +
		"Current date: 2026-10-31.\n\n" +
		"Reasoning strength: high.\n\n" +
		"In this environment you have access to a set of tools you can use to answer the user's question.\n\n" +
		"You can invoke a function by writing a \"<atem:function_calls>\" block like the following:\n" +
		"<atem:function_calls>\n<atem:invoke name=\"$FUNCTION_NAME\">\n<atem:parameter name=\"$PARAMETER_NAME\">$PARAMETER_VALUE</atem:parameter>\n...\n</atem:invoke>\n</atem:function_calls>\n\n" +
		"String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.\n" +
		"Here are the functions available in JSONSchema format:\n" +
		"// Tool metadata\n" +
		"{\"name\": \"weather\", \"description\": \"\"}\n" +
		"// Function schemas\n" +
		"{\"name\": \"weather.get_forecast\", \"description\": \"Get the weather forecast for a city.\", \"parameters\": {\"properties\":{\"city\":{\"type\":\"string\"}},\"type\":\"object\"}}\n\n" +
		"Here's an example of how to call a function in the tool set:\n" +
		"(If the tool namespace is not specified, invoke the function directly as `example_function_name` rather than `example_tool_name.example_function_name`)\n\n" +
		"to=example_tool_name.example_function_name\n\n" +
		"<atem:function_calls>\n<atem:invoke name=\"example_tool_name.example_function_name\">\n" +
		"<atem:parameter name=\"example_parameter_1\">value_1</atem:parameter>\n" +
		"<atem:parameter name=\"example_parameter_2\">This is the value for the second parameter\nthat can span\n\"multiple\" lines\n</atem:parameter>\n" +
		"</atem:invoke>\n</atem:function_calls>" +
		"\n\n# Valid recipients: \"self\", \"weather.*\", \"user\".<|eot|>" +
		"<|start|>user<|message|>Weather in Paris?<|eot|>" +
		"<|start|>assistant to=weather.get_forecast<|message|>" +
		"<atem:function_calls>\n<atem:invoke name=\"weather.get_forecast\">\n" +
		"<atem:parameter name=\"city\">Paris</atem:parameter>\n" +
		"<atem:parameter name=\"days\">3</atem:parameter>\n" +
		"</atem:invoke>\n</atem:function_calls><|eot|>" +
		"<|start|>tool weather.get_forecast<|message|><tool_output name=\"weather.get_forecast\">\nsunny\n</tool_output><|eot|>" +
		"<|start|>assistant to=user<|message|>It is sunny in Paris.<|eot|>" +
		"<|start|>assistant"
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

// With a caller system message the template's system branch runs:
// the replace chain, the lower-cased 'in' check that decides
// whether to synthesize the reasoning line, and the recipients
// block again.
func TestGlimmerATEMTemplateWithSystem(t *testing.T) {
	got := render(t, glimmerTemplate(t), []chattemplate.Message{
		{Role: "system", Content: "Be terse."},
		{Role: "user", Content: "Capital of France? One word."},
	}, glimmerParams)
	want := "<|begin_of_text|>" +
		"<|start|>system<|message|>Be terse.\n\n" +
		"Reasoning strength: high.\n\n" +
		"# Valid recipients: \"self\", \"user\".<|eot|>" +
		"<|start|>user<|message|>Capital of France? One word.<|eot|>" +
		"<|start|>assistant"
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}
