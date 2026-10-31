package chattemplate_test

import (
	"errors"
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
		{Role: "tool", Content: "t"},
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
		"tojson filter":  `{{ messages | tojson }}`,
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
	_, err := chattemplate.Render(tmpl,
		[]chattemplate.Message{{Role: "tool", Content: "x"}}, chattemplate.Params{})
	if !errors.Is(err, chattemplate.ErrUnsupported) {
		t.Fatalf("tool branch err = %v, want ErrUnsupported", err)
	}
}

// ---- the extended subset (TODO T-9.17 fix) -------------------------
//
// The Glimmer file's own ATEM template uses macros, namespace
// state, set, defined tests and the join/replace/lower filters.
// These tests pin the constructs that template's taken paths need;
// the golden test below renders the file's real template whole.

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

var glimmerParams = chattemplate.Params{
	BOSToken:            "<|begin_of_text|>",
	EOSToken:            "<|end_of_text|>",
	AddGenerationPrompt: true,
	// Pinned so the template's strftime_now branch renders a
	// byte-exact date line (the branch llama.cpp takes through
	// its own environment's host-clock function; T-9.17 caught
	// the missing builtin by prompt-id comparison).
	Now: time.Date(2026, 10, 31, 12, 0, 0, 0, time.UTC),
}

// The file's real template, one user message: the default system
// block renders (no system message given), with the cutoff, the
// current-date line the template synthesizes from strftime_now,
// and the reasoning line, plus the valid recipients list joined
// from namespace state.
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
