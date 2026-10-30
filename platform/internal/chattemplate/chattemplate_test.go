package chattemplate_test

import (
	"errors"
	"strings"
	"testing"

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
		"set statement":  `{% set x = 1 %}hi`,
		"length filter":  `{{ messages | length }}`,
		"unknown name":   `{{ model_name }}`,
		"raise call":     `{{ raise_exception('nope') }}`,
		"bad for":        `{% for m in messages %}x{% endfor %}`,
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
