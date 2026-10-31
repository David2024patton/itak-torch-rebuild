// Package chattemplate renders a model's own GGUF chat template
// into the prompt text the engine receives (TODO T-9.16).
//
// GGUF chat templates are Jinja2. This renderer implements the
// bounded subset the shipped templates actually use, and refuses
// everything else by name instead of rendering it wrong:
//
//   - literal text, {# comments #}, and {{ }} print expressions:
//     string and number literals, list literals, names, attribute
//     and subscript access (message['role'], message.role,
//     loop.last), concatenation and numeric addition with +,
//     comparisons (==, !=, in, not in), boolean logic (and, or,
//     not) with Jinja's value-returning short-circuit, the postfix
//     tests (is defined / none / string / boolean / mapping /
//     iterable), inline conditionals (a if cond else b), method
//     calls dict.get / str.split / dict.items, macro calls, the
//     namespace() builtin, the strftime_now(format) builtin the
//     reference template environment provides from the host clock
//     (llama.cpp's renderer defines it; templates test it with
//     `is defined` before calling it), and the filters trim,
//     lower, replace and join;
//   - {% for x in expr %} ... {% endfor %} over lists, with the
//     loop variable bound per iteration and loop.index / index0 /
//     first / last / length available; tuple targets (for k, v in
//     expr) bind pair items;
//   - {% if %} / {% elif %} / {% else %} / {% endif %};
//   - {% set name = expr %} and {% set ns.attr = expr %}, the
//     namespace-mutation idiom templates use for cross-iteration
//     state;
//   - {% macro name(args) %} ... {% endmacro %}, registered when
//     the definition renders and callable from print expressions;
//   - whitespace control ({{- -}}, {%- -%}, {#- -#}) on all tags.
//
// Anything outside the subset (other statements, other filters
// such as length or tojson, unknown method calls, printing an
// undefined name or a boolean) fails with ErrUnsupported naming
// the construct, and a call like raise_exception('...') refuses
// by name when the branch containing it actually renders. A chat
// request whose template cannot be rendered is a named platform
// error, never a silently flattened prompt: falling back is a
// decision the caller makes with the error in hand, not something
// this package hides.
//
// The template is attacker-controlled input (it arrives inside a
// model file's metadata), so rendering also runs under hard
// resource budgets, counted in the render state and shared by
// nested macros, loops and branches: a maximum macro call depth,
// a maximum count of evaluated nodes and loop iterations, and a
// maximum output size. A template that exhausts a budget fails
// with ErrBudget naming the limit, never with a stack overflow,
// a panic or unbounded growth: without the budgets, a
// self-recursive macro is a one-line kill switch for the serving
// process, whose fatal stack error no handler can recover.
// Legitimate templates, including this model's own ATEM
// template, render far below every budget.
//
// The implementation is split across files by stage: tokenize.go
// (the template tokenizer), parse.go (statement structure),
// values.go (the value model), expr.go (expression parsing), and
// render.go (evaluation). This file holds the package contract:
// the error values, the budgets, and the Render entry point.
package chattemplate

import (
	"errors"
	"strings"
	"time"
)

// ErrUnsupported names a template construct outside the rendered
// subset. The message carries the construct itself.
var ErrUnsupported = errors.New("chattemplate: unsupported construct")

// ErrBudget names a render that exhausted one of the resource
// budgets below. Refusing is the designed outcome for a hostile
// or pathological template: a caller maps it to the same named
// chat_template_error as any other refusal this package returns.
var ErrBudget = errors.New("chattemplate: render budget exhausted")

// The render budgets. These are safety limits, not operator
// configuration: they bound what any single template can make
// the process do, and they sit orders of magnitude above what a
// legitimate chat template needs (the model's own ATEM template
// renders a long conversation in a few thousand steps and a few
// tens of kilobytes). The depth cap alone does not bound work:
// a macro that calls two macros that each call two more stays
// shallow while its node count and output double per level, so
// the step and output budgets are the ones that stop fan-out.
const (
	// maxMacroDepth bounds nested macro calls. Jinja itself
	// leans on the interpreter recursion limit for this; here
	// the cap is explicit so recursion fails by name.
	maxMacroDepth = 64
	// maxRenderSteps bounds the total nodes rendered and loop
	// iterations executed in one Render call, across all
	// nesting. Each loop iteration over real messages costs a
	// handful of steps, so this still allows prompts built
	// from tens of thousands of messages.
	maxRenderSteps = 1 << 20
	// maxRenderOutput bounds the bytes a render may produce,
	// including text accumulated inside macro calls before it
	// reaches the print site. Real prompts are message text
	// plus template scaffolding; 16 MiB is far past any
	// legitimate render.
	maxRenderOutput = 16 << 20
)

// Message is one chat message, as templates see it.
type Message struct {
	Role    string
	Content string
}

// Params are the render-time values templates reference besides the
// messages: the special tokens from the model's file and whether
// the prompt should end ready for the assistant's turn.
type Params struct {
	BOSToken            string
	EOSToken            string
	AddGenerationPrompt bool
	// Now is the clock the strftime_now builtin reads. The zero
	// value means the current time; tests pin it so a template
	// that prints the date renders byte-exact expectations.
	Now time.Time
}

// Render builds the prompt. It is a pure function of its inputs.
func Render(template string, messages []Message, p Params) (string, error) {
	toks, err := tokenize(template)
	if err != nil {
		return "", err
	}
	nodes, err := parse(toks)
	if err != nil {
		return "", err
	}
	global := &scope{vars: map[string]value{}}
	global.vars["bos_token"] = strValue(p.BOSToken)
	global.vars["eos_token"] = strValue(p.EOSToken)
	global.vars["add_generation_prompt"] = boolValue(p.AddGenerationPrompt)
	msgVals := make([]value, len(messages))
	for i, m := range messages {
		msgVals[i] = dictValue(map[string]value{
			"role":    strValue(m.Role),
			"content": strValue(m.Content),
		})
	}
	global.vars["messages"] = listValue(msgVals)
	// The reference environment (llama.cpp's minja runtime)
	// defines strftime_now as a global function; templates like
	// this model's own test `strftime_now is defined` and call it
	// to print the current date. Registering it is what makes our
	// render of the Glimmer template carry the same "Current
	// date" line llama.cpp produces (TODO T-9.16; the omission
	// was caught by the T-9.17 prompt-id comparison).
	global.vars["strftime_now"] = builtinValue("strftime_now")
	now := p.Now
	if now.IsZero() {
		now = time.Now()
	}
	env := &env{global: global, cur: global, macros: map[string]*macroNode{}, now: now, budget: &renderBudget{}}
	var sb strings.Builder
	if err := renderNodes(nodes, env, &sb); err != nil {
		return "", err
	}
	return sb.String(), nil
}
