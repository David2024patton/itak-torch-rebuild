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
//     lower, replace, join and tojson;
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
// such as length, unknown method calls, printing an undefined
// name or a boolean) fails with ErrUnsupported naming the
// construct, and a call like raise_exception('...') refuses by
// name when the branch containing it actually renders. A chat
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
	"fmt"
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

// ErrUnresolvableTool names a tool-role message whose tool name
// the render could not resolve: the message carries no name, and
// its tool_call_id matches no tool call in the conversation (or
// it carries no id at all). The model's ATEM template prints the
// resolved name into the prompt, so rendering on without one
// would produce a prompt naming an empty tool: the silent
// flattening this package exists to refuse. Callers map it to the
// same named chat_template_error as any other refusal here.
var ErrUnresolvableTool = errors.New("chattemplate: tool message name cannot be resolved")

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

// FunctionCall is one tool invocation an assistant message
// carries, as templates see it: the function's name and its
// arguments as a mapping. The ATEM template refuses a call whose
// arguments are not a mapping (a JSON string cannot be parsed in
// the reference sandbox either), so arguments are structured
// data here, never a string.
type FunctionCall struct {
	Name      string
	Arguments map[string]any
}

// ToolCall is one entry of an assistant message's tool_calls, as
// templates see it: an id (which a later tool message's
// tool_call_id matches to find its name), a type, and the call.
type ToolCall struct {
	ID       string
	Type     string
	Function FunctionCall
}

// Message is one chat message, as templates see it. Beyond role
// and content, a message carries the fields tool-aware templates
// read: a tool message's name and tool_call_id, an assistant
// message's tool calls, its reasoning content, and the recipient
// and end_turn fields the ATEM template's assistant branch
// consults. Name and ToolCallID stay absent (not empty keys) when
// unset, so a template's message.get('name') answers none exactly
// as the reference environment answers for a message without one.
type Message struct {
	Role             string
	Content          string
	Name             string
	ToolCallID       string
	ToolCalls        []ToolCall
	ReasoningContent string
	Recipient        string
	// EndTurn distinguishes the template's end-of-turn token
	// from its end-of-message token for assistant turns. Nil
	// means the message does not say, and the template's own
	// default applies.
	EndTurn *bool
}

// ToolFunction is one function a tool definition offers, as
// templates see it: name, description, and the JSONSchema
// parameter object, carried as structured data because the
// template serializes it with tojson.
type ToolFunction struct {
	Name        string
	Description string
	Parameters  map[string]any
}

// Tool is one tool definition bound for the render: a type and
// the function it offers. A definition given without the
// {"type": "function", "function": {...}} wrapper is carried as a
// Tool whose Function is set and Type is empty; the bound value
// then is the function object itself, matching how the template
// unwraps either shape.
type Tool struct {
	Type     string
	Function ToolFunction
}

// Params are the render-time values templates reference besides the
// messages: the special tokens from the model's file, whether
// the prompt should end ready for the assistant's turn, and the
// tool definitions the request offers the model.
type Params struct {
	BOSToken            string
	EOSToken            string
	AddGenerationPrompt bool
	// Tools are the request's tool definitions, bound as the
	// `tools` value templates test and iterate. Empty means the
	// name stays undefined, so `tools is defined` and a bare
	// `{% if tools %}` answer exactly as they did before tool
	// support: the reference environment only defines the name
	// when the request carries definitions.
	Tools []Tool
	// Now is the clock the strftime_now builtin reads. The zero
	// value means the current time; tests pin it so a template
	// that prints the date renders byte-exact expectations.
	Now time.Time
}

// Render builds the prompt. It is a pure function of its inputs.
func Render(template string, messages []Message, p Params) (string, error) {
	// A tool message's name is part of the prompt the model's
	// ATEM template builds, so a message the template cannot
	// name must not render at all: resolve every tool message's
	// name up front (explicit name, else the function name of
	// the tool call its id matches, mirroring the lookup the
	// template itself performs) and refuse the unresolvable
	// ones by name.
	resolved := make([]string, len(messages))
	for i, m := range messages {
		if m.Role != "tool" {
			continue
		}
		name := m.Name
		if name == "" && m.ToolCallID != "" {
			name = toolNameForCallID(messages, m.ToolCallID)
		}
		if name == "" {
			return "", fmt.Errorf("%w: message %d has no name and no matching tool call", ErrUnresolvableTool, i)
		}
		resolved[i] = name
	}
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
		msgVals[i] = messageValue(m, resolved[i])
	}
	global.vars["messages"] = listValue(msgVals)
	if len(p.Tools) > 0 {
		toolVals := make([]value, len(p.Tools))
		for i, t := range p.Tools {
			toolVals[i] = toolValue(t)
		}
		global.vars["tools"] = listValue(toolVals)
	}
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

// toolNameForCallID finds the function name of the tool call with
// the given id anywhere in the conversation, the same lookup the
// ATEM template performs for a nameless tool message.
func toolNameForCallID(messages []Message, id string) string {
	for _, m := range messages {
		for _, tc := range m.ToolCalls {
			if tc.ID == id {
				return tc.Function.Name
			}
		}
	}
	return ""
}

// messageValue builds the mapping a template sees for one
// message. Keys follow the order reference environments build
// message dicts in (role, content, then the rest as given);
// optional fields are included only when the message carries
// them, so tests like message.get('name') and 'tool_calls' in a
// message answer as they do for a plain message upstream.
func messageValue(m Message, resolvedName string) value {
	pairs := []value{
		strValue("role"), strValue(m.Role),
		strValue("content"), strValue(m.Content),
	}
	if m.Name != "" {
		pairs = append(pairs, strValue("name"), strValue(m.Name))
	} else if resolvedName != "" {
		// A tool message whose name was resolved from its call
		// id: the template re-derives the same name itself, but
		// binding it keeps every reader of the dict consistent.
		pairs = append(pairs, strValue("name"), strValue(resolvedName))
	}
	if m.ToolCallID != "" {
		pairs = append(pairs, strValue("tool_call_id"), strValue(m.ToolCallID))
	}
	if len(m.ToolCalls) > 0 {
		calls := make([]value, len(m.ToolCalls))
		for i, tc := range m.ToolCalls {
			calls[i] = toolCallValue(tc)
		}
		pairs = append(pairs, strValue("tool_calls"), listValue(calls))
	}
	if m.ReasoningContent != "" {
		pairs = append(pairs, strValue("reasoning_content"), strValue(m.ReasoningContent))
	}
	if m.Recipient != "" {
		pairs = append(pairs, strValue("recipient"), strValue(m.Recipient))
	}
	if m.EndTurn != nil {
		pairs = append(pairs, strValue("end_turn"), boolValue(*m.EndTurn))
	}
	return orderedDict(pairs...)
}

// toolCallValue builds the mapping for one tool call: id, type,
// and the function object with its name and argument mapping.
func toolCallValue(tc ToolCall) value {
	return orderedDict(
		strValue("id"), strValue(tc.ID),
		strValue("type"), strValue(tc.Type),
		strValue("function"), orderedDict(
			strValue("name"), strValue(tc.Function.Name),
			strValue("arguments"), jsonDataValue(tc.Function.Arguments),
		),
	)
}

// toolValue builds the value for one tool definition: the
// {"type": ..., "function": {...}} wrapper when the definition
// carries a type, else the function object itself.
func toolValue(t Tool) value {
	fn := orderedDict(
		strValue("name"), strValue(t.Function.Name),
		strValue("description"), strValue(t.Function.Description),
		strValue("parameters"), jsonDataValue(t.Function.Parameters),
	)
	if t.Type == "" {
		return fn
	}
	return orderedDict(
		strValue("type"), strValue(t.Type),
		strValue("function"), fn,
	)
}

// jsonDataValue converts decoded JSON data (the shape
// encoding/json produces into any: maps with string keys,
// slices, strings, float64 numbers, booleans, nil) into the
// renderer's value model, recording each mapping's keys as an
// explicit order. A map[string]any does not retain wire order —
// encoding/json has already discarded it — so the recorded
// order is sorted: deterministic, and the same order tojson's
// default (sorted keys) produces for the same data, which is
// why ATEM parameter blocks print in sorted order today. A
// caller that builds Arguments or Parameters in code and cares
// about Jinja's insertion order must hand the fields in wire
// order; restoring that end to end means decoding with a
// streaming decoder that preserves order, a named gap recorded
// in this lane's bench note, not something this converter can
// invent after the fact. Values outside the JSON shape map to
// none; the converter is fed exclusively from decoded request
// JSON and explicitly built maps, so that arm is defensive only.
func jsonDataValue(data any) value {
	switch d := data.(type) {
	case nil:
		return value{kind: vNone}
	case string:
		return strValue(d)
	case bool:
		return boolValue(d)
	case float64:
		return numValue(d)
	case []any:
		out := make([]value, len(d))
		for i, el := range d {
			out[i] = jsonDataValue(el)
		}
		return listValue(out)
	case map[string]any:
		keys := make([]string, 0, len(d))
		for k := range d {
			keys = append(keys, k)
		}
		sortStrings(keys)
		pairs := make([]value, 0, len(d)*2)
		for _, k := range keys {
			pairs = append(pairs, strValue(k), jsonDataValue(d[k]))
		}
		return orderedDict(pairs...)
	}
	return value{kind: vNone}
}
