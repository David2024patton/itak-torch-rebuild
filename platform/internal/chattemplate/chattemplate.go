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
package chattemplate

import (
	"errors"
	"fmt"
	"strconv"
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

// ---- tokenizer -------------------------------------------------------

type tokKind int

const (
	tokText tokKind = iota
	tokPrint
	tokStmt
)

type token struct {
	kind tokKind
	body string // text content, or the tag's inner text (trimmed)
}

func tokenize(tmpl string) ([]token, error) {
	var toks []token
	rest := tmpl
	trimNextLeft := false
	for len(rest) > 0 {
		// The earliest of {{, {% and {# opens the next tag; a
		// comment's body is never lexed, so tag lookalikes inside
		// {# ... #} (the Glimmer template's header comment names
		// the {%- -%} markers) stay inert.
		i, kind := -1, tokText
		for _, cand := range []struct {
			mark string
			kind tokKind
		}{{"{{", tokPrint}, {"{%", tokStmt}, {"{#", tokText}} {
			if j := strings.Index(rest, cand.mark); j >= 0 && (i < 0 || j < i) {
				i, kind = j, cand.kind
			}
		}
		isComment := i >= 0 && strings.HasPrefix(rest[i:], "{#")
		if i < 0 {
			text := rest
			if trimNextLeft {
				text = strings.TrimLeft(text, " \t\r\n")
			}
			if text != "" {
				toks = append(toks, token{tokText, text})
			}
			break
		}
		text := rest[:i]
		if trimNextLeft {
			text = strings.TrimLeft(text, " \t\r\n")
			trimNextLeft = false
		}
		j := i + 2
		trimPrev := false
		if j < len(rest) && rest[j] == '-' {
			trimPrev = true
			j++
		}
		if trimPrev {
			text = strings.TrimRight(text, " \t\r\n")
		}
		if text != "" {
			toks = append(toks, token{tokText, text})
		}
		if isComment {
			end := strings.Index(rest[j:], "#}")
			if end < 0 {
				return nil, fmt.Errorf("%w: unclosed comment", ErrUnsupported)
			}
			inner := rest[j : j+end]
			if strings.HasSuffix(inner, "-") {
				trimNextLeft = true
			}
			rest = rest[j+end+2:]
			continue
		}
		close := "}}"
		if kind == tokStmt {
			close = "%}"
		}
		end := strings.Index(rest[j:], close)
		if end < 0 {
			return nil, fmt.Errorf("%w: unclosed tag at %q", ErrUnsupported, rest[i:min(len(rest), i+16)])
		}
		inner := rest[j : j+end]
		if strings.HasSuffix(inner, "-") {
			trimNextLeft = true
			inner = inner[:len(inner)-1]
		}
		toks = append(toks, token{kind, strings.TrimSpace(inner)})
		rest = rest[j+end+2:]
	}
	return toks, nil
}

// ---- parser ----------------------------------------------------------

type node interface{ nodeTag() }

type textNode struct{ s string }

func (textNode) nodeTag() {}

type printNode struct{ expr *vExpr }

func (printNode) nodeTag() {}

type forNode struct {
	vars []string // one name, or two for pair items
	iter *vExpr
	body []node
}

func (forNode) nodeTag() {}

type branch struct {
	cond *vExpr // nil on the else branch
	body []node
}

type ifNode struct{ branches []branch }

func (ifNode) nodeTag() {}

type setNode struct {
	target []string // a name, or name.attr for a namespace field
	expr   *vExpr
}

func (setNode) nodeTag() {}

type macroNode struct {
	name   string
	params []string
	body   []node
}

func (macroNode) nodeTag() {}

type parser struct {
	toks []token
	pos  int
}

func parse(toks []token) ([]node, error) {
	p := &parser{toks: toks}
	nodes, stop, err := p.parseBlock()
	if err != nil {
		return nil, err
	}
	if stop != "" {
		return nil, fmt.Errorf("%w: unexpected {%% %s %%} without an opening tag", ErrUnsupported, stop)
	}
	return nodes, nil
}

// parseBlock reads nodes until a statement whose keyword is a block
// terminator (elif/else/endif/endfor/endmacro); it returns that
// statement's text as the stop value without consuming it.
func (p *parser) parseBlock() (nodes []node, stop string, err error) {
	for p.pos < len(p.toks) {
		t := p.toks[p.pos]
		switch t.kind {
		case tokText:
			nodes = append(nodes, textNode{t.body})
			p.pos++
		case tokPrint:
			e, err := parseExpr(t.body)
			if err != nil {
				return nil, "", err
			}
			nodes = append(nodes, printNode{e})
			p.pos++
		case tokStmt:
			kw := stmtKeyword(t.body)
			switch kw {
			case "for":
				n, err := p.parseFor()
				if err != nil {
					return nil, "", err
				}
				nodes = append(nodes, n)
			case "if":
				n, err := p.parseIf()
				if err != nil {
					return nil, "", err
				}
				nodes = append(nodes, n)
			case "set":
				n, err := parseSet(t.body)
				if err != nil {
					return nil, "", err
				}
				nodes = append(nodes, n)
				p.pos++
			case "macro":
				n, err := p.parseMacro()
				if err != nil {
					return nil, "", err
				}
				nodes = append(nodes, n)
			case "elif", "else", "endif", "endfor", "endmacro":
				return nodes, t.body, nil
			default:
				return nil, "", fmt.Errorf("%w: {%% %s %%}", ErrUnsupported, t.body)
			}
		}
	}
	return nodes, "", nil
}

func (p *parser) parseFor() (node, error) {
	// {% for x in expr %} or {% for k, v in expr %}
	head := strings.TrimSpace(strings.TrimPrefix(p.toks[p.pos].body, "for"))
	ep := &exprParser{s: head}
	var vars []string
	for {
		ep.skipSpace()
		name, err := ep.readIdent()
		if err != nil {
			return nil, err
		}
		vars = append(vars, name)
		ep.skipSpace()
		if ep.pos < len(ep.s) && ep.s[ep.pos] == ',' {
			ep.pos++
			continue
		}
		break
	}
	if len(vars) > 2 {
		return nil, fmt.Errorf("%w: for with %d target names", ErrUnsupported, len(vars))
	}
	ep.skipSpace()
	if !ep.eatWord("in") {
		return nil, fmt.Errorf("%w: for without 'in': %q", ErrUnsupported, head)
	}
	iter, err := ep.parseCond()
	if err != nil {
		return nil, err
	}
	ep.skipSpace()
	if ep.pos != len(ep.s) {
		return nil, fmt.Errorf("%w: for head %q has trailing %q", ErrUnsupported, head, ep.s[ep.pos:])
	}
	p.pos++
	body, stop, err := p.parseBlock()
	if err != nil {
		return nil, err
	}
	if stop != "endfor" {
		return nil, fmt.Errorf("%w: for without endfor", ErrUnsupported)
	}
	p.pos++ // consume endfor
	return forNode{vars: vars, iter: iter, body: body}, nil
}

func (p *parser) parseMacro() (node, error) {
	// {% macro name(a, b) %}
	head := strings.TrimSpace(strings.TrimPrefix(p.toks[p.pos].body, "macro"))
	ep := &exprParser{s: head}
	ep.skipSpace()
	name, err := ep.readIdent()
	if err != nil {
		return nil, err
	}
	ep.skipSpace()
	if ep.pos >= len(ep.s) || ep.s[ep.pos] != '(' {
		return nil, fmt.Errorf("%w: macro %q without a parameter list", ErrUnsupported, name)
	}
	ep.pos++
	var params []string
	ep.skipSpace()
	if ep.pos < len(ep.s) && ep.s[ep.pos] != ')' {
		for {
			ep.skipSpace()
			param, err := ep.readIdent()
			if err != nil {
				return nil, err
			}
			params = append(params, param)
			ep.skipSpace()
			if ep.pos < len(ep.s) && ep.s[ep.pos] == ',' {
				ep.pos++
				continue
			}
			break
		}
	}
	ep.skipSpace()
	if ep.pos >= len(ep.s) || ep.s[ep.pos] != ')' {
		return nil, fmt.Errorf("%w: macro %q parameter list is unclosed", ErrUnsupported, name)
	}
	ep.pos++
	ep.skipSpace()
	if ep.pos != len(ep.s) {
		return nil, fmt.Errorf("%w: macro head %q has trailing %q", ErrUnsupported, head, ep.s[ep.pos:])
	}
	p.pos++
	body, stop, err := p.parseBlock()
	if err != nil {
		return nil, err
	}
	if stop != "endmacro" {
		return nil, fmt.Errorf("%w: macro without endmacro", ErrUnsupported)
	}
	p.pos++ // consume endmacro
	return macroNode{name: name, params: params, body: body}, nil
}

func parseSet(body string) (node, error) {
	// {% set name = expr %} or {% set ns.attr = expr %}
	rest := strings.TrimSpace(strings.TrimPrefix(body, "set"))
	ep := &exprParser{s: rest}
	ep.skipSpace()
	head, err := ep.readIdent()
	if err != nil {
		return nil, err
	}
	target := []string{head}
	ep.skipSpace()
	if ep.pos < len(ep.s) && ep.s[ep.pos] == '.' {
		ep.pos++
		attr, err := ep.readIdent()
		if err != nil {
			return nil, err
		}
		target = append(target, attr)
	}
	ep.skipSpace()
	if ep.pos >= len(ep.s) || ep.s[ep.pos] != '=' || strings.HasPrefix(ep.s[ep.pos:], "==") {
		return nil, fmt.Errorf("%w: set without '=': %q", ErrUnsupported, body)
	}
	ep.pos++
	e, err := ep.parseCond()
	if err != nil {
		return nil, err
	}
	ep.skipSpace()
	if ep.pos != len(ep.s) {
		return nil, fmt.Errorf("%w: set %q has trailing %q", ErrUnsupported, body, ep.s[ep.pos:])
	}
	return setNode{target: target, expr: e}, nil
}

func (p *parser) parseIf() (node, error) {
	// p.toks[p.pos] is the {% if ... %} statement.
	var branches []branch
	for {
		t := p.toks[p.pos]
		kw := stmtKeyword(t.body)
		var cond *vExpr
		if kw == "if" || kw == "elif" {
			condText := strings.TrimSpace(strings.TrimPrefix(t.body, kw))
			e, err := parseExpr(condText)
			if err != nil {
				return nil, err
			}
			cond = e
		}
		p.pos++
		body, stop, err := p.parseBlock()
		if err != nil {
			return nil, err
		}
		branches = append(branches, branch{cond, body})
		switch {
		case stop == "endif":
			p.pos++
			return ifNode{branches}, nil
		case strings.HasPrefix(stop, "elif"):
			continue // loop reads the elif as the next branch head
		case stop == "else":
			continue // same, with cond nil
		default:
			return nil, fmt.Errorf("%w: if without endif", ErrUnsupported)
		}
	}
}

func stmtKeyword(stmt string) string {
	if i := strings.IndexAny(stmt, " \t"); i >= 0 {
		return stmt[:i]
	}
	return stmt
}

// ---- values ----------------------------------------------------------

type vKind int

const (
	vUndef vKind = iota
	vNone
	vBool
	vNum
	vStr
	vList
	vDict
	vBuiltin
)

// value is a Jinja value at render time. An undefined name resolves
// to vUndef carrying the name: it is falsy and answers the defined
// test, but printing one refuses by name instead of rendering
// empty text the caller never asked for.
type value struct {
	kind vKind
	name string // vUndef: the unresolved name
	b    bool
	n    float64
	s    string
	list []value
	dict map[string]value
}

func strValue(s string) value            { return value{kind: vStr, s: s} }
func boolValue(b bool) value             { return value{kind: vBool, b: b} }
func numValue(n float64) value           { return value{kind: vNum, n: n} }
func listValue(l []value) value          { return value{kind: vList, list: l} }
func dictValue(d map[string]value) value { return value{kind: vDict, dict: d} }
func undefValue(name string) value       { return value{kind: vUndef, name: name} }
func builtinValue(name string) value     { return value{kind: vBuiltin, s: name} }

func (v value) truthy() bool {
	switch v.kind {
	case vUndef, vNone:
		return false
	case vBool:
		return v.b
	case vNum:
		return v.n != 0
	case vStr:
		return v.s != ""
	case vList:
		return len(v.list) > 0
	case vDict:
		return len(v.dict) > 0
	case vBuiltin:
		return true
	}
	return false
}

// stringify renders a value into print output. Booleans and
// undefined or none values refuse by name: a template printing one
// is asking for text the subset does not define.
func (v value) stringify() (string, error) {
	switch v.kind {
	case vStr:
		return v.s, nil
	case vNum:
		return strconv.FormatFloat(v.n, 'f', -1, 64), nil
	case vUndef:
		return "", fmt.Errorf("%w: name %q is not defined", ErrUnsupported, v.name)
	case vNone:
		return "", fmt.Errorf("%w: cannot print none", ErrUnsupported)
	case vBool:
		return "", fmt.Errorf("%w: cannot print a boolean", ErrUnsupported)
	case vBuiltin:
		return "", fmt.Errorf("%w: cannot print the builtin function %q", ErrUnsupported, v.s)
	default:
		return "", fmt.Errorf("%w: cannot print a list or mapping", ErrUnsupported)
	}
}

func valuesEqual(a, b value) bool {
	if a.kind == vNone || b.kind == vNone {
		return a.kind == b.kind
	}
	if a.kind == vUndef || b.kind == vUndef {
		return a.kind == b.kind
	}
	if a.kind != b.kind {
		return false
	}
	switch a.kind {
	case vBool:
		return a.b == b.b
	case vNum:
		return a.n == b.n
	case vStr:
		return a.s == b.s
	}
	return false
}

func valueIn(item, container value) (bool, error) {
	switch container.kind {
	case vList:
		for _, el := range container.list {
			if valuesEqual(item, el) {
				return true, nil
			}
		}
		return false, nil
	case vDict:
		if item.kind != vStr {
			return false, fmt.Errorf("%w: 'in' over a mapping needs a string key", ErrUnsupported)
		}
		_, ok := container.dict[item.s]
		return ok, nil
	case vStr:
		if item.kind != vStr {
			return false, fmt.Errorf("%w: 'in' over a string needs a string", ErrUnsupported)
		}
		return strings.Contains(container.s, item.s), nil
	}
	return false, fmt.Errorf("%w: 'in' over a non-container", ErrUnsupported)
}

// ---- expressions -----------------------------------------------------

// vExpr is the expression tree. One struct with per-kind fields,
// in the compact style of the original subset.
type vExpr struct {
	kind string // lit, num, bool, none, name, listlit, attr, index,
	// call, method, filter, add, eq, ne, in, notin, and, or, not,
	// test, cond
	s    string  // literal text, name, attribute, method or test name
	n    float64 // numeric literal
	b    bool    // boolean literal
	args []*vExpr
	kw   map[string]*vExpr
	lhs  *vExpr
	rhs  *vExpr
	cond *vExpr
	// filter arguments ride in args; the filtered value is lhs.
	neg bool // test negation: "is not defined"
}

type exprParser struct {
	s   string
	pos int
}

func parseExpr(s string) (*vExpr, error) {
	p := &exprParser{s: s}
	e, err := p.parseCond()
	if err != nil {
		return nil, err
	}
	p.skipSpace()
	if p.pos != len(p.s) {
		return nil, fmt.Errorf("%w: expression %q has trailing %q", ErrUnsupported, s, p.s[p.pos:])
	}
	return e, nil
}

// parseCond parses inline conditionals: value if condition else other.
func (p *exprParser) parseCond() (*vExpr, error) {
	val, err := p.parseOr()
	if err != nil {
		return nil, err
	}
	p.skipSpace()
	if !p.eatWord("if") {
		return val, nil
	}
	cond, err := p.parseOr()
	if err != nil {
		return nil, err
	}
	p.skipSpace()
	if !p.eatWord("else") {
		// Jinja: a missing else yields undefined. None of the
		// shipped shapes use it, but it parses and evaluates.
		return &vExpr{kind: "cond", lhs: val, cond: cond, rhs: &vExpr{kind: "none"}}, nil
	}
	other, err := p.parseCond()
	if err != nil {
		return nil, err
	}
	return &vExpr{kind: "cond", lhs: val, cond: cond, rhs: other}, nil
}

func (p *exprParser) parseOr() (*vExpr, error) {
	lhs, err := p.parseAnd()
	if err != nil {
		return nil, err
	}
	for {
		p.skipSpace()
		if !p.eatWord("or") {
			return lhs, nil
		}
		rhs, err := p.parseAnd()
		if err != nil {
			return nil, err
		}
		lhs = &vExpr{kind: "or", lhs: lhs, rhs: rhs}
	}
}

func (p *exprParser) parseAnd() (*vExpr, error) {
	lhs, err := p.parseNot()
	if err != nil {
		return nil, err
	}
	for {
		p.skipSpace()
		if !p.eatWord("and") {
			return lhs, nil
		}
		rhs, err := p.parseNot()
		if err != nil {
			return nil, err
		}
		lhs = &vExpr{kind: "and", lhs: lhs, rhs: rhs}
	}
}

func (p *exprParser) parseNot() (*vExpr, error) {
	p.skipSpace()
	if p.eatWord("not") {
		sub, err := p.parseNot()
		if err != nil {
			return nil, err
		}
		return &vExpr{kind: "not", lhs: sub}, nil
	}
	return p.parseComparison()
}

func (p *exprParser) parseComparison() (*vExpr, error) {
	lhs, err := p.parseAdditive()
	if err != nil {
		return nil, err
	}
	p.skipSpace()
	// Postfix tests: expr is [not] <test>.
	if p.eatWord("is") {
		p.skipSpace()
		neg := p.eatWord("not")
		p.skipSpace()
		name, err := p.readIdent()
		if err != nil {
			return nil, err
		}
		return &vExpr{kind: "test", s: name, lhs: lhs, neg: neg}, nil
	}
	for _, op := range []struct {
		text string
		kind string
	}{
		{"==", "eq"}, {"!=", "ne"},
	} {
		if strings.HasPrefix(p.s[p.pos:], op.text) {
			p.pos += len(op.text)
			rhs, err := p.parseAdditive()
			if err != nil {
				return nil, err
			}
			return &vExpr{kind: op.kind, lhs: lhs, rhs: rhs}, nil
		}
	}
	if p.eatWord("not") {
		p.skipSpace()
		if !p.eatWord("in") {
			return nil, fmt.Errorf("%w: 'not' without 'in' in comparison", ErrUnsupported)
		}
		rhs, err := p.parseAdditive()
		if err != nil {
			return nil, err
		}
		return &vExpr{kind: "notin", lhs: lhs, rhs: rhs}, nil
	}
	if p.eatWord("in") {
		rhs, err := p.parseAdditive()
		if err != nil {
			return nil, err
		}
		return &vExpr{kind: "in", lhs: lhs, rhs: rhs}, nil
	}
	return lhs, nil
}

func (p *exprParser) parseAdditive() (*vExpr, error) {
	lhs, err := p.parsePostfix()
	if err != nil {
		return nil, err
	}
	for {
		p.skipSpace()
		if p.pos < len(p.s) && p.s[p.pos] == '+' {
			p.pos++
			rhs, err := p.parsePostfix()
			if err != nil {
				return nil, err
			}
			lhs = &vExpr{kind: "add", lhs: lhs, rhs: rhs}
			continue
		}
		return lhs, nil
	}
}

// parsePostfix parses a primary plus its attribute, subscript,
// method-call and filter suffixes, in order.
func (p *exprParser) parsePostfix() (*vExpr, error) {
	e, err := p.parsePrimary()
	if err != nil {
		return nil, err
	}
	for {
		p.skipSpace()
		if p.pos >= len(p.s) {
			return e, nil
		}
		switch p.s[p.pos] {
		case '.':
			p.pos++
			name, err := p.readIdent()
			if err != nil {
				return nil, err
			}
			p.skipSpace()
			if p.pos < len(p.s) && p.s[p.pos] == '(' {
				args, kw, err := p.parseArgs()
				if err != nil {
					return nil, err
				}
				e = &vExpr{kind: "method", s: name, lhs: e, args: args, kw: kw}
			} else {
				e = &vExpr{kind: "attr", s: name, lhs: e}
			}
		case '[':
			p.pos++
			idx, err := p.parseCond()
			if err != nil {
				return nil, err
			}
			p.skipSpace()
			if p.pos >= len(p.s) || p.s[p.pos] != ']' {
				return nil, fmt.Errorf("%w: missing ] in expression %q", ErrUnsupported, p.s)
			}
			p.pos++
			e = &vExpr{kind: "index", lhs: e, rhs: idx}
		case '|':
			p.pos++
			p.skipSpace()
			name, err := p.readIdent()
			if err != nil {
				return nil, err
			}
			var args []*vExpr
			p.skipSpace()
			if p.pos < len(p.s) && p.s[p.pos] == '(' {
				var kw map[string]*vExpr
				args, kw, err = p.parseArgs()
				if err != nil {
					return nil, err
				}
				if len(kw) > 0 {
					return nil, fmt.Errorf("%w: filter %q with keyword arguments", ErrUnsupported, name)
				}
			}
			e = &vExpr{kind: "filter", s: name, lhs: e, args: args}
		default:
			return e, nil
		}
	}
}

func (p *exprParser) parsePrimary() (*vExpr, error) {
	p.skipSpace()
	if p.pos >= len(p.s) {
		return nil, fmt.Errorf("%w: expression ends early", ErrUnsupported)
	}
	switch c := p.s[p.pos]; {
	case c == '\'' || c == '"':
		lit, err := p.parseLiteral()
		if err != nil {
			return nil, err
		}
		return &vExpr{kind: "lit", s: lit}, nil
	case c == '(':
		p.pos++
		e, err := p.parseCond()
		if err != nil {
			return nil, err
		}
		p.skipSpace()
		if p.pos >= len(p.s) || p.s[p.pos] != ')' {
			return nil, fmt.Errorf("%w: missing ) in expression %q", ErrUnsupported, p.s)
		}
		p.pos++
		return e, nil
	case c == '{':
		p.pos++
		var pairs []*vExpr // flattened key, value, key, value
		p.skipSpace()
		if p.pos < len(p.s) && p.s[p.pos] != '}' {
			for {
				k, err := p.parseCond()
				if err != nil {
					return nil, err
				}
				p.skipSpace()
				if p.pos >= len(p.s) || p.s[p.pos] != ':' {
					return nil, fmt.Errorf("%w: mapping literal without ':' in %q", ErrUnsupported, p.s)
				}
				p.pos++
				v, err := p.parseCond()
				if err != nil {
					return nil, err
				}
				pairs = append(pairs, k, v)
				p.skipSpace()
				if p.pos < len(p.s) && p.s[p.pos] == ',' {
					p.pos++
					continue
				}
				break
			}
		}
		p.skipSpace()
		if p.pos >= len(p.s) || p.s[p.pos] != '}' {
			return nil, fmt.Errorf("%w: missing } in mapping literal %q", ErrUnsupported, p.s)
		}
		p.pos++
		return &vExpr{kind: "dictlit", args: pairs}, nil
	case c == '[':
		p.pos++
		var elems []*vExpr
		p.skipSpace()
		if p.pos < len(p.s) && p.s[p.pos] != ']' {
			for {
				el, err := p.parseCond()
				if err != nil {
					return nil, err
				}
				elems = append(elems, el)
				p.skipSpace()
				if p.pos < len(p.s) && p.s[p.pos] == ',' {
					p.pos++
					continue
				}
				break
			}
		}
		p.skipSpace()
		if p.pos >= len(p.s) || p.s[p.pos] != ']' {
			return nil, fmt.Errorf("%w: missing ] in list literal %q", ErrUnsupported, p.s)
		}
		p.pos++
		return &vExpr{kind: "listlit", args: elems}, nil
	case c >= '0' && c <= '9':
		start := p.pos
		for p.pos < len(p.s) && (isIdentChar(p.s[p.pos]) || p.s[p.pos] == '.') {
			p.pos++
		}
		n, err := strconv.ParseFloat(p.s[start:p.pos], 64)
		if err != nil {
			return nil, fmt.Errorf("%w: bad number %q", ErrUnsupported, p.s[start:p.pos])
		}
		return &vExpr{kind: "num", n: n}, nil
	case isIdentStart(c):
		name, err := p.readIdent()
		if err != nil {
			return nil, err
		}
		switch name {
		case "true":
			return &vExpr{kind: "bool", b: true}, nil
		case "false":
			return &vExpr{kind: "bool", b: false}, nil
		case "none":
			return &vExpr{kind: "none"}, nil
		}
		p.skipSpace()
		if p.pos < len(p.s) && p.s[p.pos] == '(' {
			args, kw, err := p.parseArgs()
			if err != nil {
				return nil, err
			}
			return &vExpr{kind: "call", s: name, args: args, kw: kw}, nil
		}
		return &vExpr{kind: "name", s: name}, nil
	default:
		return nil, fmt.Errorf("%w: unexpected %q in expression", ErrUnsupported, string(c))
	}
}

// parseArgs parses "(a, b, k=v)" with p.pos at the '('.
func (p *exprParser) parseArgs() (args []*vExpr, kw map[string]*vExpr, err error) {
	p.pos++ // consume (
	p.skipSpace()
	if p.pos < len(p.s) && p.s[p.pos] == ')' {
		p.pos++
		return nil, nil, nil
	}
	for {
		p.skipSpace()
		// Keyword argument: ident '=' (but not '==').
		save := p.pos
		if p.pos < len(p.s) && isIdentStart(p.s[p.pos]) {
			name, ierr := p.readIdent()
			if ierr != nil {
				return nil, nil, ierr
			}
			p.skipSpace()
			if p.pos < len(p.s) && p.s[p.pos] == '=' && !strings.HasPrefix(p.s[p.pos:], "==") {
				p.pos++
				v, verr := p.parseCond()
				if verr != nil {
					return nil, nil, verr
				}
				if kw == nil {
					kw = map[string]*vExpr{}
				}
				kw[name] = v
				goto next
			}
			p.pos = save
		}
		{
			v, verr := p.parseCond()
			if verr != nil {
				return nil, nil, verr
			}
			args = append(args, v)
		}
	next:
		p.skipSpace()
		if p.pos < len(p.s) && p.s[p.pos] == ',' {
			p.pos++
			continue
		}
		break
	}
	p.skipSpace()
	if p.pos >= len(p.s) || p.s[p.pos] != ')' {
		return nil, nil, fmt.Errorf("%w: missing ) in call %q", ErrUnsupported, p.s)
	}
	p.pos++
	return args, kw, nil
}

func (p *exprParser) parseLiteral() (string, error) {
	quote := p.s[p.pos]
	p.pos++
	var sb strings.Builder
	for p.pos < len(p.s) {
		c := p.s[p.pos]
		if c == '\\' && p.pos+1 < len(p.s) {
			switch p.s[p.pos+1] {
			case 'n':
				sb.WriteByte('\n')
			case 't':
				sb.WriteByte('\t')
			case 'r':
				sb.WriteByte('\r')
			default:
				sb.WriteByte(p.s[p.pos+1])
			}
			p.pos += 2
			continue
		}
		if c == quote {
			p.pos++
			return sb.String(), nil
		}
		sb.WriteByte(c)
		p.pos++
	}
	return "", fmt.Errorf("%w: unterminated string literal", ErrUnsupported)
}

func (p *exprParser) readIdent() (string, error) {
	start := p.pos
	for p.pos < len(p.s) && isIdentChar(p.s[p.pos]) {
		p.pos++
	}
	if start == p.pos {
		return "", fmt.Errorf("%w: expected a name in expression %q", ErrUnsupported, p.s)
	}
	return p.s[start:p.pos], nil
}

func (p *exprParser) skipSpace() {
	for p.pos < len(p.s) && (p.s[p.pos] == ' ' || p.s[p.pos] == '\t' || p.s[p.pos] == '\n' || p.s[p.pos] == '\r') {
		p.pos++
	}
}

// eatWord consumes a keyword when it stands alone (not the head of
// a longer identifier).
func (p *exprParser) eatWord(word string) bool {
	if !strings.HasPrefix(p.s[p.pos:], word) {
		return false
	}
	end := p.pos + len(word)
	if end < len(p.s) && isIdentChar(p.s[end]) {
		return false
	}
	p.pos = end
	return true
}

func isIdentStart(c byte) bool {
	return c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}

func isIdentChar(c byte) bool {
	return isIdentStart(c) || (c >= '0' && c <= '9')
}

// ---- evaluation --------------------------------------------------------

type scope struct {
	vars   map[string]value
	parent *scope
}

func (s *scope) lookup(name string) (value, bool) {
	for cur := s; cur != nil; cur = cur.parent {
		if v, ok := cur.vars[name]; ok {
			return v, true
		}
	}
	return value{}, false
}

type env struct {
	global *scope
	cur    *scope
	macros map[string]*macroNode
	now    time.Time
	budget *renderBudget
}

func (e *env) child(vars map[string]value) *env {
	return &env{global: e.global, cur: &scope{vars: vars, parent: e.cur}, macros: e.macros, now: e.now, budget: e.budget}
}

// renderBudget is the shared accounting every render in one
// Render call draws from: child scopes and macro invocations all
// carry the same pointer, so nesting cannot reset a budget.
type renderBudget struct {
	steps   int // nodes rendered plus loop iterations executed
	depth   int // macro calls currently open
	written int // bytes produced, output and intermediate alike
}

// step charges one unit of work against the step budget.
func (b *renderBudget) step() error {
	b.steps++
	if b.steps > maxRenderSteps {
		return fmt.Errorf("%w: more than %d evaluated nodes and loop iterations", ErrBudget, maxRenderSteps)
	}
	return nil
}

// write charges produced text against the output budget. Every
// sink counts, including a macro's private buffer, so text must
// fit the budget at production time, not at the print site.
func (b *renderBudget) write(n int) error {
	b.written += n
	if b.written > maxRenderOutput {
		return fmt.Errorf("%w: more than %d bytes of output", ErrBudget, maxRenderOutput)
	}
	return nil
}

// enterMacro opens a macro call against the depth budget.
func (b *renderBudget) enterMacro(name string) error {
	b.depth++
	if b.depth > maxMacroDepth {
		return fmt.Errorf("%w: macro call depth over %d at %q", ErrBudget, maxMacroDepth, name)
	}
	return nil
}

func renderNodes(nodes []node, e *env, sb *strings.Builder) error {
	for _, n := range nodes {
		if err := e.budget.step(); err != nil {
			return err
		}
		switch n := n.(type) {
		case textNode:
			if err := e.budget.write(len(n.s)); err != nil {
				return err
			}
			sb.WriteString(n.s)
		case printNode:
			v, err := eval(n.expr, e)
			if err != nil {
				return err
			}
			s, err := v.stringify()
			if err != nil {
				return err
			}
			if err := e.budget.write(len(s)); err != nil {
				return err
			}
			sb.WriteString(s)
		case setNode:
			v, err := eval(n.expr, e)
			if err != nil {
				return err
			}
			if err := assign(e, n.target, v); err != nil {
				return err
			}
		case macroNode:
			e.macros[n.name] = &n
		case forNode:
			iter, err := eval(n.iter, e)
			if err != nil {
				return err
			}
			if iter.kind != vList {
				return fmt.Errorf("%w: for over a non-list value", ErrUnsupported)
			}
			for i, item := range iter.list {
				// An empty loop body spends no node steps, so
				// iterations charge the step budget directly:
				// a loop over an attacker-sized list is work
				// even when it prints nothing.
				if err := e.budget.step(); err != nil {
					return err
				}
				vars := map[string]value{
					"loop": dictValue(map[string]value{
						"index":  numValue(float64(i + 1)),
						"index0": numValue(float64(i)),
						"first":  boolValue(i == 0),
						"last":   boolValue(i == len(iter.list)-1),
						"length": numValue(float64(len(iter.list))),
					}),
				}
				if len(n.vars) == 2 {
					if item.kind != vList || len(item.list) != 2 {
						return fmt.Errorf("%w: for pair target over a non-pair item", ErrUnsupported)
					}
					vars[n.vars[0]] = item.list[0]
					vars[n.vars[1]] = item.list[1]
				} else {
					vars[n.vars[0]] = item
				}
				if err := renderNodes(n.body, e.child(vars), sb); err != nil {
					return err
				}
			}
		case ifNode:
			for _, br := range n.branches {
				if br.cond == nil {
					if err := renderNodes(br.body, e, sb); err != nil {
						return err
					}
					break
				}
				cond, err := eval(br.cond, e)
				if err != nil {
					return err
				}
				if cond.truthy() {
					if err := renderNodes(br.body, e, sb); err != nil {
						return err
					}
					break
				}
			}
		}
	}
	return nil
}

// assign implements {% set %}: a plain name binds in the current
// scope; name.attr mutates the named mapping in place, which is
// how namespace() state survives loop iterations.
func assign(e *env, target []string, v value) error {
	if len(target) == 1 {
		e.cur.vars[target[0]] = v
		return nil
	}
	base, ok := e.cur.lookup(target[0])
	if !ok || base.kind != vDict {
		return fmt.Errorf("%w: set %s.%s on a non-mapping", ErrUnsupported, target[0], target[1])
	}
	base.dict[target[1]] = v
	return nil
}

func eval(x *vExpr, e *env) (value, error) {
	switch x.kind {
	case "lit":
		return strValue(x.s), nil
	case "num":
		return numValue(x.n), nil
	case "bool":
		return boolValue(x.b), nil
	case "none":
		return value{kind: vNone}, nil
	case "name":
		if v, ok := e.cur.lookup(x.s); ok {
			return v, nil
		}
		return undefValue(x.s), nil
	case "listlit":
		elems := make([]value, 0, len(x.args))
		for _, a := range x.args {
			v, err := eval(a, e)
			if err != nil {
				return value{}, err
			}
			elems = append(elems, v)
		}
		return listValue(elems), nil
	case "dictlit":
		d := make(map[string]value, len(x.args)/2)
		for i := 0; i+1 < len(x.args); i += 2 {
			k, err := eval(x.args[i], e)
			if err != nil {
				return value{}, err
			}
			if k.kind != vStr {
				return value{}, fmt.Errorf("%w: mapping literal key is not a string", ErrUnsupported)
			}
			v, err := eval(x.args[i+1], e)
			if err != nil {
				return value{}, err
			}
			d[k.s] = v
		}
		return dictValue(d), nil
	case "attr":
		base, err := eval(x.lhs, e)
		if err != nil {
			return value{}, err
		}
		return getAttr(base, x.s), nil
	case "index":
		base, err := eval(x.lhs, e)
		if err != nil {
			return value{}, err
		}
		idx, err := eval(x.rhs, e)
		if err != nil {
			return value{}, err
		}
		return getIndex(base, idx)
	case "add":
		l, err := eval(x.lhs, e)
		if err != nil {
			return value{}, err
		}
		r, err := eval(x.rhs, e)
		if err != nil {
			return value{}, err
		}
		out, err := addValues(l, r)
		if err != nil {
			return value{}, err
		}
		// Concatenation materializes a new string or list, so
		// it charges the output budget: a doubling chain of
		// '+' over a value produces bytes long before any
		// print site sees them.
		if err := e.budget.write(valueBytes(out)); err != nil {
			return value{}, err
		}
		return out, nil
	case "eq", "ne":
		l, err := eval(x.lhs, e)
		if err != nil {
			return value{}, err
		}
		r, err := eval(x.rhs, e)
		if err != nil {
			return value{}, err
		}
		eq := valuesEqual(l, r)
		if x.kind == "ne" {
			eq = !eq
		}
		return boolValue(eq), nil
	case "in", "notin":
		l, err := eval(x.lhs, e)
		if err != nil {
			return value{}, err
		}
		r, err := eval(x.rhs, e)
		if err != nil {
			return value{}, err
		}
		in, err := valueIn(l, r)
		if err != nil {
			return value{}, err
		}
		if x.kind == "notin" {
			in = !in
		}
		return boolValue(in), nil
	case "and":
		l, err := eval(x.lhs, e)
		if err != nil {
			return value{}, err
		}
		if !l.truthy() {
			return l, nil
		}
		return eval(x.rhs, e)
	case "or":
		l, err := eval(x.lhs, e)
		if err != nil {
			return value{}, err
		}
		if l.truthy() {
			return l, nil
		}
		return eval(x.rhs, e)
	case "not":
		l, err := eval(x.lhs, e)
		if err != nil {
			return value{}, err
		}
		return boolValue(!l.truthy()), nil
	case "test":
		return evalTest(x, e)
	case "cond":
		c, err := eval(x.cond, e)
		if err != nil {
			return value{}, err
		}
		if c.truthy() {
			return eval(x.lhs, e)
		}
		return eval(x.rhs, e)
	case "filter":
		return evalFilter(x, e)
	case "method":
		return evalMethod(x, e)
	case "call":
		return evalCall(x, e)
	}
	return value{}, fmt.Errorf("%w: expression kind %q", ErrUnsupported, x.kind)
}

func getAttr(base value, name string) value {
	if base.kind == vDict {
		if v, ok := base.dict[name]; ok {
			return v
		}
	}
	if base.kind == vUndef {
		return base
	}
	return undefValue(name)
}

func getIndex(base, idx value) (value, error) {
	switch base.kind {
	case vDict:
		if idx.kind != vStr {
			return value{}, fmt.Errorf("%w: mapping index is not a string", ErrUnsupported)
		}
		if v, ok := base.dict[idx.s]; ok {
			return v, nil
		}
		return undefValue(idx.s), nil
	case vList:
		if idx.kind != vNum {
			return value{}, fmt.Errorf("%w: list index is not a number", ErrUnsupported)
		}
		i := int(idx.n)
		if i < 0 || i >= len(base.list) {
			return value{}, fmt.Errorf("%w: list index %d out of range", ErrUnsupported, i)
		}
		return base.list[i], nil
	case vUndef:
		return base, nil
	}
	return value{}, fmt.Errorf("%w: indexing a non-container", ErrUnsupported)
}

func addValues(l, r value) (value, error) {
	switch {
	case l.kind == vStr && r.kind == vStr:
		return strValue(l.s + r.s), nil
	case l.kind == vNum && r.kind == vNum:
		return numValue(l.n + r.n), nil
	case l.kind == vList && r.kind == vList:
		out := make([]value, 0, len(l.list)+len(r.list))
		out = append(out, l.list...)
		out = append(out, r.list...)
		return listValue(out), nil
	}
	return value{}, fmt.Errorf("%w: '+' over values that do not add", ErrUnsupported)
}

// valueBytes measures the bytes a value stands for, for the
// output budget: the string itself, or the element total of a
// container. Scalars carry no production cost worth charging.
func valueBytes(v value) int {
	switch v.kind {
	case vStr:
		return len(v.s)
	case vList:
		n := 0
		for _, el := range v.list {
			n += valueBytes(el)
		}
		return n
	case vDict:
		n := 0
		for _, el := range v.dict {
			n += valueBytes(el)
		}
		return n
	}
	return 0
}

func evalTest(x *vExpr, e *env) (value, error) {
	v, err := eval(x.lhs, e)
	if err != nil {
		return value{}, err
	}
	var out bool
	switch x.s {
	case "defined":
		out = v.kind != vUndef
	case "undefined":
		out = v.kind == vUndef
	case "none":
		out = v.kind == vNone
	case "string":
		out = v.kind == vStr
	case "boolean":
		out = v.kind == vBool
	case "number":
		out = v.kind == vNum
	case "mapping":
		out = v.kind == vDict
	case "iterable":
		out = v.kind == vList || v.kind == vDict || v.kind == vStr
	default:
		return value{}, fmt.Errorf("%w: test %q", ErrUnsupported, x.s)
	}
	if x.neg {
		out = !out
	}
	return boolValue(out), nil
}

func evalFilter(x *vExpr, e *env) (value, error) {
	base, err := eval(x.lhs, e)
	if err != nil {
		return value{}, err
	}
	argStr := func(i int) (string, error) {
		if len(x.args) <= i {
			return "", fmt.Errorf("%w: filter %q needs %d arguments", ErrUnsupported, x.s, i+1)
		}
		v, err := eval(x.args[i], e)
		if err != nil {
			return "", err
		}
		if v.kind != vStr {
			return "", fmt.Errorf("%w: filter %q argument is not a string", ErrUnsupported, x.s)
		}
		return v.s, nil
	}
	switch x.s {
	case "trim":
		if base.kind != vStr {
			return value{}, fmt.Errorf("%w: filter trim over a non-string", ErrUnsupported)
		}
		return strValue(strings.TrimSpace(base.s)), nil
	case "lower":
		if base.kind != vStr {
			return value{}, fmt.Errorf("%w: filter lower over a non-string", ErrUnsupported)
		}
		return strValue(strings.ToLower(base.s)), nil
	case "replace":
		if base.kind != vStr {
			return value{}, fmt.Errorf("%w: filter replace over a non-string", ErrUnsupported)
		}
		from, err := argStr(0)
		if err != nil {
			return value{}, err
		}
		to, err := argStr(1)
		if err != nil {
			return value{}, err
		}
		out := strings.ReplaceAll(base.s, from, to)
		// A replace can multiply its input (every byte becomes
		// a longer string), so the result charges the output
		// budget like any other produced text.
		if err := e.budget.write(len(out)); err != nil {
			return value{}, err
		}
		return strValue(out), nil
	case "join":
		if base.kind != vList {
			return value{}, fmt.Errorf("%w: filter join over a non-list", ErrUnsupported)
		}
		sep, err := argStr(0)
		if err != nil {
			return value{}, err
		}
		parts := make([]string, 0, len(base.list))
		for _, el := range base.list {
			s, err := el.stringify()
			if err != nil {
				return value{}, err
			}
			parts = append(parts, s)
		}
		out := strings.Join(parts, sep)
		if err := e.budget.write(len(out)); err != nil {
			return value{}, err
		}
		return strValue(out), nil
	}
	return value{}, fmt.Errorf("%w: filter %q", ErrUnsupported, x.s)
}

func evalMethod(x *vExpr, e *env) (value, error) {
	base, err := eval(x.lhs, e)
	if err != nil {
		return value{}, err
	}
	switch x.s {
	case "get":
		if base.kind != vDict {
			return value{}, fmt.Errorf("%w: method get on a non-mapping", ErrUnsupported)
		}
		if len(x.args) < 1 || len(x.args) > 2 {
			return value{}, fmt.Errorf("%w: method get with %d arguments", ErrUnsupported, len(x.args))
		}
		key, err := eval(x.args[0], e)
		if err != nil {
			return value{}, err
		}
		if key.kind != vStr {
			return value{}, fmt.Errorf("%w: method get with a non-string key", ErrUnsupported)
		}
		if v, ok := base.dict[key.s]; ok {
			return v, nil
		}
		if len(x.args) == 2 {
			return eval(x.args[1], e)
		}
		return value{kind: vNone}, nil
	case "split":
		if base.kind != vStr {
			return value{}, fmt.Errorf("%w: method split on a non-string", ErrUnsupported)
		}
		if len(x.args) != 1 {
			return value{}, fmt.Errorf("%w: method split with %d arguments", ErrUnsupported, len(x.args))
		}
		sep, err := eval(x.args[0], e)
		if err != nil {
			return value{}, err
		}
		if sep.kind != vStr || sep.s == "" {
			return value{}, fmt.Errorf("%w: method split with a bad separator", ErrUnsupported)
		}
		parts := strings.Split(base.s, sep.s)
		out := make([]value, len(parts))
		for i, part := range parts {
			out[i] = strValue(part)
		}
		return listValue(out), nil
	case "items":
		if base.kind != vDict {
			return value{}, fmt.Errorf("%w: method items on a non-mapping", ErrUnsupported)
		}
		// Pair items in sorted-key order: Go maps do not carry
		// Jinja's insertion order, and no shipped shape iterates
		// a mapping whose order affects the prompt.
		keys := make([]string, 0, len(base.dict))
		for k := range base.dict {
			keys = append(keys, k)
		}
		sortStrings(keys)
		out := make([]value, 0, len(keys))
		for _, k := range keys {
			out = append(out, listValue([]value{strValue(k), base.dict[k]}))
		}
		return listValue(out), nil
	}
	return value{}, fmt.Errorf("%w: method call %s(...)", ErrUnsupported, x.s)
}

func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}

func evalCall(x *vExpr, e *env) (value, error) {
	if x.s == "namespace" {
		d := map[string]value{}
		for name, argExpr := range x.kw {
			v, err := eval(argExpr, e)
			if err != nil {
				return value{}, err
			}
			d[name] = v
		}
		if len(x.args) > 0 {
			return value{}, fmt.Errorf("%w: namespace() with positional arguments", ErrUnsupported)
		}
		return dictValue(d), nil
	}
	if _, isMacro := e.macros[x.s]; !isMacro {
		if v, ok := e.cur.lookup(x.s); ok && v.kind == vBuiltin {
			return evalBuiltin(v.s, x, e)
		}
	}
	m, ok := e.macros[x.s]
	if !ok {
		// raise_exception and every other unknown call refuse by
		// name when actually evaluated; untaken branches never
		// reach here.
		return value{}, fmt.Errorf("%w: function call %s(...)", ErrUnsupported, x.s)
	}
	vars := map[string]value{}
	for i, param := range m.params {
		if i < len(x.args) {
			v, err := eval(x.args[i], e)
			if err != nil {
				return value{}, err
			}
			if err := e.budget.write(valueBytes(v)); err != nil {
				return value{}, err
			}
			vars[param] = v
		} else if kwExpr, ok := x.kw[param]; ok {
			v, err := eval(kwExpr, e)
			if err != nil {
				return value{}, err
			}
			if err := e.budget.write(valueBytes(v)); err != nil {
				return value{}, err
			}
			vars[param] = v
		} else {
			vars[param] = undefValue(param)
		}
	}
	// A macro call charges a step of its own, on top of the
	// nodes its body renders, so the total call count in one
	// render is bounded directly and not only through bodies.
	if err := e.budget.step(); err != nil {
		return value{}, err
	}
	if err := e.budget.enterMacro(x.s); err != nil {
		return value{}, err
	}
	defer func() { e.budget.depth-- }()
	// Macros see the global scope plus their parameters, as in
	// Jinja; the caller's locals do not leak in.
	macroEnv := &env{global: e.global, cur: &scope{vars: vars, parent: e.global}, macros: e.macros, now: e.now, budget: e.budget}
	var sb strings.Builder
	if err := renderNodes(m.body, macroEnv, &sb); err != nil {
		return value{}, err
	}
	// The returned text is materialized here and will be
	// charged again wherever it is consumed (printed or passed
	// on as an argument); charging it at production too keeps
	// a macro that duplicates its argument a constant number
	// of times per call from growing memory a level at a time
	// while the charges land only at the final print.
	if err := e.budget.write(sb.Len()); err != nil {
		return value{}, err
	}
	return strValue(sb.String()), nil
}

// evalBuiltin runs a call to a builtin function value. The only
// builtin is strftime_now(format), the host-clock function the
// reference template environment defines; it exists so templates
// that print the current date (guarded by `is defined`, as the
// Glimmer ATEM template does) render the same text llama.cpp
// renders. The format language is the strftime subset chat
// templates use; an unknown specifier refuses by name.
func evalBuiltin(name string, x *vExpr, e *env) (value, error) {
	if name != "strftime_now" {
		return value{}, fmt.Errorf("%w: function call %s(...)", ErrUnsupported, name)
	}
	if len(x.args) != 1 || len(x.kw) > 0 {
		return value{}, fmt.Errorf("%w: strftime_now takes one format argument", ErrUnsupported)
	}
	fv, err := eval(x.args[0], e)
	if err != nil {
		return value{}, err
	}
	if fv.kind != vStr {
		return value{}, fmt.Errorf("%w: strftime_now format is not a string", ErrUnsupported)
	}
	out, err := formatStrftime(e.now, fv.s)
	if err != nil {
		return value{}, err
	}
	return strValue(out), nil
}

func formatStrftime(t time.Time, format string) (string, error) {
	var sb strings.Builder
	for i := 0; i < len(format); i++ {
		if format[i] != '%' {
			sb.WriteByte(format[i])
			continue
		}
		i++
		if i >= len(format) {
			return "", fmt.Errorf("%w: strftime format ends with a lone %%", ErrUnsupported)
		}
		switch format[i] {
		case '%':
			sb.WriteByte('%')
		case 'Y':
			fmt.Fprintf(&sb, "%04d", t.Year())
		case 'y':
			fmt.Fprintf(&sb, "%02d", t.Year()%100)
		case 'm':
			fmt.Fprintf(&sb, "%02d", int(t.Month()))
		case 'd':
			fmt.Fprintf(&sb, "%02d", t.Day())
		case 'H':
			fmt.Fprintf(&sb, "%02d", t.Hour())
		case 'M':
			fmt.Fprintf(&sb, "%02d", t.Minute())
		case 'S':
			fmt.Fprintf(&sb, "%02d", t.Second())
		case 'j':
			fmt.Fprintf(&sb, "%03d", t.YearDay())
		default:
			return "", fmt.Errorf("%w: strftime specifier %%%c", ErrUnsupported, format[i])
		}
	}
	return sb.String(), nil
}
