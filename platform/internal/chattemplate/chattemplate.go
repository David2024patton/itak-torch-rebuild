// Package chattemplate renders a model's own GGUF chat template
// into the prompt text the engine receives (TODO T-9.16).
//
// GGUF chat templates are Jinja2. This renderer implements the
// bounded subset the shipped templates actually use, and refuses
// everything else by name instead of rendering it wrong:
//
//   - literal text and {{ }} print expressions: string literals,
//     bos_token, eos_token, message['role'] / message['content']
//     (or message.role / message.content), concatenation with +,
//     and the | trim filter;
//   - {% for message in messages %} ... {% endfor %};
//   - {% if %} / {% elif %} / {% else %} / {% endif %} with == and
//     != comparisons, bare truthiness, and `not`;
//   - {% if add_generation_prompt %} and printing it is an error
//     (it is a boolean flag, not text);
//   - whitespace control ({{- -}} and {%- -%}) on both tag kinds.
//
// Anything outside the subset (other statements like {% set %},
// other filters, function calls such as raise_exception when the
// branch containing it actually renders, loop.* variables) fails
// with ErrUnsupported naming the construct. A chat request whose
// template cannot be rendered is a named platform error, never a
// silently flattened prompt: falling back is a decision the caller
// makes with the error in hand, not something this package hides.
package chattemplate

import (
	"errors"
	"fmt"
	"strings"
)

// ErrUnsupported names a template construct outside the rendered
// subset. The message carries the construct itself.
var ErrUnsupported = errors.New("chattemplate: unsupported construct")

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
	env := &env{params: p, messages: messages}
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
		iPrint := strings.Index(rest, "{{")
		iStmt := strings.Index(rest, "{%")
		i := -1
		kind := tokText
		if iPrint >= 0 && (iStmt < 0 || iPrint < iStmt) {
			i, kind = iPrint, tokPrint
		} else if iStmt >= 0 {
			i, kind = iStmt, tokStmt
		}
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
		// Opening dash: {{- or {%-
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

type printNode struct{ expr *expr }

func (printNode) nodeTag() {}

type forNode struct{ body []node }

func (forNode) nodeTag() {}

type branch struct {
	cond *expr // nil on the else branch
	body []node
}

type ifNode struct{ branches []branch }

func (ifNode) nodeTag() {}

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
// terminator (elif/else/endif/endfor); it returns that statement's
// text as the stop value without consuming it.
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
				if t.body != "for message in messages" {
					return nil, "", fmt.Errorf("%w: {%% %s %%} (only 'for message in messages' is rendered)", ErrUnsupported, t.body)
				}
				p.pos++
				body, stop, err := p.parseBlock()
				if err != nil {
					return nil, "", err
				}
				if stop != "endfor" {
					return nil, "", fmt.Errorf("%w: for without endfor", ErrUnsupported)
				}
				p.pos++ // consume endfor
				nodes = append(nodes, forNode{body})
			case "if":
				n, err := p.parseIf()
				if err != nil {
					return nil, "", err
				}
				nodes = append(nodes, n)
			case "elif", "else", "endif", "endfor":
				return nodes, t.body, nil
			default:
				return nil, "", fmt.Errorf("%w: {%% %s %%}", ErrUnsupported, t.body)
			}
		}
	}
	return nodes, "", nil
}

func (p *parser) parseIf() (node, error) {
	// p.toks[p.pos] is the {% if ... %} statement.
	var branches []branch
	for {
		t := p.toks[p.pos]
		kw := stmtKeyword(t.body)
		var cond *expr
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

// ---- expressions -----------------------------------------------------

// expr is a tiny expression tree: a value expression is a chain of
// terms joined by + ; a condition wraps one with ==, !=, or not.
// The node kinds share one struct for compactness.
type expr struct {
	kind  string // "concat", "eq", "ne", "not", "term"
	terms []*term
	lhs   *expr
	rhs   *expr
	sub   *expr
}

type term struct {
	kind  string // "literal", "path", "call"
	value string // literal text, or call name
	path  []string
	// filter is "" or "trim" (the only rendered filter).
	filter string
}

type exprParser struct {
	s   string
	pos int
}

func parseExpr(s string) (*expr, error) {
	p := &exprParser{s: s}
	e, err := p.parseCondition()
	if err != nil {
		return nil, err
	}
	p.skipSpace()
	if p.pos != len(p.s) {
		return nil, fmt.Errorf("%w: expression %q has trailing %q", ErrUnsupported, s, p.s[p.pos:])
	}
	return e, nil
}

func (p *exprParser) parseCondition() (*expr, error) {
	p.skipSpace()
	if p.eatWord("not") {
		sub, err := p.parseCondition()
		if err != nil {
			return nil, err
		}
		return &expr{kind: "not", sub: sub}, nil
	}
	lhs, err := p.parseConcat()
	if err != nil {
		return nil, err
	}
	p.skipSpace()
	if strings.HasPrefix(p.s[p.pos:], "==") {
		p.pos += 2
		rhs, err := p.parseConcat()
		if err != nil {
			return nil, err
		}
		return &expr{kind: "eq", lhs: lhs, rhs: rhs}, nil
	}
	if strings.HasPrefix(p.s[p.pos:], "!=") {
		p.pos += 2
		rhs, err := p.parseConcat()
		if err != nil {
			return nil, err
		}
		return &expr{kind: "ne", lhs: lhs, rhs: rhs}, nil
	}
	return lhs, nil
}

func (p *exprParser) parseConcat() (*expr, error) {
	first, err := p.parseTerm()
	if err != nil {
		return nil, err
	}
	e := &expr{kind: "concat", terms: []*term{first}}
	for {
		p.skipSpace()
		if p.pos < len(p.s) && p.s[p.pos] == '+' {
			p.pos++
			next, err := p.parseTerm()
			if err != nil {
				return nil, err
			}
			e.terms = append(e.terms, next)
			continue
		}
		return e, nil
	}
}

func (p *exprParser) parseTerm() (*term, error) {
	p.skipSpace()
	if p.pos >= len(p.s) {
		return nil, fmt.Errorf("%w: expression ends early", ErrUnsupported)
	}
	var t *term
	switch c := p.s[p.pos]; {
	case c == '\'' || c == '"':
		lit, err := p.parseLiteral()
		if err != nil {
			return nil, err
		}
		t = &term{kind: "literal", value: lit}
	case isIdentStart(c):
		path, err := p.parsePath()
		if err != nil {
			return nil, err
		}
		// A call like raise_exception('...') parses, then refuses
		// by name if the branch holding it actually renders.
		p.skipSpace()
		if p.pos < len(p.s) && p.s[p.pos] == '(' {
			depth := 0
			closed := false
			for p.pos < len(p.s) && !closed {
				switch p.s[p.pos] {
				case '(':
					depth++
				case ')':
					depth--
					closed = depth == 0
				}
				p.pos++
			}
			if !closed {
				return nil, fmt.Errorf("%w: unbalanced call in expression", ErrUnsupported)
			}
			t = &term{kind: "call", value: strings.Join(path, ".")}
		} else {
			t = &term{kind: "path", path: path}
		}
	default:
		return nil, fmt.Errorf("%w: unexpected %q in expression", ErrUnsupported, string(c))
	}
	// One optional filter: | trim. Anything else refuses by name.
	p.skipSpace()
	if p.pos < len(p.s) && p.s[p.pos] == '|' {
		p.pos++
		p.skipSpace()
		start := p.pos
		for p.pos < len(p.s) && isIdentChar(p.s[p.pos]) {
			p.pos++
		}
		name := p.s[start:p.pos]
		if name != "trim" {
			return nil, fmt.Errorf("%w: filter %q", ErrUnsupported, name)
		}
		t.filter = "trim"
	}
	return t, nil
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

// parsePath reads ident (.ident | ['key'])* into path segments.
func (p *exprParser) parsePath() ([]string, error) {
	readIdent := func() (string, error) {
		p.skipSpace()
		start := p.pos
		for p.pos < len(p.s) && isIdentChar(p.s[p.pos]) {
			p.pos++
		}
		if start == p.pos {
			return "", fmt.Errorf("%w: expected a name in expression %q", ErrUnsupported, p.s)
		}
		return p.s[start:p.pos], nil
	}
	first, err := readIdent()
	if err != nil {
		return nil, err
	}
	path := []string{first}
	for {
		p.skipSpace()
		if p.pos < len(p.s) && p.s[p.pos] == '.' {
			p.pos++
			seg, err := readIdent()
			if err != nil {
				return nil, err
			}
			path = append(path, seg)
			continue
		}
		if p.pos < len(p.s) && p.s[p.pos] == '[' {
			p.pos++
			p.skipSpace()
			key, err := p.parseLiteral()
			if err != nil {
				return nil, err
			}
			p.skipSpace()
			if p.pos >= len(p.s) || p.s[p.pos] != ']' {
				return nil, fmt.Errorf("%w: missing ] in expression %q", ErrUnsupported, p.s)
			}
			p.pos++
			path = append(path, key)
			continue
		}
		return path, nil
	}
}

func (p *exprParser) skipSpace() {
	for p.pos < len(p.s) && (p.s[p.pos] == ' ' || p.s[p.pos] == '\t') {
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

type env struct {
	params   Params
	messages []Message
	message  *Message // the loop variable, nil outside the loop
}

func renderNodes(nodes []node, e *env, sb *strings.Builder) error {
	for _, n := range nodes {
		switch n := n.(type) {
		case textNode:
			sb.WriteString(n.s)
		case printNode:
			v, err := evalString(n.expr, e)
			if err != nil {
				return err
			}
			sb.WriteString(v)
		case forNode:
			for i := range e.messages {
				child := &env{params: e.params, messages: e.messages, message: &e.messages[i]}
				if err := renderNodes(n.body, child, sb); err != nil {
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
				truth, err := evalBool(br.cond, e)
				if err != nil {
					return err
				}
				if truth {
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

func evalString(e *expr, env *env) (string, error) {
	switch e.kind {
	case "concat":
		var sb strings.Builder
		for _, t := range e.terms {
			s, err := evalTerm(t, env)
			if err != nil {
				return "", err
			}
			sb.WriteString(s)
		}
		return sb.String(), nil
	default:
		return "", fmt.Errorf("%w: a condition cannot be printed", ErrUnsupported)
	}
}

func evalBool(e *expr, env *env) (bool, error) {
	switch e.kind {
	case "not":
		b, err := evalBool(e.sub, env)
		return !b, err
	case "eq", "ne":
		l, err := evalString(e.lhs, env)
		if err != nil {
			return false, err
		}
		r, err := evalString(e.rhs, env)
		if err != nil {
			return false, err
		}
		if e.kind == "eq" {
			return l == r, nil
		}
		return l != r, nil
	case "concat":
		// Bare truthiness: a boolean variable's value, or a
		// non-empty string.
		if len(e.terms) == 1 {
			if b, ok := evalBoolTerm(e.terms[0], env); ok {
				return b, nil
			}
		}
		s, err := evalString(e, env)
		if err != nil {
			return false, err
		}
		return s != "", nil
	default:
		return false, fmt.Errorf("%w: expression kind %q", ErrUnsupported, e.kind)
	}
}

// evalBoolTerm resolves a term that names a boolean flag.
func evalBoolTerm(t *term, env *env) (value bool, ok bool) {
	if t.kind == "path" && len(t.path) == 1 && t.path[0] == "add_generation_prompt" {
		return env.params.AddGenerationPrompt, true
	}
	return false, false
}

func evalTerm(t *term, env *env) (string, error) {
	var out string
	switch t.kind {
	case "literal":
		out = t.value
	case "call":
		return "", fmt.Errorf("%w: function call %s(...)", ErrUnsupported, t.value)
	case "path":
		s, err := resolvePath(t.path, env)
		if err != nil {
			return "", err
		}
		out = s
	default:
		return "", fmt.Errorf("%w: term kind %q", ErrUnsupported, t.kind)
	}
	if t.filter == "trim" {
		out = strings.TrimSpace(out)
	}
	return out, nil
}

func resolvePath(path []string, env *env) (string, error) {
	switch path[0] {
	case "bos_token":
		if len(path) == 1 {
			return env.params.BOSToken, nil
		}
	case "eos_token":
		if len(path) == 1 {
			return env.params.EOSToken, nil
		}
	case "add_generation_prompt":
		return "", fmt.Errorf("%w: add_generation_prompt is a flag, not text", ErrUnsupported)
	case "message":
		if env.message == nil {
			return "", fmt.Errorf("%w: message used outside the messages loop", ErrUnsupported)
		}
		if len(path) == 2 {
			switch path[1] {
			case "role":
				return env.message.Role, nil
			case "content":
				return env.message.Content, nil
			}
		}
	}
	return "", fmt.Errorf("%w: name %q", ErrUnsupported, strings.Join(path, "."))
}
