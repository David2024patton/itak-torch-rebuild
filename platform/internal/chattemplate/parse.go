package chattemplate

import (
	"fmt"
	"strings"
)

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
