package chattemplate

import (
	"fmt"
	"strconv"
	"strings"
)

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
