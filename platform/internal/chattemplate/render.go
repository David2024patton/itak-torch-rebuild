package chattemplate

import (
	"fmt"
	"strings"
	"time"
)

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
