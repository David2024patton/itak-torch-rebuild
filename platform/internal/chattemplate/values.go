package chattemplate

import (
	"fmt"
	"strconv"
	"strings"
)

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
