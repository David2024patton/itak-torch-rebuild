package chattemplate

import (
	"fmt"
	"strings"
)

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
