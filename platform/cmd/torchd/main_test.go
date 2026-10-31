package main

import (
	"strings"
	"testing"
)

func TestAPIKeyProblem(t *testing.T) {
	// No key and no opt-out: refuse.
	t.Setenv("TORCH_ALLOW_NO_AUTH", "")
	if got := apiKeyProblem(""); !strings.Contains(got, "not set") {
		t.Fatalf("empty key problem = %q, want the not-set refusal", got)
	}
	// No key with the deliberate opt-out: allowed.
	t.Setenv("TORCH_ALLOW_NO_AUTH", "1")
	if got := apiKeyProblem(""); got != "" {
		t.Fatalf("empty key with opt-out problem = %q, want none", got)
	}
	// A short key: refused by name, with the floor stated.
	if got := apiKeyProblem("short"); !strings.Contains(got, "too short") || !strings.Contains(got, "16") {
		t.Fatalf("short key problem = %q, want the named too-short refusal", got)
	}
	// A long key: allowed.
	if got := apiKeyProblem("0123456789abcdef"); got != "" {
		t.Fatalf("16-char key problem = %q, want none", got)
	}
}
