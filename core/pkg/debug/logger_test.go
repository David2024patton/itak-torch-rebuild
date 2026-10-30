package debug

import (
	"bytes"
	"strings"
	"testing"
)

// capture redirects the global logger into a buffer at the given level
// for the duration of a test, restoring the previous state afterwards.
// Color is disabled so assertions see the plain ancestor output format:
//
//	HH:MM:SS.mmm LEVEL [component] message
func capture(t *testing.T, level Level) *bytes.Buffer {
	t.Helper()

	global.mu.Lock()
	prevLevel, prevOutput, prevColor := global.level, global.output, global.useColor
	global.mu.Unlock()

	var buf bytes.Buffer
	SetOutput(&buf)
	SetColor(false)
	SetLevel(level)

	t.Cleanup(func() {
		SetOutput(prevOutput)
		SetColor(prevColor)
		SetLevel(prevLevel)
	})
	return &buf
}

// TestLevelValues pins the numeric level values the TODO names:
// Error 0, Warn 1, Info 2, Debug 3, most severe first.
func TestLevelValues(t *testing.T) {
	if LevelError != 0 {
		t.Errorf("LevelError = %d, want 0", LevelError)
	}
	if LevelWarn != 1 {
		t.Errorf("LevelWarn = %d, want 1", LevelWarn)
	}
	if LevelInfo != 2 {
		t.Errorf("LevelInfo = %d, want 2", LevelInfo)
	}
	if LevelDebug != 3 {
		t.Errorf("LevelDebug = %d, want 3", LevelDebug)
	}
}

// TestLevelFiltering verifies that messages more verbose than the
// configured level are dropped, and messages at or above the
// configured severity are emitted.
func TestLevelFiltering(t *testing.T) {
	buf := capture(t, LevelWarn)

	Debug("filter", "debug should be dropped")
	Info("filter", "info should be dropped")
	Warn("filter", "warn should appear")
	Error("filter", "error should appear")

	out := buf.String()
	if strings.Contains(out, "should be dropped") {
		t.Errorf("filtered levels leaked into output:\n%s", out)
	}
	if !strings.Contains(out, "warn should appear") {
		t.Errorf("WARN message missing from output:\n%s", out)
	}
	if !strings.Contains(out, "error should appear") {
		t.Errorf("ERROR message missing from output:\n%s", out)
	}
	if got := strings.Count(strings.TrimRight(out, "\n"), "\n") + 1; got != 2 {
		t.Errorf("expected exactly 2 emitted lines, got %d:\n%s", got, out)
	}
}

// TestLevelFilteringDebugShowsAll verifies the other end of the range:
// at LevelDebug every severity is emitted.
func TestLevelFilteringDebugShowsAll(t *testing.T) {
	buf := capture(t, LevelDebug)

	Error("filter", "e")
	Warn("filter", "w")
	Info("filter", "i")
	Debug("filter", "d")

	out := buf.String()
	for _, want := range []string{"ERROR", "WARN", "INFO", "DEBUG"} {
		if !strings.Contains(out, want) {
			t.Errorf("level %s missing from output at LevelDebug:\n%s", want, out)
		}
	}
}

// TestSetLevelAtRuntime verifies SetLevel changes filtering without a
// restart: a message dropped before the change is emitted after it.
func TestSetLevelAtRuntime(t *testing.T) {
	buf := capture(t, LevelError)

	Info("filter", "before change")
	SetLevel(LevelInfo)
	Info("filter", "after change")

	out := buf.String()
	if strings.Contains(out, "before change") {
		t.Errorf("INFO emitted while level was LevelError:\n%s", out)
	}
	if !strings.Contains(out, "after change") {
		t.Errorf("INFO not emitted after SetLevel(LevelInfo):\n%s", out)
	}
}

// TestComponentTagging verifies the emitted line carries the component
// tag in brackets, the level name, and the formatted message.
func TestComponentTagging(t *testing.T) {
	buf := capture(t, LevelInfo)

	Info("scheduler", "job %d started", 42)

	out := buf.String()
	if !strings.Contains(out, "[scheduler]") {
		t.Errorf("component tag missing from output:\n%s", out)
	}
	if !strings.Contains(out, "INFO") {
		t.Errorf("level name missing from output:\n%s", out)
	}
	if !strings.Contains(out, "job 42 started") {
		t.Errorf("formatted message missing from output:\n%s", out)
	}
}

// TestParseLevel covers the ancestor's string-to-level mapping,
// including the "warning" alias and the LevelInfo default.
func TestParseLevel(t *testing.T) {
	cases := []struct {
		in   string
		want Level
	}{
		{"debug", LevelDebug},
		{"DEBUG", LevelDebug},
		{"info", LevelInfo},
		{"warn", LevelWarn},
		{"warning", LevelWarn},
		{"error", LevelError},
		{"bogus", LevelInfo},
		{"", LevelInfo},
	}
	for _, tc := range cases {
		if got := ParseLevel(tc.in); got != tc.want {
			t.Errorf("ParseLevel(%q) = %d, want %d", tc.in, got, tc.want)
		}
	}
}
