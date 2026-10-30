package debug

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestStepLoggerRecordAndCount(t *testing.T) {
	tmp := t.TempDir()

	sl, err := NewStepLogger(tmp)
	if err != nil {
		t.Fatalf("NewStepLogger error: %v", err)
	}
	defer sl.Close()

	sl.Record(StepUserMessage, "", "", "hello", "", nil)
	sl.Record(StepDelegation, "orchestrator", "", "delegate to scout", "", nil)
	sl.Record(StepAgentStart, "scout", "", "", "", nil)

	if sl.StepCount() != 3 {
		t.Errorf("expected 3 steps, got %d", sl.StepCount())
	}
}

func TestStepLoggerIDs(t *testing.T) {
	tmp := t.TempDir()

	sl, err := NewStepLogger(tmp)
	if err != nil {
		t.Fatalf("NewStepLogger error: %v", err)
	}
	defer sl.Close()

	sl.Record(StepUserMessage, "", "", "first", "", nil)
	sl.Record(StepToolInvoked, "scout", "file_read", "/tmp/test.txt", "", nil)

	steps := sl.Steps()
	if steps[0].ID != 0 {
		t.Errorf("first step ID should be 0, got %d", steps[0].ID)
	}
	if steps[1].ID != 1 {
		t.Errorf("second step ID should be 1, got %d", steps[1].ID)
	}
}

func TestStepLoggerRecordError(t *testing.T) {
	tmp := t.TempDir()
	sl, _ := NewStepLogger(tmp)
	defer sl.Close()

	sl.RecordError("scout", "file not found")

	steps := sl.Steps()
	if len(steps) != 1 {
		t.Fatalf("expected 1 step, got %d", len(steps))
	}
	if steps[0].Type != StepError {
		t.Errorf("expected type 'error', got %q", steps[0].Type)
	}
}

func TestStepLoggerRecordTimed(t *testing.T) {
	tmp := t.TempDir()
	sl, _ := NewStepLogger(tmp)
	defer sl.Close()

	start := time.Now().Add(-500 * time.Millisecond)
	sl.RecordTimed(StepLLMRequest, "scout", "", "test prompt", start, "response")

	steps := sl.Steps()
	if len(steps) != 1 {
		t.Fatalf("expected 1 step, got %d", len(steps))
	}
	if steps[0].DurationMs < 400 {
		t.Errorf("expected duration >= 400ms, got %d", steps[0].DurationMs)
	}
}

func TestStepLoggerJSONLFile(t *testing.T) {
	tmp := t.TempDir()
	sl, _ := NewStepLogger(tmp)

	sl.Record(StepUserMessage, "", "", "hello", "", nil)
	sl.Record(StepDelegation, "orch", "", "to scout", "", nil)
	sl.Close()

	// Find the trace file.
	traceDir := filepath.Join(tmp, "traces")
	entries, err := os.ReadDir(traceDir)
	if err != nil {
		t.Fatalf("read trace dir: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("expected 1 trace file, got %d", len(entries))
	}

	// Read and verify JSONL content.
	data, _ := os.ReadFile(filepath.Join(traceDir, entries[0].Name()))
	lines := splitLines(string(data))
	validLines := 0
	for _, line := range lines {
		if line == "" {
			continue
		}
		var step Step
		if err := json.Unmarshal([]byte(line), &step); err != nil {
			t.Errorf("invalid JSON line: %v", err)
		}
		validLines++
	}
	if validLines != 2 {
		t.Errorf("expected 2 valid JSON lines, got %d", validLines)
	}
}

func TestLoadTrace(t *testing.T) {
	tmp := t.TempDir()
	sl, _ := NewStepLogger(tmp)

	sl.Record(StepUserMessage, "", "", "hello", "", nil)
	sl.Record(StepToolInvoked, "scout", "file_read", "/tmp/test", "content", nil)
	sl.Record(StepFinalResponse, "scout", "", "", "all done", nil)
	sl.Close()

	// Find the trace file.
	traces, err := ListTraces(tmp)
	if err != nil {
		t.Fatalf("ListTraces error: %v", err)
	}
	if len(traces) != 1 {
		t.Fatalf("expected 1 trace, got %d", len(traces))
	}

	// Load it back.
	steps, err := LoadTrace(traces[0])
	if err != nil {
		t.Fatalf("LoadTrace error: %v", err)
	}
	if len(steps) != 3 {
		t.Errorf("expected 3 steps, got %d", len(steps))
	}
	if steps[0].Type != StepUserMessage {
		t.Errorf("first step should be user_message, got %q", steps[0].Type)
	}
	if steps[2].Output != "all done" {
		t.Errorf("last step output should be 'all done', got %q", steps[2].Output)
	}
}

func TestListTracesEmpty(t *testing.T) {
	tmp := t.TempDir()

	traces, err := ListTraces(tmp)
	if err != nil {
		t.Fatalf("ListTraces error: %v", err)
	}
	if len(traces) != 0 {
		t.Errorf("expected 0 traces for empty dir, got %d", len(traces))
	}
}

func TestSplitLines(t *testing.T) {
	tests := []struct {
		input string
		want  int
	}{
		{"a\nb\nc", 3},
		{"a\r\nb\r\nc", 3},
		{"single", 1},
		{"", 0},
		{"a\n\nb", 3}, // includes empty line
	}
	for _, tc := range tests {
		got := splitLines(tc.input)
		if len(got) != tc.want {
			t.Errorf("splitLines(%q): expected %d lines, got %d", tc.input, tc.want, len(got))
		}
	}
}
