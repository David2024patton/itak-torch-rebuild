package debug

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// StepType represents the kind of step recorded in a workflow trace.
type StepType string

const (
	StepUserMessage   StepType = "user_message"
	StepDelegation    StepType = "delegation"
	StepAgentStart    StepType = "agent_start"
	StepAgentComplete StepType = "agent_complete"
	StepLLMRequest    StepType = "llm_request"
	StepLLMResponse   StepType = "llm_response"
	StepToolInvoked   StepType = "tool_invoked"
	StepToolResult    StepType = "tool_result"
	StepFinalResponse StepType = "final_response"
	StepError         StepType = "error"
	StepMemoryOp      StepType = "memory_op"
	StepFailover      StepType = "failover"
)

// Step is a single recorded event in an agent workflow.
type Step struct {
	ID         int                    `json:"id"`
	Type       StepType               `json:"type"`
	Timestamp  time.Time              `json:"timestamp"`
	Agent      string                 `json:"agent,omitempty"`
	Tool       string                 `json:"tool,omitempty"`
	Input      string                 `json:"input,omitempty"`
	Output     string                 `json:"output,omitempty"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
	DurationMs int64                  `json:"duration_ms,omitempty"`
	Error      string                 `json:"error,omitempty"`
}

// StepLogger records every step of an agent workflow for replay.
type StepLogger struct {
	mu      sync.Mutex
	steps   []Step
	nextID  int
	dataDir string
	file    *os.File // JSONL append file
}

// NewStepLogger creates a step logger that writes to a JSONL trace file.
func NewStepLogger(dataDir string) (*StepLogger, error) {
	traceDir := filepath.Join(dataDir, "traces")
	if err := os.MkdirAll(traceDir, 0o755); err != nil {
		return nil, fmt.Errorf("create trace dir: %w", err)
	}

	// Use timestamp-based filename.
	filename := fmt.Sprintf("trace_%s.jsonl", time.Now().Format("20060102_150405"))
	tracePath := filepath.Join(traceDir, filename)

	f, err := os.OpenFile(tracePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, fmt.Errorf("create trace file: %w", err)
	}

	Info("trace", "Step logger started: %s", tracePath)

	return &StepLogger{
		steps:   make([]Step, 0),
		dataDir: traceDir,
		file:    f,
	}, nil
}

// Record logs a step in real-time.
func (sl *StepLogger) Record(stepType StepType, agent, tool, input, output string, metadata map[string]interface{}) {
	sl.mu.Lock()
	defer sl.mu.Unlock()

	step := Step{
		ID:        sl.nextID,
		Type:      stepType,
		Timestamp: time.Now(),
		Agent:     agent,
		Tool:      tool,
		Input:     input,
		Output:    output,
		Metadata:  metadata,
	}
	sl.nextID++
	sl.steps = append(sl.steps, step)

	// Write to file in real-time.
	if sl.file != nil {
		line, err := json.Marshal(step)
		if err == nil {
			sl.file.Write(line)
			sl.file.Write([]byte("\n"))
			sl.file.Sync()
		}
	}

	Debug("trace", "[%d] %s %s/%s", step.ID, step.Type, step.Agent, step.Tool)
}

// RecordError logs an error step.
func (sl *StepLogger) RecordError(agent, errMsg string) {
	sl.Record(StepError, agent, "", "", "", map[string]interface{}{"error": errMsg})
}

// RecordTimed logs a step with duration tracking.
func (sl *StepLogger) RecordTimed(stepType StepType, agent, tool, input string, startTime time.Time, output string) {
	sl.mu.Lock()
	defer sl.mu.Unlock()

	step := Step{
		ID:         sl.nextID,
		Type:       stepType,
		Timestamp:  time.Now(),
		Agent:      agent,
		Tool:       tool,
		Input:      input,
		Output:     output,
		DurationMs: time.Since(startTime).Milliseconds(),
	}
	sl.nextID++
	sl.steps = append(sl.steps, step)

	if sl.file != nil {
		line, _ := json.Marshal(step)
		sl.file.Write(line)
		sl.file.Write([]byte("\n"))
		sl.file.Sync()
	}

	Debug("trace", "[%d] %s %s/%s (%dms)", step.ID, step.Type, step.Agent, step.Tool, step.DurationMs)
}

// Steps returns all recorded steps.
func (sl *StepLogger) Steps() []Step {
	sl.mu.Lock()
	defer sl.mu.Unlock()
	return append([]Step{}, sl.steps...)
}

// StepCount returns the number of recorded steps.
func (sl *StepLogger) StepCount() int {
	sl.mu.Lock()
	defer sl.mu.Unlock()
	return len(sl.steps)
}

// Close closes the trace file.
func (sl *StepLogger) Close() {
	if sl.file != nil {
		sl.file.Close()
		sl.file = nil
	}
	Info("trace", "Step logger closed (%d steps recorded)", len(sl.steps))
}

// LoadTrace reads a trace file for replay.
func LoadTrace(path string) ([]Step, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var steps []Step
	for _, line := range splitLines(string(data)) {
		if line == "" {
			continue
		}
		var step Step
		if err := json.Unmarshal([]byte(line), &step); err != nil {
			continue
		}
		steps = append(steps, step)
	}
	return steps, nil
}

// ListTraces returns all available trace files.
func ListTraces(dataDir string) ([]string, error) {
	traceDir := filepath.Join(dataDir, "traces")
	entries, err := os.ReadDir(traceDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	var traces []string
	for _, e := range entries {
		if !e.IsDir() && filepath.Ext(e.Name()) == ".jsonl" {
			traces = append(traces, filepath.Join(traceDir, e.Name()))
		}
	}
	return traces, nil
}

// splitLines splits text into lines (handles \r\n and \n).
func splitLines(text string) []string {
	var lines []string
	start := 0
	for i := 0; i < len(text); i++ {
		if text[i] == '\n' {
			line := text[start:i]
			if len(line) > 0 && line[len(line)-1] == '\r' {
				line = line[:len(line)-1]
			}
			lines = append(lines, line)
			start = i + 1
		}
	}
	if start < len(text) {
		lines = append(lines, text[start:])
	}
	return lines
}
