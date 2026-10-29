package diag_test

import (
	"testing"

	"github.com/David2024patton/itak-torch-rebuild/platform/internal/diag"
)

func TestRegistryCounts(t *testing.T) {
	r := diag.New()
	r.RecordRequest("GET /v1/models")
	r.RecordRequest("GET /v1/models")
	r.RecordRequest("POST /v1/chat/completions")
	r.RecordError()
	r.RecordAuthFailure()
	r.RecordEngineCall()
	r.RecordEngineError()

	s := r.Snapshot()
	if s.Requests != 3 || s.Errors != 1 || s.AuthFailures != 1 {
		t.Fatalf("Snapshot = %+v, want requests=3 errors=1 auth_failures=1", s)
	}
	if s.EngineCalls != 1 || s.EngineErrors != 1 {
		t.Fatalf("engine counters = %d/%d, want 1/1", s.EngineCalls, s.EngineErrors)
	}
	if s.Routes["GET /v1/models"] != 2 || s.Routes["POST /v1/chat/completions"] != 1 {
		t.Fatalf("route counts = %v", s.Routes)
	}
}
