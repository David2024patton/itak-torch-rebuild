package health

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/David2024patton/itak-torch-rebuild/core/pkg/registry"
)

func TestHeartbeatIntervalIsConfigWithDefault(t *testing.T) {
	if got := (HeartbeatConfig{}).interval(); got != DefaultHeartbeatInterval {
		t.Errorf("unset Interval resolved to %v, want the documented default %v", got, DefaultHeartbeatInterval)
	}
	if got := (HeartbeatConfig{Interval: -time.Second}).interval(); got != DefaultHeartbeatInterval {
		t.Errorf("negative Interval resolved to %v, want default %v", got, DefaultHeartbeatInterval)
	}
	custom := HeartbeatConfig{Interval: 250 * time.Millisecond}
	if got := custom.interval(); got != 250*time.Millisecond {
		t.Errorf("configured Interval resolved to %v, want 250ms honored (cadence must come from config)", got)
	}
	if DefaultHeartbeatInterval != 10*time.Second {
		t.Errorf("DefaultHeartbeatInterval = %v, want the original measured default 10s", DefaultHeartbeatInterval)
	}
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

func healthyChecker() *SimpleChecker {
	return &SimpleChecker{ModuleName: "torch", Version: "test", StartTime: time.Now()}
}

func testConfig(reg *registry.ServiceRegistry, checker Checker) HeartbeatConfig {
	return HeartbeatConfig{
		Interval: 10 * time.Millisecond,
		Registry: reg,
		Endpoint: registry.ServiceEndpoint{
			Module: "torch", Host: "127.0.0.1", Port: 11435, Protocol: registry.ProtocolREST,
		},
		Checker: checker,
	}
}

func TestStartHeartbeatRegistersAndDeregistersOnCancel(t *testing.T) {
	reg := registry.New(0)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		StartHeartbeat(ctx, testConfig(reg, healthyChecker()))
		close(done)
	}()

	waitFor(t, "endpoint to be discoverable", func() bool {
		return len(reg.Discover("torch")) == 1
	})

	cancel()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("StartHeartbeat did not return after context cancel")
	}
	if got := reg.All()["torch"]; len(got) != 0 {
		t.Fatalf("endpoint still registered after cancel: %d records, want deregistration", len(got))
	}
}

func TestStartHeartbeatMarksUnhealthy(t *testing.T) {
	reg := registry.New(0)
	checker := &SimpleChecker{
		ModuleName: "torch",
		StartTime:  time.Now(),
		CheckFn: func() (Status, map[string]Check) {
			return StatusUnhealthy, nil
		},
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go StartHeartbeat(ctx, testConfig(reg, checker))

	waitFor(t, "endpoint to be flagged unhealthy", func() bool {
		eps := reg.All()["torch"]
		return len(eps) == 1 && !eps[0].Healthy
	})
	if got := reg.Discover("torch"); len(got) != 0 {
		t.Fatalf("unhealthy endpoint still discoverable: %d, want 0", len(got))
	}
}

func TestStartHeartbeatNilCheckerIsLivenessOnly(t *testing.T) {
	reg := registry.New(0)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		StartHeartbeat(ctx, testConfig(reg, nil))
		close(done)
	}()

	waitFor(t, "endpoint to be discoverable with nil checker", func() bool {
		return len(reg.Discover("torch")) == 1
	})
	cancel()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("StartHeartbeat with nil Checker did not return after cancel")
	}
}

func TestHandlerStatusCodes(t *testing.T) {
	cases := []struct {
		status Status
		want   int
	}{
		{StatusHealthy, http.StatusOK},
		{StatusDegraded, http.StatusOK},
		{StatusStarting, http.StatusServiceUnavailable},
		{StatusUnhealthy, http.StatusServiceUnavailable},
	}
	for _, tc := range cases {
		checker := &SimpleChecker{
			ModuleName: "torch",
			Version:    "1.2.3",
			StartTime:  time.Now(),
			CheckFn: func() (Status, map[string]Check) {
				return tc.status, nil
			},
		}
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, "/health", nil)
		Handler(checker).ServeHTTP(rec, req)

		if rec.Code != tc.want {
			t.Errorf("status %s: HTTP %d, want %d", tc.status, rec.Code, tc.want)
		}
		var report Report
		if err := json.NewDecoder(rec.Body).Decode(&report); err != nil {
			t.Fatalf("status %s: decode report: %v", tc.status, err)
		}
		if report.Module != "torch" || report.Status != tc.status || report.Version != "1.2.3" {
			t.Errorf("report = %+v, want module=torch status=%s version=1.2.3", report, tc.status)
		}
		if report.Timestamp.IsZero() {
			t.Errorf("status %s: handler must stamp Timestamp", tc.status)
		}
	}
}
