package health

import (
	"encoding/json"
	"net/http"
	"time"
)

// Status represents a module's health state.
type Status string

const (
	StatusHealthy   Status = "healthy"
	StatusDegraded  Status = "degraded"  // Running but with issues
	StatusUnhealthy Status = "unhealthy" // Not functional
	StatusStarting  Status = "starting"  // Still initializing
)

// Report is the standard health check response returned by every module.
type Report struct {
	Module    string            `json:"module"`
	Status    Status            `json:"status"`
	Version   string            `json:"version"`
	Uptime    string            `json:"uptime"`
	Checks    map[string]Check  `json:"checks,omitempty"`
	Metadata  map[string]string `json:"metadata,omitempty"`
	Timestamp time.Time         `json:"timestamp"`
}

// Check is a single sub-check within a health report (e.g., "gpu_available", "model_loaded").
type Check struct {
	Status  Status `json:"status"`
	Message string `json:"message,omitempty"`
}

// Checker is the interface each module implements to report its health.
type Checker interface {
	// Check returns the current health report for this module.
	HealthCheck() Report
}

// Handler returns an http.HandlerFunc that serves the health endpoint.
// Mount this at GET /health on every iTaK module.
//
//	mux.HandleFunc("GET /health", health.Handler(myModule))
func Handler(checker Checker) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		report := checker.HealthCheck()
		report.Timestamp = time.Now()

		w.Header().Set("Content-Type", "application/json")

		switch report.Status {
		case StatusHealthy:
			w.WriteHeader(http.StatusOK)
		case StatusDegraded:
			w.WriteHeader(http.StatusOK) // 200 but degraded in body
		case StatusStarting:
			w.WriteHeader(http.StatusServiceUnavailable)
		default:
			w.WriteHeader(http.StatusServiceUnavailable)
		}

		json.NewEncoder(w).Encode(report)
	}
}

// SimpleChecker is a convenience implementation for modules that just
// need to report basic status. For complex checks, implement Checker directly.
type SimpleChecker struct {
	ModuleName string
	Version    string
	StartTime  time.Time
	CheckFn    func() (Status, map[string]Check) // Optional custom checks
}

// HealthCheck returns a Report built from the SimpleChecker fields.
func (sc *SimpleChecker) HealthCheck() Report {
	status := StatusHealthy
	var checks map[string]Check

	if sc.CheckFn != nil {
		status, checks = sc.CheckFn()
	}

	return Report{
		Module:    sc.ModuleName,
		Status:    status,
		Version:   sc.Version,
		Uptime:    time.Since(sc.StartTime).Truncate(time.Second).String(),
		Checks:    checks,
		Timestamp: time.Now(),
	}
}
