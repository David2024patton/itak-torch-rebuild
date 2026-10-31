package auth_test

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/David2024patton/itak-torch-rebuild/platform/internal/auth"
)

type recorder struct{ fails int }

func (r *recorder) RecordAuthFailure() { r.fails++ }

func okHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
}

func serve(h http.Handler, path string, headers map[string]string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodGet, path, nil)
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func TestBearerKeyAccepted(t *testing.T) {
	h := auth.New("s3cret", nil).Wrap(okHandler())
	rec := serve(h, "/v1/models", map[string]string{"Authorization": "Bearer s3cret"})
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
}

func TestAPIKeyHeaderAccepted(t *testing.T) {
	h := auth.New("s3cret", nil).Wrap(okHandler())
	rec := serve(h, "/v1/models", map[string]string{"X-API-Key": "s3cret"})
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
}

func TestMissingKeyRejected(t *testing.T) {
	rec := &recorder{}
	h := auth.New("s3cret", rec).Wrap(okHandler())
	out := serve(h, "/v1/models", nil)
	if out.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401", out.Code)
	}
	if rec.fails != 1 {
		t.Fatalf("recorded failures = %d, want 1", rec.fails)
	}
}

func TestWrongKeyRejected(t *testing.T) {
	h := auth.New("s3cret", nil).Wrap(okHandler())
	rec := serve(h, "/v1/models", map[string]string{"Authorization": "Bearer wrong"})
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401", rec.Code)
	}
}

func TestEmptyConfiguredKeyFailsClosed(t *testing.T) {
	h := auth.New("", nil).Wrap(okHandler())
	rec := serve(h, "/v1/models", map[string]string{"Authorization": "Bearer anything"})
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401 (empty configured key must fail closed)", rec.Code)
	}
}

func TestExemptPathSkipsKey(t *testing.T) {
	h := auth.New("s3cret", nil, "/healthz").Wrap(okHandler())
	rec := serve(h, "/healthz", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 for exempt path", rec.Code)
	}
	rec = serve(h, "/debug", nil)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("/debug must not be exempt: status = %d, want 401", rec.Code)
	}
}

// TestRepeatedFailuresAreThrottled is the Wave 1 auth hygiene fix:
// the first failures are answered at once, then each failure from
// the same source is delayed, and a success resets the count.
func TestRepeatedFailuresAreThrottled(t *testing.T) {
	h := auth.New("s3cret", nil).Wrap(okHandler())
	wrong := map[string]string{"Authorization": "Bearer wrong"}
	right := map[string]string{"Authorization": "Bearer s3cret"}

	// The threshold failures are immediate.
	for i := 0; i < 3; i++ {
		start := time.Now()
		rec := serve(h, "/v1/models", wrong)
		if rec.Code != http.StatusUnauthorized {
			t.Fatalf("failure %d status = %d, want 401", i+1, rec.Code)
		}
		if elapsed := time.Since(start); elapsed > 150*time.Millisecond {
			t.Fatalf("failure %d took %s; the threshold failures must not be delayed", i+1, elapsed)
		}
	}
	// The next failure pays the first throttle step (200ms).
	start := time.Now()
	if rec := serve(h, "/v1/models", wrong); rec.Code != http.StatusUnauthorized {
		t.Fatalf("throttled failure status = %d, want 401", rec.Code)
	}
	if elapsed := time.Since(start); elapsed < 150*time.Millisecond {
		t.Fatalf("fourth failure answered in %s; want the throttle delay", elapsed)
	}
	// A success answers at once and clears the count, so the
	// failure after it is immediate again.
	start = time.Now()
	if rec := serve(h, "/v1/models", right); rec.Code != http.StatusOK {
		t.Fatalf("success status = %d, want 200", rec.Code)
	}
	if elapsed := time.Since(start); elapsed > 150*time.Millisecond {
		t.Fatalf("success took %s; successes are never throttled", elapsed)
	}
	start = time.Now()
	if rec := serve(h, "/v1/models", wrong); rec.Code != http.StatusUnauthorized {
		t.Fatalf("post-reset failure status = %d, want 401", rec.Code)
	}
	if elapsed := time.Since(start); elapsed > 150*time.Millisecond {
		t.Fatalf("failure after a success took %s; the count must have reset", elapsed)
	}
}
