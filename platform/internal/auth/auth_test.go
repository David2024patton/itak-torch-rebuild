package auth_test

import (
	"net/http"
	"net/http/httptest"
	"testing"

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
