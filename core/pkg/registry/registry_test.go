package registry

import (
	"net/http/httptest"
	"testing"
	"time"
)

func testEndpoint() ServiceEndpoint {
	return ServiceEndpoint{
		Module:   "torch",
		Host:     "127.0.0.1",
		Port:     11435,
		Protocol: ProtocolREST,
		Metadata: map[string]string{"backend": "vulkan"},
	}
}

func TestRegisterAndDiscover(t *testing.T) {
	r := New(0)
	ep := testEndpoint()
	r.Register(ep)

	got := r.Discover("torch")
	if len(got) != 1 {
		t.Fatalf("Discover returned %d endpoints, want 1", len(got))
	}
	if got[0].Module != "torch" || got[0].Host != "127.0.0.1" || got[0].Port != 11435 {
		t.Errorf("endpoint identity = %+v, want torch@127.0.0.1:11435", got[0])
	}
	if got[0].Protocol != ProtocolREST {
		t.Errorf("protocol = %q, want %q", got[0].Protocol, ProtocolREST)
	}
	if !got[0].Healthy {
		t.Error("freshly registered endpoint must be healthy")
	}
	if got[0].LastSeen.IsZero() {
		t.Error("Register must stamp LastSeen")
	}
	if got[0].Metadata["backend"] != "vulkan" {
		t.Errorf("metadata = %v, want backend=vulkan carried through", got[0].Metadata)
	}
	if got[0].Address() != "127.0.0.1:11435" {
		t.Errorf("Address() = %q, want 127.0.0.1:11435", got[0].Address())
	}

	one, err := r.DiscoverOne("torch")
	if err != nil {
		t.Fatalf("DiscoverOne: %v", err)
	}
	if one.Address() != "127.0.0.1:11435" {
		t.Errorf("DiscoverOne = %q, want 127.0.0.1:11435", one.Address())
	}
}

func TestRegisterUpdatesInPlace(t *testing.T) {
	r := New(0)
	ep := testEndpoint()
	r.Register(ep)

	ep.Metadata = map[string]string{"backend": "rocm"}
	r.Register(ep)

	all := r.All()["torch"]
	if len(all) != 1 {
		t.Fatalf("re-register created %d records, want 1 (update in place)", len(all))
	}
	if all[0].Metadata["backend"] != "rocm" {
		t.Errorf("metadata after update = %v, want backend=rocm", all[0].Metadata)
	}
}

func TestHealthFlagBehavior(t *testing.T) {
	r := New(0)
	ep := testEndpoint()
	r.Register(ep)

	r.MarkUnhealthy("torch", "127.0.0.1", 11435)
	if got := r.Discover("torch"); len(got) != 0 {
		t.Fatalf("Discover returned %d endpoints after MarkUnhealthy, want 0", len(got))
	}
	all := r.All()["torch"]
	if len(all) != 1 {
		t.Fatalf("All lost the unhealthy endpoint: %d records, want 1", len(all))
	}
	if all[0].Healthy {
		t.Error("All must show Healthy=false after MarkUnhealthy")
	}
	if _, err := r.DiscoverOne("torch"); err == nil {
		t.Error("DiscoverOne must error when no healthy endpoint exists")
	}

	r.Heartbeat("torch", "127.0.0.1", 11435)
	if got := r.Discover("torch"); len(got) != 1 || !got[0].Healthy {
		t.Errorf("after Heartbeat, Discover = %+v, want the endpoint back and healthy", got)
	}
}

func TestDeregister(t *testing.T) {
	r := New(0)
	r.Register(testEndpoint())
	r.Deregister("torch", "127.0.0.1", 11435)

	if got := r.All()["torch"]; len(got) != 0 {
		t.Fatalf("All after Deregister = %d records, want 0", len(got))
	}
	// Deregistering an absent endpoint is a no-op, not a panic.
	r.Deregister("torch", "127.0.0.1", 11435)
	r.Deregister("never-registered", "127.0.0.1", 1)
}

func TestTTLExpiry(t *testing.T) {
	r := New(30 * time.Millisecond)
	r.Register(testEndpoint())

	time.Sleep(90 * time.Millisecond)
	if got := r.Discover("torch"); len(got) != 0 {
		t.Fatalf("stale endpoint survived TTL: Discover = %d, want 0", len(got))
	}
	if got := r.All()["torch"]; len(got) != 0 {
		t.Fatalf("stale endpoint was filtered but not evicted: All = %d, want 0", len(got))
	}
}

func TestClientTimeoutIsConfigWithDefault(t *testing.T) {
	if got := (ClientConfig{}).timeout(); got != DefaultClientTimeout {
		t.Errorf("unset Timeout resolved to %v, want the documented default %v", got, DefaultClientTimeout)
	}
	custom := ClientConfig{Timeout: 250 * time.Millisecond}
	if got := custom.timeout(); got != 250*time.Millisecond {
		t.Errorf("configured Timeout resolved to %v, want 250ms honored (limit must come from config)", got)
	}
	if c := NewClient("http://127.0.0.1:1", ""); c.httpClient.Timeout != DefaultClientTimeout {
		t.Errorf("NewClient timeout = %v, want default %v", c.httpClient.Timeout, DefaultClientTimeout)
	}
	if c := NewClientWithConfig(custom); c.httpClient.Timeout != 250*time.Millisecond {
		t.Errorf("NewClientWithConfig timeout = %v, want 250ms", c.httpClient.Timeout)
	}
}

func TestClientServerRoundTrip(t *testing.T) {
	reg := New(0)
	srv := httptest.NewServer(NewServer(reg).Handler())
	defer srv.Close()

	c := NewClient(srv.URL, "")
	ep := testEndpoint()
	if err := c.Register(ep); err != nil {
		t.Fatalf("client Register: %v", err)
	}

	got, err := c.Discover("torch")
	if err != nil {
		t.Fatalf("client Discover: %v", err)
	}
	if len(got) != 1 || got[0].Address() != "127.0.0.1:11435" || !got[0].Healthy {
		t.Fatalf("client Discover = %+v, want the one registered healthy endpoint", got)
	}

	reg.MarkUnhealthy("torch", "127.0.0.1", 11435)
	if err := c.Heartbeat("torch", "127.0.0.1", 11435); err != nil {
		t.Fatalf("client Heartbeat: %v", err)
	}
	got, err = c.Discover("torch")
	if err != nil || len(got) != 1 {
		t.Fatalf("after remote Heartbeat, Discover = %+v (err %v), want endpoint healthy again", got, err)
	}

	all, err := c.All()
	if err != nil {
		t.Fatalf("client All: %v", err)
	}
	if len(all["torch"]) != 1 {
		t.Fatalf("client All = %v, want one torch endpoint", all)
	}

	if err := c.Deregister("torch", "127.0.0.1", 11435); err != nil {
		t.Fatalf("client Deregister: %v", err)
	}
	got, err = c.Discover("torch")
	if err != nil {
		t.Fatalf("client Discover after Deregister: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("Discover after Deregister = %d endpoints, want 0", len(got))
	}
}
