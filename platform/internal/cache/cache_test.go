package cache_test

import (
	"testing"
	"time"

	"github.com/David2024patton/itak-torch-rebuild/platform/internal/cache"
)

func TestSetGetRoundTrip(t *testing.T) {
	c := cache.New[string](time.Minute)
	c.Set("k", "v")
	got, ok := c.Get("k")
	if !ok || got != "v" {
		t.Fatalf("Get = %q, %v; want %q, true", got, ok, "v")
	}
}

func TestMissOnAbsentKey(t *testing.T) {
	c := cache.New[string](time.Minute)
	if _, ok := c.Get("never-set"); ok {
		t.Fatal("Get on absent key returned ok")
	}
}

func TestTTLExpiryWithInjectedClock(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	c := cache.New[string](30 * time.Second)
	c.SetClock(func() time.Time { return now })
	c.Set("k", "v")

	now = now.Add(29 * time.Second)
	if _, ok := c.Get("k"); !ok {
		t.Fatal("entry expired before its TTL")
	}
	now = now.Add(2 * time.Second) // 31s after Set
	if _, ok := c.Get("k"); ok {
		t.Fatal("entry still present after its TTL lapsed")
	}
	if c.Len() != 0 {
		t.Fatalf("expired entry not dropped on read: Len = %d", c.Len())
	}
}

func TestZeroTTLDisablesRetention(t *testing.T) {
	c := cache.New[string](0)
	c.Set("k", "v")
	if _, ok := c.Get("k"); ok {
		t.Fatal("zero-TTL cache retained an entry")
	}
	if c.Len() != 0 {
		t.Fatalf("zero-TTL cache Len = %d, want 0", c.Len())
	}
}

func TestPurgeExpired(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	c := cache.New[int](10 * time.Second)
	c.SetClock(func() time.Time { return now })
	c.Set("a", 1)
	now = now.Add(5 * time.Second)
	c.Set("b", 2)
	now = now.Add(6 * time.Second) // a is 11s old (expired), b is 6s old
	if n := c.PurgeExpired(); n != 1 {
		t.Fatalf("PurgeExpired removed %d, want 1", n)
	}
	if _, ok := c.Get("b"); !ok {
		t.Fatal("unexpired entry was purged")
	}
}

func TestStatsCountHitsAndMisses(t *testing.T) {
	c := cache.New[string](time.Minute)
	c.Set("k", "v")
	_, _ = c.Get("k")      // hit
	_, _ = c.Get("absent") // miss
	s := c.Stats()
	if s.Hits != 1 || s.Misses != 1 || s.Entries != 1 {
		t.Fatalf("Stats = %+v, want hits=1 misses=1 entries=1", s)
	}
}
