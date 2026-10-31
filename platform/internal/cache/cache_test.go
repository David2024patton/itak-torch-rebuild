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

func TestEntryCapEvictsSoonestToExpire(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	c := cache.NewWithLimit[string](time.Minute, 2)
	c.SetClock(func() time.Time { return now })
	c.Set("a", "1") // expires first
	now = now.Add(time.Second)
	c.Set("b", "2")
	now = now.Add(time.Second)
	c.Set("c", "3") // crosses the cap: "a" is closest to expiry
	if c.Len() != 2 {
		t.Fatalf("Len = %d, want 2 at the cap", c.Len())
	}
	if _, ok := c.Get("a"); ok {
		t.Fatal("soonest-to-expire entry survived eviction")
	}
	if _, ok := c.Get("b"); !ok {
		t.Fatal("entry b was wrongly evicted")
	}
	if _, ok := c.Get("c"); !ok {
		t.Fatal("freshly set entry c was evicted")
	}
}

func TestEntryCapDropsExpiredBeforeEvictingLive(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	c := cache.NewWithLimit[string](10*time.Second, 2)
	c.SetClock(func() time.Time { return now })
	c.Set("old", "1")
	now = now.Add(9 * time.Second)
	c.Set("live", "2")
	now = now.Add(2 * time.Second) // old is 11s old (expired), live is 2s old
	c.Set("new", "3")
	if _, ok := c.Get("live"); !ok {
		t.Fatal("live entry evicted while an expired one was available to drop")
	}
	if _, ok := c.Get("new"); !ok {
		t.Fatal("freshly set entry missing")
	}
	if c.Len() != 2 {
		t.Fatalf("Len = %d, want 2", c.Len())
	}
}

func TestEntryCapOverwriteDoesNotEvict(t *testing.T) {
	c := cache.NewWithLimit[string](time.Minute, 2)
	c.Set("a", "1")
	c.Set("b", "2")
	c.Set("a", "3") // overwrite of an existing key needs no room
	if c.Len() != 2 {
		t.Fatalf("Len = %d, want 2", c.Len())
	}
	if got, ok := c.Get("a"); !ok || got != "3" {
		t.Fatalf("Get(a) = %q, %v; want the overwritten value", got, ok)
	}
	if _, ok := c.Get("b"); !ok {
		t.Fatal("overwrite evicted an unrelated entry")
	}
}
