// Package cache is the platform's exact-match response cache.
//
// One module, one job: store a completed response under a caller-built
// key and hand it back until its TTL lapses. No fuzzy matching, no
// semantic claims. Retention is bounded: a configured entry cap evicts
// the soonest-to-expire entries (expired ones first), so unique-key
// traffic cannot grow the map for the life of the process. Expiry is
// lazy (checked on Get) plus an explicit PurgeExpired for janitors.
// The clock is injectable so TTL behavior is proven in tests without
// sleeping.
package cache

import (
	"sync"
	"sync/atomic"
	"time"
)

type entry[V any] struct {
	value     V
	expiresAt time.Time
}

// Cache is a goroutine-safe exact-match TTL cache.
type Cache[V any] struct {
	mu         sync.Mutex
	items      map[string]entry[V]
	ttl        time.Duration
	maxEntries int // <= 0 means no cap
	now        func() time.Time

	hits   atomic.Int64
	misses atomic.Int64
}

// New builds a cache whose entries live for ttl, with no entry cap.
// A ttl <= 0 disables retention: Set becomes a no-op and every Get
// misses, which is the honest "cache off" mode rather than a silent
// infinite cache.
func New[V any](ttl time.Duration) *Cache[V] {
	return NewWithLimit[V](ttl, 0)
}

// NewWithLimit builds a cache capped at maxEntries entries. When a
// Set of a new key would cross the cap, expired entries are dropped
// first, then the entries closest to expiry, until there is room.
// A maxEntries <= 0 means no cap.
func NewWithLimit[V any](ttl time.Duration, maxEntries int) *Cache[V] {
	if maxEntries < 0 {
		maxEntries = 0
	}
	return &Cache[V]{
		items:      make(map[string]entry[V]),
		ttl:        ttl,
		maxEntries: maxEntries,
		now:        time.Now,
	}
}

// SetClock replaces the time source. Tests use it to control expiry.
func (c *Cache[V]) SetClock(now func() time.Time) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = now
}

// Get returns the stored value and true, or the zero value and false when
// the key is absent or expired.
func (c *Cache[V]) Get(key string) (V, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.items[key]
	if !ok || !c.now().Before(e.expiresAt) {
		if ok {
			delete(c.items, key) // expired on read
		}
		c.misses.Add(1)
		var zero V
		return zero, false
	}
	c.hits.Add(1)
	return e.value, true
}

// Set stores value under key for the cache's TTL, evicting to the
// entry cap first when key is new (see NewWithLimit).
func (c *Cache[V]) Set(key string, value V) {
	if c.ttl <= 0 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, exists := c.items[key]; !exists && c.maxEntries > 0 {
		c.makeRoomLocked()
	}
	c.items[key] = entry[V]{value: value, expiresAt: c.now().Add(c.ttl)}
}

// makeRoomLocked frees one slot for a new key: expired entries go
// first, then the entries with the earliest expiry, until the cache
// is below its cap. Caller holds mu.
func (c *Cache[V]) makeRoomLocked() {
	if len(c.items) < c.maxEntries {
		return
	}
	now := c.now()
	for k, e := range c.items {
		if len(c.items) < c.maxEntries {
			return
		}
		if !now.Before(e.expiresAt) {
			delete(c.items, k)
		}
	}
	for len(c.items) >= c.maxEntries {
		var victim string
		var victimExp time.Time
		first := true
		for k, e := range c.items {
			if first || e.expiresAt.Before(victimExp) {
				victim, victimExp, first = k, e.expiresAt, false
			}
		}
		delete(c.items, victim)
	}
}

// Len reports the number of stored entries (including not-yet-purged
// expired ones).
func (c *Cache[V]) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.items)
}

// PurgeExpired drops every expired entry and returns how many went.
func (c *Cache[V]) PurgeExpired() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	now := c.now()
	n := 0
	for k, e := range c.items {
		if !now.Before(e.expiresAt) {
			delete(c.items, k)
			n++
		}
	}
	return n
}

// Stats is the cache's debug summary.
type Stats struct {
	Entries    int   `json:"entries"`
	MaxEntries int   `json:"max_entries"`
	Hits       int64 `json:"hits"`
	Misses     int64 `json:"misses"`
}

// Stats snapshots counters for the /debug endpoint.
func (c *Cache[V]) Stats() Stats {
	c.mu.Lock()
	maxEntries := c.maxEntries
	c.mu.Unlock()
	return Stats{
		Entries:    c.Len(),
		MaxEntries: maxEntries,
		Hits:       c.hits.Load(),
		Misses:     c.misses.Load(),
	}
}
