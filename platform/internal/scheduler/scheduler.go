// Package scheduler is the platform's request scheduling gate.
//
// It bounds how many engine jobs run at once and queues the rest in two
// priority lanes. The design answers a specific defect in the old
// platform: its batch scheduler never drained its high-priority queue
// (scheduler.go:310-313 over there), so elevated requests hung forever.
// Here both lanes are served by one grant loop: whenever a slot frees,
// the head of the high lane goes first, then the normal lane, and the
// loop keeps granting until slots or waiters run out. The tests prove
// each lane drains on its own and that both drain together.
package scheduler

import (
	"context"
	"errors"
	"sync"
)

// Priority selects a scheduling lane.
type Priority int

const (
	// PriorityNormal is the default lane for chat and embedding traffic.
	PriorityNormal Priority = iota
	// PriorityHigh is for operator and interactive traffic.
	PriorityHigh
)

// ErrClosed is returned to waiters and new callers after Close.
var ErrClosed = errors.New("scheduler: closed")

type waiter struct {
	grant chan error // buffered 1: receives nil on grant, ErrClosed on Close
}

// Scheduler is a goroutine-safe priority gate over engine work.
type Scheduler struct {
	mu      sync.Mutex
	max     int
	running int
	high    []*waiter
	normal  []*waiter
	closed  bool
}

// New builds a scheduler allowing maxConcurrent jobs at once. Values
// below 1 are clamped to 1: a zero-slot scheduler would hang every
// caller, which is a configuration error, not a mode.
func New(maxConcurrent int) *Scheduler {
	if maxConcurrent < 1 {
		maxConcurrent = 1
	}
	return &Scheduler{max: maxConcurrent}
}

// Acquire blocks until a slot is granted, the context ends, or the
// scheduler closes. The returned release func frees the slot and must be
// called exactly once; it is safe to call more than once.
func (s *Scheduler) Acquire(ctx context.Context, p Priority) (release func(), err error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil, ErrClosed
	}
	if s.running < s.max && len(s.high) == 0 && len(s.normal) == 0 {
		s.running++
		s.mu.Unlock()
		return s.releaseFunc(), nil
	}
	w := &waiter{grant: make(chan error, 1)}
	if p == PriorityHigh {
		s.high = append(s.high, w)
	} else {
		s.normal = append(s.normal, w)
	}
	s.mu.Unlock()

	select {
	case gerr := <-w.grant:
		if gerr != nil {
			return nil, gerr
		}
		return s.releaseFunc(), nil
	case <-ctx.Done():
		s.mu.Lock()
		removed := s.removeLocked(w)
		s.mu.Unlock()
		if removed {
			return nil, ctx.Err()
		}
		// The grant raced the cancellation and won: take the slot only
		// to hand it straight back, then report the context error.
		if gerr := <-w.grant; gerr == nil {
			s.release()
		}
		return nil, ctx.Err()
	}
}

func (s *Scheduler) removeLocked(w *waiter) bool {
	for i, q := range s.high {
		if q == w {
			s.high = append(s.high[:i], s.high[i+1:]...)
			return true
		}
	}
	for i, q := range s.normal {
		if q == w {
			s.normal = append(s.normal[:i], s.normal[i+1:]...)
			return true
		}
	}
	return false
}

func (s *Scheduler) releaseFunc() func() {
	var once sync.Once
	return func() { once.Do(s.release) }
}

func (s *Scheduler) release() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.running--
	if s.closed {
		return
	}
	s.grantLocked()
}

// grantLocked hands free slots to waiters, high lane first, until slots
// or waiters are exhausted. Caller holds mu.
func (s *Scheduler) grantLocked() {
	for s.running < s.max {
		var w *waiter
		switch {
		case len(s.high) > 0:
			w = s.high[0]
			s.high = s.high[1:]
		case len(s.normal) > 0:
			w = s.normal[0]
			s.normal = s.normal[1:]
		default:
			return
		}
		s.running++
		w.grant <- nil
	}
}

// Close fails every queued waiter with ErrClosed and rejects new
// Acquires. Running jobs are untouched; their releases still land.
func (s *Scheduler) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return
	}
	s.closed = true
	for _, w := range s.high {
		w.grant <- ErrClosed
	}
	for _, w := range s.normal {
		w.grant <- ErrClosed
	}
	s.high, s.normal = nil, nil
}

// Stats is the scheduler's debug summary.
type Stats struct {
	Max          int `json:"max"`
	Running      int `json:"running"`
	HighQueued   int `json:"high_queued"`
	NormalQueued int `json:"normal_queued"`
}

// Stats snapshots the gate for the /debug endpoint.
func (s *Scheduler) Stats() Stats {
	s.mu.Lock()
	defer s.mu.Unlock()
	return Stats{
		Max:          s.max,
		Running:      s.running,
		HighQueued:   len(s.high),
		NormalQueued: len(s.normal),
	}
}
