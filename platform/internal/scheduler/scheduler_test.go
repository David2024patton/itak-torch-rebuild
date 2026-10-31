package scheduler_test

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/David2024patton/itak-torch-rebuild/platform/internal/scheduler"
)

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

// holdSlot takes the only slot of a max=1 scheduler.
func holdSlot(t *testing.T, s *scheduler.Scheduler) func() {
	t.Helper()
	rel, err := s.Acquire(context.Background(), scheduler.PriorityNormal)
	if err != nil {
		t.Fatalf("Acquire holder: %v", err)
	}
	return rel
}

// queueWaiter starts a goroutine that acquires, records its name in the
// completion order, and releases immediately.
func queueWaiter(s *scheduler.Scheduler, p scheduler.Priority, name string,
	mu *sync.Mutex, order *[]string, wg *sync.WaitGroup,
) {
	wg.Add(1)
	go func() {
		defer wg.Done()
		rel, err := s.Acquire(context.Background(), p)
		if err != nil {
			return
		}
		mu.Lock()
		*order = append(*order, name)
		mu.Unlock()
		rel()
	}()
}

func TestBothLanesDrain(t *testing.T) {
	s := scheduler.New(1, 16)
	releaseHolder := holdSlot(t, s)

	var mu sync.Mutex
	var order []string
	var wg sync.WaitGroup
	for i := 0; i < 3; i++ {
		queueWaiter(s, scheduler.PriorityHigh, "high", &mu, &order, &wg)
		queueWaiter(s, scheduler.PriorityNormal, "normal", &mu, &order, &wg)
	}
	waitFor(t, "all six waiters queued", func() bool {
		st := s.Stats()
		return st.HighQueued+st.NormalQueued == 6
	})

	releaseHolder()
	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatalf("lanes did not drain; completed order so far: %v", order)
	}
	if len(order) != 6 {
		t.Fatalf("completed %d jobs, want 6", len(order))
	}
}

// TestHighLaneDrainsAlone is the direct regression shape for the old
// platform bug, where the high-priority queue was never served.
func TestHighLaneDrainsAlone(t *testing.T) {
	s := scheduler.New(1, 16)
	releaseHolder := holdSlot(t, s)

	var mu sync.Mutex
	var order []string
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		queueWaiter(s, scheduler.PriorityHigh, "high", &mu, &order, &wg)
	}
	waitFor(t, "high lane queued", func() bool { return s.Stats().HighQueued == 4 })

	releaseHolder()
	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("high-priority lane did not drain")
	}
	if len(order) != 4 {
		t.Fatalf("completed %d high jobs, want 4", len(order))
	}
}

func TestHighBeforeNormal(t *testing.T) {
	s := scheduler.New(1, 16)
	releaseHolder := holdSlot(t, s)

	var mu sync.Mutex
	var order []string
	var wg sync.WaitGroup
	queueWaiter(s, scheduler.PriorityNormal, "normal", &mu, &order, &wg)
	waitFor(t, "normal queued", func() bool { return s.Stats().NormalQueued == 1 })
	queueWaiter(s, scheduler.PriorityHigh, "high", &mu, &order, &wg)
	waitFor(t, "high queued", func() bool { return s.Stats().HighQueued == 1 })

	releaseHolder()
	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("waiters did not finish")
	}
	if len(order) != 2 || order[0] != "high" || order[1] != "normal" {
		t.Fatalf("grant order = %v, want [high normal]", order)
	}
}

func TestContextCancelWithdrawsCleanly(t *testing.T) {
	s := scheduler.New(1, 16)
	releaseHolder := holdSlot(t, s)

	ctx, cancel := context.WithCancel(context.Background())
	errCh := make(chan error, 1)
	go func() {
		_, err := s.Acquire(ctx, scheduler.PriorityHigh)
		errCh <- err
	}()
	waitFor(t, "waiter queued", func() bool { return s.Stats().HighQueued == 1 })
	cancel()
	select {
	case err := <-errCh:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Acquire err = %v, want context.Canceled", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("cancelled waiter never returned")
	}
	if got := s.Stats().HighQueued; got != 0 {
		t.Fatalf("cancelled waiter still queued: %d", got)
	}

	// The withdrawal must not eat a slot: after the holder releases, a
	// fresh Acquire succeeds immediately.
	releaseHolder()
	rel, err := s.Acquire(context.Background(), scheduler.PriorityNormal)
	if err != nil {
		t.Fatalf("Acquire after cancel+release: %v", err)
	}
	rel()
}

func TestCloseFailsWaitersAndRejectsNew(t *testing.T) {
	s := scheduler.New(1, 16)
	releaseHolder := holdSlot(t, s)
	defer releaseHolder()

	errCh := make(chan error, 1)
	go func() {
		_, err := s.Acquire(context.Background(), scheduler.PriorityNormal)
		errCh <- err
	}()
	waitFor(t, "waiter queued", func() bool { return s.Stats().NormalQueued == 1 })

	s.Close()
	select {
	case err := <-errCh:
		if !errors.Is(err, scheduler.ErrClosed) {
			t.Fatalf("waiter err = %v, want ErrClosed", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("Close did not fail the queued waiter")
	}
	if _, err := s.Acquire(context.Background(), scheduler.PriorityNormal); !errors.Is(err, scheduler.ErrClosed) {
		t.Fatalf("Acquire after Close err = %v, want ErrClosed", err)
	}
}

// TestQueueFullRefusesByName is the Wave 1 bound: a lane at its
// configured waiter cap refuses the next Acquire immediately with
// ErrQueueFull instead of queueing without bound.
func TestQueueFullRefusesByName(t *testing.T) {
	for _, lane := range []scheduler.Priority{scheduler.PriorityNormal, scheduler.PriorityHigh} {
		s := scheduler.New(1, 2)
		releaseHolder := holdSlot(t, s)

		errCh := make(chan error, 2)
		for i := 0; i < 2; i++ {
			go func() {
				_, err := s.Acquire(context.Background(), lane)
				errCh <- err
			}()
		}
		waitFor(t, "lane filled to its cap", func() bool {
			st := s.Stats()
			return st.HighQueued+st.NormalQueued == 2
		})

		// The next waiter is refused at once, by name; it must not
		// block behind the full lane.
		refused := make(chan error, 1)
		go func() {
			_, err := s.Acquire(context.Background(), lane)
			refused <- err
		}()
		select {
		case err := <-refused:
			if !errors.Is(err, scheduler.ErrQueueFull) {
				t.Fatalf("lane %v: Acquire past the cap err = %v, want ErrQueueFull", lane, err)
			}
		case <-time.After(3 * time.Second):
			t.Fatalf("lane %v: Acquire past the cap blocked instead of refusing", lane)
		}

		// Draining the lane makes room again: once the holder and
		// one waiter release, a fresh Acquire is granted.
		releaseHolder()
		if err := <-errCh; err != nil {
			t.Fatalf("lane %v: queued waiter err = %v, want grant", lane, err)
		}
	}
}

// TestZeroWaiterCapDisablesQueueing pins the cap=0 mode: nothing
// waits, any Acquire that cannot be granted at once refuses.
func TestZeroWaiterCapDisablesQueueing(t *testing.T) {
	s := scheduler.New(1, 0)
	releaseHolder := holdSlot(t, s)
	defer releaseHolder()
	if _, err := s.Acquire(context.Background(), scheduler.PriorityNormal); !errors.Is(err, scheduler.ErrQueueFull) {
		t.Fatalf("Acquire with a zero waiter cap err = %v, want ErrQueueFull", err)
	}
}
