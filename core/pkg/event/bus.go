// Package event is the itak-core event bus, ported from the GOAgent
// repo's pkg/eventbus (PLAN S4, TODO T-1.2). The shape is the
// ancestor's: a channel-based fan-out bus. Publishers call Publish;
// subscribers call Subscribe to receive a channel of the events whose
// topics match their filter (no filter means all topics), and
// Unsubscribe when done. Delivery is non-blocking: an event for a
// subscriber whose channel buffer is full is dropped for that
// subscriber rather than stalling the publisher.
package event

import (
	"encoding/json"
	"sync"
	"time"

	"github.com/David2024patton/itak-torch-rebuild/core/pkg/debug"
)

// ── Topic constants ────────────────────────────────────────────────
// Every subsystem publishes to one of these typed topics.

const (
	// User lifecycle.
	TopicUserInput = "user.input"
	TopicUserExit  = "user.exit"

	// Orchestrator lifecycle.
	TopicOrchestratorThinking   = "orchestrator.thinking"
	TopicOrchestratorDelegation = "orchestrator.delegation"
	TopicOrchestratorSynthesis  = "orchestrator.synthesis"

	// Focused agent lifecycle.
	TopicAgentStart      = "agent.start"
	TopicAgentToolCall   = "agent.tool_call"
	TopicAgentToolResult = "agent.tool_result"
	TopicAgentComplete   = "agent.complete"
	TopicAgentError      = "agent.error"

	// Memory operations.
	TopicMemoryFactSaved       = "memory.fact_saved"
	TopicMemoryFactRecalled    = "memory.fact_recalled"
	TopicMemoryEntityTracked   = "memory.entity_tracked"
	TopicMemoryReflectionAdded = "memory.reflection_added"

	// Browser operations.
	TopicBrowserNavigate   = "browser.navigate"
	TopicBrowserScreenshot = "browser.screenshot"
	TopicBrowserClose      = "browser.close"

	// Debug / trace.
	TopicDebugTrace = "debug.trace"

	// File System observability.
	TopicFileSystemActivity = "fs.activity"

	// System-level.
	TopicSystemStatus   = "system.status"
	TopicSystemReady    = "system.ready"
	TopicSystemShutdown = "system.shutdown"
)

// ── Event ──────────────────────────────────────────────────────────

// Event is a single typed message flowing through the bus.
type Event struct {
	Topic     string                 `json:"topic"`
	Agent     string                 `json:"agent,omitempty"`
	Tool      string                 `json:"tool,omitempty"`
	Data      map[string]interface{} `json:"data,omitempty"`
	Message   string                 `json:"message,omitempty"`
	Timestamp time.Time              `json:"ts"`
}

// FileSystemEvent provides structured details about file system operations.
// These details are typically mapped into the Data field of an Event.
type FileSystemEvent struct {
	Action string `json:"action"` // "read", "write", "list", etc.
	Path   string `json:"path"`
	Size   int64  `json:"size,omitempty"`
}

// JSON returns the event serialized as a JSON byte slice.
func (e Event) JSON() []byte {
	b, _ := json.Marshal(e)
	return b
}

// ── Subscriber ─────────────────────────────────────────────────────

// subscriber is a single listener on the bus.
type subscriber struct {
	ch     chan Event
	topics map[string]bool // nil = all topics
	id     int
}

// ── EventBus ───────────────────────────────────────────────────────

// EventBus is a fan-out pub/sub hub backed by Go channels.
// Publishers call Publish(). Subscribers call Subscribe() to get a
// channel of events, and Unsubscribe() when they are done.
type EventBus struct {
	mu          sync.RWMutex
	subscribers map[int]*subscriber
	nextID      int
	closed      bool
}

// New creates an empty event bus ready for use.
func New() *EventBus {
	return &EventBus{
		subscribers: make(map[int]*subscriber),
	}
}

// Publish sends an event to every subscriber whose topic filter matches.
// Non-blocking: if a subscriber's channel is full, the event is dropped
// for that subscriber (the dashboard can reconnect and catch up).
func (b *EventBus) Publish(e Event) {
	if e.Timestamp.IsZero() {
		e.Timestamp = time.Now()
	}

	b.mu.RLock()
	defer b.mu.RUnlock()

	if b.closed {
		return
	}

	debug.Debug("eventbus", "Publish [%s] msg=%s subs=%d", e.Topic, truncateBus(e.Message, 80), len(b.subscribers))

	for _, sub := range b.subscribers {
		if sub.topics != nil && !sub.topics[e.Topic] {
			continue // subscriber filtered this topic out
		}
		// Non-blocking send.
		select {
		case sub.ch <- e:
		default:
			// Channel full, drop event for this subscriber.
			debug.Warn("eventbus", "Dropped event [%s] for subscriber #%d (channel full)", e.Topic, sub.id)
		}
	}
}

// Subscribe returns a channel that receives events matching the given
// topics. Pass zero topics to receive ALL events.
// The returned int is the subscription ID used for Unsubscribe().
func (b *EventBus) Subscribe(bufSize int, topics ...string) (int, <-chan Event) {
	if bufSize < 1 {
		bufSize = 64
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	id := b.nextID
	b.nextID++

	var topicMap map[string]bool
	if len(topics) > 0 {
		topicMap = make(map[string]bool, len(topics))
		for _, t := range topics {
			topicMap[t] = true
		}
	}

	sub := &subscriber{
		ch:     make(chan Event, bufSize),
		topics: topicMap,
		id:     id,
	}
	b.subscribers[id] = sub

	debug.Debug("eventbus", "Subscribe #%d (buf=%d, topics=%d, total_subs=%d)", id, bufSize, len(topics), len(b.subscribers))

	return id, sub.ch
}

// Unsubscribe removes a subscriber and closes its channel.
func (b *EventBus) Unsubscribe(id int) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if sub, ok := b.subscribers[id]; ok {
		close(sub.ch)
		delete(b.subscribers, id)
		debug.Debug("eventbus", "Unsubscribe #%d (remaining=%d)", id, len(b.subscribers))
	}
}

// Close shuts down the bus and closes all subscriber channels.
func (b *EventBus) Close() {
	b.mu.Lock()
	defer b.mu.Unlock()

	debug.Info("eventbus", "Closing event bus (%d subscribers)", len(b.subscribers))
	b.closed = true
	for id, sub := range b.subscribers {
		close(sub.ch)
		delete(b.subscribers, id)
	}
}

// SubscriberCount returns the number of active subscribers.
func (b *EventBus) SubscriberCount() int {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return len(b.subscribers)
}

// ── Helper constructors ────────────────────────────────────────────

// NewEvent creates an event with the given topic and optional message.
func NewEvent(topic, message string) Event {
	return Event{
		Topic:     topic,
		Message:   message,
		Timestamp: time.Now(),
	}
}

// AgentEvent creates an event scoped to a specific agent.
func AgentEvent(topic, agent, message string) Event {
	return Event{
		Topic:     topic,
		Agent:     agent,
		Message:   message,
		Timestamp: time.Now(),
	}
}

// ToolEvent creates an event for a tool call or result.
func ToolEvent(topic, agent, tool, message string, data map[string]interface{}) Event {
	return Event{
		Topic:     topic,
		Agent:     agent,
		Tool:      tool,
		Message:   message,
		Data:      data,
		Timestamp: time.Now(),
	}
}

// truncateBus shortens a string for bus debug logging.
func truncateBus(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "..."
}
