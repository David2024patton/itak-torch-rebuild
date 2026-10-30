package event

import (
	"testing"
	"time"
)

func TestNewEventBus(t *testing.T) {
	bus := New()
	if bus == nil {
		t.Fatal("New() returned nil")
	}
	if bus.SubscriberCount() != 0 {
		t.Fatalf("expected 0 subscribers, got %d", bus.SubscriberCount())
	}
}

func TestPublishSubscribe(t *testing.T) {
	bus := New()
	defer bus.Close()

	subID, ch := bus.Subscribe(16)
	if subID < 0 {
		t.Fatal("Subscribe returned negative ID")
	}

	// Publish an event.
	bus.Publish(Event{
		Topic:   TopicUserInput,
		Message: "hello world",
	})

	// Should receive the event within 1 second.
	select {
	case e := <-ch:
		if e.Topic != TopicUserInput {
			t.Errorf("expected topic %q, got %q", TopicUserInput, e.Topic)
		}
		if e.Message != "hello world" {
			t.Errorf("expected message %q, got %q", "hello world", e.Message)
		}
		if e.Timestamp.IsZero() {
			t.Error("expected non-zero timestamp")
		}
	case <-time.After(1 * time.Second):
		t.Fatal("timed out waiting for event")
	}
}

func TestTopicFiltering(t *testing.T) {
	bus := New()
	defer bus.Close()

	// Subscribe only to agent events.
	_, agentCh := bus.Subscribe(16, TopicAgentStart, TopicAgentComplete)

	// Subscribe to all events.
	_, allCh := bus.Subscribe(16)

	// Publish a user event (should NOT arrive on agentCh).
	bus.Publish(NewEvent(TopicUserInput, "test message"))

	// Publish an agent event (should arrive on both).
	bus.Publish(AgentEvent(TopicAgentStart, "scout", "list files"))

	// allCh should get both events.
	select {
	case e := <-allCh:
		if e.Topic != TopicUserInput {
			t.Errorf("allCh: expected %q first, got %q", TopicUserInput, e.Topic)
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("allCh: timed out on first event")
	}

	select {
	case e := <-allCh:
		if e.Topic != TopicAgentStart {
			t.Errorf("allCh: expected %q second, got %q", TopicAgentStart, e.Topic)
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("allCh: timed out on second event")
	}

	// agentCh should only get the agent event, not user.input.
	select {
	case e := <-agentCh:
		if e.Topic != TopicAgentStart {
			t.Errorf("agentCh: expected %q, got %q", TopicAgentStart, e.Topic)
		}
		if e.Agent != "scout" {
			t.Errorf("agentCh: expected agent %q, got %q", "scout", e.Agent)
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("agentCh: timed out waiting for agent event")
	}
}

func TestUnsubscribe(t *testing.T) {
	bus := New()
	defer bus.Close()

	subID, ch := bus.Subscribe(16)

	bus.Publish(NewEvent(TopicSystemReady, "ready"))

	// Should receive.
	select {
	case <-ch:
	case <-time.After(500 * time.Millisecond):
		t.Fatal("should have received event before unsubscribe")
	}

	// Unsubscribe.
	bus.Unsubscribe(subID)

	if bus.SubscriberCount() != 0 {
		t.Fatalf("expected 0 subscribers after unsubscribe, got %d", bus.SubscriberCount())
	}
}

func TestClose(t *testing.T) {
	bus := New()

	_, ch := bus.Subscribe(16)
	bus.Close()

	// Channel should be closed after bus.Close().
	select {
	case _, ok := <-ch:
		if ok {
			t.Error("expected channel to be closed after bus.Close()")
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("channel was not closed after bus.Close()")
	}

	// Publishing to a closed bus should not panic.
	bus.Publish(NewEvent(TopicSystemShutdown, "bye"))
}

func TestEventJSON(t *testing.T) {
	e := ToolEvent(TopicAgentToolCall, "scout", "shell", "running ls", map[string]interface{}{
		"command": "ls -la",
	})

	data := e.JSON()
	if len(data) == 0 {
		t.Fatal("JSON() returned empty data")
	}

	// Should contain the topic.
	s := string(data)
	if !contains(s, `"topic":"agent.tool_call"`) {
		t.Errorf("JSON missing topic, got: %s", s)
	}
	if !contains(s, `"agent":"scout"`) {
		t.Errorf("JSON missing agent, got: %s", s)
	}
	if !contains(s, `"tool":"shell"`) {
		t.Errorf("JSON missing tool, got: %s", s)
	}
}

func TestMultipleSubscribers(t *testing.T) {
	bus := New()
	defer bus.Close()

	const numSubs = 10
	channels := make([]<-chan Event, numSubs)
	for i := 0; i < numSubs; i++ {
		_, ch := bus.Subscribe(16)
		channels[i] = ch
	}

	if bus.SubscriberCount() != numSubs {
		t.Fatalf("expected %d subscribers, got %d", numSubs, bus.SubscriberCount())
	}

	bus.Publish(NewEvent(TopicSystemReady, "test"))

	// All subscribers should receive the event.
	for i, ch := range channels {
		select {
		case e := <-ch:
			if e.Topic != TopicSystemReady {
				t.Errorf("sub %d: expected %q, got %q", i, TopicSystemReady, e.Topic)
			}
		case <-time.After(500 * time.Millisecond):
			t.Fatalf("sub %d: timed out", i)
		}
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsStr(s, substr))
}

func containsStr(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
