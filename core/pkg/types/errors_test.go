package types

import (
	"errors"
	"testing"
)

func TestSentinelErrors(t *testing.T) {
	sentinels := []error{
		ErrNotFound, ErrUnauthorized, ErrForbidden, ErrTimeout,
		ErrUnavailable, ErrInvalidInput, ErrResourceExhausted,
		ErrModelNotLoaded, ErrInternal,
	}
	for _, sentinel := range sentinels {
		if sentinel == nil || sentinel.Error() == "" {
			t.Errorf("sentinel error %v is nil or empty", sentinel)
		}
		if !errors.Is(sentinel, sentinel) {
			t.Errorf("errors.Is(%v, itself) = false", sentinel)
		}
	}
}

func TestWrapError(t *testing.T) {
	err := WrapError("torch", "Inference", ErrModelNotLoaded)

	if got, want := err.Error(), "torch.Inference: model not loaded"; got != want {
		t.Errorf("Error() = %q, want %q", got, want)
	}
	if !errors.Is(err, ErrModelNotLoaded) {
		t.Error("errors.Is(wrapped, ErrModelNotLoaded) = false, want true")
	}

	var modErr *ModuleError
	if !errors.As(err, &modErr) {
		t.Fatal("errors.As did not recover *ModuleError")
	}
	if modErr.Module != "torch" || modErr.Op != "Inference" {
		t.Errorf("ModuleError = %q.%q, want torch.Inference", modErr.Module, modErr.Op)
	}
	if modErr.Version != Version {
		t.Errorf("ModuleError.Version = %d, want %d", modErr.Version, Version)
	}
}
