package types

import "errors"

// Standard error types for cross-module communication.
// Every module should return these instead of custom error strings
// so callers can switch on error type without parsing messages.
var (
	// ErrNotFound indicates the requested resource does not exist.
	ErrNotFound = errors.New("not found")

	// ErrUnauthorized indicates missing or invalid credentials.
	ErrUnauthorized = errors.New("unauthorized")

	// ErrForbidden indicates the caller lacks permission for the operation.
	ErrForbidden = errors.New("forbidden")

	// ErrTimeout indicates the operation exceeded its deadline.
	ErrTimeout = errors.New("timeout")

	// ErrUnavailable indicates the target service is not reachable.
	ErrUnavailable = errors.New("service unavailable")

	// ErrInvalidInput indicates the request contained malformed data.
	ErrInvalidInput = errors.New("invalid input")

	// ErrResourceExhausted indicates a quota or limit has been reached
	// (e.g., token budget, rate limit, memory).
	ErrResourceExhausted = errors.New("resource exhausted")

	// ErrModelNotLoaded indicates the requested model is not currently loaded.
	ErrModelNotLoaded = errors.New("model not loaded")

	// ErrInternal indicates an unexpected internal failure.
	ErrInternal = errors.New("internal error")
)

// ModuleError wraps an error with the originating module name,
// making cross-module error traces readable.
type ModuleError struct {
	Version int    // Schema version of this data format
	Module  string // e.g. "torch", "shield", "gateway"
	Op      string // e.g. "Inference", "Redact", "Route"
	Err     error
}

// Error returns the module-qualified error string.
func (e *ModuleError) Error() string {
	return e.Module + "." + e.Op + ": " + e.Err.Error()
}

// Unwrap returns the underlying error for errors.Is and errors.As.
func (e *ModuleError) Unwrap() error {
	return e.Err
}

// WrapError creates a ModuleError for consistent cross-module error reporting.
func WrapError(module, op string, err error) error {
	return &ModuleError{Version: Version, Module: module, Op: op, Err: err}
}
