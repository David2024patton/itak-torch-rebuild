// Package types provides the shared primitive types used across all
// iTaK modules: the inference request and response contracts, chat
// messages, and model descriptors that the Go platform, the console
// tooling, and the Rust engine (through the itak-types mirror) all
// compile against.
//
// The shapes in this package are the measured call-site surface from
// the original Torch tree (pkg/torch/core_adapter.go and its test are
// the importing call sites), carried over from the rescued original
// Core. Fields are added only when a call site uses them.
package types

// Version is the schema version of every data format in this package.
// Each struct carries it in a Version field so data written today
// stays readable as the contracts evolve; a zero Version marks data
// written before versioning landed.
const Version = 1
