//! K5: RoPE rotation plus KV append. Not yet implemented.
//!
//! Rotates Q and K (sliding-window layers only; global layers are
//! NoPE in this architecture) at the sequence position and writes
//! K and V into the layer's KV storage in the same pass: paged
//! slots for global layers, ring slots for sliding layers. The
//! rotation pairing is llama.cpp's NORM mode (adjacent channels)
//! for muse-glimmer, the convention `cpu-compute` pins and the
//! arch-llama RoPE fix confirmed from the other side. Until this
//! kernel lands, `attention` refuses by name (see K6's module).

// This module is the K5 slot: its refusal is raised from the
// attention step, which K5 and K6 jointly block. It carries no
// code until the kernel lands; the slot exists so the module map
// in `kernels` matches BACKEND.md Section 3 one to one.
