# gpu package STAND-IN for Lane C2's server proof only.
#
# This directory is NOT src/gpu/ (that package is Lane C1's).
# It lives under tests/gpu_standin/ and is put on the include
# path AFTER src/ (`-I src -I tests/gpu_standin`) only when
# running Lane C2's proof before C1's real package is
# integrable. It implements, minimally, exactly the contract
# surface Lane C2 codes against (WAVE_C_INTERFACES.md):
# GpuRuntime + copy helpers, DeviceWeights accessors,
# DeviceKvCache. The coordinator's integration runs the same
# Lane C2 code against the real C1 package and re-runs the
# proof; nothing in src/ imports this stand-in by path.
