# gpu package: the MAX device plumbing for the Mojo/MAX
# engine (Wave C lane C1, 2026-11-07).
#
# context.mojo: GpuRuntime (DeviceContext ownership, budget
# accounting against MOJO_ENGINE_GPU_BUDGET_BYTES, upload /
# download / zero-filled buffer helpers, named refusals).
# weights.mojo: DeviceWeights (one device buffer per tensor
# slot, uploaded once from the CPU GlimmerModel's cache).
# device_kv.mojo: DeviceKvCache (B1's layout on device, one-row
# staging + scatter append, whole-buffer copies forbidden).
