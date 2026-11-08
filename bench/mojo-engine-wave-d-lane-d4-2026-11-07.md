# Mojo engine Wave D, Lane D4: activation residency + device KV window wrap

Date: 2026-11-07 (server windows ran 2026-11-08 04:06-04:14
UTC, 23:06-23:14 EST). Branch `lane/mojo-d4-residency`, cut
from `mojo-max-engine` head
2c1802c4d9a764899f56178e673e183d70301749 (remote branch head
verified equal before the cut; remote `main` verified at
b416987a736b0a63c9f7815ec4b4fbc84e4dd2dd and never touched).
Lane commits are listed at the bottom; the coordinator
integrates and pushes.

## (a) Activation residency - DONE, full chain

`gpu_layer_step` now runs `gpu_layer_step_resident`
(src/model/gpu_layer.mojo): the layer's input row uploads
once, and the residual stream, norm outputs, Q/K/V rows,
the gated attention output, and the MLP intermediates stay
as device buffers passed op to op. The KV append is
device-to-device (`DeviceKvCache.append_device`), the
residual adds are device adds (`residual_add_kernel`, the
same plain f32 per-element add the host add did), and the
output row downloads once. `gpu_logits` chains the same way
(one upload, one download). The Wave C step survives as
`gpu_layer_step_hostchain`, the A/B baseline.

Remaining host crossings, stated exactly: (1) the layer
boundary itself - one hidden-row upload + one download per
layer, the contract's boundary shape, crossed 52 times per
model step by the model-level chain (embedding and logits
in gpu_model still hand host rows between layers; a
cross-layer resident chain is a different change); (2) no
other activation crosses inside a layer. Past a KV wrap,
attention's K/V source is a device-to-device visible-order
gather, not a host crossing.

Correctness (server proof, C2 harness shape, fresh KV per
step, rung-2 input construction): the resident row is
numerically IDENTICAL to the hostchain row at every printed
figure, both layers, every round - layer 0 cosine
0.9999999999985569, max rel 2.33e-06, sum 65.22703143954277
vs counted 65.22733633872122; layer 3 cosine
0.9999999999983837, max rel 8.41e-07, sum 157.77902115881443
vs counted 157.77914571017027. The in-process CPU rows match
the counted sums exactly (sum_rel 0.0). Bars were cosine
>= 0.999999, max rel <= 1e-4: RESIDENCY PROOF PASS.

Timings (interleaved rounds, warmup excluded, medians of 4;
window 2 at microsecond resolution, logs
~/gpu-wave-d4b/residency_us.log read in the foreground):
- Layer 0 (sliding): hostchain median 3,578 us, resident
  median 1,871 us - 1.91x.
- Layer 3 (global): hostchain median 2,964 us, resident
  median 1,847 us - 1.60x.
Per-round pairs (us, host/resident): layer 0: 4257/2919,
3519/1831, 3578/1800, 2931/1871; layer 3: 4074/2856,
2964/1762, 2792/1785, 2782/1847. Context for the Wave C
record: Wave C's single cold steps printed 16 ms / 15 ms;
this proof's warmup round printed hostchain 8,828 / 8,193 us
and resident 3,641 / 3,170 us. The warm per-layer saving is
about 1.1-1.7 ms, which over 52 layers matches the
end-to-end signal below.

End-to-end regression on the resident path (same window):
the C2 layer proof passes unchanged (GPU_LAYER_PROOF PASS;
its gpu_ms now reads 14 / 13 because that harness's single
step is the resident one; LOGITS_PROOF argmax 150015 exact),
and the teacher-forced device walk passes 8/8 with Wave C's
exact winning logits, at 54-56 ms per full step vs Wave C's
recorded ~109 ms per teacher-forced step (total_ms 843 for
the 8-step walk's measured section; the runner decode
figure is the coordinator's timing-harness leg, not this
lane's).

## (b) Device KV window wrap - DONE, refusal retired

`DeviceKvCache` appends now wrap the ring with B1's exact
semantics (slot = position % window, slot_position tracks
the holder, stored/first_visible unchanged in meaning);
`kv_append_refusal` no longer returns
`gpu_kv_window_wrapped` for any append. Readers get
`DeviceKvCache.gather_visible`: the raw buffer when its slot
order already is the visible order (global layers, and
windowed layers whose first visible position lands on slot
0 - including first_visible exactly 2048), otherwise
`kv_visible_gather_kernel` materializes the visible rows in
absolute-position order into a scratch buffer (at most
window x row floats, inside the scratch envelope) that
attention_kernel consumes unchanged. The resident attention
(`attention_device_resident`) reads through it; the legacy
hostchain step still refuses past the window by the old
name, stated in its header - the name survives ONLY there.

Proofs (tests/server_proof_gpu_kv_wrap.mojo, WRAP PROOF
PASS; 83 WRAP EXACT lines, 0 mismatches):
- Synthetic window-4 replay, positions 0..11 (three wraps),
  device vs B1 CPU cache: after EVERY append, stored_len and
  first_visible agree and every visible row compares
  element-exact - read-backs AND the full visible-order
  gather buffer. Evicted reads refuse
  gpu_kv_position_evicted on device.
- Real 2048 window, Glimmer geometry (2 x 128), positions
  0..2059 on a windowed layer plus a global sibling: reads at
  12, 100, 2047, 2048, 2049, 2059 element-exact vs CPU; the
  full 2048-row visible gather element-exact for K and V;
  stored 2048, first_visible 12 on both caches.
- SYNTHETIC GENERATION PAST 2048 (named synthetic): a decode
  loop on a small synthetic config (window 2048, 2 KV heads
  x 32, 4 Q heads), positions 0..2063, driven entirely on
  device through the resident path (append_device +
  attention_device_resident), no CPU fallback, no wrap
  refusal. Device attention vs the CPU twin
  (attention_cpu_rotated_cache over B1 visible rows,
  ROPE_NONE by construction) at the sampled positions:
  cosine >= 0.99999999998 and max rel <= 1.01e-05 everywhere,
  including 2047 (max rel 6.82e-06), 2048 (5.70e-06), 2049
  (6.56e-06), and 2063 (1.00e-05, cosine 0.9999999999808027).
  The real model at that length is a coordinator-window
  decision, not this lane's, per the brief.
- The C1 plumbing proof's wrap legs were updated to the new
  parity (both caches accept 4-on-window-4 and 2048/2049,
  wrapped rows element-exact): PROOF GPU_PLUMBING PASS, 45
  PROOF EXACT lines, hash leg byte-exact as before.

B1's CPU cache semantics were not changed (no edit to
src/kv/); the device cache mirrors it.

## VM tally

tests/run_tests.sh on the lane head (2b9f763 at suite time):
PASS - 23 TestSuite files + seam-on diag rerun + config
probes + engine hello (1.4), 126 tests, 0 failed. Inside
that: test_gpu_plumbing now 7/7 (wrap refusal expectations
replaced by B1-parity legality + the visible-order slot/gather
arithmetic), test_gpu_layer_vm 5/5. GPU mains compile-verified
with `mojo build --target-accelerator gfx1100` (layer,
generate, wrap, residency, plumbing proofs, exit 0). Changed
files are `mojo format` clean.

For the coordinator's run_tests.sh wiring: no new TestSuite
files (the two new proofs are server proofs, not TestSuite);
test_gpu_plumbing.mojo is already wired and its content
changed - no wiring change needed.

## Server closeout (as observed, both windows)

Window 1 (~/gpu-wave-d4/, staged 2b9f7634f06b8ae9c222747d3f884b689764c4af;
staged sha256 of gpu_layer.mojo 4427a5f4... and device_kv.mojo
60c13190... equal to the worktree's): phase lock waited
04:05:41 -> acquired 04:06:55 (a sibling lane held it);
PORT8080 0 before and after; VRAM_PRESTOP 2,584,600,576 B;
only cortex-embed.service stopped (it is a USER service on
this host: `systemctl --user`); VRAM_DRAINED 33,271,808 B
(the exact Wave C drained figure); all five proofs exit 0
(wrap, residency, plumbing, layer, generate);
TENANT_EMBED active after restore; VRAM_RESTORED
2,584,600,576 B; WAVE_D4_WINDOW_DONE 04:10:24 UTC.
Window 2 (~/gpu-wave-d4b/, staged e26da1bbad1e614fa1d67956c628cdaa02c153e9,
residency proof only): same bookends - drained 33,271,808 B,
restored 2,584,600,576 B, port 8080 0/0, embed active,
PROOF_EXIT 0. cortex-cortex-1 and cortex-cortex-surrealdb-1
were Up throughout and never stopped. Glimmer serving down;
llama-server untouched (llama-laguna.service is in a
pre-existing crash loop - its model path does not exist,
166,888 restarts at window time; it was not touched, and its
brief load attempts did not disturb either window's proofs:
one 886 ms hostchain outlier in window 1's residency round 0
is recorded in that log and excluded by the median).
No reboot. Both windows closed hours clear of the
2026-11-08 02:00-04:00 ET maintenance window.

## Commits (lane/mojo-d4-residency, not pushed)

- a1847c6 device KV window wrap (ring appends, visible-order
  gather, append_device; retirement of gpu_kv_window_wrapped
  from DeviceKvCache; VM plumbing 7/7)
- ffb31ce activation residency (resident layer chain,
  gpu_logits chained, hostchain kept as baseline)
- 2b9f763 wrap + residency server proofs; plumbing proof
  wrap legs updated to B1 parity
- e26da1b residency proof at microsecond resolution
- (this note's commit) the lane record
