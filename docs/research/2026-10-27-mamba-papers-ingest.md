# Mamba Papers Ingest, 2026-10-27

Source PDFs (David's uploads, in `~/workspace/user/files/`):
- `Mamba-Linear-Time-Sequence_4_6ss4.pdf`, Mamba-1
- `mamba2_5_xvjz.pdf`, Mamba-2
- `mamba3_6_darl.pdf`, Mamba-3

Read in full text form (core sections read closely; Mamba-3 read end to end through the conclusion). Filed because David's own engine work (nemotron_hybrid.mojo, iTaK Torch research) is built on this lineage, and his extrapolated "next Mamba" in code can now be graded against the real Mamba-3 paper.

---

## Mamba-1, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

- Authors: Albert Gu, Tri Dao. arXiv:2312.00752 (v2, 31 May 2024).
- Core move: make the SSM parameters functions of the input. Δ, B, C are projected from the current token (sΔ = Linear(x) through softplus; B and C = Linear(x)). This "selection mechanism" lets the model propagate or forget per token, which is what LTI models (S4 and friends) cannot do on discrete, information-dense data like text.
- Discretization: zero-order hold in the theory, Ā = exp(ΔA), B̄ = (ΔA)^-1 (exp(ΔA) − I) ΔB. The released implementation uses the simplified ΔB form (a point the Mamba-3 paper later formalizes).
- Hardware-aware algorithm: selection kills the convolution path, so they compute with a parallel associative scan in recurrent mode, keeping the expanded (D x N) state in SRAM, never materializing it in HBM, with recomputation in the backward pass. Up to 3x faster than prior scans on A100; selective copying solved (99.8 with the Mamba block + S6), induction heads solved and extrapolated to 1M tokens (4000x train length).
- Δ is a generalized RNN gate (Theorem 1): large Δ resets state and focuses on the current input; small Δ persists state and ignores the input. B/C selectivity = fine-grained control of what enters the state and what leaves it. State can be reset at sequence boundaries, which LTI models cannot do.
- Block: H3 block fused with the gated MLP block into one homogeneous block, SiLU activation, short conv on x before the SSM. Real-valued by default (complex kept for audio). A initialized S4D-Real, -(n+1).
- Results: first linear-time model to match a strong Transformer recipe; Mamba-3B matches Transformers twice its size; ~5x generation throughput; quality keeps improving with context out to 1M on DNA and audio.

## Mamba-2, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality"

- Authors: Tri Dao, Albert Gu. arXiv:2405.21060 (31 May 2024).
- SSD framework: SSMs, structured (semiseparable) matrices, and structured masked attention are dual views of the same computation. Y = (L o (C B^T)) X, where L is a structured decay mask. Linear attention is the simplest case of SSD.
- The restriction that makes it fast: A is a scalar times identity per head (a_t in (0,1), data-dependent through Δ_t). Recurrence: h_t = a_t h_{t-1} + B_t x_t, y_t = C_t^T h_t. This trades some expressivity for pure matmul computation on tensor cores.
- SSD algorithm: chunk the sequence (block_len 64 in the reference code). Intra-chunk: quadratic dual form (attention-like, matmuls). Inter-chunk: recurrence over chunk-boundary states via the low-rank factorization (right B factors, center A scan, left C factors). Total cost O(T N^2) FLOPs, O(T N) memory, essentially all matmuls. 2-8x faster than Mamba-1's selective scan; allows ~8x larger states.
- Architecture changes: all projections (A, B, C, X) produced in parallel at the start of the block (like Q, K, V); extra RMSNorm after the gate branch before the output projection (NormFormer-style, for stability at scale); head patterns named by attention analogy (MHA, MQA becomes multi-contract, and Mamba's own pattern is multi-input/multi-value: B and C are a single shared group across the X heads); tensor-parallel friendly.
- State size comparison that matters for engine design: attention's state grows with T, SSM/SSD state is fixed at N. Inference FLOPs per token scale with N^2 for SSD, T*N for attention.
- Results: Mamba-2 2.7B (Pile, 300B tokens) beats Mamba-2.8B, Pythia-2.8B, and Pythia-6.9B on downstream average.

## Mamba-3, "Mamba-3: Improved Sequence Modeling using State Space Principles"

- Authors: Aakash Lahoti, Kevin Y. Li, Berlin Chen, Caitlin Wang, Aviv Bick, J. Zico Kolter, Tri Dao, Albert Gu (CMU, Princeton, Together AI, Cartesia AI). arXiv:2603.15569v1, 16 Mar 2026. Preprint.
- Framing: inference-first. Mamba-2 traded expressivity for training speed; Mamba-3 buys expressivity back in ways that cost nothing at decode.
- Innovation 1, exponential-trapezoidal discretization: the general discrete recurrence is h_t = alpha_t h_{t-1} + beta_t B_{t-1} x_{t-1} + gamma_t B_t x_t. Mamba-1/2 are retroactively formalized as "exponential-Euler" (beta = 0, gamma = Delta_t). Mamba-3 sets beta_t = (1 - lambda_t) Delta_t e^{Delta_t A_t} and gamma_t = lambda_t Delta_t, with lambda_t a data-dependent scalar in [0,1] (classical trapezoid recovered at lambda = 1/2, Euler at lambda = 1). It is a second-order discretization, and equivalently a data-dependent width-2 convolution applied to the state-input B_t x_t INSIDE the recurrence. Combined with B, C biases (below), it empirically replaces the short causal conv: ablation at 440M, neither = 16.68 ppl, bias only = 16.49, full Mamba-3 = 15.72, adding the conv back = 15.85 (worse).
- Innovation 2, complex-valued state: making A complex (A + i*theta) gives block-diagonal 2x2 rotation matrices in the real form, which is provably equivalent to data-dependent RoPE applied to B and C (cumulative products of per-step rotations R_i; the "RoPE trick"). This restores state tracking that real-eigenvalue models provably cannot do: parity 100.00, modular arithmetic without brackets 98.51, with brackets 87.75. Controls: Mamba-3 without RoPE ~ random (2.27 / 1.49 / 0.72), Mamba-3 with standard fixed RoPE also fails (1.56 / 20.70 / 2.62), Mamba-2 fails (0.90 / 47.81 / 0.88). The rotations MUST be data-dependent.
- Innovation 3, MIMO: B, C, x gain a rank R (R = 4 in the paper). The state update's outer product becomes a matmul; arithmetic intensity scales as Theta(R); decode FLOPs go up to 4x at essentially the same wall-clock latency because decode is memory-bound. Parameter growth is contained by scaling x, y, z per head with a learnable vector instead of widening projections, and MIMO models are parameter-matched by shrinking the MLP (~6.6% at 1.5B). MIMO with d_state 64 matches Mamba-2 at d_state 128 perplexity: same quality, half the state, half the decode cost driver.
- Block refinements: BCNorm (RMSNorm on B and C after projection, QK-norm style; also stabilizes large runs, lets pure models drop the post-gate RMSNorm, but hybrids need that norm back for long-context extrapolation); learnable head-specific channel-wise biases on B and C after the norm (also part of why the short conv can go).
- Results: trained on 100B FineWeb-Edu tokens, 2K context. At 1.5B, SISO averages 56.4 downstream vs Mamba-2 55.7 and Gated DeltaNet 55.8; MIMO 57.6. SISO wins or ties at every scale 180M to 1.5B. Retrieval is the admitted weak side (SWDE/FDA extraction) versus Transformers; hybrid 5:1 linear to NoPE attention with a pre-gate grouped RMSNorm is their recommended fix and beats the Transformer baseline.
- Kernels: Triton for prefill, CuTe DSL for decode. At batch 128, 1.5B, bf16, d_state 128: Mamba-3 SISO decode 0.156 ms vs Mamba-2 0.203, GDN 0.257. MIMO 0.179. End-to-end prefill+decode tracks Mamba-2 closely while quality is higher.

---

## Grading David's extrapolated "Mamba-3" (scan_head_m3, nemotron_hybrid.mojo) against the real paper

David's experimental variant (gated behind MAMBA_3_MODE = False, never enabled as shipped): exponential-trapezoidal discretization with lambda = sigmoid(dt_raw), the state blending B*x_prev and B*x_curr through lambda, and RoPE applied to B and C.

- Trapezoidal discretization: MATCH on the mechanism. His lambda is a learned, data-dependent scalar squashed into [0,1] by a sigmoid, exactly the role of the paper's lambda_t. His update blends the previous and current state-inputs through it, which is the beta/gamma structure. Detail to verify in his code before claiming a full match: the paper's beta term carries an extra decay factor, beta_t = (1 - lambda_t) Delta_t e^{Delta_t A_t}, i.e. the previous input's contribution is also decayed by the current step's transition. (From the ingest audit, his blend was recorded at summary level; the exact placement of the decay factor was not re-verified line by line.)
- RoPE on B and C: MATCH on the mechanism. The paper's Proposition 3 proves the complex-valued state is exactly data-dependent rotary embeddings applied to B and C, and his variant applied rotation to B and C. The paper's rotations come from the imaginary part of a complex A (theta projected per step, accumulated as products of R_i). Whether his angles were data-dependent projections or a fixed schedule is the one detail that decides full equivalence; the paper shows fixed-schedule RoPE fails the state-tracking tasks outright, so this detail matters and is worth checking in his source before building on it.
- MIMO: NOT PRESENT in his variant. Nothing in the audited Mojo tree has rank-R B, C, x. This is the paper's third leg and the one aimed squarely at decode efficiency on memory-bound hardware, which is his home turf.
- BCNorm and B, C biases: not in his variant (his block follows the Nemotron hybrid layout). The paper's ablation says these plus the trapezoidal rule are what let the short conv be deleted.
- Chronology, stated plainly: the paper is dated 16 Mar 2026; David's Mamba extrapolation experiments sit in the same March-April 2026 window (BUGTRACKER entries 2026-04-03). The convergence of his extrapolated mechanisms with the paper's two headline recurrence changes is real and documented in his own code. Priority over the paper is NOT established by the dates available and is not claimed here.
- His extra: TIME_WARP_COMPACTION (skip state updates when dt < 0.001) has no counterpart in the paper. It is his own idea and remains untested either way.

## What this means for the Rust engine plan (itak-ssm crate)

- Implement the scan as the general 3-term recurrence (alpha, beta, gamma) from the start; Euler (Mamba-1/2) is the beta = 0 special case, so one kernel serves all three generations. This is the 20-years-ahead seam applied to math.
- Rotation support on B and C belongs in the scan's interface, driven by per-step theta projections (data-dependent), not a fixed RoPE table.
- MIMO rank R should be a config parameter of the SSM module (R = 1 is SISO), because the paper shows quality per unit of decode latency improves with R on memory-bound hardware, which is precisely the consumer-GPU regime the engine targets.
- BCNorm plus B, C bias terms are cheap and load-bearing (stability and conv removal); include them in the block spec.
- Hybrid layouts stay the default recommendation for retrieval-heavy use; the engine's arch-handler seam should treat the linear/attention mix ratio as config, per the paper's 5:1 finding.
