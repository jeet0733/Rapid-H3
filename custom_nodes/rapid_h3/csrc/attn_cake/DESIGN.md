# attn_cake: config 80's INT8 attention on a CAKE-style schedule

Goal: a faster INT8 self-attention for MiniMax-H3 on GB10 (sm_121, mma.sync only, 48 SMs, ~99 KB smem/SM,
INT8 IMMA peak ≈252 TOPS at ≈2.3 GHz) with **no change in the math**: same quantization, same softmax, error
vs fp32 no worse than comfy-kitchen's (ck) and ideally equal to our config 80.

**Result.** Configs 94–99 are bit-identical to config 80 (`torch.equal` on every tested shape). Plain
CAKE/FA3 ping-pong is 8–10% *slower* on GB10; the reason is measured below. A different ordering, the
**softmax mutex** (config 94), is faster than config 80 per call by 3.9–4.0% at the 5 s shapes and 2.1% at
10 s, and faster than ck by 6.6–6.7% and 4.6%. That is under the 5% bar set for this candidate. Measured
numbers are in §5. **Config 90** (§7, 2026-10-27) enters the mutex after the row max: still bit-identical, another
1.8–2.4% per call. **Status:** `ts_h2` / `ts_s6` run config 190 (= 90 with the q/k RMSNorm + RoPE fused into
the pre-pass, §6) since 2026-10-27; 194 (= 94 fused) from 2026-10-26 until then.

## 1. The starting point (2026-10-26, before this candidate)

**Call site.** `comfy/ldm/minimax/model.py:174-201` (line numbers from ComfyUI 0.39, the image used then):
- Q/K get RMSNorm + split-half RoPE in place (ck `rms_rope_split_half_`).
- Then `optimized_attention(q, k, v, heads=56, mask=None, skip_reshape=True)` on `[1, 56, S, 128]` bf16.
- One self-attention per DiT block, 50 blocks (`model.py:475`, `num_layers=50`).
- Text-only token-refiner calls have S < 1024 and fall through.
- The presets run CFG 1 (`BasicGuider`), so a step makes **50 calls**.

**Shape.**
- Non-causal, no mask, one joint segment (video + audio + text packed into one sequence).
- Batch 1, H = 56, D = 128.
- S = 22,332 (ts_h2 0.7 draft) and 44,992 (full-size 5 s refine), dumped from real renders (docs/kernels.md).
- 10 s at 1344×768 ≈ 75–80k (docs/kernels.md, "Speed"). We bench 78,592 (61 latent frames × 1008 + audio/text).
- Because there is one segment per call, CAKE's varlen machinery (cu_seqlens, per-segment K mean, tail masks)
  reduces to per-head statistics, and its LPT order reduces to round robin over equal units.

**comfy-kitchen** (`comfy_kitchen/sage_attention.py`, `_int8_attention_cuda`). The quantizer is one fused
binary call, `_C.sage_sdpa`, and the main kernel is `qk_int_sv_i8_attn_kernel<128,128,16,128,128>`:
- `cta_k = 128` for D ≥ 128 and S > 1024.
- Q is INT8 with 32 scales per 128-row block (per-thread row groups), after a signed Hadamard-128 rotation.
- K is INT8 with 4 scales per 128-key block (per-thread key groups), smoothed by subtracting a sampled
  per-head **anchor token** (`anchor_indices`), also H128-rotated.
- V is INT8 with one scale per (head, channel), padded to `cta_k`.
- Q·Kᵀ is `mma.m16n8k32.s8`, softmax is fp32 exp2 with an integer-domain row max.
- P is quantized to u8 against each 128-key tile's own row max, P·V is u8×s8 IMMA with int32 accumulation,
  and a per-tile fold goes into the fp32 output.

**Config 80** (`../attn_int8.cu`, `attn_fwd8<2,false>`, line 2250; prepass in `int8_attention`, line 2658):

Same structure as ck, with finer quantization:
- **Pre-pass:**
  - `stats_partial`/`stats_final` (:80, :95): per-head channel means of K and V and V min/max, in a
    deterministic two-level reduction.
  - `quant_qk` (:128): Q and K to INT8 with a **per-token** scale after `hadamard128` (:111). K is
    mean-smoothed first, an exact softmax shift. Per-key `(scale, −1.5·2²³·scale)` pairs (`KSB`) make
    dequantization one FFMA on the magic-biased int32.
  - `quant_v` (:450): V to INT8 per channel, mean-smoothed, stored transposed `[H][128][Spad]` with keys
    permuted inside 32-key groups, so the Q·Kᵀ accumulator fragment is directly the A operand of P·V.
- **Main kernel:**
  - Layout: 8 warps × 16 query rows, 128-key tiles, 2-stage cp.async ring, one `__syncthreads` per tile.
  - Score dequant is FFMA per key. The row max is a 4-shuffle reduction. exp2 runs in fp32.
  - P is u8 against the tile max, packed by `cvt.pack.sat.u8` (F2IP).
  - P·V is `mma.m16n8k32.u8.s8` into int32 starting at 1.5·2²³, so its bits are already a float.
  - The fold is `O = a·O + ts·acc`, two FFMAs per element per tile. Row sums come from the quantized P via `dp4a`.
  - In-warp interleave: the exp2/packing of chunk kc runs between the P·V IMMAs of chunk kc−1, and half 0's
    fold runs between half 1's IMMAs.
- **Accuracy:** error vs fp32 is 0.81–0.98% against ck's 0.90–1.23% on real q/k/v (docs/kernels.md).
- **Speed:** 2–2.7% faster than ck. The main kernel runs at ≈165–170 TFLOPS.

## 2. The CAKE structure mapped onto INT8 operands

CAKE (flashinfer PR 5539, FP8 E4M3 for SM120) does the following:
- persistent, 1 CTA/SM, 256 threads, 8 warps × 16 rows;
- a two-stage TMA ring for 128-key K/V tiles;
- FA3-style ping-pong between warp groups {0–3} and {4–7} on named barriers;
- `mma.sync.m16n8k32 kind::f8f6f4` with fp32 accumulation;
- FP8 P against the running max, so there is no per-tile fold;
- per-128-key K scales;
- an LPT unit table built on the host.

Its rel-L2 error is 5.1–5.3%, about 5× ours, so its numerics are not usable here. Only the schedule is taken.

| | CAKE (FP8) | attn_cake (INT8, = config 80 math) |
|---|---|---|
| Q·Kᵀ | e4m3 m16n8k32, f32 acc | s8 m16n8k32, int32 acc from 1.5·2²³ (bits = float), per-key (scale, bias) FFMA |
| softmax | fp32 exp2, running max | fp32 exp2, P u8 vs **tile** max, `ts = 2^(tm−m)`, `a = 2^(m_old−m)` |
| P·V | e4m3 P × e4m3 V → f32 directly | u8 P × s8 V → int32 per tile, then `O = a·O + ts·acc` (2 FFMA/elt) |
| K smoothing | per-segment channel mean | per-head channel mean (one segment), before the H128 rotation |
| loads | TMA 3D, 128 B swizzle | TMA 3D (`CU_TENSOR_MAP_SWIZZLE_128B` = config 80's `chunk ^ (row & 7)`), KSB by 1 KB `cp.async.bulk` |
| ring | 2 stages, K/V full + empty mbarriers | same; K and V(+KSB) have separate full/empty barriers |
| producer | warp 0, elected lane | warp 4 lane 0 (in the trailing group, so its empty-waits are short) |
| schedule | (segment, head, Q tile) LPT units | (head, Q tile) units, head-major round robin (equal work → LPT = RR) |

**Folding K smoothing and scales so the math equals config 80's.** The kernel consumes config 80's
pre-pass outputs unchanged (`q8, qs, k8, ksb, vt, vscale, vmean`). It runs the same per-element
operations in the same order:
- `t = fma(acc_bits, s_j, b_j)`;
- `tm = max`, the 2 shuffles, `a`, `ts`;
- `ex2(fma(t, c, b))` → `cvt.pack.sat.u8`, then `dp4a` row sums;
- `l = fma(ts, i2f(rs), l·a)`;
- `O = fma(ts, acc_bits, fma(O, a, −ts·2²³·1.5))`;
- the same epilogue.

The output is therefore **bit-identical** to config 80. The bench checks this on every shape.

**Budgets per warp group** (4 warps × 16 rows; per thread):
- **Registers:** the live set is about 64 fp32 O + 16 Q fragments + 64 scores (later 16 packed P) + 64 int32
  P·V accumulators + addresses ≈ 220 at the P·V peak. ptxas uses 255 with 0 spills (config 80: 255 with 8 B).
- **Shared memory:** 2 × (16 KB K + 16 KB Vᵀ + 1 KB KSB) + barriers + 1 KB alignment = 69.6 KB, so 1 CTA/SM.

**Cycle model.** One m16n8k32 IMMA is 8192 ops. At 252 TOPS / (48 SM × 4 SMSP × 2.3 GHz) it occupies a
sub-partition's tensor pipe for ≈14.3 cycles. A 128-key tile is 128 IMMAs per warp, so 2 warps per SMSP
give 256 IMMAs ≈ 3,660 cycles at peak.

Each warp also issues per tile:
- 64 LDSM.x4 and 16 LDS.128;
- 64 dequant FFMA, ≈64 FMNMX, 4 SHFL;
- 64 FFMA + 64 MUFU.EX2, 32 F2IP, 16 DP4A;
- 128 fold FFMA.

That is ≈5.3 non-IMMA instructions per IMMA (config 80: 809 SASS per 128 IMMA). By `bench/cuda/imma_warps.cu`,
this mix caps the pipe at ≈215–220 TFLOPS. Config 80 reaches ≈166, about 77% of that cap. The gap is phases in
which both warps of a sub-partition sit in the softmax critical path (row-max shuffles → exp2 → first P·V)
while the tensor pipe idles.

## 3. Why the CAKE ping-pong itself does not work on GB10 (measured)

FA3/CAKE ping-pong makes exactly one warp per sub-partition issue IMMAs at any time. On Hopper that is free,
because wgmma is asynchronous. With mma.sync the warp must issue every IMMA itself, so its single-warp
issue rate becomes the ceiling. `bench/cuda/ldsm_mma.cu`, 2026-10-26:

| IMMA per LDSM.x4 | 1 warp / SMSP | 2 warps / SMSP |
|---|---:|---:|
| 2 (our Q·Kᵀ and P·V pattern) | **203.8** | 242.3 |
| 4 | 221.7 | 244.5 |
| 8 | 228.9 | 246.8 |

So even a perfect ping-pong is capped at 204 TFLOPS before the other group's softmax steals issue slots.
4 IMMA per LDSM would need 32 rows per warp, and the registers for that don't exist.

Measured with ncu at 8 heads, S = 36,288:

| config | design | TFLOPS (main) | tensor pipe active | notes |
|---|---|---:|---:|---|
| 80 | reference | 165–170 | 66–73% | `math_pipe_throttle` 1.7 per issue |
| old 94 (`<0,1>`) | CAKE ping-pong, QK / softmax / PV / fold | 150 | 64% | `math_pipe_throttle` ≈0: the pipe starves |
| 95 (`<1,1>`) | ping-pong with 80's in-section interleave | 155 | | |
| 97 (`<16,1>`) | FA3 merged phase [PV(n) \| QK(n+1)], 2 hand-offs/tile | 118–126 | | spills, worse schedule |
| 99 (`<3,0>`) | persistent + TMA ring, no ordering | 166–170 | | = config 80 |
| **96 / 94** | **softmax mutex** (below) | **174–175** | **76.4%** | barrier 0.8, `wait` 2.24 per issue |

(Configs 94 and 97 were reassigned during the session. The ping-pong rows were measured before that.)

## 4. The kept design: softmax mutex (configs 94, 96)

The order is inverted relative to the ping-pong:
- The **softmax sections** of the two warp groups alternate. Each section runs from the row-max shuffles
  through the exp2/u8-packing of the 4 chunks, interleaved with P·V half 0's IMMAs as in config 80. It uses
  the same named-barrier protocol: `bar.sync(1+wg)` before, `bar.arrive(2−wg)` after, and group 1 pre-arrives once.
- The tensor sections overlap freely.

Both warps of a sub-partition keep issuing IMMAs, which keeps the 242-TFLOPS two-warp ceiling. The groups are
held roughly half a softmax apart, so the pipe always has a warp outside the latency-bound shuffle → ex2 chain.

The rest is from CAKE:
- persistent 48 CTAs;
- the TMA ring with separate K and V/KSB mbarriers and a producer thread in the trailing group;
- no `__syncthreads` in the loop;
- mbarrier waits with a suspend-time hint, so waiting warps sleep instead of spinning.

Config 94 also splits Q·Kᵀ into groups of 2 key pairs, giving 4 independent accumulator chains instead of 2.
That is worth ≈1%, within noise.

Variants tried (all bit-identical):
- the end of the exclusive region after chunk 0/1/2/3 of P·V half 0 (SYNC 2–5): within ±1%;
- 8 accumulator chains in Q·Kᵀ: same;
- k-step-outer Q·Kᵀ with the dequant inside the region (`<1,2>`): −2%.

## 5. Results and projection

**Run.** `bench/attn_cake.py --tokens 22332 44992 78592 --configs 94 95 96 97 98 99 --kernel --rounds 5 --iters 3`
(2026-10-26). Raw data: `bench/results/attn_cake_synth.json`.

**Inputs.** Synthetic `[1, 56, S, 128]` q/k/v with H3-like statistics: unit-RMS rows, 4 outlier channels per
head with a large per-head K offset (what smoothing removes), and heavier-tailed V. The real q/k/v dumps of
round 4 were not available for this run (§7 uses real dumps).

**Correctness.** All new configs are `torch.equal` to config 80 at S = 1000, 4096, 22,332, 44,992 and 78,592
(8 and 56 heads, including S not a multiple of 128). Their error on real q/k/v is therefore config 80's
documented 0.81–0.98%, against ck's 0.90–1.23%.

Error vs fp32 SDPA on the same bf16 inputs, ‖o−ref‖/‖ref‖ (max |o−ref| / max |ref| in brackets):

| S | ck | config 80 | config 94 |
|---|---:|---:|---:|
| 22,332 | 2.40% (5.3%) | 2.06% (4.3%) | 2.06% (4.3%) |
| 44,992 | 2.48% (6.0%) | 2.12% (5.0%) | 2.12% (5.0%) |
| 78,592 | 2.51% (5.1%) | 2.17% (4.1%) | 2.17% (4.1%) |

The synthetic outliers make every kernel's error about 2.3× its real-data value; the ranking is the same.

**Speed** (ms per call including the pre-pass; TFLOPS = 4·S²·128·56 / t; main kernel in brackets):

| S | ck | config 80 | **config 94** | 94 vs 80 | 94 vs ck |
|---|---:|---:|---:|---:|---:|
| 22,332 (draft) | 98.84 (164 TF) | 96.21 (162 TF) | **92.34 (175 TF)** | −4.0% | −6.6% |
| 44,992 (5 s refine) | 383.1 (162 TF) | 371.7 (163 TF) | **357.3 (170 TF)** | −3.9% | −6.7% |
| 78,592 (10 s, est. S) | 1149.7 (159 TF) | 1120.3 (162 TF) | **1096.4 (167 TF)** | −2.1% | −4.6% |

Other configs at the same shapes:
- 96 (mutex, 1-pair Q·Kᵀ) equals 94 within 0.2%.
- 98/99 (persistent + TMA, no ordering) are −1 to −2% vs 80.
- 95 (ping-pong) is +6 to +7% slower than 80.
- 97 (merged-phase ping-pong) is +29 to +36% slower.

**Per-clip projection** (CFG 1, so 50 calls per step; ts_h2 = 6 draft steps + 1 refine; ts_s6 = 6 full-size steps):

| | vs config 80 | vs ck (the default before 2026-10-26) |
|---|---:|---:|
| ts_h2, 5 s | 6·50·3.9 ms + 50·14.4 ms ≈ **1.9 s** | 6·50·6.5 ms + 50·25.8 ms ≈ **3.2 s** |
| ts_s6, 5 s | 6·50·14.4 ms ≈ **4.3 s** | 6·50·25.8 ms ≈ **7.7 s** |
| ts_h2, 10 s (draft S ≈ 39k est.) | ≈ 3.3 + 1.2 ≈ **4.5 s** | ≈ 6 + 2.7 ≈ **8.7 s** |
| ts_s6, 10 s | 6·50·23.9 ms ≈ **7.2 s** | 6·50·53.3 ms ≈ **16 s** |

**What limits it:**
1. The ≥5%-over-config-80 bar was not met: −3.9 to −4.0% at the 5 s shapes and −2.1% at 10 s.
2. The schedule is capped by mma.sync. A strict ping-pong gives up the second IMMA-issuing warp per
   sub-partition (204 vs 242 TFLOPS ceiling). Any schedule that keeps both warps issuing has to live with
   ≈5.3 non-IMMA instructions per IMMA, which costs ≈12% of the pipe (`imma_warps`). The `wait` stall (IMMA
   cadence, ptxas padding) is now the largest, at 2.24 per issue; `math_pipe_throttle` is 1.0.
3. The pre-pass is fixed. It costs ≈10.4 ms at 22k and ≈15.9 ms at 45k (ms per call minus main kernel,
   `bench/results/attn_cake_synth.json`; memory-bound, two passes over K and V that the mean smoothing needs).
   It is ≈11% of a draft call and isn't touched here.
4. The gain shrinks with S: main-kernel TFLOPS go 175 → 170 → 167, while config 80 stays at 162. At 78.6k one
   head's K+Vᵀ is 20 MB, close to the 24 MB L2. 48 persistent CTAs drifting apart within a 614-tile unit
   likely lose L2 reuse. Unverified, but it is the next lever: split the key range or re-synchronise the
   CTAs per head.

## Files
- `attn_cake.cuh`: kernels `attn_cake<VAR,SYNC,TB,SMX>` and `attn_cake_m<PP>`, TMA map encoding via
  `cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled")` (no `-lcuda`), `cake::launch`.
- `../attn_int8.cu`: `#include`s it after the anonymous namespace and dispatches configs 94–99 and 90 / 89 / 79
  (§7; the pre-pass is shared, so the INT8 operands are config 80's).
- `../../../../bench/attn_cake.py`: bit-identity vs config 80, error vs fp32, ms/call, TFLOPS, main-kernel ms.
- `../../../../bench/attn_softmax_r6.py`: configs 90 / 89 / 79 vs 94 on the fused rms/rope and text-bias paths.

## 6. q/k RMSNorm + RoPE fused into the pre-pass (configs 1xx, 2xx)

Before this, each block ran comfy-kitchen's in-place `rms_rope_split_half_` on q and k inside the qkv buffer
(read and write q and k, 4 × 320 MB at 22k), and then our pre-pass read them again. Measured per call
(`bench/rmsrope_fuse.py`, 2026-10-26):

| S | ck rms+rope | stats | quant_qk | quant_v | pre-pass + rms/rope | fused pre-pass | saved |
|---|---:|---:|---:|---:|---:|---:|---:|
| 22,332 | 5.98 ms | 3.42 | 4.66 | 2.30 | 16.5 ms | **9.8 ms** (`stats_partial_rr` 3.07, `quant_qk_rr` 4.47) | 6.1–6.7 ms |
| 44,992 | 11.52 ms | 6.80 | 9.39 | 4.74 | 32.1 ms | **19.0 ms** (5.90, 8.80) | 11.5–13 ms |

ck's kernel ran at 210–217 GB/s, so it was all memory traffic. The fused pre-pass reads raw q/k once in the stats
pass (K only) and once in the quant pass. It recomputes the norm and rotation in registers each time, and it runs
at the same speed as the unfused kernels (one warp per (token, head), heads fastest, so a token's rope table row
serves all 56 heads from L1).

**Bit-identity.** `rms_rope_row` (`../attn_int8.cu`) reproduces ck 0.2.37's
`rope_kernel<bf16,bf16,bf16,1,1,1,1,1>`, which we read from its sm_120f PTX:
- the per-lane `fma` chain over channels l, l+32, l+64, l+96;
- a shfl-down add tree, then `rsqrt.approx.ftz(eps + div.approx.ftz(ss, 128))`;
- `bf16((r·x)·w)`;
- `bf16(fma(a, t00, b·t01))` / `bf16(fma(a, t10, b·t11))`.

All fp32 ops are `.ftz` inline PTX. The stats pass keeps `stats_partial`'s per-channel token order, so kmean,
vmean and vscale are bitwise equal. `torch.equal` holds:
- on q and k after rms/rope, on all 8 operands and on the attention output (S = 1000, 4096, 22,332, 44,992; 2 seeds);
- in a real ts_h2 render (config 294, which checks every call): 350/350 calls (300 at S = 22,310, 50 at 44,455)
  equal in output and in every operand;
- for the final latents (video and audio) of ts_h2 and ts_s6 renders with 194, against the defaults.

**Use.** `RapidH3Int8Attention` config 100 + c (`rapidh3_attn=190` in `harness/workflow.py`'s presets since 2026-10-27, 194 before) tags its override.
`_fused_block` then calls `attn_int8.attention_rr` on the raw q/k instead of ck rms/rope + `optimized_attention`.
Config 200 + c does the same and checks each call against the unfused path; the result goes to
`/data/output/rapidh3_rrcheck.jsonl` (slow, for verification only). Any other override set later (Shadow, DumpQKV,
sparse) drops the tag, and the block falls back to the unfused path.

## 7. Softmax variants (attention round 6): configs 90, 89, 79

All three are config 94 (`attn_cake<7, 2>`) with a template switch `SMX` in the softmax; pre-pass, schedule and
TMA ring are unchanged, and each has a fused-rms/rope `1xx` form (190, 189, 179) like 194. Measured with
`bench/attn_cake.py` on real q/k/v (an er13 render dumped with `RapidH3DumpQKV`: ts_h2 draft S = 22,332 and ts_s6
S = 44,992, step 1 block 25 / step 5 blocks 5 and 45, all 56 heads) plus the synthetic 10 s shape (S = 78,592).
Raw data: `bench/results/attn_softmax_r6.json` (2026-10-27, 7 rounds × 3 iterations, interleaved).

| | S = 22,332 ms/call (main) | S = 44,992 ms/call (main) | S = 78,592 ms/call (main) | error vs fp32 (real, 6 dumps) |
|---|---:|---:|---:|---|
| 94 (= 194's kernel) | 93.87 (83.85) | 363.09 (345.02) | 1087.6 (1051.3) | 0.803–1.017% |
| **90** | **92.15 (82.83), −1.8%** | **354.40 (334.36), −2.4%** | **1063.3 (1037.4), −2.2%** | identical (bit-identical output) |
| 89 | 93.76 (83.66), −0.1% | 360.74 (343.44), −0.6% | 1079.8 (1053.4), −0.7% | identical to 4 digits (+0.0001 pt once) |
| 79 | 96.14 (86.05), +2.4% | 370.61 (351.63), +2.1% | 1106.7 (1076.3), +1.8% | +0.001–0.003 pt (0.806–1.019%) |

ms/call includes the pre-pass; "main" is the attention kernel alone (torch profiler). Over four separate runs the
main kernel of 90 was 1.2–2.6% faster than 94's at 22k and 2.3–3.1% at 45k.

**Config 79 (half-precision exps): slower, rejected.** The idea was two exps per MUFU op with
`ex2.approx.f16x2`. sm_121 has no paired MUFU.EX2: ptxas (CUDA 13.0) turns `ex2.approx.f16x2` into two
`MUFU.EX2.F16` plus a PRMT that re-packs the halves (the same on sm_90a, sm_100a and sm_120a; `ex2.approx.ftz.bf16x2`
likewise becomes two `MUFU.EX2.BF16`). Per 4 scores the fp32 path issues 4 FFMA + 4 MUFU + 2 F2IP; the half path
issues 4 FFMA + 2 F2FP (pack the arguments) + 4 MUFU + 2 PRMT (re-pack) + 2 HADD2 (+1024, which puts the rounded
integer in the low mantissa byte) + 1 PRMT (gather the 4 bytes), i.e. 15 instead of 10. The kernel has 80 more SASS
instructions per tile and is 2–3% slower. Precision would have been fine: the u8 target is 254 instead of 255 (so
that the f16 exp error can never round up to 256, which the magic add would turn into byte 0), and the error
moves by +0.001–0.003 points.

**Config 89 (lazy rescale): exact, but nothing to gain in this scheme.** In FlashAttention-4 the saving is
the O·a multiply skipped while the running max stands still. Here P is quantized against each tile's own max, so
every tile needs `ts = 2^((tm − m)c)` anyway, and the rescale was already free: the fold
`O = fma(ts, acc, fma(O, a, −ts·MAGIC))` spends its second FFMA on removing the 1.5·2²³ bias of the P·V
accumulator, and `a` rides along. A lazy max alone (a = 1) leaves 2 FFMA per element. Config 89 therefore also
keeps the magic bias in O: the fold is `O' = fma(ts, acc_bits, O')` (1 FFMA), `B += ts` per row, and a
warp-uniform pass `O' = a·O' − a·MAGIC·B` runs only when a row of the warp moves its max (by more than τ = 8 in
exp2 units, FA4's threshold; first tile always) or every 8 tiles, which keeps |O'| ≤ ~8·MAGIC·ts so its
rounding stays at the level of the old fold's intermediate (which already rounded at ~MAGIC·ts). The epilogue
removes the rest. This drops ≈56 of ≈680 non-IMMA instructions per tile and warp, yet the kernel is only
0.1–0.7% faster: half 0's fold sits between half 1's IMMAs and half 1's fold after the V stage is released, both
outside the softmax critical section, so those FFMAs were filling issue slots the tensor pipe left free.
Combined with config 90's schedule it was 3–6% *slower* than 90 alone (22k main 84.9 vs 79.8 ms, 45k 339.9 vs
329.4 ms). Not pursued as a default.

**Config 90 (the mutex starts after the row max): bit-identical, faster.** Config 94's softmax section, the one
the two warp groups take in turn, starts at the row-max shuffles. Config 90 moves `bar.sync` down past the four
shuffles and the per-row scalars (n, a, ts, b), so the section is the exps / u8 packing interleaved with
P·V half 0, as before. Each group then runs its latency-bound shuffle → MUFU chain while the other group holds
the section, instead of serializing it. Arithmetic and order of operations are unchanged, so the output is
`torch.equal` to config 94 (and 80) on every tested case: S = 1,000, 4,096, 22,332 (8 heads, synthetic), the 6
real dumps and S = 78,592 (56 heads), and on the fused rms/rope path and the text-bias path
(`bench/attn_softmax_r6.py`, S = 1,000–44,992, 2 seeds; the text bias accepts config 90 too). Variants tried,
all bit-identical and all slower than 90: ending the section earlier (after chunk 2 / 1 / 0 of P·V half 0:
+2%, +4%, +10% main kernel at 22k) and computing P chunk 0 before the section (+5%).

**Per-clip projection for 190 vs 194** (50 calls per step): ts_h2 5 s 300 × 1.7 ms + 50 × 8.7 ms ≈ **0.95 s**;
ts_s6 5 s 300 × 8.7 ms ≈ **2.6 s**; ts_s6 10 s 300 × 24 ms ≈ 7 s.

**Status:** 190 is the attention in `ts_h2` / `ts_s6` since 2026-10-27 (bit-identical to 194, so no quality check
was needed); `rapidh3_attn=194` runs the previous kernel. 189 and 179 stay opt-in for reference.

**End to end (2026-10-27).** The 6 screening shots rendered with ts_h2 / ts_s6
(194) and their `_a190` variants (seed 7): all 12 pairs have identical final latents (sha256) and identical mp4
frames and audio (`ffmpeg -f framemd5`), so quality scoring is moot. Steady-state synced steps: ts_h2 draft
10.52 → 10.44 s, refine 29.30 → 28.82 s (≈ −0.9 s per clip, 0.7%); ts_s6 29.74 → 29.30 s per step (≈ −2.6 s per
clip, 1.3%). In contiguous per-config blocks (3 other shots, first render of each block dropped) the clips were
123.4 → 120.3 s (ts_h2) and 205.7–206.9 → 203.5–205.5 s (ts_s6). Config 89's single-step divergence vs bf16
(RapidH3Shadow, ts_h2, er11 / er41 / ef31) is 1.84% video / 4.77% audio against 194's 1.88% / 4.76%.
