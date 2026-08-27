# SeedLM — from-scratch replication

An implementation of

> Shafipour, Harrison, Horton, Marker, Bedayat, Mehta, Rastegari, Najibi, Naderiparizi.
> **SeedLM: Compressing LLM Weights into Seeds of Pseudo-Random Generators.** ICLR 2025.
> [arXiv:2410.10714](https://arxiv.org/abs/2410.10714)

written entirely from the paper text. **No official code was released** — Apple's
ML Research page for SeedLM links only to the preprint, and there is no Apple
repository or well-known third-party port. Everything here comes from Sections
3.1–3.4, Table 1, Table 6, and Algorithms 1–3.

Evaluated on synthetic weights and on 70 real tensors from Llama-2-7B and
Qwen2.5-7B. The metric throughout is reconstruction error, not perplexity — see
the caveat at the end of the real-weights section.

```bash
python -m pytest tests/ -q            # 56 checks against the paper, ~3s
python validate.py                    # synthetic validation report, ~80s
python results/independent_check.py   # second implementation, shares no code

# real weights: extract slices on the machine holding the checkpoint...
python3 extract_llama.py ~/llama2-7b slices.safetensors   # or any HF-style checkpoint
# ...then compare against a data-free baseline (~14 min, 2 cores)
python real_weights.py slices.safetensors --label Llama-2-7B \
    --jobs $(nproc) --n-search 131072
```

`real_weights.py` needs only numpy. It reports SeedLM against three data-free
baselines at matched bits, plus the mean group peak per tensor, and writes
everything to `results/real_weights_<label>.json` — that JSON is self-contained,
so the weight slices are not needed for any downstream analysis.

```python
import numpy as np
from seedlm import compress, decompress, pack, SEEDLM_4BIT

W  = np.random.default_rng(0).normal(0, 0.02, (256, 256)).astype(np.float32)
ct = compress(W, SEEDLM_4BIT)     # searches all 65535 seeds per block
ct.bits_per_element()             # -> 4.0
What = decompress(ct)             # rebuild from seeds + coefficients
blob = pack(ct)                   # the actual bitstream, len == n_blocks * 32 / 8
```

## What the method does

Cut the weight tensor into blocks of `C` contiguous elements. For each block
`w ∈ R^C`, search every `K`-bit LFSR seed; each seed `s` names a pseudo-random
basis `U(s) ∈ R^{C×P}` read off the LFSR state sequence and mapped into
`[-1, 1]`. Take the least-squares coefficients `t = U(s)⁺w`, quantise them to
`P` 4-bit mantissas with one shared 4-bit exponent, and keep the seed whose
quantised reconstruction is closest. Store only the seed and the coefficients:
`K + 4 + 4P` bits per block, so `M = (K + 4 + 4P)/C` bits per weight.

The point is not that this beats a codebook on rate–distortion; it is that the
basis never has to be read from DRAM. It is regenerated on-chip by a few XOR
gates, trading compute — which is ~200× cheaper per bit than a DRAM read — for
memory traffic.

## Layout

| File | Paper section |
|---|---|
| `seedlm/lfsr.py` | 3.1, A.1 (Table 6 taps), A.2 (Figure 4), A.3 (Algorithm 1) |
| `seedlm/quant.py` | 3.2 — 4-bit mantissas + shared 4-bit exponent |
| `seedlm/compress.py` | 3.3 / A.5 (Algorithm 3 search), A.4 (Algorithm 2 reconstruction), Eq. 3 |
| `seedlm/pack.py` | Eq. 3, made checkable — real bitstream serialisation |
| `tests/test_seedlm.py` | 56 assertions, each naming what it checks |
| `validate.py` | the synthetic report below |
| `extract_llama.py` | stdlib-only safetensors slicer; runs where the checkpoint is |
| `real_weights.py` | the real-weight comparison and its data-free baseline |
| `results/pooled_predictor.json` | per-tensor group-peak vs margin, both models |
| `report/seedlm-baseline.html` | the findings write-up, self-contained |

## Reproduced without modification

* **Figure 4 worked example.** `V(4)` for `K=3, C=4, P=2` comes out
  `[[2,5],[6,7],[3,1],[4,2]]`, matching the paper exactly — which pins down the
  shift direction and the off-by-one on when the state is recorded.
* **Table 6.** All 23 tap sets are maximal-length (walked directly for `K ≤ 20`;
  confirmed for the rest through the multiplicative order of the GF(2)
  transition matrix).
* **Table 1 / Eq. 3.** `(8,3,16) → 4.000` and `(12,4,16) → 3.000` bits per
  element, confirmed against an actual packed bitstream, not just arithmetic.
* **Section 3.3 memory claim.** The pseudo-inverse cache is 6.29 MB — matching
  the paper's "at most 6.3MB" for the 4-bit config in fp32, or the 3-bit config
  in fp16. The 3-bit config in fp32 is 12.6 MB, so the quoted figure implies one
  of those two readings.

## Results on synthetic weights

Synthetic weights, `σ = 0.02` (typical LLM tensor scale), full 65535-seed search:

| Config | Gaussian | Student-t(4) | 1% outliers |
|---|---|---|---|
| 4-bit | 0.1230 | 0.1255 | 0.1182 |
| 3-bit | 0.2257 | 0.2341 | 0.2177 |

(relative Frobenius error). Heavier tails cost only ~2%, which is consistent
with the method being data-free — it never fits a distribution in the first
place.

**What the seed search buys.** One fixed basis with exact coefficients leaves
`sqrt((C-P)/C) = 0.7906` of the energy behind; measured 0.7851. Searching all
`2^16` seeds takes that to 0.1228 — a **6.4× reduction**, bought entirely with
the 16 bits spent naming the seed:

| seeds searched | 1 | 256 | 4096 | 65535 |
|---|---|---|---|---|
| rel. error | 0.785 | 0.303 | 0.188 | 0.123 |

Splitting the loss: with the winning seed but *unquantised* coefficients the
error is 0.1072, so the 4-bit mantissas add only 15% on top. Almost all of the
work is done by choosing the basis, not by resolving the coefficients — which is
the paper's actual thesis, and it holds up.

## Results on real Llama-2-7B weights

35 tensors from `meta-llama/Llama-2-7b-hf` — 7 projection types (q/k/v/o,
gate/up/down) at layers 0, 8, 16, 23, 31 — as contiguous row slices, 32,768
weights each, full 65,535-seed search. 13.6 min on two cores.

**Real weights are bimodal.** Excess kurtosis has a median of 0.27 but a range
of 0.01 to 77.4. Every MLP projection is essentially Gaussian (0.01–0.27,
max/std ≈ 5); the heavy tails live entirely in attention, concentrated at layer
0 (q_proj 77.4, k_proj 36.0, o_proj 28.4, max/std up to 64).

**Against a data-free baseline the paper never runs.** Block floating point —
integer mantissas, one fp16 scale per group of 128, MSE-optimal clip; reads only
the weights, so exactly as data-free as SeedLM:

| | bits/weight | median rel. error | SeedLM wins |
|---|---|---|---|
| SeedLM 4-bit | 4.000 | 0.1234 | — |
| block-FP g=128 | 4.125 | **0.1018** | 2/35 |
| block-FP g=512 | 4.031 | 0.1083 | 5/35 |
| block-FP per-channel | **4.004** | 0.1104 | 5/35 |

Same at 3 bits: 0.2256 vs 0.1965, 2/35. The margin is not bought with the extra
bits — at 4.004 bits, 0.1% above SeedLM's exact 4.000, the baseline still wins
on 30 of 35.

**The two wins are the two most heavy-tailed tensors**, by a wide margin:
`L0.attn.q_proj` (kurtosis 77.4) at 1.91×, `L0.attn.k_proj` (36.0) at 1.71×.
Everywhere else SeedLM loses at 0.81–0.88×. Spearman(kurtosis, margin) = +0.29.

**What SeedLM actually owns is a flat error profile.** Its error spans
0.1189–0.1457 (σ = 0.0047) across all 35 tensors; block-FP spans 0.0993–0.2268
(σ = 0.0274), 5.8× the spread. SeedLM is a fixed-rate coder whose error barely
depends on the weight distribution, because it never fits one. Uniform
quantization is better on well-behaved tensors and collapses on pathological
ones.

**The caveat cuts in SeedLM's favour.** Reconstruction error is not perplexity.
The two tensors where uniform quantization collapses are early-layer attention
q/k — exactly the weights the quantization literature treats as
accuracy-critical. A method that is mediocre on all 35 can beat one that is
excellent on 33 and catastrophic on 2, if those 2 dominate the loss. That would
reconcile the paper's perplexity numbers with everything above, and it is
directly checkable by quantizing both ways and measuring WikiText-2. Not done
here.

## Two models, and a corrected mechanism

Qwen2.5-7B, sampled identically (35 tensors, GQA, bf16, ~18T training tokens
against Llama-2's 2T):

| | Llama-2-7B | Qwen2.5-7B |
|---|---|---|
| SeedLM 4-bit (4.000 bits) | 0.1234 | 0.1247 |
| block-FP fp16 g=128 (4.125 bits) | **0.1018** | **0.1062** |
| SeedLM wins | 2/35 | 1/35 |
| SeedLM error spread (σ) | 0.0047 | 0.0047 |
| block-FP error spread (σ) | 0.0274 | 0.0070 |

**The heavy-tail explanation is wrong.** From Llama alone it looked like SeedLM
wins on heavy-tailed tensors — its two wins were the two highest-kurtosis ones.
Qwen falsifies that: it is *more* heavy-tailed (median excess kurtosis 1.26 vs
0.27, max 156 vs 77) and SeedLM does worse there. The kurtosis correlation
flips sign between models, +0.29 to −0.13.

**The right predictor is mean group peak** — for each group of 128 weights,
`max|w| / rms|w|`, averaged over groups. A group-wise quantiser sets its step
from the max and must represent the rms, so that is the ratio it feels;
kurtosis is a whole-tensor statistic and misses it.

| predictor | Llama | Qwen | pooled (70 tensors) |
|---|---|---|---|
| excess kurtosis | +0.29 | −0.13 | +0.32 |
| **mean group peak** | **+0.76** | **+0.71** | **+0.73** |

The separation is exact: every tensor where SeedLM wins scores 3.87–6.24, every
tensor where it loses scores 2.79–3.86, no overlap across two architectures.
Purely Gaussian weights score 2.83, and 65 of the 70 real tensors are below 3.2
— at the granularity a quantiser works on, almost all real weights are Gaussian.

SeedLM wins when outliers are **concentrated enough to wreck an individual
quantisation group**, not when the tensor is globally heavy-tailed. Qwen's fat
tails are diffuse; Llama-2's layer-0 attention outliers are piled into
particular groups.

## Findings — where the paper is wrong, ambiguous, or worth knowing about

**1. The shared-exponent rule as printed cannot be what they ran.**
Section 3.2 gives `e = max_i floor(log2|t_i|)`. That places `max|t| / 2^e` in
`[1, 2)`, so rounding can only ever return `{-2,-1,0,1,2}` — 5 of the 16
available codes. Implemented literally it gives 0.1912 instead of 0.1233 at
4 bits, 1.55× worse, which would not have produced the reported accuracy.
Shifting the exponent down by ~2 so the mantissas span the full range
(`e = ceil(log2(max|t| / 7.5))`) recovers it. Both are implemented; select with
`exponent_mode="paper"` or `"range"` (default). **This is the one place the
replication has to deviate from the text.**

**2. Figure 4 and Algorithm 2 define two different seed→matrix maps.**
Figure 4 runs the LFSR forward *from the seed*. Algorithm 2 step 5 instead
indexes a cached full cycle at offset `s mod length`. These are different
functions of `s`. They expose the same *set* of candidate matrices, so no
accuracy number changes — but a decoder built to one convention and a seed
found under the other will reconstruct garbage. Both are implemented
(`build_V_direct`, `build_V_cached`), the test suite proves the sets are equal,
and `CompressedTensor.hardware_seeds()` converts a search-space offset into the
register state a real LFSR must be loaded with.

**3. The 4-bit exponent is an absolute range, and LLM weights sit near its floor.**
SeedLM has no per-tensor or per-channel scale, and `e` is stored in 4 bits, so
`e ∈ [-8, 7]`. The method is therefore only scale-invariant while the
coefficients stay inside that window. Section 3.4 tunes `(C,P,K)` on
**standard normal** `w`, where the window is comfortable — so their own design
analysis could not have surfaced this:

| σ | 1e0 | 1e-1 | 2e-2 | 1e-2 | 1e-3 | 1e-4 |
|---|---|---|---|---|---|---|
| rel. error | 0.123 | 0.123 | 0.122 | 0.129 | 0.413 | 0.586 |
| % blocks at exponent floor | 0 | 0 | 46 | 98 | 73 | 0 |

Llama-scale tensors (σ ≈ 0.01–0.02) are inside the envelope but with roughly one
octave of headroom, and 98% of blocks already pinned to the floor at σ = 0.01.
A tensor with a smaller norm falls out of it.

> **Corrected by the real-weight run.** This predicted the floor would matter at
> Llama scale, and it mostly doesn't. Real tensors do sit on it — a median of
> 63.5% of blocks pinned, worst tensor 99.1% — but rescaling each tensor to unit
> std, so the window never binds, moves the median error from 0.1235 to 0.1231.
> A 0.4% effect; only the 99.1% tensor improves materially (0.1401 → 0.1240).
> The floor is a real design fragility, not a material accuracy cost on this
> checkpoint, and SeedLM's ~20% deficit against block-FP is intrinsic to the
> pseudo-random basis rather than an artifact of the exponent spec.

**4. Below that floor, the search starts exploiting near-singular bases.**
Once well-conditioned bases quantise to all-zero coefficients, the only seeds
that reconstruct anything are the ill-conditioned ones, whose huge coefficients
survive quantisation. Median condition number of the *chosen* seeds:

| σ | 1e-2 | 1e-3 | 1e-4 |
|---|---|---|---|
| median cond(U) of chosen seeds | 2.4 | 6.6 | **166,630** |

against a median of 2.6 across all 65535 seeds. The degradation is graceful in
error terms but the mechanism is pathological, and it is worth knowing about
before trusting the method on an unusually small tensor.

**5. Ill-conditioning also breaks the obvious pseudo-inverse implementation.**
Consecutive seeds are overlapping windows of one sequence, so some bases are
nearly collinear: cond(U) is 2.6 at the median but reaches 5e5. Computing
`(UᵀU)⁻¹Uᵀ` squares that, which in float32 destroys ~0.8% of seeds. The
codebook uses an SVD in float64 instead — 0.4s for all 65535, so there is no
reason not to.

**6. The 65535 candidate bases are nowhere near 65535 independent bases.**
Because `U(s)` is a sliding window, column `p` of `U(s)` *is* column `p-1` of
`U(s+1)`. So `U(s)·(0,t₁,t₂)` and `U(s+1)·(t₁,t₂,0)` are the same weight block:
the candidate set contains exact ties, and the arg-min over seeds is genuinely
non-unique. This bounds how much diversity the search has to work with, and it
means any test asserting *which* seed wins is asserting a round-off outcome
rather than an invariant.

## Not implemented

Out of scope: the Section 3.4 `(C,P,K)` grid search that produced Table 1;
AWQ/OmniQuant/GPTQ as baselines (they need calibration data and their own
repos); end-to-end perplexity and zero-shot evaluation; the FPGA RTL of
Section 4.2.

The perplexity run is the one that matters most — see the caveat under real
weights. Everything here is weight-space reconstruction error, which is a proxy.

## Scaling up

The search is exhaustive as the paper specifies, costing roughly 3–5 ms per
block on two CPU cores. That is fine for validation and far too slow for a whole
7B checkpoint: 6.5B linear weights at `C=8` is 812M blocks, or about five weeks.
The 35-tensor study above samples 32,768 weights per tensor instead, which is
ample for stable error statistics.

The route to scaling, not taken here in order to keep the implementation
faithful: the unquantised least-squares error is a valid *lower bound* on the
quantised error and costs one small matmul, so one can rank all seeds by that
bound, evaluate the full quantised path on only the top few, and confirm
exactness by checking that the best discarded bound exceeds the best achieved
error. That prunes ~99.9% of the work while still returning the exact arg-min.

## Reproducing the real-weight numbers

`extract_llama.py` runs on the machine holding the checkpoint and needs nothing
installed — no numpy, no torch, no safetensors package. It parses the
safetensors header, seeks to the byte ranges it wants, and writes a ~70 MB valid
safetensors file, reading only those ranges rather than loading the 13.5 GB of
shards.

It takes **contiguous row slices**, which matters: SeedLM's blocks are `C`
adjacent elements in row-major order, so sampling scattered weights would
destroy the exact structure the method sees. Four evenly spaced row chunks per
tensor keep each block intact while spreading the sample across the tensor.

```bash
hf download meta-llama/Llama-2-7b-hf \
    --include "*.safetensors" --include "*.json" --local-dir ~/llama2-7b
python3 extract_llama.py ~/llama2-7b slices.safetensors
python real_weights.py slices.safetensors --label Llama-2-7B \
    --jobs $(nproc) --n-search 131072
```

`real_weights.py` needs only numpy. It reports SeedLM against three data-free
baselines at matched bits, plus the mean group peak per tensor, and writes
everything to `results/real_weights_<label>.json` — that JSON is self-contained,
so the weight slices are not needed for any downstream analysis.

The slices are Meta's weights under the Llama 2 license and are not included in
this bundle.
