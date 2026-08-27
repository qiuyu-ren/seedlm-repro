#!/usr/bin/env python3
"""Run the SeedLM replication against real transformer weight tensors.

Model-agnostic: anything with HF Llama-style tensor names works, including
GQA architectures where k/v projections are narrower than q/o, and bf16
checkpoints as well as fp16.

Answers three questions the synthetic study left open:

1. Are real transformer weights near-Gaussian or heavy-tailed?  That decides
   which row of the synthetic comparison is the representative one, and hence
   whether SeedLM actually beats a trivial data-free baseline.
2. Does SeedLM beat data-free block floating point at matched bits on real
   tensors?  The paper never runs a data-free baseline.
3. Is the 4-bit shared exponent's absolute range a real problem at the scale
   real weights occupy?  The synthetic sweep predicted about one octave of
   headroom; on Llama-2-7B it turned out to cost only 0.4%.

    python real_weights.py slices.safetensors --label Llama-2-7B
    python real_weights.py slices.safetensors --label Qwen2.5-7B
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

from seedlm import SEEDLM_3BIT, SEEDLM_4BIT, compress, decompress, relative_error

RESULTS = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
def load_safetensors(path):
    """Minimal reader; handles F16/BF16/F32 without a torch dependency."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        a, b = meta["data_offsets"]
        raw, dt = data[a:b], meta["dtype"]
        if dt == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        elif dt == "BF16":
            # bfloat16 is the top 16 bits of a float32
            u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            arr = u.view(np.float32) if u.dtype == np.uint32 else u.astype(np.uint32).view(np.float32)
        elif dt == "F32":
            arr = np.frombuffer(raw, dtype=np.float32).copy()
        else:
            raise ValueError(f"unsupported dtype {dt} for {name}")
        out[name] = arr.reshape(meta["shape"])
    return out


def excess_kurtosis(x):
    x = np.asarray(x, np.float64).ravel()
    d = x - x.mean()
    m2 = (d * d).mean()
    return float((d ** 4).mean() / (m2 * m2) - 3.0) if m2 > 0 else 0.0


# ---------------------------------------------------------------------------
def blockfp_opt(W, mant_bits, group, exp_bits=4, grid=None):
    """Data-free block floating point with a per-group MSE-optimal clip.

    Integer mantissas plus one power-of-two exponent per group -- the same
    storage idea SeedLM uses for its own shared exponent, and a shift rather
    than a multiply on the decode side.  Reads only the weights, so it is as
    data-free as SeedLM is.  bits/element = mant_bits + exp_bits/group.
    """
    grid = np.linspace(0.25, 1.0, 76) if grid is None else grid
    flat = np.asarray(W, np.float64).ravel()
    pad = (-flat.size) % group
    if pad:
        flat = np.concatenate([flat, np.zeros(pad)])
    X = flat.reshape(-1, group)

    qmax = 2 ** (mant_bits - 1) - 1
    lo, hi = -(1 << (exp_bits - 1)), (1 << (exp_bits - 1)) - 1
    amax0 = np.abs(X).max(1, keepdims=True)
    amax0[amax0 == 0] = 1.0

    best = best_err = None
    for c in grid:
        s = 2.0 ** np.clip(np.ceil(np.log2(amax0 * c / (qmax + 0.5))), lo, hi)
        Q = np.clip(np.rint(X / s), -qmax - 1, qmax) * s
        e = ((X - Q) ** 2).sum(1)
        if best is None:
            best, best_err = Q, e
        else:
            t = e < best_err
            best = np.where(t[:, None], Q, best)
            best_err = np.where(t, e, best_err)

    out = best.ravel()
    if pad:
        out = out[:-pad]
    return out.reshape(np.shape(W)).astype(np.float32), mant_bits + exp_bits / group


def blockfp_fp16(W, mant_bits, group, grid=None):
    """Group-wise integer quantisation with a real fp16 scale.

    What AWQ and GPTQ actually use, and the strongest data-free baseline here:
    no power-of-two floor, so it is not handicapped by the same absolute
    exponent range that constrains SeedLM.  bits/element = mant_bits + 16/group.
    """
    grid = np.linspace(0.25, 1.0, 76) if grid is None else grid
    flat = np.asarray(W, np.float64).ravel()
    pad = (-flat.size) % group
    if pad:
        flat = np.concatenate([flat, np.zeros(pad)])
    X = flat.reshape(-1, group)

    qmax = 2 ** (mant_bits - 1) - 1
    amax0 = np.abs(X).max(1, keepdims=True)
    amax0[amax0 == 0] = 1.0

    best = best_err = None
    for c in grid:
        s = np.maximum((amax0 * c / qmax).astype(np.float16).astype(np.float64), 1e-30)
        Q = np.clip(np.rint(X / s), -qmax - 1, qmax) * s
        e = ((X - Q) ** 2).sum(1)
        if best is None:
            best, best_err = Q, e
        else:
            t = e < best_err
            best = np.where(t[:, None], Q, best)
            best_err = np.where(t, e, best_err)

    out = best.ravel()
    if pad:
        out = out[:-pad]
    return out.reshape(np.shape(W)).astype(np.float32), mant_bits + 16 / group


def group_peak(w, group=128):
    """Per-group max / rms, the statistic a group-wise quantiser actually feels.

    Its step size is set by the group's largest magnitude while the signal it
    must represent is the group's rms, so this ratio -- not global kurtosis --
    is what decides whether a grid can cope.  Purely Gaussian weights score
    2.83 for groups of 128.  Across Llama-2-7B and Qwen2.5-7B this separates
    the tensors SeedLM beats block floating point on (3.87-6.24) from those it
    loses (2.79-3.86) with no overlap, while kurtosis flips sign between the
    two models.
    """
    x = np.asarray(w, np.float64).ravel()
    x = x[: (len(x) // group) * group].reshape(-1, group)
    rms = np.sqrt((x * x).mean(1))
    rms[rms == 0] = 1.0
    gp = np.abs(x).max(1) / rms
    return float(gp.mean()), float(np.percentile(gp, 99)), float(gp.max())


def _one_tensor(job):
    """Compress one tensor slice every way.  Module level so it is picklable.

    The exhaustive seed search is dominated by elementwise work in the
    coefficient quantiser, which numpy runs single-threaded, so spreading
    tensors across processes is close to a linear speed-up.
    """
    name, w = job
    rec = {"name": name}
    gp, gp99, gpmax = group_peak(w)
    rec["group_peak"], rec["group_peak_p99"], rec["group_peak_max"] = gp, gp99, gpmax

    for bits, cfg in ((4, SEEDLM_4BIT), (3, SEEDLM_3BIT)):
        ct = compress(w, cfg)
        rec[f"seedlm_{bits}"] = relative_error(w, decompress(ct))
        if bits == 4:
            rec["frac_at_exp_floor"] = float((ct.e == -8).mean())
            rec["e_min"], rec["e_max"] = int(ct.e.min()), int(ct.e.max())

        q, b = blockfp_opt(w, bits, 128)
        rec[f"blockfp_{bits}"] = relative_error(w, q)
        rec[f"blockfp_{bits}_bits"] = b

        q, b = blockfp_fp16(w, bits, 128)
        rec[f"blockfp16_{bits}"] = relative_error(w, q)
        rec[f"blockfp16_{bits}_bits"] = b

        q, b = blockfp_fp16(w, bits, len(w))          # per-channel, ~bits + 0.001
        rec[f"blockfp16_pc_{bits}"] = relative_error(w, q)
        rec[f"blockfp16_pc_{bits}_bits"] = b
    return rec


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slices")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-search", type=int, default=65536,
                    help="weights per tensor fed to the exhaustive seed search")
    ap.add_argument("--label", default="LLAMA",
                    help="model name for the report and the results filename")
    ap.add_argument("--jobs", type=int, default=0,
                    help="worker processes (default: all cores). Each holds its "
                         "own codebook, ~250MB peak.")
    args = ap.parse_args()
    label = args.label.upper()
    slug = args.label.lower().replace("/", "_").replace(" ", "-")

    n_search = 16384 if args.quick else args.n_search
    n_jobs = args.jobs if args.jobs > 0 else max(1, cpu_count())
    tensors = load_safetensors(args.slices)
    names = sorted(tensors)
    print(f"loaded {len(names)} real weight tensors from {args.slices}"
          f"   [{args.label}]\n")

    # -- 1. distribution ----------------------------------------------------
    print("=" * 82)
    print(f"1.  WHAT DO REAL {label} WEIGHTS ACTUALLY LOOK LIKE?")
    print("=" * 82)
    rng = np.random.default_rng(0)
    ref = {
        "Gaussian": rng.normal(0, 1, 400000),
        "Student-t(4)": rng.standard_t(4, 400000),
    }
    mix = rng.normal(0, 1, 400000)
    mix[rng.random(400000) < 0.01] *= 8.0
    ref["1% outliers"] = mix
    print("  reference distributions used in the synthetic study:")
    for k, v in ref.items():
        print(f"      {k:<14} excess kurtosis {excess_kurtosis(v):8.2f}   "
              f"max/std {np.abs(v).max() / v.std():6.1f}")

    print(f"\n  {'tensor':<44} {'std':>9} {'kurtosis':>9} {'max/std':>8}")
    stats = []
    for nm in names:
        w = tensors[nm].astype(np.float64)
        s, k = w.std(), excess_kurtosis(w)
        r = np.abs(w).max() / s
        stats.append(dict(name=nm, std=float(s), kurtosis=k, max_over_std=float(r)))
        print(f"  {nm.replace('model.layers.', 'L'):<44} {s:>9.5f} {k:>9.2f} {r:>8.1f}")
    ks = np.array([x["kurtosis"] for x in stats])
    sd = np.array([x["std"] for x in stats])
    print(f"\n  median excess kurtosis {np.median(ks):.2f}, "
          f"range {ks.min():.2f} to {ks.max():.2f}")
    print(f"  weight std: median {np.median(sd):.5f}, "
          f"range {sd.min():.5f} to {sd.max():.5f}")

    # -- 2. matched-bit comparison -----------------------------------------
    print("\n" + "=" * 82)
    print("2.  SEEDLM vs DATA-FREE BLOCK FLOATING POINT, MATCHED BITS")
    print("=" * 82)
    print(f"  {n_search} weights per tensor, full 65535-seed search, "
          f"{n_jobs} worker(s).")
    print(f"  SeedLM is exactly 4.000 / 3.000 bits. Baselines at g=128:")
    print(f"    BFP2  = shared power-of-two exponent   (4.031 / 3.031 bits)")
    print(f"    BFP16 = fp16 scale, as AWQ/GPTQ use    (4.125 / 3.125 bits)\n")
    print(f"  {'tensor':<38} {'SL-4':>7} {'BFP2':>7} {'BFP16':>7} {'SL-3':>7} "
          f"{'BFP16':>7} {'gpeak':>6} {'%flr':>6}")

    rows = []
    t0 = time.time()
    jobs = [(nm, tensors[nm].ravel()[:n_search].astype(np.float32)) for nm in names]
    with Pool(processes=n_jobs) as pool:
        for rec in pool.imap(_one_tensor, jobs):
            rows.append(rec)
            print(f"  {rec['name'].replace('model.layers.', 'L'):<38} "
                  f"{rec['seedlm_4']:>7.4f} {rec['blockfp_4']:>7.4f} "
                  f"{rec['blockfp16_4']:>7.4f} {rec['seedlm_3']:>7.4f} "
                  f"{rec['blockfp16_3']:>7.4f} {rec['group_peak']:>6.2f} "
                  f"{100 * rec['frac_at_exp_floor']:>5.1f}%", flush=True)
    rows.sort(key=lambda r: r["name"])
    print(f"\n  ({time.time() - t0:.0f}s)")

    for bits in (4, 3):
        sl = np.array([r[f"seedlm_{bits}"] for r in rows])
        print(f"\n  {bits}-bit -- SeedLM median {np.median(sl):.4f}")
        for key, lab in ((f"blockfp_{bits}", "block-FP pow2-exp  g=128"),
                         (f"blockfp16_{bits}", "block-FP fp16      g=128"),
                         (f"blockfp16_pc_{bits}", "block-FP fp16  per-channel")):
            bf = np.array([r[key] for r in rows])
            b = rows[0][key + "_bits"]
            print(f"      vs {lab}  ({b:.3f} bits): {np.median(bf):.4f}   "
                  f"SeedLM wins {int((sl < bf).sum())}/{len(rows)}")

    # -- group peak, the cross-model predictor -----------------------------
    gp = np.array([r["group_peak"] for r in rows])
    m4 = np.array([r["blockfp16_4"] / r["seedlm_4"] for r in rows])
    ku = np.array([next(x["kurtosis"] for x in stats if x["name"] == r["name"])
                   for r in rows])
    def _sp(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])
    print(f"\n  Predicting the margin (block-FP fp16 / SeedLM, 4-bit):")
    print(f"      Spearman vs excess kurtosis   {_sp(ku, m4):+.2f}")
    print(f"      Spearman vs mean group peak   {_sp(gp, m4):+.2f}")
    print(f"      group peak: median {np.median(gp):.2f}, max {gp.max():.2f}, "
          f"{int((gp > 4).sum())}/{len(rows)} above 4.0  (Gaussian = 2.83)")

    # -- 3. exponent range --------------------------------------------------
    print("\n" + "=" * 82)
    print(f"3.  IS THE 4-BIT EXPONENT FLOOR A REAL PROBLEM AT {label} SCALE?")
    print("=" * 82)
    fl = np.array([r["frac_at_exp_floor"] for r in rows])
    emin = min(r["e_min"] for r in rows)
    emax = max(r["e_max"] for r in rows)
    print(f"  exponents used across all tensors: [{emin}, {emax}]  "
          f"(representable range is [-8, 7])")
    print(f"  blocks pinned to the floor: median {100 * np.median(fl):.1f}%, "
          f"worst tensor {100 * fl.max():.1f}%")
    print(f"  headroom below the smallest tensor std "
          f"({sd.min():.5f}): {np.log2(sd.min() / 2e-3):.1f} octaves "
          f"before the synthetic study's collapse point")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"real_weights_{slug}.json").write_text(
        json.dumps(dict(stats=stats, comparison=rows), indent=2))
    print(f"\n  written to {RESULTS}/real_weights_{slug}.json")


if __name__ == "__main__":
    sys.exit(main())
