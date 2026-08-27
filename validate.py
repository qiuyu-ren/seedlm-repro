#!/usr/bin/env python3
"""End-to-end validation of the SeedLM replication on synthetic weights.

Writes a human-readable report to stdout and a machine-readable summary to
``results/validation.json``.

    python validate.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from seedlm import (
    LFSR_TAPS,
    SEEDLM_3BIT,
    SEEDLM_4BIT,
    Codebook,
    build_V_direct,
    compress,
    decompress,
    is_maximal_length,
    relative_error,
)

RESULTS = Path(__file__).parent / "results"


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def nmse(W, What) -> float:
    """Normalised mean squared error, ||W-What||^2 / ||W||^2."""
    d = np.asarray(W, np.float64) - np.asarray(What, np.float64)
    return float((d * d).sum() / (np.asarray(W, np.float64) ** 2).sum())


def synth_weights(shape, kind="gaussian", sigma=0.02, seed=0):
    """Stand-ins for a real LLM weight tensor.

    ``gaussian``  matches the paper's own design-space assumption (Section 3.4).
    ``heavy``     Student-t(4), closer to the outlier-heavy reality of trained
                  transformer weights.
    ``mixture``   99% narrow Gaussian plus 1% wide, a crude outlier model.
    """
    rng = np.random.default_rng(seed)
    n = int(np.prod(shape))
    if kind == "gaussian":
        w = rng.normal(0, 1, n)
    elif kind == "heavy":
        w = rng.standard_t(4, n)
        w /= w.std()
    elif kind == "mixture":
        w = rng.normal(0, 1, n)
        out = rng.random(n) < 0.01
        w[out] *= 8.0
        w /= w.std()
    else:
        raise ValueError(kind)
    return (w * sigma).astype(np.float32).reshape(shape)


# ---------------------------------------------------------------------------
def check_paper_claims(out: dict) -> None:
    rule("1.  CLAIMS CHECKED DIRECTLY AGAINST THE PAPER")

    v = build_V_direct(K=3, seed=4, C=4, P=2)
    fig4 = np.array_equal(v, np.array([[2, 5], [6, 7], [3, 1], [4, 2]]))
    print(f"  Figure 4 worked example V(4) reproduces exactly ......... {fig4}")

    maximal = {K: is_maximal_length(K) for K in sorted(LFSR_TAPS) if K <= 18}
    print(
        f"  Table 6 taps give maximal-length cycles (K=2..18) ....... "
        f"{all(maximal.values())}  [{sum(maximal.values())}/{len(maximal)}]"
    )

    print(f"  Eq. 3 bit budget, Table 1 configurations:")
    for nm, cfg in (("4-bit", SEEDLM_4BIT), ("3-bit", SEEDLM_3BIT)):
        print(
            f"      {nm}: C={cfg.C:2d} P={cfg.P} K={cfg.K}"
            f"  ->  ({cfg.K}+4+4*{cfg.P})/{cfg.C} = {cfg.bits_per_element:g} bits/element"
        )

    print("  Section 3.3 pseudo-inverse cache (paper: 'at most 6.3MB'):")
    mem = {}
    for nm, cfg in (("4-bit", SEEDLM_4BIT), ("3-bit", SEEDLM_3BIT)):
        cb = Codebook.get(cfg)
        mb32 = cb.Upinv.nbytes / 1e6
        mem[nm] = mb32
        print(
            f"      {nm}: U+ is {cb.Upinv.shape} = {mb32:.2f} MB fp32"
            f" / {mb32 / 2:.2f} MB fp16"
        )

    out["claims"] = {
        "figure4_exact": bool(fig4),
        "table6_all_maximal_to_K18": bool(all(maximal.values())),
        "bits_4bit": SEEDLM_4BIT.bits_per_element,
        "bits_3bit": SEEDLM_3BIT.bits_per_element,
        "pinv_mb_fp32": mem,
    }


# ---------------------------------------------------------------------------
def check_reconstruction(out: dict, n_weights: int) -> None:
    rule("2.  RECONSTRUCTION QUALITY ON SYNTHETIC WEIGHTS")
    print("  Full search over all 65535 seeds, sigma = 0.02 (typical LLM scale).")
    print(f"  {'config':<8} {'distribution':<12} {'rel.err':>9} {'NMSE':>10} "
          f"{'blocks':>8} {'time':>8}")
    rows = []
    for nm, cfg in (("4-bit", SEEDLM_4BIT), ("3-bit", SEEDLM_3BIT)):
        for kind in ("gaussian", "heavy", "mixture"):
            W = synth_weights((n_weights,), kind=kind, sigma=0.02, seed=1)
            t0 = time.time()
            ct = compress(W, cfg)
            dt = time.time() - t0
            What = decompress(ct)
            r, m = relative_error(W, What), nmse(W, What)
            print(f"  {nm:<8} {kind:<12} {r:>9.4f} {m:>10.5f} "
                  f"{ct.n_blocks:>8d} {dt:>7.1f}s")
            rows.append(dict(config=nm, dist=kind, rel_err=r, nmse=m,
                             blocks=ct.n_blocks, seconds=dt,
                             bits=ct.bits_per_element()))
    out["reconstruction"] = rows


# ---------------------------------------------------------------------------
def exact_coefficient_error(W, ct) -> float:
    """Rel. error the chosen seeds would give with unquantised coefficients.

    Separates the two sources of loss: how good the selected basis is, versus
    how much the 4-bit mantissas throw away on top of it.
    """
    cfg = ct.cfg
    cb = Codebook.get(cfg)
    flat = np.asarray(W, np.float32).reshape(-1)
    if ct.pad:
        flat = np.concatenate([flat, np.zeros(ct.pad, np.float32)])
    blocks = flat.reshape(-1, cfg.C)
    idx = ct.seeds.astype(np.int64)
    t = np.einsum("npc,nc->np", cb.Upinv[idx], blocks)
    rec = np.einsum("ncp,np->nc", cb.U[idx], t)
    return relative_error(blocks, rec)


def check_search_gain(out: dict, n_blocks: int) -> None:
    rule("3.  WHAT THE SEED SEARCH ACTUALLY BUYS")
    cfg = SEEDLM_4BIT
    W = synth_weights((n_blocks * cfg.C,), sigma=0.02, seed=2)

    # Anchor: a rank-P projection onto a *fixed* random subspace, with exact
    # (unquantised) coefficients.  For an arbitrary P-dim subspace and isotropic
    # w this leaves (C-P)/C of the energy behind.
    floor = np.sqrt((cfg.C - cfg.P) / cfg.C)
    print(f"  Anchor - one fixed random basis, exact coefficients:")
    print(f"      rel.err -> sqrt((C-P)/C) = {floor:.4f}\n")
    print(f"  {'seeds searched':>15} {'seed bits':>10} {'rel.err':>9} "
          f"{'basis only':>12} {'quant. cost':>12}")

    rows = []
    for n in (1, 4, 16, 64, 256, 1024, 4096, 16384, 65535):
        ct = compress(W, cfg, n_candidate_seeds=n)
        r = relative_error(W, decompress(ct))
        basis = exact_coefficient_error(W, ct)
        print(f"  {n:>15d} {np.log2(n):>10.1f} {r:>9.4f} {basis:>12.4f} "
              f"{r / basis:>11.2f}x")
        rows.append(dict(n_seeds=int(n), rel_err=r, basis_only=basis))
    gain = rows[0]["rel_err"] / rows[-1]["rel_err"]
    print(f"\n  Searching 2^16 seeds instead of 1 reduces the error {gain:.2f}x.")
    print(f"  'basis only' is the same seed with unquantised coefficients, so")
    print(f"  the last column is what the 4-bit mantissas cost on top of it.")
    out["search_gain"] = dict(floor=floor, rows=rows, gain=gain)


# ---------------------------------------------------------------------------
def check_exponent_rule(out: dict, n_blocks: int) -> None:
    rule("4.  THE SHARED-EXPONENT RULE (Section 3.2)")
    print("  'paper' is the rule exactly as printed, e = max_i floor(log2|t_i|).")
    print("  'range' shifts it down so the 4-bit mantissas are actually used.\n")
    print(f"  {'config':<8} {'exponent rule':<15} {'rel.err':>9} {'mantissa codes used':>21}")
    rows = []
    for nm, cfg in (("4-bit", SEEDLM_4BIT), ("3-bit", SEEDLM_3BIT)):
        W = synth_weights((n_blocks * cfg.C,), sigma=0.02, seed=3)
        for mode in ("paper", "range"):
            ct = compress(W, cfg, exponent_mode=mode)
            r = relative_error(W, decompress(ct))
            used = len(np.unique(ct.q))
            print(f"  {nm:<8} {mode:<15} {r:>9.4f} {used:>13d} / 16")
            rows.append(dict(config=nm, mode=mode, rel_err=r, codes_used=int(used)))
    out["exponent_rule"] = rows


# ---------------------------------------------------------------------------
def check_exponent_range(out: dict, n_blocks: int) -> None:
    rule("5.  DIAGNOSTIC - THE 4-BIT EXPONENT HAS AN ABSOLUTE RANGE")
    print("  SeedLM has no per-tensor or per-channel scale, and e is stored in")
    print("  4 bits, so e is confined to [-8, 7].  The method is therefore only")
    print("  scale invariant while the coefficients stay inside that window.")
    print("  Section 3.4 tunes (C,P,K) on standard-normal w (sigma=1), where the")
    print("  window is comfortable; real weight tensors sit far lower.\n")
    print(f"  {'sigma':>9} {'rel.err':>9} {'e range':>12} {'% at floor':>11} "
          f"{'median cond(U) chosen':>22}")
    cfg = SEEDLM_4BIT
    cond = np.linalg.cond(Codebook.get(cfg).U.astype(np.float64))
    rows = []
    for sigma in (1.0, 1e-1, 2e-2, 1e-2, 1e-3, 1e-4, 1e-5):
        W = synth_weights((n_blocks * cfg.C,), sigma=sigma, seed=4)
        ct = compress(W, cfg)
        r = relative_error(W, decompress(ct))
        at_floor = float((ct.e == -8).mean())
        med_cond = float(np.median(cond[ct.seeds.astype(np.int64)]))
        print(f"  {sigma:>9.0e} {r:>9.4f} {f'[{ct.e.min()},{ct.e.max()}]':>12} "
              f"{100 * at_floor:>10.1f}% {med_cond:>22.1f}")
        rows.append(dict(sigma=sigma, rel_err=r, e_min=int(ct.e.min()),
                         e_max=int(ct.e.max()), frac_at_floor=at_floor,
                         median_cond_chosen=med_cond))
    best = min(r["rel_err"] for r in rows)
    worst = max(r["rel_err"] for r in rows)
    print(f"\n  Scale invariance would make every row identical; spread is "
          f"{worst / best:.1f}x.")
    print(f"  Median cond(U) over all 65535 seeds is {np.median(cond):.1f}.  Once the")
    print(f"  exponent floor forces well-conditioned bases to quantise to zero, the")
    print(f"  search is left picking near-singular ones, whose huge coefficients are")
    print(f"  the only ones that survive - the method is outside its operating range.")
    out["exponent_range"] = dict(rows=rows, median_cond_all=float(np.median(cond)))


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller problem sizes")
    args = ap.parse_args()

    n_weights = 4096 if args.quick else 16384
    n_blocks = 128 if args.quick else 512

    print("SeedLM replication - validation report")
    print("Shafipour et al., ICLR 2025, arXiv:2410.10714")
    print("Implemented from the paper; no official code was released.")

    out: dict = {}
    t0 = time.time()
    check_paper_claims(out)
    check_reconstruction(out, n_weights)
    check_search_gain(out, n_blocks)
    check_exponent_rule(out, n_blocks)
    check_exponent_range(out, n_blocks)

    rule("DONE")
    print(f"  total wall clock: {time.time() - t0:.1f}s")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "validation.json").write_text(json.dumps(out, indent=2))
    print(f"  summary written to {RESULTS / 'validation.json'}")


if __name__ == "__main__":
    main()
