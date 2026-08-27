#!/usr/bin/env python3
"""Independent re-derivation of the headline numbers.

Deliberately imports NOTHING from the ``seedlm`` package: a second, separately
written implementation straight from the paper, structured differently (one
Python loop per block rather than a tiled block x seed grid) so that a shared
bug is unlikely to survive in both.  Only the final numbers are compared.
"""

import numpy as np

TAPS = {16: (0, 1, 3, 12)}


def cycle(K):
    """Algorithm 1, run from state 1 for the whole period."""
    taps, st, out = TAPS[K], 1, []
    for _ in range((1 << K) - 1):
        b = 0
        for t in taps:
            b ^= (st >> t) & 1
        st = (st >> 1) | (b << (K - 1))
        out.append(st)
    return np.array(out, dtype=np.int64)


def bases(K, C, P):
    """All U(s), built by explicit index arithmetic (no sliding-window view)."""
    seq = cycle(K)
    L = len(seq)
    half = 1 << (K - 1)
    u = (seq - half) / (half - 1)
    idx = (np.arange(L)[:, None] + np.arange(C * P)[None, :]) % L
    return u[idx].reshape(L, C, P)


def quant(t, mode):
    """4-bit mantissas + shared 4-bit exponent. float64 logs, unlike the library."""
    amax = np.abs(t).max(-1)
    amax = np.where(amax == 0, 1.0, amax).astype(np.float64)
    if mode == "paper":
        e = np.floor(np.log2(amax))
    else:
        e = np.ceil(np.log2(amax / 7.5))
    e = np.clip(e, -8, 7)
    s = 2.0 ** e
    q = np.clip(np.round(t / s[..., None]), -8, 7)
    return q, e, q * s[..., None]


def run(C, P, K, W, mode="range"):
    U = bases(K, C, P)
    Up = np.linalg.pinv(U)                       # SVD route, float64
    blocks = W.reshape(-1, C).astype(np.float64)
    rec = np.empty_like(blocks)
    for i, w in enumerate(blocks):
        t = np.einsum("npc,c->np", Up, w)
        _, _, th = quant(t, mode)
        r = np.einsum("ncp,np->nc", U, th)
        rec[i] = r[np.argmin(((r - w) ** 2).sum(-1))]
    return np.linalg.norm(blocks - rec) / np.linalg.norm(blocks)


rng = np.random.default_rng(2024)
print("Independent re-implementation (nothing imported from seedlm/)\n")

# (a) one fixed basis, exact coefficients -> sqrt((C-P)/C)
C, P = 8, 3
U = bases(16, C, P)
W = rng.normal(0, 0.02, (20000, C))
errs = []
for s in (0, 7, 1234, 40000, 65000):
    Us = U[s]
    rec = W @ np.linalg.pinv(Us).T @ Us.T
    errs.append(np.linalg.norm(W - rec) / np.linalg.norm(W))
print(f"(a) single fixed basis, exact coeffs : {np.mean(errs):.4f} "
      f"(spread {min(errs):.4f}-{max(errs):.4f})")
print(f"    theory sqrt((C-P)/C)             : {np.sqrt((C - P) / C):.4f}")

# (b) full 65535-seed search
n = 400
for name, (C, P, K) in (("4-bit", (8, 3, 16)), ("3-bit", (12, 4, 16))):
    W = rng.normal(0, 0.02, n * C)
    print(f"(b) {name} full search, rel.err        : {run(C, P, K, W):.4f}")

# (c) the paper's literal exponent rule
t = rng.normal(0, 0.02, (20000, 3))
for mode in ("paper", "range"):
    q, _, th = quant(t, mode)
    print(f"(c) mode={mode:6s} mantissa codes used : {len(np.unique(q)):2d}/16   "
          f"coeff rel.err {np.linalg.norm(t - th) / np.linalg.norm(t):.4f}")
