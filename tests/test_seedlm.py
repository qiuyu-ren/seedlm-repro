"""Validation of the SeedLM replication against the published paper.

Every test names the part of the paper it checks.  Run with::

    python -m pytest tests/ -v
    python tests/test_seedlm.py        # no pytest required
"""

from __future__ import annotations

import numpy as np
import pytest

from seedlm import (
    LFSR_TAPS,
    SEEDLM_3BIT,
    SEEDLM_4BIT,
    Codebook,
    CompressedTensor,
    SeedLMConfig,
    build_U_all,
    build_V_cached,
    build_V_direct,
    compress,
    decompress,
    dequantize,
    full_cycle,
    is_maximal_length,
    lfsr_sequence,
    lfsr_state_for_offset,
    normalize_states,
    quantize,
)
from seedlm.compress import _quantize_fast
from seedlm.pack import pack, packed_nbytes, unpack
from seedlm.quant import QuantSpec


# ===========================================================================
# Section 3.1 / Appendix A.1 / A.3 -- the LFSR itself
# ===========================================================================
def test_figure4_worked_example():
    """Appendix A.2, Figure 4: V(4) for K=3, C=4, P=2 is printed in the paper."""
    expected = np.array([[2, 5], [6, 7], [3, 1], [4, 2]])
    assert np.array_equal(build_V_direct(K=3, seed=4, C=4, P=2), expected)


def test_algorithm1_first_states_by_hand():
    """Algorithm 1 step-by-step for K=3, taps (0,1), seed 4.

    4 = 0b100 -> feedback 0^0 = 0, shift right -> 0b010 = 2
    2 = 0b010 -> feedback 0^1 = 1, shift right -> 0b101 = 5
    5 = 0b101 -> feedback 1^0 = 1, shift right -> 0b110 = 6
    """
    assert lfsr_sequence(3, 4, 6).tolist() == [2, 5, 6, 7, 3, 1]


def test_sequence_starts_after_the_seed():
    """results[0] is the successor of the seed, not the seed itself."""
    seq = lfsr_sequence(16, seed=12345, length=4)
    assert seq[0] != 12345


@pytest.mark.parametrize("K", [k for k in sorted(LFSR_TAPS) if k <= 18])
def test_table6_taps_are_maximal_length(K):
    """Appendix A.1 claims every tabulated tap set gives a maximal-length LFSR."""
    assert is_maximal_length(K), f"K={K} taps {LFSR_TAPS[K]} are not maximal length"


def test_cycle_returns_to_start():
    """A maximal-length cycle wraps back to state 1."""
    for K in (3, 8, 12, 16):
        assert full_cycle(K)[-1] == 1


def test_all_zero_state_never_occurs():
    """Section 3.1: the all-zero state is absorbing and must be excluded."""
    for K in (3, 8, 16):
        assert (full_cycle(K) != 0).all()


# ===========================================================================
# Section 3.2 -- normalisation of U(s)
# ===========================================================================
@pytest.mark.parametrize("K", [3, 8, 12, 16])
def test_normalisation_formula(K):
    """U(s) = (V(s) - 2**(K-1)) / (2**(K-1) - 1)."""
    cyc = full_cycle(K)
    half = 1 << (K - 1)
    expected = (cyc.astype(np.float64) - half) / (half - 1)
    assert np.allclose(normalize_states(cyc, K, dtype=np.float64), expected)


@pytest.mark.parametrize("K", [3, 8, 12, 16])
def test_normalisation_range_is_pm_one(K):
    """Section 3.2: normalisation puts the entries of U inside [-1, 1]."""
    u = normalize_states(full_cycle(K), K, dtype=np.float64)
    assert u.min() >= -1.0 and u.max() <= 1.0
    # The full nonzero state range is used, so the bounds are attained.
    assert np.isclose(u.min(), -1.0) and np.isclose(u.max(), 1.0)
    assert abs(u.mean()) < 1e-9


# ===========================================================================
# Figure 4 vs Algorithm 2: the two seed conventions
# ===========================================================================
def test_direct_and_cached_yield_the_same_candidate_set():
    """The paper specifies two different seed->matrix maps.

    Figure 4 runs the LFSR from the seed; Algorithm 2 step 5 indexes a cached
    cycle at ``s % length``.  They label rotations differently but expose the
    same set of candidate matrices, so the achievable reconstruction error --
    and therefore every accuracy number -- is unaffected.
    """
    K, C, P = 8, 4, 2
    cyc = full_cycle(K)
    n = len(cyc)
    direct = {build_V_direct(K, s, C, P).tobytes() for s in range(1, n + 1)}
    cached = {build_V_cached(cyc, o, C, P).tobytes() for o in range(n)}
    assert len(direct) == len(cached) == n
    assert direct == cached


def test_offset_to_hardware_state_is_exact():
    """A cached-mode offset must be translatable to a real register state."""
    K, C, P = 8, 4, 2
    cyc = full_cycle(K)
    for off in range(len(cyc)):
        state = lfsr_state_for_offset(cyc, off)
        assert np.array_equal(
            build_V_cached(cyc, off, C, P), build_V_direct(K, state, C, P)
        )


def test_window_wraps_around_end_of_cycle():
    """Algorithm 2 step 6: 'if the slice exceeds length, cycle through states'."""
    K, C, P = 3, 4, 2  # C*P = 8 > 2**3 - 1 = 7, so wrapping is forced
    cyc = full_cycle(K)
    V = build_V_cached(cyc, offset=0, C=C, P=P)
    assert V.ravel()[7] == V.ravel()[0]


# ===========================================================================
# Section 3.2 -- coefficient quantisation
# ===========================================================================
def test_quantised_values_fit_their_bit_widths():
    spec = QuantSpec(4, 4)
    rng = np.random.default_rng(0)
    for scale in (1e-6, 1e-3, 1.0, 1e3):
        t = rng.normal(0, scale, (500, 3)).astype(np.float32)
        q, e = quantize(t, spec, mode="range")
        assert q.min() >= -8 and q.max() <= 7
        assert e.min() >= -8 and e.max() <= 7


def test_dequantisation_is_exactly_q_times_two_to_the_e():
    spec = QuantSpec(4, 4)
    rng = np.random.default_rng(1)
    t = rng.normal(0, 0.02, (200, 3)).astype(np.float32)
    q, e = quantize(t, spec, mode="range")
    assert np.array_equal(
        dequantize(q, e, dtype=np.float64),
        q.astype(np.float64) * 2.0 ** e.astype(np.float64)[:, None],
    )


@pytest.mark.parametrize("mode", ["paper", "range"])
def test_search_quantiser_matches_reference_quantiser(mode):
    """The optimised in-loop quantiser must be bit-identical to the readable one."""
    spec = QuantSpec(4, 4)
    rng = np.random.default_rng(2)
    for scale in (1e-6, 1e-3, 1e-2, 1.0, 10.0, 1e3):
        t = rng.normal(0, scale, (2000, 3)).astype(np.float32)
        q_ref, e_ref = quantize(t, spec, mode=mode)
        q_fast, e_fast, t_hat = _quantize_fast(t, spec, mode)
        assert np.array_equal(q_ref.astype(np.int32), q_fast.astype(np.int32))
        assert np.array_equal(e_ref.astype(np.int32), e_fast.astype(np.int32))
        assert np.allclose(t_hat, dequantize(q_ref, e_ref))


@pytest.mark.parametrize("mode", ["paper", "range"])
def test_exponent_is_exact_at_power_of_two_boundaries(mode):
    """Gaussian samples never probe the case where the exponent rule can break.

    The shared exponent is a floor/ceil of a log, so the only inputs that can
    go wrong are the ones within an ULP of a power of two -- which random data
    hits with probability ~1e-7 and a test therefore never sees.  Constructing
    them deliberately: computing the exponent with a float32 ``log2`` gets
    these wrong (returning -7 where -8 is correct, which doubles the scale of
    every mantissa in the vector); ``frexp`` does not.
    """
    spec = QuantSpec(4, 4)
    edges = []
    for k in range(-12, 8):
        p = np.float32(2.0) ** np.float32(k)
        edges += [p, np.nextafter(p, np.float32(0)), np.nextafter(p, np.float32(1e30))]
        for m in (spec.q_max + 0.5, spec.q_max, 1.0):
            v = np.float32(p * m)
            edges += [v, np.nextafter(v, np.float32(0)),
                      np.nextafter(v, np.float32(1e30))]
    t = np.array([[v, v / 2, -v / 4] for v in edges], dtype=np.float32)

    q_ref, e_ref = quantize(t, spec, mode=mode)
    q_fast, e_fast, _ = _quantize_fast(t, spec, mode)
    assert np.array_equal(e_ref.astype(np.int32), e_fast.astype(np.int32))
    assert np.array_equal(q_ref.astype(np.int32), q_fast.astype(np.int32))

    # ...and both must agree with an exact float64 evaluation of the rule.
    amax = np.abs(t).max(axis=-1).astype(np.float64)
    if mode == "paper":
        exact = np.floor(np.log2(amax))
    else:
        exact = np.ceil(np.log2(amax / (spec.q_max + 0.5)))
    exact = np.clip(exact, spec.e_min, spec.e_max)
    assert np.array_equal(e_fast.astype(np.int64), exact.astype(np.int64))


def test_paper_exponent_rule_wastes_mantissa_bits():
    """Section 3.2's literal rule e = max_i floor(log2|t_i|) cannot be what ran.

    It forces max|t| / 2**e into [1, 2), so rounding can only return
    {-2, -1, 0, 1, 2}: 5 of the 16 codes are reachable and the coefficient
    error is several times larger than necessary.  Recorded here because it is
    the one place the replication has to deviate from the text to reproduce the
    reported accuracy.
    """
    spec = QuantSpec(4, 4)
    rng = np.random.default_rng(3)
    t = rng.normal(0, 0.02, (20000, 3)).astype(np.float32)

    q_paper, e_paper = quantize(t, spec, mode="paper")
    q_range, e_range = quantize(t, spec, mode="range")

    assert set(np.unique(q_paper).tolist()) <= {-2, -1, 0, 1, 2}
    assert set(np.unique(q_range).tolist()) == set(range(-7, 8))

    err_paper = np.linalg.norm(t - dequantize(q_paper, e_paper))
    err_range = np.linalg.norm(t - dequantize(q_range, e_range))
    assert err_paper > 3 * err_range


def test_zero_vector_quantises_to_zero():
    spec = QuantSpec(4, 4)
    t = np.zeros((5, 3), dtype=np.float32)
    q, e = quantize(t, spec, mode="range")
    assert (q == 0).all()
    assert np.array_equal(dequantize(q, e), t)


# ===========================================================================
# Section 3.3 -- the pseudo-inverse cache
# ===========================================================================
def test_pseudo_inverse_satisfies_moore_penrose_conditions():
    cfg = SeedLMConfig(C=8, P=3, K=10)
    cb = Codebook(cfg, dtype=np.float64)
    idx = np.random.default_rng(0).integers(0, cfg.n_seeds, 200)
    U, Up = cb.U[idx], cb.Upinv[idx]
    assert np.allclose(U @ Up @ U, U, atol=1e-8)
    assert np.allclose(Up @ U @ Up, Up, atol=1e-8)
    assert np.allclose((U @ Up).transpose(0, 2, 1), U @ Up, atol=1e-8)
    assert np.allclose((Up @ U).transpose(0, 2, 1), Up @ U, atol=1e-8)


def test_pseudo_inverse_survives_ill_conditioned_bases():
    """Overlapping windows make a small fraction of the K=16 bases ill conditioned.

    Consecutive seeds are sliding windows over one LFSR sequence, so some of
    them are nearly collinear: the condition number of U(s) is ~2.6 at the
    median but reaches 5e5 in the tail.  Forming ``(U^T U)^-1 U^T`` squares
    that, which in float32 destroys those seeds' coefficients.  The SVD route
    used by :class:`Codebook` must stay accurate.
    """
    cb = Codebook.get(SEEDLM_4BIT)
    U64 = cb.U.astype(np.float64)
    cond = np.linalg.cond(U64)
    assert cond.max() > 1e4, "expected an ill-conditioned tail"

    worst = np.argsort(cond)[-50:]
    ref = np.linalg.pinv(U64[worst])  # accurate pinv of the *stored* bases

    svd_rel = np.linalg.norm(ref - cb.Upinv[worst]) / np.linalg.norm(ref)

    Ut = U64[worst].transpose(0, 2, 1).astype(np.float32)
    Uw = cb.U[worst]
    normal_eq = np.linalg.inv(Ut @ Uw) @ Ut
    neq_rel = np.linalg.norm(ref - normal_eq) / np.linalg.norm(ref)

    assert svd_rel < 1e-6, f"SVD pinv lost accuracy: {svd_rel:.2e}"
    assert neq_rel > 1e-3, "expected float32 normal equations to be inaccurate"
    assert neq_rel > 100 * svd_rel


# ===========================================================================
# Eq. 3 -- the bit budget
# ===========================================================================
def test_published_configs_hit_their_bit_budgets():
    """Table 1: (C,P,K) = (8,3,16) -> 4 bits, (12,4,16) -> 3 bits."""
    assert SEEDLM_4BIT.bits_per_element == 4.0
    assert SEEDLM_3BIT.bits_per_element == 3.0
    assert SEEDLM_4BIT.bits_per_block == 32
    assert SEEDLM_3BIT.bits_per_block == 36


def test_eq3_matches_actual_payload():
    """The stored arrays must really occupy (K + 4 + 4P) bits per block."""
    rng = np.random.default_rng(0)
    for cfg in (SEEDLM_4BIT, SEEDLM_3BIT):
        W = rng.normal(0, 0.02, (16, cfg.C * 4)).astype(np.float32)
        ct = compress(W, cfg, n_candidate_seeds=64)
        assert ct.bits_per_element() == cfg.bits_per_element
        # seed field must be wide enough, mantissas and exponent must fit
        assert ct.seeds.max() < 2 ** cfg.K
        assert ct.q.min() >= -8 and ct.q.max() <= 7
        assert ct.e.min() >= -8 and ct.e.max() <= 7


# ===========================================================================
# Algorithm 3 -- the search
# ===========================================================================
def _reference_search(w, cfg, cb, mode="range"):
    """Algorithm 3 transcribed literally, one block at a time."""
    best_j, best_q, best_e, best_norm = -1, None, None, np.inf
    for j in range(cfg.n_seeds):
        t = cb.Upinv[j] @ w
        q, e = quantize(t, cfg.spec, mode=mode)
        r = w - cb.U[j] @ dequantize(q, e, dtype=w.dtype)
        norm = float(r @ r)
        if norm < best_norm:
            best_j, best_q, best_e, best_norm = j, q, e, norm
    return best_j, best_q, best_e, best_norm


def test_vectorised_search_matches_literal_algorithm3():
    """The chunked implementation must achieve what the pseudocode achieves.

    The invariant is the *reconstruction*, not the seed index.  Adjacent seeds
    are overlapping windows of one LFSR sequence, so the candidate set contains
    exact ties (see ``test_adjacent_seeds_are_structurally_degenerate``); which
    member of a tied group wins is decided by summation order.  Asserting seed
    equality would be asserting a round-off outcome.
    """
    cfg = SeedLMConfig(C=8, P=3, K=9)  # 511 seeds -- small enough to brute force
    cb = Codebook.get(cfg)
    rng = np.random.default_rng(7)
    W = rng.normal(0, 0.02, (12, cfg.C)).astype(np.float32)

    ct = compress(W, cfg, block_chunk=5, seed_chunk=37)  # deliberately ragged tiles
    t_hat = dequantize(ct.q, ct.e)
    for i, w in enumerate(W):
        j, q, e, norm = _reference_search(w, cfg, cb)
        mine = cb.U[int(ct.seeds[i])] @ t_hat[i]
        assert np.allclose(np.sum((w - mine) ** 2), norm, rtol=1e-5, atol=1e-12)
        theirs = cb.U[j] @ dequantize(q, e, dtype=np.float32)
        assert np.allclose(mine, theirs, atol=1e-6)


def test_adjacent_seeds_are_structurally_degenerate():
    """The 2**K candidate bases are far from 2**K independent bases.

    U(s) is a sliding window over one LFSR sequence, so column p of U(s) equals
    column p-1 of U(s+1).  A coefficient vector whose last entry is zero under
    seed s therefore produces exactly the same reconstruction as a shifted
    coefficient vector under seed s+1.  This caps how much genuine diversity
    the seed search has to work with, and it is why the arg-min over seeds is
    not unique.
    """
    cfg = SeedLMConfig(C=8, P=3, K=10)
    cb = Codebook.get(cfg)
    s = 123
    # Columns overlap by construction.
    assert np.array_equal(cb.U[s][:, 1:], cb.U[s + 1][:, :-1])

    # ...so these two (seed, coefficient) pairs are the same weight block.
    a = cb.U[s] @ np.array([0.0, 2.0, -3.0], dtype=np.float32)
    b = cb.U[s + 1] @ np.array([2.0, -3.0, 0.0], dtype=np.float32)
    assert np.allclose(a, b, atol=1e-7)


def test_result_is_independent_of_tiling():
    """Chunk sizes are a performance knob and must not change the answer."""
    cfg = SeedLMConfig(C=8, P=3, K=10)
    rng = np.random.default_rng(8)
    W = rng.normal(0, 0.02, (40, cfg.C)).astype(np.float32)
    a = compress(W, cfg, block_chunk=7, seed_chunk=13)
    b = compress(W, cfg, block_chunk=64, seed_chunk=4096)
    assert np.array_equal(a.seeds, b.seeds)
    assert np.array_equal(a.q, b.q)
    assert np.array_equal(a.e, b.e)


def test_reported_error_survives_the_round_trip():
    """Decompressing must reproduce exactly the block the search scored."""
    cfg = SEEDLM_4BIT
    cb = Codebook.get(cfg)
    rng = np.random.default_rng(9)
    W = rng.normal(0, 0.02, (32, cfg.C)).astype(np.float32)
    ct = compress(W, cfg, n_candidate_seeds=2048)
    What = decompress(ct)

    t = dequantize(ct.q, ct.e)
    for i in range(len(W)):
        manual = cb.U[int(ct.seeds[i])] @ t[i]
        assert np.allclose(manual, What[i], atol=1e-6)


def test_searching_more_seeds_never_hurts():
    """Monotonicity: the arg-min is over a growing candidate set."""
    cfg = SEEDLM_4BIT
    rng = np.random.default_rng(10)
    W = rng.normal(0, 0.02, (64, cfg.C)).astype(np.float32)
    errs = []
    for n in (1, 16, 256, 4096):
        ct = compress(W, cfg, n_candidate_seeds=n)
        errs.append(np.linalg.norm(W - decompress(ct)))
    assert all(errs[i] >= errs[i + 1] - 1e-9 for i in range(len(errs) - 1))
    assert errs[0] > errs[-1], "the seed search should actually be buying something"


# ===========================================================================
# Plumbing
# ===========================================================================
def test_padding_for_sizes_that_are_not_a_multiple_of_C():
    cfg = SEEDLM_4BIT
    rng = np.random.default_rng(11)
    W = rng.normal(0, 0.02, (7, 5)).astype(np.float32)  # 35 elements, C = 8
    ct = compress(W, cfg, n_candidate_seeds=256)
    assert ct.pad == 5 and ct.n_blocks == 5
    assert decompress(ct).shape == W.shape


def test_shape_and_dtype_are_preserved():
    cfg = SEEDLM_4BIT
    rng = np.random.default_rng(12)
    for shape in [(16, 16), (4, 8, 8), (64,)]:
        W = rng.normal(0, 0.02, shape).astype(np.float32)
        What = decompress(compress(W, cfg, n_candidate_seeds=128))
        assert What.shape == W.shape and What.dtype == np.float32


def test_packed_payload_is_exactly_eq3_bits():
    """Serialise for real and check the byte count against (K + 4 + 4P)/C."""
    rng = np.random.default_rng(14)
    for cfg in (SEEDLM_4BIT, SEEDLM_3BIT):
        W = rng.normal(0, 0.02, (cfg.C * 300,)).astype(np.float32)
        ct = compress(W, cfg, n_candidate_seeds=512)
        buf = pack(ct)
        assert len(buf) == packed_nbytes(ct)
        assert len(buf) * 8 >= ct.n_blocks * cfg.bits_per_block
        assert len(buf) * 8 - ct.n_blocks * cfg.bits_per_block < 8  # only tail pad
        assert 8 * len(buf) / W.size == pytest.approx(
            cfg.bits_per_element, abs=8 / W.size
        )


def test_pack_unpack_round_trip_is_lossless():
    rng = np.random.default_rng(15)
    for cfg in (SEEDLM_4BIT, SEEDLM_3BIT):
        W = rng.normal(0, 0.02, (37, 11)).astype(np.float32)
        ct = compress(W, cfg, n_candidate_seeds=512)
        back = unpack(pack(ct), ct.shape, cfg, pad=ct.pad)
        assert np.array_equal(back.seeds, ct.seeds)
        assert np.array_equal(back.q, ct.q)
        assert np.array_equal(back.e, ct.e)
        assert np.array_equal(decompress(back), decompress(ct))


def test_pack_survives_extreme_field_values():
    """Two's complement fields must round-trip at their limits, not just typically."""
    cfg = SEEDLM_4BIT
    n = 6
    ct = CompressedTensor(
        seeds=np.array([0, 1, 2**16 - 2, 2**16 - 2, 7, 9], dtype=np.uint32),
        q=np.array([[-8, 7, 0], [7, -8, -1], [0, 0, 0],
                    [-8, -8, -8], [7, 7, 7], [1, -1, 2]], dtype=np.int8),
        e=np.array([-8, 7, 0, -1, 3, -8], dtype=np.int8),
        shape=(n * cfg.C,),
        cfg=cfg,
    )
    back = unpack(pack(ct), ct.shape, cfg, pad=0)
    assert np.array_equal(back.seeds, ct.seeds)
    assert np.array_equal(back.q, ct.q)
    assert np.array_equal(back.e, ct.e)


def test_hardware_seed_translation():
    cfg = SeedLMConfig(C=8, P=3, K=9)
    cb = Codebook.get(cfg)
    rng = np.random.default_rng(13)
    W = rng.normal(0, 0.02, (8, cfg.C)).astype(np.float32)
    ct = compress(W, cfg)
    hw = ct.hardware_seeds(cb)
    assert (hw >= 1).all() and (hw <= cfg.n_seeds).all()
    for off, state in zip(ct.seeds, hw):
        assert np.array_equal(
            build_V_cached(cb.cycle, int(off), cfg.C, cfg.P),
            build_V_direct(cfg.K, int(state), cfg.C, cfg.P),
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
