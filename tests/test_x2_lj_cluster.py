import numpy as np
import pytest

from lab.problems.base import InvalidSubmission
from lab.problems.x2_lj_cluster import (
    N, BinaryLJCluster, basin_hop, builder, draw_params, energy, energy_grad, pair_params,
    submission_of,
)
from lab.scoring import scaled


@pytest.fixture(scope="module")
def p():
    return BinaryLJCluster(0)


@pytest.fixture(scope="module")
def ref(p):
    return builder(p.seed)[1]


def test_params_seeded_and_nonstandard():
    a, b = draw_params(0), draw_params(1)
    assert a == draw_params(0) and a != b
    for q in (a, b):  # database instances use additive sigma_AB and unit eps
        assert abs(q["sigma_AB"] - (1 + q["sigma_BB"]) / 2) > 1e-3
        assert abs(q["eps_AB"] - 1) > 1e-3 or abs(q["eps_BB"] - 1) > 1e-3


def test_gradient_matches_finite_difference(ref):
    _, X, t = ref
    rng = np.random.default_rng(0)
    x = X.ravel() + rng.normal(0, 0.01, 3 * N)
    s, e = pair_params(t, draw_params(0))
    g = energy_grad(x, s, e)[1]
    for k in rng.choice(3 * N, 5, replace=False):
        d = np.zeros(3 * N)
        d[k] = 1e-6
        fd = (energy_grad(x + d, s, e)[0] - energy_grad(x - d, s, e)[0]) / 2e-6
        assert fd == pytest.approx(g[k], rel=1e-5, abs=1e-6)


def test_invalid_rejected(p, ref):
    _, X, t = ref
    good = submission_of(X, t)
    bad = [None, {}, {**good, "positions": good["positions"][:-1]},
           {**good, "types": ["A"] * N},
           {**good, "types": good["types"][:-1] + ["C"]},
           {**good, "positions": [[0.0, 0.0]] * N},
           {**good, "positions": [["1", 0, 0]] + good["positions"][1:]}]
    nan = [row[:] for row in good["positions"]]
    nan[3][1] = float("nan")
    close = [row[:] for row in good["positions"]]
    close[1] = [close[0][0] + 0.3, close[0][1], close[0][2]]
    bad += [{**good, "positions": nan}, {**good, "positions": close}]
    for s in bad:
        with pytest.raises(InvalidSubmission):
            p.verify(s)


def test_energy_exact_and_invariant(p, ref):
    E, X, t = ref
    assert p.verify(submission_of(X, t)) == pytest.approx(E, abs=1e-9)
    Q = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))[0]
    assert p.verify(submission_of(X @ Q.T + 5.0, t)) == pytest.approx(E, abs=1e-8)
    # swapping one A and one B changes the energy (types matter)
    t2 = t.copy()
    i, j = np.flatnonzero(t == 0)[0], np.flatnonzero(t == 1)[0]
    t2[i], t2[j] = 1, 0
    assert p.verify(submission_of(X, t2)) == pytest.approx(energy(X, t2, p.params))
    assert p.verify(submission_of(X, t2)) > E


def test_anchor_scores(p, ref):
    a = p.anchors()
    base = builder(p.seed)[0]
    assert a.reference < a.baseline - 5  # reference meaningfully better than baseline
    assert a.optimum == pytest.approx(1.05 * a.reference)
    assert scaled(p.verify(submission_of(base[1], base[2])), a) == pytest.approx(0.0, abs=1e-9)
    assert scaled(p.verify(submission_of(ref[1], ref[2])), a) == pytest.approx(0.5, abs=1e-9)


def test_more_search_scores_higher(p, ref):
    a = p.anchors()
    E, X, t = basin_hop(p.params, 1000, np.random.default_rng(101), start=ref)
    s = scaled(p.verify(submission_of(X, t)), a)
    assert E < a.reference - 1e-6 and 0.5 < s < 1
