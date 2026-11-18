import math

import numpy as np
import pytest

from lab.problems.base import InvalidSubmission
from lab.problems.m2_circles import N, TOL, CirclePacking26, grid_packing, objective
from lab.scoring import scaled

# Best known packing (sum 2.6359830849...), reproduced by multistart SLSQP + perturbation.
BEST = [
    [0.5966412163973, 0.7424170495016, 0.0958423257455],
    [0.8969394798584, 0.4846008026519, 0.1030605201416],
    [0.0961513340458, 0.6820800429326, 0.0961513340458],
    [0.103467233358, 0.4825955822105, 0.103467233358],
    [0.500571630861, 0.9060726627226, 0.0939273372775],
    [0.7602894720725, 0.7636735693834, 0.0691806763572],
    [0.1051825602687, 0.273952839624, 0.1051825602688],
    [0.2730942856884, 0.5960427019081, 0.1006003678187],
    [0.2406475984391, 0.762958863654, 0.0694401937113],
    [0.904267670693, 0.6832585349731, 0.095732329307],
    [0.9153604993042, 0.0846395006958, 0.0846395006958],
    [0.3140569780132, 0.9074079050486, 0.0925920949514],
    [0.8892209872093, 0.8892209872093, 0.1107790127907],
    [0.5013319244966, 0.5299634197532, 0.1370104301238],
    [0.297390396301, 0.3816658444526, 0.11514888016],
    [0.0849262624549, 0.0849262624549, 0.0849262624549],
    [0.6868841900296, 0.907608448429, 0.092391551571],
    [0.7052539409511, 0.3869235534096, 0.1120770889503],
    [0.8932098553714, 0.2747832833508, 0.1067901446286],
    [0.7053905112182, 0.1302211010652, 0.1302211010652],
    [0.5044682393256, 0.2753426167714, 0.1176296880465],
    [0.4047802670603, 0.7420494434606, 0.0960189757583],
    [0.728370148514, 0.5976347963877, 0.0998983505928],
    [0.2976904749108, 0.1332585727708, 0.1332585727708],
    [0.1111561794104, 0.8888438205896, 0.1111561794105],
    [0.5027155537959, 0.078860372916, 0.078860372916],
]


@pytest.fixture(scope="module")
def p():
    return CirclePacking26(0)


def sub(c):
    return {"circles": [list(map(float, row)) for row in c]}


@pytest.mark.parametrize("bad", [
    None, {}, {"circles": []}, {"circles": grid_packing()[:25]},
    {"circles": grid_packing() + [[0.5, 0.5, 0.01]]},
    {"circles": [[0.5, 0.5]] * N}, {"circles": [["0.5", 0.5, 0.1]] * N},
    {"circles": [[0.5, 0.5, True]] * N},
])
def test_malformed_rejected(p, bad):
    with pytest.raises(InvalidSubmission):
        p.verify(bad)


def test_infeasible_rejected(p):
    def mutate(i, k, v):
        c = [row[:] for row in BEST]
        c[i][k] = v
        return sub(c)
    for s in (mutate(0, 2, 0.0), mutate(0, 2, -0.1), mutate(0, 0, float("nan")),
              mutate(0, 2, BEST[0][2] + 1e-6), mutate(15, 0, BEST[15][0] - 1e-6)):
        with pytest.raises(InvalidSubmission):
            p.verify(s)
    c = grid_packing()
    c[1][:2] = c[0][:2]  # coincident centres
    with pytest.raises(InvalidSubmission):
        p.verify(sub(c))


def test_tolerance_cannot_inflate(p):
    c = np.array(BEST)
    inflated = c.copy()
    inflated[:, 2] += 0.4 * TOL  # pairs violate by 0.8 TOL: accepted
    assert p.verify(sub(inflated)) <= p.verify(sub(c)) + 1e-12


def test_baseline_scores_zero(p):
    a = p.anchors()
    assert p.verify(sub(grid_packing())) == pytest.approx(13 / 6)
    assert scaled(p.verify(sub(grid_packing())), a) == pytest.approx(0.0, abs=1e-9)


def test_best_known_scores_half(p):
    a = p.anchors()
    m = p.verify(sub(BEST))
    assert m == pytest.approx(2.6359830849, abs=1e-9)
    assert m <= a.reference  # reproducing the record is not a new record
    assert scaled(m, a) == pytest.approx(0.5, abs=1e-6)


def test_scale_monotone_and_bound(p):
    a = p.anchors()
    assert a.optimum == pytest.approx(math.sqrt(N / math.pi))
    rnd = 2.62  # typical single SLSQP multistart result
    s = [scaled(x, a) for x in (rnd, a.reference, 2.6365, 2.6375)]
    assert 0 < s[0] < 0.5 < s[2] < s[3] == 1.0  # beating the record by RECORD_MARGIN earns 1.0
    assert objective(np.array(BEST)) < a.optimum
