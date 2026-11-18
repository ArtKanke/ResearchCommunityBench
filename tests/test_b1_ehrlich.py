import numpy as np
import pytest

from lab.problems.b1_ehrlich import ALPHABET, BUDGET, EhrlichDesign, _ga
from lab.problems.base import InvalidSubmission
from lab.scoring import scaled


def seq(x):
    return "".join(ALPHABET[i] for i in x)


@pytest.fixture(scope="module", params=[0, 1, 2])
def p(request):
    return EhrlichDesign(request.param)


def test_invalid(p):
    for bad in [{}, {"sequence": 3}, {"sequence": "A" * (p.L - 1)}, {"sequence": "B" * p.L}, []]:
        with pytest.raises(InvalidSubmission):
            p.verify(bad)
    with pytest.raises(InvalidSubmission):
        p.oracle({"sequences": ["A" * p.L] * 51})
    with pytest.raises(InvalidSubmission):
        p.oracle({"sequences": "A" * p.L})


def test_feasibility_and_oracle(p):
    e = p._e
    banned = np.argwhere(~e.feas)[0]
    x = e.witness.copy()
    x[:2] = banned
    assert p.verify({"sequence": seq(x)}) == 0.0
    out = p.oracle({"sequences": [seq(e.witness), seq(x)]})
    assert out["_cost"] == 2 and out["values"] == [1.0, 0.0]


def test_anchors(p):
    a = p.anchors()
    assert 0 < a.baseline < a.reference < a.optimum == 1.0
    assert scaled(a.baseline, a) == pytest.approx(0.0)
    assert scaled(a.reference, a) == pytest.approx(0.5)
    assert scaled(p.verify({"sequence": seq(p._e.witness)}), a) == 1.0
    # an independent GA run lands near the reference; the optimum scores higher
    ga = _ga(p._e, np.random.default_rng(777), budget=BUDGET)
    assert scaled(ga, a) < scaled(1.0, a)
    assert "motif" in p.statement() and seq(p._e.witness) not in p.statement()


def test_better_is_higher(p):
    e, rng = p._e, np.random.default_rng(0)
    X = e.random_feasible(rng, 200)
    f = e.f(X)
    lo, hi = X[np.argmin(f)], p._e.witness
    assert p.verify({"sequence": seq(hi)}) > p.verify({"sequence": seq(lo)})
