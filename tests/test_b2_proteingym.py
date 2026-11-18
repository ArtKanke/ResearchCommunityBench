import io

import numpy as np
import pandas as pd
import pytest

from lab.problems.b2_proteingym import ASSAYS, ProteinFitness
from lab.problems.base import InvalidSubmission
from lab.scoring import scaled


@pytest.fixture(scope="module")
def p():
    return ProteinFitness(0)


def test_public_files_hide_labels(p):
    f = p.public_files()
    assert sum(map(len, f.values())) < 20e6
    test, dev, train = (pd.read_csv(io.BytesIO(f[k])) for k in ("test.csv", "dev.csv", "train.csv"))
    assert list(test.columns) == ["assay", "mutant"] and list(dev.columns) == ["assay", "mutant"]
    pos = lambda d: set(zip(d.assay, d.mutant.str[1:-1]))  # noqa: E731
    assert not pos(train) & pos(test) and not pos(train) & pos(dev) and not pos(dev) & pos(test)
    assert len(test) == len(p._split("test")) and set(test.assay) == set(ASSAYS)


def test_invalid(p):
    n = len(p._split("test"))
    for bad in [{}, {"predictions": [0.0] * (n - 1)}, {"predictions": [0.0] * (n - 1) + [np.nan]},
                {"predictions": "x"}, {"predictions": [[0.0]] * n}]:
        with pytest.raises(InvalidSubmission):
            p.verify(bad)
    with pytest.raises(InvalidSubmission):
        p.dev_eval({"predictions": [0.0] * n})


def test_anchors_and_ordering(p):
    a = p.anchors()
    assert a.optimum < a.reference < a.baseline < 1.2
    assert scaled(a.baseline, a) == pytest.approx(0.0)
    assert scaled(a.reference, a) == pytest.approx(0.5)
    d = p._split("test")
    y = d.DMS_score.to_numpy()
    assert p.verify({"predictions": [0.0] * len(d)}) == 1.0
    assert p.verify({"predictions": y.tolist()}) == pytest.approx(0.0, abs=1e-12)
    rng = np.random.default_rng(0)
    sd = d.groupby("assay").DMS_score.transform("std").to_numpy()
    raws = [p.verify({"predictions": (y + k * sd * rng.normal(size=len(y))).tolist()}) for k in (3, 1, 0.3)]
    assert raws[0] > raws[1] > raws[2]
    s = [scaled(r, a) for r in raws]
    assert s[0] < s[1] < s[2]


def test_dev_eval(p):
    d = p._split("dev")
    assert p.dev_eval({"predictions": d.DMS_score.tolist()}) == pytest.approx(0.0, abs=1e-12)
