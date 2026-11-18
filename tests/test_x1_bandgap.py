import io

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from lab.problems.base import InvalidSubmission
from lab.problems.x1_bandgap import DATA, BandGap, _load, featurize, parse_formula
from lab.scoring import scaled


@pytest.fixture(scope="module")
def p():
    return BandGap(0)


def test_split_is_matbench_fold0(p):
    df, _ = _load()
    assert len(df) == 4604
    test = next(iter(KFold(5, shuffle=True, random_state=18012019).split(df)))[1]
    assert (df.split == "test").sum() == 921 and (df.loc[test, "split"] == "test").all()
    assert df.loc[test[:3], "id"].tolist() == ["mb-expt-gap-0008", "mb-expt-gap-0010", "mb-expt-gap-0019"]


def test_public_files(p):
    f = p.public_files()
    assert sum(map(len, f.values())) < 20e6
    for k in ("test.csv", "dev.csv"):
        assert list(pd.read_csv(io.BytesIO(f[k])).columns) == ["id", "composition"]
    train = pd.read_csv(io.BytesIO(f["train.csv"]))
    test = pd.read_csv(io.BytesIO(f["test.csv"]))
    assert not set(train.id) & set(test.id)
    ns = {}
    exec(f["featurize.py"].decode(), ns)
    assert ns["parse_formula"]("Ag(AuS)2") == {"Ag": 1.0, "Au": 2.0, "S": 2.0}


def test_parser():
    assert parse_formula("Ag0.5Ge1Pb1.75S4") == {"Ag": 0.5, "Ge": 1.0, "Pb": 1.75, "S": 4.0}
    assert parse_formula("Ca(Fe(CN)6)2") == {"Ca": 1.0, "Fe": 2.0, "C": 12.0, "N": 12.0}
    for bad in ("Ag(S", "ag2", "Fe2O3)"):
        with pytest.raises(ValueError):
            parse_formula(bad)
    df, _ = _load()
    X = featurize(df.composition[:50], DATA / "elements.csv")
    assert X.shape[0] == 50 and np.isfinite(X.to_numpy()).all()


def test_invalid(p):
    n = len(p._split("test"))
    for bad in [{}, {"predictions": [0.0] * (n + 1)}, {"predictions": [np.inf] * n}, {"predictions": None},
                {"predictions": ["a"] * n}]:
        with pytest.raises(InvalidSubmission):
            p.verify(bad)
    with pytest.raises(InvalidSubmission):
        p.dev_eval({"predictions": [0.0] * n})


def test_anchors_and_ordering(p):
    a = p.anchors()
    assert 1.0 < a.baseline < 1.2 and a.reference == 0.2723
    assert scaled(a.baseline, a) == pytest.approx(0.0)
    assert scaled(a.reference, a) == pytest.approx(0.5)
    y = p._split("test").gap.to_numpy()
    assert p.verify({"predictions": y.tolist()}) == 0.0
    rng = np.random.default_rng(0)
    raws = [p.verify({"predictions": (y + rng.normal(0, s, len(y))).tolist()}) for s in (1.0, 0.4, 0.1)]
    assert raws[0] > raws[1] > raws[2]
    assert scaled(raws[0], a) < scaled(raws[1], a) < scaled(raws[2], a)
    d = p._split("dev")
    assert p.dev_eval({"predictions": d.gap.tolist()}) == 0.0
