"""C2: submission formats, split hygiene, and the anchor scale."""

from __future__ import annotations

import csv
import io

import numpy as np
import pytest

from lab.problems.base import InvalidSubmission
from lab.problems.c2_solubility import EMIN, REFERENCE_MAE, AqSolubility
from lab.scoring import scaled


@pytest.fixture(scope="module")
def p():
    return AqSolubility(0)


@pytest.fixture(scope="module")
def truth(p):
    return np.array([float(r["Y"]) for r in p.test])


def offset_by(truth, delta):
    """A submission whose MAE is exactly `delta`."""
    return {"predictions": (truth + delta).tolist()}


# ----------------------------------------------------------------- bad submissions


@pytest.mark.parametrize("bad", [
    {},
    {"predictions": []},
    {"predictions": None},
    {"predictions": "0.0"},
    {"preds": [0.0]},
    [0.0],
    "0.0",
])
def test_malformed_submissions_rejected(p, bad):
    with pytest.raises(InvalidSubmission):
        p.verify(bad)


def test_wrong_length_rejected(p):
    with pytest.raises(InvalidSubmission):
        p.verify({"predictions": [0.0] * (len(p.test) - 1)})
    with pytest.raises(InvalidSubmission):
        p.verify({"predictions": [0.0] * (len(p.test) + 1)})


def test_non_finite_rejected(p):
    preds = [0.0] * len(p.test)
    for bad in (float("nan"), float("inf"), None, "x"):
        preds[17] = bad
        with pytest.raises(InvalidSubmission):
            p.verify({"predictions": preds})


def test_dict_missing_a_row_rejected(p):
    d = {str(i): 0.0 for i in range(len(p.test))}
    d.pop("123")
    with pytest.raises(InvalidSubmission):
        p.verify({"predictions": d})


def test_dev_submission_must_be_dev_sized(p):
    with pytest.raises(InvalidSubmission):
        p.dev_eval({"predictions": [0.0] * len(p.test)})
    assert p.dev_eval({"predictions": [0.0] * len(p.dev)}) > 0


# --------------------------------------------------------------- accepted formats


def test_three_formats_agree(p, truth):
    preds = (truth + 0.25).tolist()
    by_list = p.verify({"predictions": preds})
    by_index = p.verify({"predictions": {str(i): v for i, v in enumerate(preds)}})
    by_smiles = p.verify({"predictions": {r["Drug"]: v for r, v in zip(p.test, preds)}})
    assert by_list == pytest.approx(0.25)
    assert by_index == pytest.approx(by_list)
    assert by_smiles == pytest.approx(by_list)


def test_perfect_predictions_score_zero_mae_and_one(p, truth):
    a = p.anchors()
    assert p.verify({"predictions": truth.tolist()}) == pytest.approx(0.0)
    assert scaled(0.0, a) == pytest.approx(1.0)


# --------------------------------------------------------------- public file hygiene


def test_public_files_hide_every_hidden_label(p):
    files = p.public_files()
    assert set(files) == {"train.csv", "dev.csv", "test.csv"}
    for name in ("dev.csv", "test.csv"):
        rows = list(csv.DictReader(io.StringIO(files[name].decode())))
        assert rows and set(rows[0]) == {"Drug_ID", "Drug"}
    train = list(csv.DictReader(io.StringIO(files["train.csv"].decode())))
    assert set(train[0]) == {"Drug_ID", "Drug", "Y"}
    assert len(train) == len(p.train)


def test_statement_leaks_no_labels(p, truth):
    text = p.statement()
    assert not any(f"{y:.4f}" in text for y in truth[:200])


def test_splits_are_disjoint_and_complete(p):
    # Drug_ID is a compound name and repeats in AqSolDB; the SMILES are unique.
    smiles = [{r["Drug"] for r in rows} for rows in (p.train, p.dev, p.test)]
    assert all(len(s) == len(rows) for s, rows in zip(smiles, (p.train, p.dev, p.test)))
    assert len(smiles[0] & smiles[1]) == len(smiles[0] & smiles[2]) == len(smiles[1] & smiles[2]) == 0
    assert len(p.train) + len(p.dev) + len(p.test) == 9982  # TDC: 9,982 drugs
    assert len(p.test) == 1997  # TDC's own 20% scaffold holdout


def test_dev_is_a_scaffold_shift(p):
    from lab.problems.c2_solubility import _scaffold

    train_scaffolds = {_scaffold(r["Drug"]) for r in p.train}
    assert not {_scaffold(r["Drug"]) for r in p.dev} & train_scaffolds


def test_split_is_stable_across_instances():
    a, b = AqSolubility(0), AqSolubility(3)
    assert [r["Drug_ID"] for r in a.test] == [r["Drug_ID"] for r in b.test]
    assert [r["Drug_ID"] for r in a.dev] == [r["Drug_ID"] for r in b.dev]


# --------------------------------------------------------------------- anchor scale


def test_anchor_endpoints(p, truth):
    a = p.anchors()
    assert a.optimum == 0.0 and a.ref_kind == "P"
    assert a.baseline > a.reference > a.emin == EMIN
    assert a.reference == REFERENCE_MAE
    assert scaled(p.verify(offset_by(truth, a.baseline)), a) == pytest.approx(0.0)
    assert scaled(p.verify(offset_by(truth, a.reference)), a) == pytest.approx(0.5)
    assert scaled(p.verify(offset_by(truth, a.emin)), a) == pytest.approx(1.0)


def test_better_is_higher(p, truth):
    a = p.anchors()
    maes = [a.baseline + 0.5, a.baseline, 1.1, a.reference, 0.6, 0.45, a.emin]
    scores = [scaled(p.verify(offset_by(truth, m)), a) for m in maes]
    assert scores == sorted(scores)
    assert scores[0] == pytest.approx(0.0) and scores[-1] == pytest.approx(1.0)


def test_predicting_the_training_mean_scores_zero(p):
    mean = np.mean([float(r["Y"]) for r in p.train])
    raw = p.verify({"predictions": [mean] * len(p.test)})
    assert raw > p.anchors().baseline
    assert scaled(raw, p.anchors()) == pytest.approx(0.0)
