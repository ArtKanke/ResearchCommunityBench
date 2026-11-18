"""C1: oracle validation, the AUC normalisation, and the anchor scale."""

from __future__ import annotations

import random

import pytest

from lab.problems.base import InvalidSubmission
from lab.problems.c1_rediscovery import (
    BUDGET,
    MAX_PER_QUERY,
    TOP_K,
    SealedTargetRediscovery,
    _auc_top_k,
    _ga,
    _pool,
)
from lab.scoring import scaled


@pytest.fixture(scope="module")
def p():
    return SealedTargetRediscovery(0)


def ask(p, smiles):
    return p.oracle({"smiles": smiles})


def scaled_one_threshold(p):
    """The raw AUC at which the log scale saturates at 1.0."""
    return p.anchors().optimum - p.anchors().emin


# --------------------------------------------------------------- oracle validation


@pytest.mark.parametrize("query", [{}, {"smiles": []}, {"smiles": 7}, {"smiles": {"a": 1}},
                                   {"smiles": ["CCO"] * (MAX_PER_QUERY + 1)}])
def test_bad_queries_rejected(p, query):
    with pytest.raises(InvalidSubmission):
        p.oracle(query)


def test_single_smiles_string_accepted(p):
    assert ask(p, "CCO")["similarity"][0] is not None


def test_invalid_smiles_returns_null_but_costs_budget(p):
    r = ask(p, ["CCO", "not-a-molecule", "C(C(C("])
    assert r["similarity"][0] is not None
    assert r["similarity"][1] is None and r["similarity"][2] is None
    assert r["_cost"] == 3  # every molecule is charged, valid or not


def test_similarity_is_a_probability(p):
    sims = ask(p, list(_pool()[:20]))["similarity"]
    assert all(0.0 <= s <= 1.0 for s in sims)


def test_non_dict_submission_rejected(p):
    for bad in ([], "x", 3, None):
        with pytest.raises(InvalidSubmission):
            p.verify(bad)


# ------------------------------------------------------------------- AUC behaviour


def test_perfect_from_first_query_scores_one(p):
    """The attainable ceiling is 1 - (TOP_K - 1)/(2 * BUDGET): filling the top ten costs
    ten queries. `optimum = 1.0` sits just above it, far above emin."""
    perfect = _auc_top_k([(f"perfect{i}", 1.0) for i in range(TOP_K)])
    assert perfect == pytest.approx(1.0 - (TOP_K - 1) / (2 * BUDGET))
    assert perfect > scaled_one_threshold(p)
    assert _auc_top_k(p._score([p._target_smiles])) == pytest.approx(1.0 / TOP_K)


def test_target_found_late_scores_less_than_found_early(p):
    pool = list(_pool()[:200])
    early = p._score([p._target_smiles] + pool)
    late = p._score(pool + [p._target_smiles])
    assert _auc_top_k(early) > _auc_top_k(late) > 0.0


def test_duplicates_cannot_fill_the_top_ten(p):
    """Ten copies of the target are worth one molecule, not ten."""
    dup = _auc_top_k(p._score([p._target_smiles] * TOP_K))
    distinct = _auc_top_k([(f"distinct{i}", 1.0) for i in range(TOP_K)])
    assert dup == pytest.approx(1.0 / TOP_K)
    assert distinct == pytest.approx(1.0 - (TOP_K - 1) / (2 * BUDGET))


def test_wasted_queries_lower_the_score(p):
    good = p._score([p._target_smiles])
    wasted = p._score(["junk"] * 100 + [p._target_smiles])
    assert _auc_top_k(wasted) < _auc_top_k(good)


def test_stopping_early_is_not_punished(p):
    """A short log is padded with its final top-10 value (mol_opt `finish=True`)."""
    scored = p._score(list(_pool()[:50]))
    assert _auc_top_k(scored) == pytest.approx(_auc_top_k(scored + scored[-1:] * 500))


def test_more_queries_never_hurt(p):
    """g is non-decreasing, so appending anything cannot lower the score."""
    scored = p._score(list(_pool()[:40]))
    for extra in ([("junk", None)] * 20, p._score(list(_pool()[100:140]))):
        assert _auc_top_k(scored + extra) >= _auc_top_k(scored) - 1e-12


def test_queries_past_the_budget_are_ignored(p):
    scored = p._score(list(_pool()[:5]))
    late = [("perfect", 1.0)]
    assert _auc_top_k(scored + late, budget=5) == _auc_top_k(scored, budget=5)
    assert _auc_top_k(scored + late, budget=6) > _auc_top_k(scored, budget=6)


def test_score_from_oracle_log_reads_server_log_shapes(p):
    q = SealedTargetRediscovery(0)
    smiles = list(_pool()[:30])
    flat = q.score_from_oracle_log([{"smiles": smiles[:10]}, {"smiles": smiles[10:]}])
    nested = q.score_from_oracle_log([{"query": {"smiles": smiles}, "response": {"similarity": []}}])
    assert flat == pytest.approx(nested)


def test_verify_matches_the_oracle_log(p):
    q = SealedTargetRediscovery(0)
    smiles = list(_pool()[:40])
    ask(q, smiles)
    assert q.verify({}) == pytest.approx(q.score_from_oracle_log([{"smiles": smiles}]))


# ------------------------------------------------------------- statement and files


def test_nothing_public_leaks_the_target(p):
    from rdkit import Chem

    target = p._target_smiles
    blobs = [p.statement()] + [v.decode() for v in p.public_files().values()]
    for text in blobs:
        assert target not in text
    pool = set(_pool())
    assert target not in pool
    # and no public molecule is the target under a different SMILES spelling
    assert Chem.MolToSmiles(Chem.MolFromSmiles(target)) == target


def test_public_pool_is_the_advertised_size(p):
    pool = p.public_files()["pool.smi"].decode().split()
    assert len(pool) == len(_pool()) == len(set(pool))


def test_seeds_seal_different_targets():
    targets = {SealedTargetRediscovery(s)._target_smiles for s in range(6)}
    assert len(targets) == 6


# --------------------------------------------------------------------- anchor scale


def test_baseline_scores_zero(p):
    """Random sampling from the public pool, end to end through the oracle."""
    q = SealedTargetRediscovery(0)
    draw = random.Random(4242).sample(_pool(), BUDGET)
    for i in range(0, BUDGET, MAX_PER_QUERY):
        ask(q, draw[i : i + MAX_PER_QUERY])
    raw = q.verify({})
    assert scaled(raw, q.anchors()) < 0.15


def test_anchor_endpoints(p):
    a = p.anchors()
    assert a.baseline < a.reference < a.optimum == 1.0
    assert scaled(a.baseline, a) == pytest.approx(0.0)
    assert scaled(a.reference, a) == pytest.approx(0.5)
    assert scaled(1.0 - a.emin, a) == pytest.approx(1.0)


def test_better_is_higher(p):
    a = p.anchors()
    scores = [scaled(raw, a) for raw in (a.baseline, a.reference, 0.5, 0.6, 1.0)]
    assert scores == sorted(scores)
    assert scores[-1] == pytest.approx(1.0)


def test_reference_ga_lands_near_half(p):
    """A fresh GA run at the same budget should score around 0.5 (GA seed spread is wide)."""
    raw = _auc_top_k(_ga(p._target_fp, BUDGET, 777))
    assert 0.2 < scaled(raw, p.anchors()) < 0.8
