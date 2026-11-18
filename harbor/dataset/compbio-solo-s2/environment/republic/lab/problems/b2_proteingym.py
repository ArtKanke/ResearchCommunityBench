"""B2. Supervised protein fitness prediction at unseen positions (ProteinGym v1.3, MIT).

Data: ProteinGym v1.3 supervised CV files, https://marks.hms.harvard.edu/proteingym/
ProteinGym_v1.3/cv_folds_singles_substitutions.zip and reference file DMS_substitutions.csv
(MIT License, (c) 2023 OATML-Markslab; Notin et al., NeurIPS 2023). Two single-mutant DMS
assays, chosen small (CPU-feasible), singles-only, with no sibling assay of the same protein
in ProteinGym and two different phenotypes:
  OTC_HUMAN_Lo_2023          enzyme activity, 1,570 variants, L = 354
  ESTA_BACSU_Nutschel_2020   thermostability (T50), 2,172 variants, L = 212

Split: ProteinGym's `fold_modulo_5` (mutated-position rank mod 5, position-disjoint).
Hidden test = fold 0. Dev (labels withheld, server dev_eval) = fold-1 positions with rank
mod 10 == 1 (about half of fold 1). Public train = everything else (~70%).

Metric: 1 - Spearman rho per assay, averaged (constant predictions count as rho = 0).
Anchors: baseline = ridge on one-hot features (position x residue, plus wild-type and mutant
residue identity so it can transfer to unseen positions), fit on the public train set;
reference = Kermut (Groth et al. 2024), the best published supervised model on the
ProteinGym fold_modulo_5 leaderboard for these assays (OTC 0.806, ESTA 0.601; 5-fold CV
means trained on 80%, here agents train on ~70%); emin = 1 - 0.95.
"""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .base import Anchors, InvalidSubmission, Problem, ProblemInfo

DATA = Path(__file__).resolve().parents[2] / "data" / "B2"
URL = "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/"
ASSAYS = ["OTC_HUMAN_Lo_2023", "ESTA_BACSU_Nutschel_2020"]
KERMUT = {"OTC_HUMAN_Lo_2023": 0.806, "ESTA_BACSU_Nutschel_2020": 0.601}
AA = "ACDEFGHIKLMNPQRSTVWY"
META_COLS = ["DMS_id", "UniProt_ID", "source_organism", "seq_len", "selection_assay",
             "coarse_selection_type", "title", "year", "jo", "target_seq"]


def _get(url):
    with urllib.request.urlopen(url, timeout=300) as r:
        return r.read()


def fetch_data():
    """Download ProteinGym files once and write data/B2/{variants,assays}.csv."""
    out = DATA / "variants.csv"
    if out.exists():
        return
    DATA.mkdir(parents=True, exist_ok=True)
    raw = DATA / "raw"
    raw.mkdir(exist_ok=True)
    zpath, rpath = raw / "cv_folds_singles_substitutions.zip", raw / "DMS_substitutions.csv"
    if not zpath.exists():
        zpath.write_bytes(_get(URL + zpath.name))
    if not rpath.exists():
        rpath.write_bytes(_get(URL + rpath.name))
    ref = pd.read_csv(rpath).set_index("DMS_id")
    rows = []
    with zipfile.ZipFile(zpath) as z:
        for a in ASSAYS:
            d = pd.read_csv(z.open(f"cv_folds_singles_substitutions/{a}.csv"))
            assert not d.mutant.str.contains(":").any()
            pos = d.mutant.str[1:-1].astype(int)
            rank = pos.map({p: k for k, p in enumerate(sorted(pos.unique()))})
            assert (rank % 5 == d.fold_modulo_5).all()  # verified ProteinGym fold definition
            split = np.where(d.fold_modulo_5 == 0, "test", np.where(rank % 10 == 1, "dev", "train"))
            rows.append(pd.DataFrame({"assay": a, "mutant": d.mutant, "DMS_score": d.DMS_score,
                                      "split": split}))
    pd.concat(rows).to_csv(out, index=False)
    ref.loc[ASSAYS].reset_index()[META_COLS].to_csv(DATA / "assays.csv", index=False)


@lru_cache(maxsize=1)
def _load():
    fetch_data()
    return pd.read_csv(DATA / "variants.csv"), pd.read_csv(DATA / "assays.csv").set_index("DMS_id")


def _rho(pred, y):
    if np.ptp(pred) == 0:
        return 0.0
    r = spearmanr(pred, y).statistic
    return float(r) if np.isfinite(r) else 0.0


def _onehot(mutants, L):
    """position x mutant residue, plus wild-type and mutant residue identities."""
    m = pd.Series(mutants)
    pos = m.str[1:-1].astype(int).to_numpy() - 1
    wt = m.str[0].map(AA.index).to_numpy()
    mt = m.str[-1].map(AA.index).to_numpy()
    X = np.zeros((len(m), L * 20 + 40))
    i = np.arange(len(m))
    X[i, pos * 20 + mt] = 1
    X[i, L * 20 + wt] = 1
    X[i, L * 20 + 20 + mt] = 1
    return X


@lru_cache(maxsize=1)
def _baseline():
    from sklearn.linear_model import RidgeCV

    var, meta = _load()
    rhos = []
    for a in ASSAYS:
        d = var[var.assay == a]
        tr, te = d[d.split == "train"], d[d.split == "test"]
        L = int(meta.loc[a, "seq_len"])
        m = RidgeCV(alphas=np.logspace(-2, 3, 11)).fit(_onehot(tr.mutant, L), tr.DMS_score)
        rhos.append(_rho(m.predict(_onehot(te.mutant, L)), te.DMS_score.to_numpy()))
    return 1 - float(np.mean(rhos)), rhos


class ProteinFitness(Problem):
    info = ProblemInfo(
        id="B2",
        domain="biology",
        title="Protein fitness prediction at unseen positions",
        kind="generalization",
        metric="1 - Spearman rho on hidden variants, mean over 2 assays (lower is better)",
        higher_is_better=False,
        dev_budget=20,
    )

    def _split(self, name):
        var, _ = _load()
        return var[var.split == name].reset_index(drop=True)

    def statement(self) -> str:
        var, meta = _load()
        n = var.groupby(["assay", "split"]).size().unstack()
        rows = "\n".join(
            f"| {a} | {meta.loc[a, 'source_organism']} | {meta.loc[a, 'seq_len']} | "
            f"{meta.loc[a, 'selection_assay']} | {n.loc[a, 'train']} | {n.loc[a, 'dev']} | {n.loc[a, 'test']} |"
            for a in ASSAYS)
        return f"""# B2 · Protein fitness prediction at unseen positions

Two deep mutational scanning (DMS) assays of single amino-acid substitutions, from
ProteinGym v1.3. Predict the measured fitness of variants at **positions never seen in
training**: test and dev variants sit at mutated positions that do not occur in `train.csv`
(ProteinGym's modulo split, fold 0 held out).

| assay | organism | length | phenotype | train | dev | test |
|---|---|---|---|---|---|---|
{rows}

**Files:** `train.csv` (assay, mutant, DMS_score; higher = fitter), `dev.csv` and `test.csv`
(assay, mutant; no labels), `assays.json` (wild-type sequence and assay metadata). Mutant
strings are `<wt><pos><mut>` with 1-based positions in the wild-type sequence.

**Submission** (JSON): `{{"predictions": [p_1, ..., p_n]}}`, one finite number per row of
`test.csv`, in file order. Only the ranking within each assay matters. `lab eval` scores the
same format against `dev.csv` (dev rows, in order) and is limited to {self.info.dev_budget} calls.

**Metric:** 1 − Spearman ρ per assay, averaged over the two assays (0 is perfect; a
constant prediction counts as ρ = 0). Baseline: ridge regression on one-hot features.
Reference: the best published supervised model on ProteinGym's modulo split for these
assays (Kermut, mean ρ ≈ 0.70, which also uses protein language-model embeddings and
structure). The test variants are public data: do not look up their published labels.
"""

    def public_files(self):
        _, meta = _load()
        info = {a: {k: (v.item() if hasattr(v, "item") else v) for k, v in meta.loc[a].items()}
                for a in ASSAYS}
        out = {"assays.json": json.dumps(info, indent=1).encode()}
        for name, cols in [("train", ["assay", "mutant", "DMS_score"]), ("dev", ["assay", "mutant"]),
                           ("test", ["assay", "mutant"])]:
            buf = io.StringIO()
            self._split(name)[cols].to_csv(buf, index=False)
            out[f"{name}.csv"] = buf.getvalue().encode()
        return out

    def _score(self, submission, name):
        d = self._split(name)
        p = submission.get("predictions") if isinstance(submission, dict) else None
        try:
            pred = np.asarray(p, dtype=float)
        except (TypeError, ValueError):
            raise InvalidSubmission("predictions must be a list of numbers")
        if pred.shape != (len(d),) or not np.all(np.isfinite(pred)):
            raise InvalidSubmission(f"predictions must be {len(d)} finite numbers ({name}.csv order)")
        y = d.DMS_score.to_numpy()
        return 1 - float(np.mean([_rho(pred[d.assay == a], y[d.assay == a]) for a in ASSAYS]))

    def verify(self, submission) -> float:
        return self._score(submission, "test")

    def dev_eval(self, submission) -> float:
        return self._score(submission, "dev")

    def anchors(self) -> Anchors:
        base, _ = _baseline()
        return Anchors(baseline=base, reference=1 - float(np.mean(list(KERMUT.values()))), optimum=0.0,
                       emin=0.05, ref_kind="P",
                       ref_source="Kermut, ProteinGym v1.3 supervised leaderboard, fold_modulo_5 "
                                  "(OTC_HUMAN_Lo_2023 0.806, ESTA_BACSU_Nutschel_2020 0.601; 5-fold CV mean)")
