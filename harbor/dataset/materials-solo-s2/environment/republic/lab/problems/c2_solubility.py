"""C2. Aqueous solubility on the TDC ADMET benchmark group's scaffold split.

Predict log-solubility (log mol/L) from SMILES for the 1997 held-out molecules of
TDC's `Solubility_AqSolDB` benchmark. The split is TDC's own: a Murcko-scaffold split
holding out 20% of the 9982 molecules, shipped pre-split by TDC so the test set is
byte-identical to the one the public leaderboard is computed on. Test labels stay here.

Metric: MAE in log mol/L, lower is better.

Anchors:
  baseline  = Morgan fingerprint (radius 2, 2048 bits) + RandomForest(500), trained on
              exactly the labelled data agents get, computed here and cached.
  reference = 0.741, the best MAE on the TDC leaderboard (MiniMol), ref_kind "P",
              https://tdcommons.ai/benchmark/admet_group/06aqsol/ (runner-up Chemprop-RDKit
              0.761; the leaderboard's +/- are the sd over the 5 required seeds, MiniMol
              +/- 0.013, so the reference is a genuine published record, not a fitted curve).
  optimum   = 0 (a loss), emin = 0.35 log units.

emin justification. AqSolDB aggregates aqueous-solubility measurements from nine
heterogeneous literature sources, so the labels carry interlaboratory measurement error,
and no model can do better than that error in expectation. Avdeef (ADMET & DMPK 2015,
3(2):84-109, doi:10.5599/admet.3.2.193) puts the "currently expected interlaboratory
reproducibility" of equilibrium solubility at 0.6-0.7 log units, and reports (ADMET & DMPK
2019, 7(3):210-219, doi:10.5599/admet.698) average interlaboratory standard deviations of
0.15 (CheqSol) and 0.24 (shake-flask) log units for carefully standardised measurements.
Taking a label-noise sd of ~0.44 log units -- between the best-practice and heterogeneous
regimes -- and E|N(0, sd)| = sd * sqrt(2/pi) gives an irreducible MAE of ~0.35 log units.
That is less than half the published reference, so a 1.0 here means a model essentially at
the data's noise floor.

Data (downloaded by `fetch_data()`, Harvard Dataverse, TDC's own pre-split files):
  https://dataverse.harvard.edu/api/access/datafile/4426004  (admet_group.zip, TDC
  benchmark2id["admet_group"]), member admet_group/solubility_aqsoldb/{train_val,test}.csv
  TDC dataverse: https://doi.org/10.7910/DVN/21LKWG; TDC datasets are CC BY 4.0.
  Underlying data: Sorkun, Khetan & Er, "AqSolDB, a curated reference set of aqueous
  solubility and 2D descriptors for a diverse set of compounds", Sci Data 6, 143 (2019),
  https://doi.org/10.1038/s41597-019-0151-1.
"""

from __future__ import annotations

import csv
import io
import json
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from .base import Anchors, InvalidSubmission, Problem, ProblemInfo

RDLogger.DisableLog("rdApp.*")

DATA = Path(__file__).resolve().parents[2] / "data" / "C2"
RAW = DATA / "raw"  # gitignored: the download, kept so fetch_data() is a no-op on rerun
GROUP_URL = "https://dataverse.harvard.edu/api/access/datafile/4426004"
MEMBERS = {"train_val.csv": "admet_group/solubility_aqsoldb/train_val.csv",
           "test.csv": "admet_group/solubility_aqsoldb/test.csv"}

DEV_FRAC = 0.15
DEV_BUDGET = 20
REFERENCE_MAE = 0.741
EMIN = 0.35

_FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fetch_data() -> None:
    """Download TDC's admet_group bundle and keep the two solubility split files."""
    RAW.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    blob = RAW / "admet_group.zip"
    if not blob.exists():
        with urllib.request.urlopen(GROUP_URL, timeout=300) as r:  # noqa: S310 (fixed https URL)
            blob.write_bytes(r.read())
    with zipfile.ZipFile(blob) as z:
        for out, member in MEMBERS.items():
            (DATA / out).write_bytes(z.read(member))


def _rows(name: str) -> list[dict]:
    path = DATA / name
    if not path.exists():
        fetch_data()
    return list(csv.DictReader(io.StringIO(path.read_text())))


def _scaffold(smiles: str) -> str:
    """Bemis-Murcko framework: strip terminal non-ring atoms until only rings and linkers
    remain. Done by hand because `Chem.MurckoDecompose` raises "bad any cast" in RDKit
    2026.09. Acyclic molecules collapse to "" and so share one group."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles  # unparseable: its own group
    em = Chem.RWMol(mol)
    stripped = True
    while stripped and em.GetNumAtoms():
        stripped = False
        for atom in em.GetAtoms():
            if atom.GetDegree() <= 1 and not atom.IsInRing():
                em.RemoveAtom(atom.GetIdx())
                stripped = True
                break
    try:
        return Chem.MolToSmiles(em.GetMol())
    except Exception:
        return smiles


@lru_cache(maxsize=1)
def _splits() -> tuple[tuple[dict, ...], tuple[dict, ...], tuple[dict, ...]]:
    """(train, dev, test). Dev is a deterministic scaffold carve-out of TDC's train_val,
    cached to split_train.csv / split_dev.csv so a server start costs no scaffold work."""
    test = _rows("test.csv")
    if (DATA / "split_train.csv").exists() and (DATA / "split_dev.csv").exists():
        return tuple(_rows("split_train.csv")), tuple(_rows("split_dev.csv")), tuple(test)
    train_val = _rows("train_val.csv")
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(train_val):
        groups.setdefault(_scaffold(r["Drug"]), []).append(i)
    order = sorted(groups.values(), key=lambda g: (-len(g), g[0]))  # DeepChem/TDC style
    n_dev = int(round(DEV_FRAC * len(train_val)))
    dev_idx: set[int] = set()
    for g in reversed(order):  # smallest, most unusual scaffolds become dev
        if len(dev_idx) >= n_dev:
            break
        dev_idx.update(g)
    train = tuple(r for i, r in enumerate(train_val) if i not in dev_idx)
    dev = tuple(r for i, r in enumerate(train_val) if i in dev_idx)
    (DATA / "split_train.csv").write_bytes(_csv(train, labelled=True))
    (DATA / "split_dev.csv").write_bytes(_csv(dev, labelled=True))
    return train, dev, tuple(test)


def _xy(rows):
    """Morgan features and labels, skipping the few rows RDKit cannot parse."""
    mols = [(Chem.MolFromSmiles(r["Drug"]), r) for r in rows]
    ok = [(m, r) for m, r in mols if m is not None]
    x = np.array([_FPGEN.GetFingerprintAsNumPy(m) for m, _ in ok], dtype=np.uint8)
    return x, np.array([float(r["Y"]) for _, r in ok])


def _csv(rows, labelled: bool) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["Drug_ID", "Drug", "Y"] if labelled else ["Drug_ID", "Drug"])
    for r in rows:
        w.writerow([r["Drug_ID"], r["Drug"]] + ([r["Y"]] if labelled else []))
    return buf.getvalue().encode()


class AqSolubility(Problem):
    def __init__(self, seed: int = 0):
        super().__init__(seed)  # the split is TDC's and fixed; seed changes nothing
        self.train, self.dev, self.test = _splits()
        self.info = ProblemInfo(
            id="C2",
            domain="chemistry",
            title="Aqueous solubility, TDC scaffold split",
            kind="generalization",
            metric="MAE in log mol/L (lower is better)",
            higher_is_better=False,
            dev_budget=DEV_BUDGET,
        )

    def statement(self) -> str:
        return f"""# C2 · Aqueous solubility (TDC `Solubility_AqSolDB`, scaffold split)

Predict **log aqueous solubility in log mol/L** from a SMILES string. The data is AqSolDB
(9,982 compounds) on the Therapeutics Data Commons ADMET benchmark group's Murcko-scaffold
split: 20% of the molecules are held out as the test set, which is the same test set the
public TDC leaderboard is scored on. Test molecules sit on scaffolds you have never seen.

**Public files**
- `train.csv` -- `Drug_ID,Drug,Y`: {len(self.train):,} labelled molecules (`Y` is log mol/L).
- `dev.csv` -- `Drug_ID,Drug`: {len(self.dev):,} molecules carved off the training set on
  unusual scaffolds. Labels are held here. `lab eval C2 <file>` scores predictions on it
  and returns the MAE, at most **{DEV_BUDGET} times** per run: it is your validation signal
  for a shift like the test set's, so do not burn it on hyperparameter noise.
- `test.csv` -- `Drug_ID,Drug`: {len(self.test):,} molecules, labels hidden. This is what you
  are scored on.

**Submission** (JSON), any one of these, all equivalent:
- `{{"predictions": [y_0, ..., y_{len(self.test) - 1}]}}` -- one number per row of `test.csv`, in file order;
- `{{"predictions": {{"0": y_0, "1": y_1, ...}}}}` -- keyed by row index;
- `{{"predictions": {{"<SMILES>": y, ...}}}}` -- keyed by the `Drug` column verbatim.

Every row must be covered and every value finite. For `dev.csv`, submit the same shape
sized to `dev.csv`. Two test SMILES are hypervalent and do not parse in RDKit; they are part
of the benchmark, so predict something for them anyway.

**Metric:** mean absolute error in log mol/L, lower is better. A Morgan-fingerprint random
forest on `train.csv` is the baseline; the reference is **MAE {REFERENCE_MAE}**, the best
result on the TDC leaderboard for this benchmark (MiniMol). Beating {REFERENCE_MAE} is a
record on a live public leaderboard -- though note that the leaderboard entries trained on
all {len(self.train) + len(self.dev):,} non-test molecules while you hold
{len(self.train):,} labelled and {len(self.dev):,} unlabelled, so factor that into your claim.
"""

    def public_files(self):
        return {
            "train.csv": _csv(self.train, labelled=True),
            "dev.csv": _csv(self.dev, labelled=False),
            "test.csv": _csv(self.test, labelled=False),
        }

    def _mae(self, submission, rows) -> float:
        preds = submission.get("predictions") if isinstance(submission, dict) else None
        n = len(rows)
        if isinstance(preds, dict):
            out = []
            for i, r in enumerate(rows):
                v = preds.get(str(i), preds.get(i, preds.get(r["Drug"])))
                if v is None:
                    raise InvalidSubmission(f"no prediction for row {i} ({r['Drug']})")
                out.append(v)
            preds = out
        if not isinstance(preds, list) or len(preds) != n:
            raise InvalidSubmission(
                f"'predictions' must be a list of {n} numbers (file order), or a dict keyed by "
                f"row index or by the SMILES in the 'Drug' column"
            )
        try:
            y = np.asarray(preds, dtype=float)
        except (TypeError, ValueError):
            raise InvalidSubmission("predictions must be numbers")
        if y.shape != (n,) or not np.all(np.isfinite(y)):
            raise InvalidSubmission(f"predictions must be {n} finite numbers")
        truth = np.array([float(r["Y"]) for r in rows])
        return float(np.mean(np.abs(y - truth)))

    def verify(self, submission) -> float:
        return self._mae(submission, self.test)

    def dev_eval(self, submission) -> float:
        return self._mae(submission, self.dev)

    def anchors(self) -> Anchors:
        return Anchors(
            baseline=_baseline_mae(),
            reference=REFERENCE_MAE,
            optimum=0.0,
            emin=EMIN,
            ref_kind="P",
            ref_source="TDC ADMET leaderboard, Solubility_AqSolDB, best MAE 0.741 (MiniMol), "
                       "https://tdcommons.ai/benchmark/admet_group/06aqsol/",
        )


@lru_cache(maxsize=1)
def _baseline_mae() -> float:
    """Morgan + RandomForest on the public train split, scored on the hidden test set."""
    cache = DATA / "anchors.json"
    if cache.exists():
        return json.loads(cache.read_text())["baseline"]
    from sklearn.ensemble import RandomForestRegressor

    train, _, test = _splits()
    xtr, ytr = _xy(train)
    xte, yte = _xy(test)
    model = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=0).fit(xtr, ytr)
    mae = float(np.mean(np.abs(model.predict(xte) - yte)))
    if not mae > REFERENCE_MAE:
        raise RuntimeError(f"C2 baseline MAE {mae:.3f} is not worse than the reference {REFERENCE_MAE}")
    cache.write_text(json.dumps({"baseline": mae, "reference": REFERENCE_MAE,
                                 "model": "Morgan r2/2048 + RandomForest(500) on train.csv"}, indent=1))
    return mae
