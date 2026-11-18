"""X1. Experimental band gap from composition (Matbench v0.1 `matbench_expt_gap`).

Data: https://ml.materialsproject.org/projects/matbench_expt_gap.json.gz (4,604 compositions,
sha256 783e7d14...; Matbench, Dunn et al. npj Comput. Mater. 2020, MIT License; labels from
Zhuo, Mansouri Tehrani & Brgoch, J. Phys. Chem. Lett. 2018 SI). Folds: Matbench v0.1 uses
KFold(n_splits=5, shuffle=True, random_state=18012019) on the dataset order; verified
identical to matbench/matbench_v0.1_validation.json. Hidden test = fold 0 (921 entries);
public train = the other 3,683 minus a dev split of 368 (fixed RNG) whose labels are withheld.

Elemental properties shipped to agents: the Magpie element tables (Ward et al. 2016) as
distributed in matminer (src/matminer/utils/data_files/magpie_elementdata, BSD-3-style
license, LBNL), mostly derived from Mathematica ElementData.

Metric: MAE (eV) on fold 0. Anchors: baseline = train-mean predictor (computed; Matbench
`dummy` gives 1.0965 on this fold); reference = leaderboard top, Darwin (Xie et al. 2023),
fold-0 MAE 0.2723 (5-fold mean 0.2865; MODNet v0.1.12: fold-0 0.3121, mean 0.3327), from
matbench/benchmarks/*/results.json.gz. emin = 0.1 eV: Matbench dropped compositions whose
reported gaps span more than 0.1 eV, so ~0.1 eV is the scale of experimental disagreement
(label noise) that remains, and the practical floor for composition-only prediction.
"""

from __future__ import annotations

import gzip
import io
import json
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .base import Anchors, InvalidSubmission, Problem, ProblemInfo
from .x1_featurize import featurize, parse_formula  # noqa: F401  (shipped to agents)

DATA = Path(__file__).resolve().parents[2] / "data" / "X1"
URL = "https://ml.materialsproject.org/projects/matbench_expt_gap.json.gz"
MAGPIE = ("https://raw.githubusercontent.com/hackingmaterials/matminer/main/src/matminer/utils/"
          "data_files/magpie_elementdata/")
PROPS = ["Number", "MendeleevNumber", "AtomicWeight", "MeltingT", "Column", "Row", "CovalentRadius",
         "Electronegativity", "NsValence", "NpValence", "NdValence", "NfValence", "NValence",
         "NsUnfilled", "NpUnfilled", "NdUnfilled", "NfUnfilled", "NUnfilled", "GSvolume_pa",
         "GSbandgap", "GSmagmom", "SpaceGroupNumber", "FirstIonizationEnergy", "ElectronAffinity",
         "Density", "Polarizability"]
SYMBOLS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se "
           "Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb "
           "Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm "
           "Bk Cf Es Fm Md No Lr").split()
FOLD_SEED, N_DEV, DEV_SEED = 18012019, 368, 7


def _get(url):
    with urllib.request.urlopen(url, timeout=300) as r:
        return r.read()


def fetch_data():
    """Download once; write data/X1/{gap,elements}.csv."""
    if (DATA / "gap.csv").exists() and (DATA / "elements.csv").exists():
        return
    from sklearn.model_selection import KFold

    raw = DATA / "raw"
    (raw / "magpie").mkdir(parents=True, exist_ok=True)
    src = raw / "matbench_expt_gap.json.gz"
    if not src.exists():
        src.write_bytes(_get(URL))
    d = json.load(gzip.open(src))
    df = pd.DataFrame(d["data"], columns=["composition", "gap"])
    df.insert(0, "id", [f"mb-expt-gap-{i + 1:04d}" for i in range(len(df))])
    df["split"] = "train"
    test = next(iter(KFold(5, shuffle=True, random_state=FOLD_SEED).split(df)))[1]
    df.loc[test, "split"] = "test"
    tr = np.flatnonzero(df.split == "train")
    df.loc[np.random.default_rng(DEV_SEED).choice(tr, N_DEV, replace=False), "split"] = "dev"
    df.to_csv(DATA / "gap.csv", index=False)
    el = {"symbol": SYMBOLS}
    for p in PROPS:
        f = raw / "magpie" / f"{p}.table"
        if not f.exists():
            f.write_bytes(_get(MAGPIE + f.name))
        vals = f.read_text().split("\n")[: len(SYMBOLS)]
        el[p] = pd.to_numeric(pd.Series(vals).str.strip(), errors="coerce")
    pd.DataFrame(el).to_csv(DATA / "elements.csv", index=False)


@lru_cache(maxsize=1)
def _load():
    fetch_data()
    return pd.read_csv(DATA / "gap.csv"), pd.read_csv(DATA / "elements.csv")


class BandGap(Problem):
    info = ProblemInfo(
        id="X1",
        domain="materials",
        title="Experimental band gap from composition",
        kind="generalization",
        metric="MAE in eV on hidden compositions (lower is better)",
        higher_is_better=False,
        dev_budget=20,
    )

    def _split(self, name):
        df, _ = _load()
        return df[df.split == name].reset_index(drop=True)

    def statement(self) -> str:
        n = {k: len(self._split(k)) for k in ("train", "dev", "test")}
        return f"""# X1 · Experimental band gap from composition

Predict the **experimentally measured band gap** (eV) of an inorganic solid from its chemical
formula alone (no structure). Data: Matbench `matbench_expt_gap` (curated from Zhuo et al.
2018); about half of all entries are metals with gap exactly 0. The hidden test set is
Matbench fold 0.

**Files:** `train.csv` ({n['train']} rows: id, composition, gap), `dev.csv` ({n['dev']}) and
`test.csv` ({n['test']}) (id, composition; no labels), `elements.csv` (Magpie elemental
property table per element symbol, as used by matminer's Magpie featurizer; empty = missing),
`featurize.py` (formula parser and a mean/min/max/range/deviation composition featurizer
over `elements.csv`; pure numpy/pandas). Formulas may contain parentheses and fractional
amounts, e.g. `Ag0.5Ge1Pb1.75S4`, `Ag(AuS)2`.

**Submission** (JSON): `{{"predictions": [g_1, ..., g_n]}}`, one finite number (eV) per row of
`test.csv`, in file order. `lab eval` scores the same format against `dev.csv` (dev rows in
order), at most {self.info.dev_budget} calls.

**Metric:** mean absolute error in eV (0 is perfect). Baseline: predicting the training mean
(about 1.1 eV). Reference: the best Matbench leaderboard entry on this fold (about 0.27 eV).
The test compositions are public data: do not look up their published labels.
"""

    def public_files(self):
        _, el = _load()
        out = {"featurize.py": Path(__file__).with_name("x1_featurize.py").read_bytes()}
        buf = io.StringIO()
        el.to_csv(buf, index=False)
        out["elements.csv"] = buf.getvalue().encode()
        for name, cols in [("train", ["id", "composition", "gap"]), ("dev", ["id", "composition"]),
                           ("test", ["id", "composition"])]:
            buf = io.StringIO()
            self._split(name)[cols].to_csv(buf, index=False)
            out[f"{name}.csv"] = buf.getvalue().encode()
        return out

    def _mae(self, submission, name):
        d = self._split(name)
        p = submission.get("predictions") if isinstance(submission, dict) else None
        try:
            pred = np.asarray(p, dtype=float)
        except (TypeError, ValueError):
            raise InvalidSubmission("predictions must be a list of numbers")
        if pred.shape != (len(d),) or not np.all(np.isfinite(pred)):
            raise InvalidSubmission(f"predictions must be {len(d)} finite numbers ({name}.csv order)")
        return float(np.mean(np.abs(pred - d.gap.to_numpy())))

    def verify(self, submission) -> float:
        return self._mae(submission, "test")

    def dev_eval(self, submission) -> float:
        return self._mae(submission, "dev")

    def anchors(self) -> Anchors:
        mean = float(self._split("train").gap.mean())
        base = self.verify({"predictions": [mean] * len(self._split("test"))})
        return Anchors(baseline=base, reference=0.2723, optimum=0.0, emin=0.1, ref_kind="P",
                       ref_source="Matbench v0.1 leaderboard top (Darwin), matbench_expt_gap fold-0 MAE "
                                  "0.2723 eV (5-fold mean 0.2865; MODNet v0.1.12 0.3121 / 0.3327)")
