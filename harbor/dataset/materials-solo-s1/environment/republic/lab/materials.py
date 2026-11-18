"""Materials environment helpers: formula parsing, a Magpie-lite featurizer and the reference
baseline (gradient boosting). The same file is published to agents as
/data/baselines/materials_baseline.py, so the comparator in the `prediction` profile is
transparent: agents know exactly what they have to beat.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

PROPS = ["Number", "MendeleevNumber", "AtomicWeight", "MeltingT", "Column", "Row", "CovalentRadius",
         "Electronegativity", "NsValence", "NpValence", "NdValence", "NfValence", "NValence",
         "NUnfilled", "GSvolume_pa", "GSbandgap", "GSmagmom"]
_TOKEN = re.compile(r"([A-Z][a-z]?|\(|\)|\[|\]|\d*\.?\d+)")


def parse_formula(formula: str) -> dict[str, float]:
    """'Ag(AuS)2' -> {'Ag': 1, 'Au': 2, 'S': 2}. Handles nesting and decimal amounts."""
    stack = [Counter()]
    toks = _TOKEN.findall(formula.replace(" ", ""))
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in "([":
            stack.append(Counter())
        elif t in ")]":
            group = stack.pop()
            mult = 1.0
            if i + 1 < len(toks) and re.fullmatch(r"\d*\.?\d+", toks[i + 1]):
                mult = float(toks[i + 1])
                i += 1
            for el, n in group.items():
                stack[-1][el] += n * mult
        elif re.fullmatch(r"[A-Z][a-z]?", t):
            n = 1.0
            if i + 1 < len(toks) and re.fullmatch(r"\d*\.?\d+", toks[i + 1]):
                n = float(toks[i + 1])
                i += 1
            stack[-1][t] += n
        i += 1
    return dict(stack[0])


def featurize(compositions, elements: pd.DataFrame) -> np.ndarray:
    """Fraction-weighted mean, min, max and range of elemental properties."""
    table = elements.set_index("symbol")[PROPS].astype(float)
    table = table.fillna(table.mean())
    rows = []
    for comp in compositions:
        amt = parse_formula(str(comp))
        els = [e for e in amt if e in table.index]
        if not els:
            rows.append(np.full(4 * len(PROPS), np.nan))
            continue
        w = np.array([amt[e] for e in els], float)
        w /= w.sum()
        v = table.loc[els].to_numpy()
        rows.append(np.concatenate([w @ v, v.min(0), v.max(0), v.max(0) - v.min(0)]))
    X = np.vstack(rows)
    return np.where(np.isnan(X), np.nanmean(X, axis=0), X)


def baseline_predict(train: pd.DataFrame, inputs: pd.DataFrame, elements: pd.DataFrame, task: str, seed: int = 0) -> np.ndarray:
    """Reference baseline: Magpie-lite features + histogram gradient boosting (scikit-learn, 500 rounds)."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    Xtr, Xte = featurize(train["composition"], elements), featurize(inputs["composition"], elements)
    params = dict(max_iter=500, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=1.0, random_state=seed)
    if task == "classification":
        return HistGradientBoostingClassifier(**params).fit(Xtr, train["target"].astype(int)).predict(Xte).astype(float)
    return HistGradientBoostingRegressor(**params).fit(Xtr, train["target"].astype(float)).predict(Xte)


if __name__ == "__main__":  # usage inside lab exec: python materials_baseline.py expt_gap
    import os
    import sys

    data = os.environ.get("DATA_DIR", "/data")
    ds = sys.argv[1]
    tr = pd.read_csv(f"{data}/{ds}/train.csv")
    el = pd.read_csv(f"{data}/elements/elements.csv")
    import json
    task = json.load(open(f"{data}/{ds}/meta.json"))["task"]
    from sklearn.model_selection import KFold
    errs = []
    for a, b in KFold(5, shuffle=True, random_state=0).split(tr):
        p = baseline_predict(tr.iloc[a], tr.iloc[b], el, task)
        y = tr.iloc[b]["target"].to_numpy(float)
        errs.append(np.mean(np.abs(p - y)) if task == "regression" else np.mean((p > .5) != (y > .5)))
    print(json.dumps({"dataset": ds, "cv_error": float(np.mean(errs))}))
