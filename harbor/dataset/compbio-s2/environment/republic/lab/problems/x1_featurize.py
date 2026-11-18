"""Composition parsing and Magpie-style featurization (numpy/pandas only)."""
import re
import numpy as np
import pandas as pd

_TOK = re.compile(r"([A-Z][a-z]?|\(|\)|\d+(?:\.\d+)?|\.\d+)")


def parse_formula(s):
    """{element: amount} for formulas with nested parentheses and fractional counts."""
    toks = _TOK.findall(s)
    if "".join(toks) != s:
        raise ValueError(f"cannot parse {s!r}")
    stack, i = [{}], 0
    while i < len(toks):
        t = toks[i]
        i += 1
        mult = 1.0
        if i < len(toks) and (toks[i][0].isdigit() or toks[i][0] == "."):
            mult, i = float(toks[i]), i + 1
        if t == "(":
            if mult != 1.0:
                raise ValueError(f"number after '(' in {s!r}")
            stack.append({})
        elif t == ")":
            if len(stack) == 1:
                raise ValueError(f"unbalanced parentheses in {s!r}")
            grp = stack.pop()
            for k, v in grp.items():
                stack[-1][k] = stack[-1].get(k, 0.0) + v * mult
        elif t[0].isalpha():
            stack[-1][t] = stack[-1].get(t, 0.0) + mult
        else:
            raise ValueError(f"cannot parse {s!r}")
    if len(stack) != 1:
        raise ValueError(f"unbalanced parentheses in {s!r}")
    return stack[0]


def featurize(formulas, elements_csv="elements.csv"):
    """Fraction-weighted mean, avg. deviation, min, max, range of every elemental property,
    plus number of elements and L2/L3 norms of the fractions."""
    el = pd.read_csv(elements_csv).set_index("symbol")
    el = el.fillna(el.mean())
    rows = []
    for f in formulas:
        c = parse_formula(f)
        x = np.array(list(c.values()))
        x = x / x.sum()
        P = el.loc[list(c)].to_numpy()
        mu = x @ P
        feats = [mu, x @ np.abs(P - mu), P.min(0), P.max(0), P.max(0) - P.min(0)]
        rows.append(np.r_[np.concatenate(feats), len(c), np.linalg.norm(x, 2), np.linalg.norm(x, 3)])
    names = [f"{s}_{p}" for s in ("mean", "avgdev", "min", "max", "range") for p in el.columns]
    return pd.DataFrame(rows, columns=names + ["n_elements", "frac_l2", "frac_l3"])
