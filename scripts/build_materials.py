"""Build the materials-science research environment: data/domains/materials/{public,hidden}.

Layout: public/ (agents, labeled), eval/ (hidden inputs only, mounted at /eval during
verification), hidden/ (labels; read only by the server, never mounted).

Composition-only Matbench datasets (CPU-friendly, no structure tools needed), each split
75/25 at random with a fixed seed. Agents see public/<ds>/train.csv (labeled) and a dataset
card. The hidden quarter is used only by the evaluator after the run: verification code gets
the hidden *inputs* (no labels); the server scores predictions against the hidden labels itself.

  .venv/bin/python scripts/build_materials.py
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import sys
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "data", "domains", "materials")
RAW = os.path.join(ROOT, "data", "domains", "_raw")
URL = "https://ml.materialsproject.org/projects/{}.json.gz"

DATASETS = {
    "expt_gap": ("matbench_expt_gap", "gap expt", "regression", "eV",
                 "Experimental band gaps of inorganic compounds (Zhuo, Mansouri Tehrani & Brgoch, J. Phys. Chem. Lett. 2018), "
                 "as curated in Matbench (Dunn et al., npj Comput. Mater. 2020). Zero gap = metal."),
    "expt_is_metal": ("matbench_expt_is_metal", "is_metal", "classification", "",
                      "Whether a compound is metallic from experiment (Zhuo et al. 2018), as curated in Matbench."),
    "steels": ("matbench_steels", "yield strength", "regression", "MPa",
               "Yield strength of steel alloys from composition (Citrine Informatics data), as curated in Matbench. Small: 312 alloys."),
    "glass": ("matbench_glass", "gfa", "classification", "",
              "Glass-forming ability of binary/ternary metallic alloys from melt-spinning experiments "
              "(Kawazoe et al., Landolt-Boernstein), as curated in Matbench."),
}
HIDDEN_FRACTION = 0.25
SEED = 20261114


def hidden_key(composition: str, seed: int) -> bool:
    """A composition is hidden in every dataset or in none (expt_gap and expt_is_metal share compositions;
    an independent split leaked 71% of hidden metal labels through the public band gaps)."""
    import hashlib
    import sys

    sys.path.insert(0, ROOT)
    from lab.materials import parse_formula

    amt = parse_formula(composition)
    tot = sum(amt.values()) or 1
    canon = ",".join(f"{e}:{amt[e] / tot:.4f}" for e in sorted(amt))
    h = int(hashlib.sha256(f"{seed}|{canon}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < HIDDEN_FRACTION


def fetch(name):
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, name + ".json.gz")
    if not os.path.exists(path):
        req = urllib.request.Request(URL.format(name), headers={"User-Agent": "Mozilla/5.0 (republic benchmark build)"})
        with urllib.request.urlopen(req, timeout=300) as r, open(path, "wb") as f:
            f.write(r.read())
    d = json.load(gzip.open(path))
    return pd.DataFrame(d["data"], columns=d["columns"])


def main(out=OUT, seed=SEED):
    """Write one environment. Each benchmark task (seed) gets its own random hidden quarter."""
    if os.path.exists(out):
        shutil.rmtree(out)
    rng = np.random.default_rng(seed)
    index = []
    for key, (name, target, task, units, blurb) in DATASETS.items():
        df = fetch(name).rename(columns={target: "target"})
        df["composition"] = df["composition"].astype(str)
        if task == "classification":
            df["target"] = df["target"].astype(int)
        df.insert(0, "id", [f"{key}-{i:05d}" for i in range(len(df))])
        hidden = np.array([hidden_key(c, seed) for c in df["composition"]])
        pub, hid = df[~hidden], df[hidden]
        pdir, hdir, edir = (os.path.join(out, d, key) for d in ("public", "hidden", "eval"))
        os.makedirs(pdir), os.makedirs(hdir), os.makedirs(edir)
        pub[["id", "composition", "target"]].to_csv(os.path.join(pdir, "train.csv"), index=False)
        hid[["id", "composition"]].to_csv(os.path.join(edir, "inputs.csv"), index=False)  # mounted for verification
        hid[["id", "composition"]].to_csv(os.path.join(hdir, "inputs.csv"), index=False)  # server only, with labels
        hid[["id", "target"]].to_csv(os.path.join(hdir, "labels.csv"), index=False)
        meta = {"id": key, "source": name, "task": task, "units": units, "n_public": int(len(pub)), "n_hidden": int(len(hid)),
                "metric": "MAE" if task == "regression" else "error rate"}
        json.dump(meta, open(os.path.join(hdir, "meta.json"), "w"), indent=1)
        json.dump(meta, open(os.path.join(pdir, "meta.json"), "w"), indent=1)
        with open(os.path.join(pdir, "README.md"), "w") as f:
            f.write(f"# {key}\n\n{blurb}\n\n- task: {task}" + (f" (target in {units})" if units else " (target 0/1)") +
                    f"\n- train.csv: {len(pub)} rows, columns id, composition, target\n"
                    f"- A further {len(hid)} compositions are held by the evaluator and never shown.\n"
                    "- License: Matbench data (MIT/CC-BY as distributed by Materials Project).\n")
        index.append(meta)
        print(key, len(pub), "public /", len(hid), "hidden")
    el = os.path.join(out, "public", "elements")
    os.makedirs(el)
    if not os.path.exists(os.path.join(ROOT, "data", "X1", "elements.csv")):  # the Magpie table comes from the X1 build
        sys.path.insert(0, ROOT)
        from lab.problems.x1_bandgap import fetch_data
        fetch_data()
    shutil.copy(os.path.join(ROOT, "data", "X1", "elements.csv"), os.path.join(el, "elements.csv"))
    with open(os.path.join(el, "README.md"), "w") as f:
        f.write("# elements\n\nPer-element properties (Magpie table, Ward et al., npj Comput. Mater. 2016; via matminer, BSD). "
                "One row per element symbol.\n")
    bl = os.path.join(out, "public", "baselines")
    os.makedirs(bl)
    shutil.copy(os.path.join(ROOT, "lab", "materials.py"), os.path.join(bl, "materials_baseline.py"))
    with open(os.path.join(bl, "README.md"), "w") as f:
        f.write("# baselines\n\nmaterials_baseline.py: the evaluator's reference baseline (Magpie-lite composition features + "
                "random forest). `python materials_baseline.py <dataset>` prints its 5-fold CV error on train.csv. "
                "Claims under the `prediction` profile are compared against this model on the hidden data.\n")
    json.dump(index, open(os.path.join(out, "index.json"), "w"), indent=1)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    main(a.out, a.seed)
