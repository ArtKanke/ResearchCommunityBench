"""Build the computational-biology and AI/ML research environments (materials: build_materials.py).

Same layout for every domain, so the verifier is shared:
  public/<dataset>/   train data with labels, README.md (card), meta.json   -> agents (/lab/data)
  eval/<dataset>/     hidden inputs only                                  -> verifier, mounted at $EVAL_DIR
  hidden/<dataset>/   labels.csv (id, target), meta.json, and the comparator for the `prediction`
                      profile: baseline.csv (id, pred; paired test) or reference.json ({"error": e})
"""

from __future__ import annotations

import io
import json
import os
import pickle
import shutil
import tarfile
import urllib.request
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "domains", "_raw")


def _get(url, name):
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (republic benchmark build)"})
        with urllib.request.urlopen(req, timeout=600) as r, open(path + ".tmp", "wb") as f:
            shutil.copyfileobj(r, f)
        os.replace(path + ".tmp", path)
    return path


def _write(out, key, meta, card, public: dict, eval_: dict, labels: pd.DataFrame, comparator):
    for part in ("public", "eval", "hidden"):
        os.makedirs(os.path.join(out, part, key), exist_ok=True)
    for name, obj in public.items():
        _save(obj, os.path.join(out, "public", key, name))
    for name, obj in eval_.items():
        _save(obj, os.path.join(out, "eval", key, name))
    labels.to_csv(os.path.join(out, "hidden", key, "labels.csv"), index=False)
    for part in ("public", "hidden"):
        json.dump(meta, open(os.path.join(out, part, key, "meta.json"), "w"), indent=1)
    if isinstance(comparator, pd.DataFrame):
        comparator.to_csv(os.path.join(out, "hidden", key, "baseline.csv"), index=False)
    else:
        json.dump(comparator, open(os.path.join(out, "hidden", key, "reference.json"), "w"), indent=1)
    open(os.path.join(out, "public", key, "README.md"), "w").write(card)
    print(f"  {key}: {meta}")


def _save(obj, path):
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path, index=False)
    elif isinstance(obj, dict):
        np.savez_compressed(path, **obj)
    else:
        open(path, "w").write(obj)


# ----------------------------------------------------------------------------- computational biology

ASSAYS = ["BLAT_ECOLX_Firnberg_2014", "BLAT_ECOLX_Stiffler_2015", "RASK_HUMAN_Weng_2022_abundance",
          "RASK_HUMAN_Weng_2022_binding-DARPin_K55", "KCNE1_HUMAN_Muhammad_2023_expression",
          "KCNE1_HUMAN_Muhammad_2023_function", "CASP3_HUMAN_Roychowdhury_2020"]
AA = "ACDEFGHIKLMNPQRSTVWY"


def _hidden(protein: str, mutant: str, seed: int) -> bool:
    """A mutant of a protein is hidden in every assay of that protein or in none (sister assays of the same
    protein, e.g. two beta-lactamase scans at Spearman 0.94, would otherwise leak hidden labels)."""
    import hashlib

    h = int(hashlib.sha256(f"{seed}|{protein}|{mutant}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < 0.25


def _onehot(seqs):
    L = max(len(s) for s in seqs)
    X = np.zeros((len(seqs), L * 20), np.float32)
    for i, s in enumerate(seqs):
        for j, a in enumerate(s):
            k = AA.find(a)
            if k >= 0:
                X[i, j * 20 + k] = 1
    return X


def build_compbio(out, seed):
    from sklearn.linear_model import Ridge

    if os.path.exists(out):
        shutil.rmtree(out)
    pg = os.path.join(ROOT, "data", "B2", "raw")
    ref = pd.read_csv(os.path.join(pg, "DMS_substitutions.csv")).set_index("DMS_id")
    z = zipfile.ZipFile(os.path.join(pg, "cv_folds_singles_substitutions.zip"))
    rng = np.random.default_rng(seed)
    for dms in ASSAYS:
        df = pd.read_csv(z.open(f"cv_folds_singles_substitutions/{dms}.csv"))
        df = df[["mutant", "mutated_sequence", "DMS_score"]].rename(columns={"mutated_sequence": "sequence", "DMS_score": "target"})
        df.insert(0, "id", [f"{dms[:24]}-{i:05d}" for i in range(len(df))])
        hidden = np.array([_hidden(ref.loc[dms, "UniProt_ID"], m, seed) for m in df["mutant"]])
        pub, hid = df[~hidden], df[hidden]
        m = Ridge(alpha=1.0).fit(_onehot(pub.sequence.tolist()), pub.target)
        base = pd.DataFrame({"id": hid.id, "pred": m.predict(_onehot(hid.sequence.tolist()))})
        r = ref.loc[dms]
        meta = {"id": dms, "task": "regression", "units": "DMS score", "metric": "MAE", "n_public": int(len(pub)),
                "n_hidden": int(len(hid)), "protein": r["UniProt_ID"], "selection_type": r["coarse_selection_type"]}
        card = (f"# {dms}\n\nDeep mutational scan: {r['title']} ({r['first_author']}, {r['year']}). "
                f"Protein {r['UniProt_ID']} ({r['source_organism']}), {r['seq_len']} residues; phenotype: "
                f"{r['selection_assay']} ({r['coarse_selection_type']}). From ProteinGym v1.3 (Notin et al., NeurIPS 2023; MIT).\n\n"
                f"- train.csv: {len(pub)} single substitutions: id, mutant (e.g. A102E), sequence (full mutated sequence), target (DMS score; higher = fitter)\n"
                f"- A further {len(hid)} variants are held by the evaluator.\n- Wild-type sequence: {r['target_seq']}\n"
                "- Evaluated by mean absolute error on the DMS score scale.\n")
        _write(out, dms, meta, card, {"train.csv": pub[["id", "mutant", "sequence", "target"]]},
               {"inputs.csv": hid[["id", "mutant", "sequence"]]}, hid[["id", "target"]], base)
    bl = os.path.join(out, "public", "baselines")
    os.makedirs(bl)
    open(os.path.join(bl, "README.md"), "w").write(
        "# baselines\n\nThe evaluator's comparator for `prediction` claims on each assay: one-hot encoding of the mutated "
        "sequence (position x amino acid) + ridge regression (alpha=1), trained on train.csv. Code: compbio_baseline.py.\n")
    open(os.path.join(bl, "compbio_baseline.py"), "w").write(
        '"""One-hot + ridge baseline (what `prediction` claims must beat)."""\nimport numpy as np\nfrom sklearn.linear_model import Ridge\n'
        f'AA = "{AA}"\n\n\ndef onehot(seqs):\n    L = max(len(s) for s in seqs)\n    X = np.zeros((len(seqs), L * 20), np.float32)\n'
        '    for i, s in enumerate(seqs):\n        for j, a in enumerate(s):\n            k = AA.find(a)\n            if k >= 0:\n'
        '                X[i, j * 20 + k] = 1\n    return X\n\n\ndef baseline_predict(train, inputs):\n'
        '    return Ridge(alpha=1.0).fit(onehot(train.sequence.tolist()), train.target).predict(onehot(inputs.sequence.tolist()))\n')


# ----------------------------------------------------------------------------- AI/ML

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
AGNEWS = "https://huggingface.co/datasets/fancyzhx/ag_news/resolve/main/data/{}-00000-of-00001.parquet"
CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def build_aiml(out, seed):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    if os.path.exists(out):
        shutil.rmtree(out)
    # CIFAR-10: public = the 50k training images; hidden = the 10k official test images
    tf = tarfile.open(_get(CIFAR_URL, "cifar-10-python.tar.gz"))

    def batch(name):
        d = pickle.load(tf.extractfile(f"cifar-10-batches-py/{name}"), encoding="bytes")
        return d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), np.array(d[b"labels"])
    xs, ys = zip(*[batch(f"data_batch_{i}") for i in range(1, 6)])
    xtr, ytr = np.concatenate(xs), np.concatenate(ys)
    xte, yte = batch("test_batch")
    idtr = np.array([f"cifar-tr-{i:05d}" for i in range(len(xtr))])
    idte = np.array([f"cifar-te-{i:05d}" for i in range(len(xte))])
    meta = {"id": "cifar10", "task": "classification", "units": "", "metric": "error rate", "n_public": int(len(xtr)),
            "n_hidden": int(len(xte)), "classes": CIFAR_CLASSES}
    card = ("# cifar10\n\nCIFAR-10 (Krizhevsky 2009): 32x32 colour images, 10 classes "
            f"({', '.join(CIFAR_CLASSES)}).\n\n- train.npz: x (50000, 32, 32, 3) uint8, y (50000,) int labels 0-9, id\n"
            "- The evaluator holds the 10,000 official test images; at verification they are at "
            "$EVAL_DIR/cifar10/inputs.npz (x, id). Predictions: integer class ids.\n- Metric: error rate. "
            "Reference for `prediction` claims: 6% error (airbench94, Jordan 2024, arXiv 2404.00498: 94% accuracy in seconds on one A100).\n")
    _write(out, "cifar10", meta, card, {"train.npz": {"x": xtr, "y": ytr, "id": idtr}},
           {"inputs.npz": {"x": xte, "id": idte}}, pd.DataFrame({"id": idte, "target": yte}),
           {"error": 0.06, "source": "airbench94 (Jordan 2024, arXiv 2404.00498)"})
    # AG News: public = 40k sampled training articles; hidden = the 7,600 official test articles
    rng = np.random.default_rng(seed)
    tr = pd.read_parquet(_get(AGNEWS.format("train"), "agnews-train.parquet"))
    te = pd.read_parquet(_get(AGNEWS.format("test"), "agnews-test.parquet"))
    tr = tr.iloc[rng.permutation(len(tr))[:40000]].reset_index(drop=True)
    tr.insert(0, "id", [f"ag-tr-{i:05d}" for i in range(len(tr))])
    te.insert(0, "id", [f"ag-te-{i:05d}" for i in range(len(te))])
    tr, te = tr.rename(columns={"label": "target"}), te.rename(columns={"label": "target"})
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    clf = LogisticRegression(max_iter=2000, C=4.0).fit(vec.fit_transform(tr.text), tr.target)
    base = pd.DataFrame({"id": te.id, "pred": clf.predict(vec.transform(te.text))})
    meta = {"id": "agnews", "task": "classification", "units": "", "metric": "error rate", "n_public": int(len(tr)),
            "n_hidden": int(len(te)), "classes": ["World", "Sports", "Business", "Sci/Tech"]}
    card = ("# agnews\n\nAG News topic classification (Zhang et al. 2015), 4 classes: 0 World, 1 Sports, 2 Business, 3 Sci/Tech.\n\n"
            f"- train.csv: {len(tr)} articles: id, text, target\n- The evaluator holds the 7,600 official test articles "
            "($EVAL_DIR/agnews/inputs.csv: id, text at verification).\n- Metric: error rate. Comparator for `prediction` claims: "
            "TF-IDF (1-2 grams) + logistic regression trained on train.csv (about 8% error).\n")
    _write(out, "agnews", meta, card, {"train.csv": tr[["id", "text", "target"]]}, {"inputs.csv": te[["id", "text"]]},
           te[["id", "target"]], base)
    bl = os.path.join(out, "public", "baselines")
    os.makedirs(bl)
    open(os.path.join(bl, "README.md"), "w").write(
        "# baselines\n\n- cifar10: `prediction` claims must beat 6% error (airbench94 reference).\n"
        "- agnews: `prediction` claims must beat TF-IDF + logistic regression (the evaluator's per-article predictions).\n"
        "- Starter code: train_cifar.py (a small ResNet-style CNN, a few minutes on the GPU) shows the data and "
        "prediction-file formats. It is a starting point, not a reference.\n")
    open(os.path.join(bl, "train_cifar.py"), "w").write(STARTER_CIFAR)


STARTER_CIFAR = '''"""Minimal CIFAR-10 trainer showing the formats (not a strong model).
Without $EVAL_DIR: trains on 45k, reports held-out error to $OUT_DIR/metrics.json.
With $EVAL_DIR: trains on all 50k and writes $OUT_DIR/predictions.csv for the hidden test images."""
import json, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

D, E, O = os.environ["DATA_DIR"], os.environ.get("EVAL_DIR"), os.environ["OUT_DIR"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
d = np.load(f"{D}/cifar10/train.npz")
x, y = torch.tensor(d["x"]).permute(0, 3, 1, 2).float() / 255, torch.tensor(d["y"])
if E:
    t = np.load(f"{E}/cifar10/inputs.npz"); xv, ids = torch.tensor(t["x"]).permute(0, 3, 1, 2).float() / 255, t["id"]
    xt, yt = x, y
else:
    xt, yt, xv, yv = x[:45000], y[:45000], x[45000:], y[45000:]
net = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(), nn.Linear(256, 10)).to(dev)
opt = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4, nesterov=True)
for epoch in range(8):
    perm = torch.randperm(len(xt))
    for i in range(0, len(xt), 256):
        b = perm[i:i + 256]
        loss = F.cross_entropy(net(xt[b].to(dev)), yt[b].to(dev))
        opt.zero_grad(); loss.backward(); opt.step()
net.eval()
with torch.no_grad():
    pred = torch.cat([net(xv[i:i + 1000].to(dev)).argmax(1).cpu() for i in range(0, len(xv), 1000)]).numpy()
if E:
    import pandas as pd
    pd.DataFrame({"id": ids, "pred": pred}).to_csv(f"{O}/predictions.csv", index=False)
else:
    json.dump({"val_error": round(float((pred != yv.numpy()).mean()), 4)}, open(f"{O}/metrics.json", "w"))
'''


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("domain", choices=["compbio", "aiml"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261114)
    a = ap.parse_args()
    (build_compbio if a.domain == "compbio" else build_aiml)(a.out, a.seed)
