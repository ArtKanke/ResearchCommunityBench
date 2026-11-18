"""One-hot + ridge baseline (what `prediction` claims must beat)."""
import numpy as np
from sklearn.linear_model import Ridge
AA = "ACDEFGHIKLMNPQRSTVWY"


def onehot(seqs):
    L = max(len(s) for s in seqs)
    X = np.zeros((len(seqs), L * 20), np.float32)
    for i, s in enumerate(seqs):
        for j, a in enumerate(s):
            k = AA.find(a)
            if k >= 0:
                X[i, j * 20 + k] = 1
    return X


def baseline_predict(train, inputs):
    return Ridge(alpha=1.0).fit(onehot(train.sequence.tolist()), train.target).predict(onehot(inputs.sequence.tolist()))
