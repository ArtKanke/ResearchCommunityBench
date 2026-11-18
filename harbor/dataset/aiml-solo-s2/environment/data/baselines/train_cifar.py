"""Minimal CIFAR-10 trainer showing the formats (not a strong model).
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
