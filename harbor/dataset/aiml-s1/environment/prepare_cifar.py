"""Download CIFAR-10 at image build time and keep only the training batches (test labels never enter the image)."""
import io, os, pickle, sys, tarfile, urllib.request
import numpy as np
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
data = urllib.request.urlopen("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", timeout=600).read()
tf = tarfile.open(fileobj=io.BytesIO(data))
xs, ys = [], []
for i in range(1, 6):
    d = pickle.load(tf.extractfile(f"cifar-10-batches-py/data_batch_{i}"), encoding="bytes")
    xs.append(d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)); ys.append(np.array(d[b"labels"]))
x, y = np.concatenate(xs), np.concatenate(ys)
np.savez_compressed(f"{out}/train.npz", x=x, y=y, id=np.array([f"cifar-tr-{i:05d}" for i in range(len(x))]))
print("cifar10 train", x.shape)
