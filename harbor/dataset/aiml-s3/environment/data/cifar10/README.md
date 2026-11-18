# cifar10

CIFAR-10 (Krizhevsky 2009): 32x32 colour images, 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

- train.npz: x (50000, 32, 32, 3) uint8, y (50000,) int labels 0-9, id
- The evaluator holds the 10,000 official test images; at verification they are at $EVAL_DIR/cifar10/inputs.npz (x, id). Predictions: integer class ids.
- Metric: error rate. Reference for `prediction` claims: 6% error (airbench94, Jordan 2024, arXiv 2404.00498: 94% accuracy in seconds on one A100).
