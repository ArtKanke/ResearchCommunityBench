# baselines

materials_baseline.py: the evaluator's reference baseline (Magpie-lite composition features + random forest). `python materials_baseline.py <dataset>` prints its 5-fold CV error on train.csv. Claims under the `prediction` profile are compared against this model on the hidden data.
