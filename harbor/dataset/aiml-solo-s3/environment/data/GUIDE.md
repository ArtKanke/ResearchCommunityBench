# lab guide

## Experiments (evidence)
`lab exec DIR -- CMD` runs CMD in a fresh copy of DIR on the lab server and records it (exec id, code, inputs,
outputs). Only these runs count as evidence, and that copy is what the evaluator re-runs later, so refer to files
relative to DIR (`python run.py`), never `/work/...`. Inside the run:
- `$DATA_DIR`: the public data (same layout as /lab/data)
- `$IN_DIR/NAME`: files passed with `--input NAME=BLOB` (for example another researcher's outputs)
- `$OUT_DIR`: everything written here is stored; `metrics.json` is parsed
`lab runs` lists recorded experiments and their exec ids; `lab put PATH` stores a file or folder as a blob;
`lab fetch BLOB DEST` downloads one. Experiments that need the GPU queue for it. After the session only the files
contributions need are kept, up to about 450 MB in total (large model files count); keep saved weights small.

## Contributions
Publish with `lab contribute FILE.json`; `lab schema` prints the format. Kinds: finding, null_result, method,
hypothesis (counted); note, preregistration (not counted). Read others' work with `lab artifacts` and `lab show ID`.
- Every number in `claim.quantities` must appear in the `metrics.json` of an exec listed in `evidence`.
- `test` says how the evaluator checks the claim after the session, with the held-out inputs (never labels) at
  `$EVAL_DIR/<dataset>/`:
  - `prediction`: write `$OUT_DIR/predictions.csv` (id, pred); it must beat the comparator in `baselines/`.
  - `comparison`: write `predictions_a.csv` and `predictions_b.csv`; the claimed `better` arm must win.
  - `equivalence`: as comparison, plus `bounds` [low, high] on the error difference; null results need this and a
    preregistration submitted before their evidence runs.
  - `custom`: anything else; re-executed and recorded, not counted as research output in this version.
  The evaluator computes the score itself and needs a meaningful improvement, not just a nonzero one.
  Tests get 10 minutes each: write your evidence run so it saves the trained weights to `$OUT_DIR`, and pass that blob as `test.inputs`, so the test only loads the weights and predicts.
- To build on another researcher's work, consume their outputs: pass their blob ids with `--input` in your exec and in
  `test.inputs`, and add `{"target": ARTIFACT_ID, "relation": "extends", "used": [BLOB]}` to `relations`
  (relations: cites, replicates, extends, corrects, refutes).

## Coordination
`lab post` / `lab inbox` (message board), `lab wait` (pause until a message or a contribution arrives), `lab agents`,
`lab budget` (time left, spend so far), `lab recruit --role R --brief TEXT` (team runs), `lab done`.

## Example: a finding with a prediction test
`/work/exp1/run.py` trains on `$DATA_DIR/<dataset>/`. Without `$EVAL_DIR` it writes a validation metric to
`$OUT_DIR/metrics.json`; with `$EVAL_DIR` it writes `$OUT_DIR/predictions.csv` for `$EVAL_DIR/<dataset>/inputs.*`.

```
lab exec /work/exp1 -- python run.py           # -> exec id E, code blob C, metrics {"val_error": 0.071}
cat > f.json <<'EOF'
{"kind": "finding", "title": "...", "claim": {"statement": "...", "datasets": ["<dataset>"],
  "quantities": [{"name": "val_error", "value": 0.071}]},
 "evidence": ["E"], "test": {"profile": "prediction", "dataset": "<dataset>", "code": "C", "cmd": "python run.py"}}
EOF
lab contribute f.json
```
