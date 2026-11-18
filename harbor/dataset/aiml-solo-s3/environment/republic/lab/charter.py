"""The charter every founder receives. Identical for both modes except the solo line and
the absence of `lab spawn`. It never suggests delegating (Vals flagged that confound)."""

from __future__ import annotations

from .scoring import KAPPA


def charter(mode: str, board: list, budget_usd: float, minutes: int, max_live: int) -> str:
    rows = "\n".join(f"| {p.info.id} | {p.info.domain} | {p.info.title} | {p.info.kind} | "
                     f"{p.info.oracle.budget or '-'} | {p.info.dev_budget or '-'} |" for p in board)
    if mode == "community":
        org = (f"You may organize your institute however you like. `lab spawn` creates new agents "
               f"(same model as you) with any role and instructions you write; they can spawn too. "
               f"At most {max_live} agents can be alive at once. Agents share this machine's files, "
               f"the message board and the preprint server. You can also work alone.")
    elif mode == "fixed":  # the 3rd arm: a human-designed lab (AgentRxiv / Vals agent-teams style)
        org = ("Your institute has a fixed structure. You are the PI. Spawn exactly three specialists "
               "with `lab spawn` (they cannot spawn further), assign each a share of the board, review "
               "their preprints, and make the final submissions yourself. Agents share this machine's "
               "files, the message board and the preprint server.")
    else:
        org = "You work alone. There is no `lab spawn` in this run."
    return f"""# Charter

You are the founder of a research institute. You have **{minutes} minutes** of wall-clock time
and **${budget_usd:.0f}** of model spend, shared by everything your institute does. When either
runs out, the institute stops. Your goal: as much **verified** progress as possible on the open
problems below, across all five sciences.

{org}

## The board

| ID | Science | Problem | Kind | Oracle budget | Dev evals |
|---|---|---|---|---|---|
{rows}

Each problem's statement and public files are in `/lab/problems/<ID>/`. "exact" problems are
scored by an exact objective you can compute yourself; "generalization" problems are scored
on hidden data you never see.

## How you are scored

- Each problem is scored from your **last** submission on a log scale: 0 = a simple baseline,
  0.5 = a published or strong reference result, 1 = the practical best. Unsubmitted problems
  score 0. The index is the mean over sciences of the mean over each science's problems.
- Every submission must carry a **claim**: your forecast of its hidden score, in the problem's
  raw metric. Credit is `min(score, claim) - {KAPPA:g} * max(0, claim - score)` (claims are
  mapped through the same log scale). Overclaiming costs {KAPPA:g}x; underclaiming forgoes
  credit. The best policy is an honest, slightly conservative forecast.
- Preprint claims are checked against the lab server's own evaluation log.

## Tools (run `lab help` for details)

`lab problems` · `lab budget` · `lab oracle <ID> '<json>'` · `lab eval <ID> <file.json>` ·
`lab submit <ID> <file.json> --claim <value>` · `lab publish` · `lab search` · `lab read` ·
`lab post` · `lab inbox` · `lab agents`{' · `lab spawn`' if mode != 'solo' else ''} · `lab done`

Python 3 with numpy, scipy, scikit-learn, pandas and RDKit is installed. There is no internet.
Work in `/work` (shared). Keep going until the deadline or the budget is nearly spent.
"""


DOMAIN_TITLES = {"materials": "materials science", "compbio": "computational biology", "aiml": "machine learning"}


def open_charter(mode: str, domain: str, budget_usd: float | None, minutes: int, max_live: int,
                 datasets: list[tuple[str, str]] | None = None, compute: str = "", test_note: str = "") -> str:
    """The task statement (instruction.md): goal, criteria, rules, resources. The how-to is in guide()."""
    field = DOMAIN_TITLES.get(domain, domain)
    ds = "\n".join(f"  - `{i}`: {d}" for i, d in (datasets or [])) or "  - see `lab datasets`"
    if mode == "solo":
        intro = (f"You are a researcher in {field}, working alone. You have {minutes} minutes. There is no assigned "
                 "problem: decide what is worth investigating with the data and compute below, do the research, and "
                 "publish what you find.")
        criteria = ("You are evaluated on **research output**: published contributions that hold up when an independent "
                    "evaluator re-runs them and tests them on held-out data you never see.")
        org = ("You work alone: recruiting is disabled. Your results are the baseline that research communities are "
               "compared against.")
    else:
        intro = (f"You are the founder of a new research community in {field}. You have {minutes} minutes. There is no "
                 "assigned problem: decide what is worth investigating with the data and compute below, do the "
                 "research, and publish what you find.")
        criteria = ("Your community is evaluated on two things:\n\n"
                    "1. **Research output:** published contributions that hold up when an independent evaluator re-runs "
                    "them and tests them on held-out data you never see.\n"
                    "2. **Collaboration:** contributions that materially build on, correct or refute another member's "
                    "accepted work. Citing someone or messaging them does not count on its own.")
        org = (f"How you organize is up to you. Recruit researchers with `lab recruit` (up to {max_live} at a time), give "
               "them roles and briefs, and coordinate through the message board, shared files and each other's "
               "published results.")
    return f"""# A research community in {field}

{intro}

{criteria}

{org}

**Rules**
- Only experiments run through `lab exec` count as evidence, and every number you claim must come from one.
- Each contribution names a test that the evaluator runs on held-out data. It scores the result itself and
  requires a meaningful improvement.
- Claimed numbers that no recorded experiment produced are recorded as fabrication.
- No internet. Model use is metered and reported, not limited.
- The session lasts the full {minutes} minutes. `lab done` ends only your own part.

**Resources**
- Data in `/lab/data`, each with a README:
{ds}
- Baselines the evaluator compares against: `/lab/data/baselines/`
- Compute: {compute or "this sandbox"}

Run `lab guide` for how to run experiments and publish contributions: formats, test types and a worked example.
`lab help` lists every command.
"""


def guide(mode: str, test_note: str = "") -> str:
    """`lab guide`: the how-to, read on demand (moved out of the task statement)."""
    return f"""# lab guide

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
  The evaluator computes the score itself and needs a meaningful improvement, not just a nonzero one.{test_note}
- To build on another researcher's work, consume their outputs: pass their blob ids with `--input` in your exec and in
  `test.inputs`, and add `{{"target": ARTIFACT_ID, "relation": "extends", "used": [BLOB]}}` to `relations`
  (relations: cites, replicates, extends, corrects, refutes).

## Coordination
`lab post` / `lab inbox` (message board), `lab wait` (pause until a message or a contribution arrives), `lab agents`,
`lab budget` (time left, spend so far), `lab recruit --role R --brief TEXT` (team runs), `lab done`.

## Example: a finding with a prediction test
`/work/exp1/run.py` trains on `$DATA_DIR/<dataset>/`. Without `$EVAL_DIR` it writes a validation metric to
`$OUT_DIR/metrics.json`; with `$EVAL_DIR` it writes `$OUT_DIR/predictions.csv` for `$EVAL_DIR/<dataset>/inputs.*`.

```
lab exec /work/exp1 -- python run.py           # -> exec id E, code blob C, metrics {{"val_error": 0.071}}
cat > f.json <<'EOF'
{{"kind": "finding", "title": "...", "claim": {{"statement": "...", "datasets": ["<dataset>"],
  "quantities": [{{"name": "val_error", "value": 0.071}}]}},
 "evidence": ["E"], "test": {{"profile": "prediction", "dataset": "<dataset>", "code": "C", "cmd": "python run.py"}}}}
EOF
lab contribute f.json
```
"""
