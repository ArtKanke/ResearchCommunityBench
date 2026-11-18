"""results/main_summary.json: one row per model x domain x mode (mean and standard error over seeds), the team/solo
multiples, and per-task rows with Valkyrie run IDs.

Rewards come from the re-verification (scripts/reverify_modal.py: the official verifier plus R_frontier and
best-gain) when present, else from Valkyrie's evaluation. Tasks that failed on infrastructure are listed, not scored.

  .venv/bin/python scripts/build_summary.py
"""

import glob
import json
import math
import os
import sqlite3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = {"6.1-sol": "gpt-6.1-sol", "opus": "claude-opus-5-5", "6-luna": "gpt-6-luna", "haiku": "claude-haiku-5-5"}
INFRA_FAILED = {("6-luna", "materials-s1"): "Valkyrie verifier failed after the 60-minute session (Modal sandbox connection refused); excluded by decision made before rescoring. Rescored offline: 11 accepted, best gain 0.22"}
METRICS = ["R_replicated", "R_frontier", "C_collaborative", "mean_best_gain", "datasets_advanced", "submitted_credited",
           "cost_usd", "agents_total", "messages", "experiments", "model_calls", "session_used",
           "fabrication", "miscited_evidence", "not_reproducible", "not_replicated", "duplicates", "auditor_pending",
           "accepted_consumed_by_others", "built_on_failed_work", "replications_of_failed_work"]


def session_used(store):
    """Share of the wall clock in use: when the last researcher stopped, over the session length."""
    db = sqlite3.connect(os.path.join(store, "lab.sqlite"))
    created, deadline = db.execute("select created, deadline from runs").fetchone()
    last = db.execute("select max(coalesce(ended, ?)) from agents", (deadline,)).fetchone()[0]
    return round(min(1.0, (last - created) / (deadline - created)), 3)


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    m = sum(xs) / len(xs)
    se = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs)) if len(xs) > 1 else None
    return {"mean": round(m, 4), "se": round(se, 4) if se is not None else None, "n": len(xs)}


def main():
    runs = dict(line.split() for line in open(os.path.join(ROOT, "results/main_runs.txt")) if line.strip())
    tasks, failed = [], []
    for key, full in MODELS.items():
        for f in sorted(glob.glob(os.path.join(ROOT, f"results/main/{key}/*.task.json"))):
            t = json.load(open(f))
            tid = t["task_id"]
            row = {"model": full, "task": tid, "domain": tid.split("-")[0], "mode": "solo" if "-solo-" in tid else "team",
                   "seed": int(tid[-1]), "valkyrie_run_id": runs[full], "valkyrie_task_id": t["id"]}
            if (key, tid) in INFRA_FAILED:
                failed.append(row | {"reason": INFRA_FAILED[(key, tid)]})
                continue
            rv = os.path.join(ROOT, f"results/main/{key}/reverify/{tid}.json")
            if os.path.exists(rv):
                rw, row["source"] = json.load(open(rv))["rewards"], "reverify"
            else:
                rw, row["source"] = t["evaluation_result"]["rewards"], "valkyrie"
            row["valkyrie_R_replicated"] = t["evaluation_result"]["rewards"]["R_replicated"]
            store = os.path.join(ROOT, f"results/main/{key}/outputs/{tid}/artifacts/lab/store/store")
            row |= {k: rw.get(k) for k in METRICS if k != "session_used"}
            row["session_used"] = session_used(store) if os.path.exists(os.path.join(store, "lab.sqlite")) else None
            tasks.append(row)

    groups = {}
    for r in tasks:
        groups.setdefault((r["model"], r["domain"], r["mode"]), []).append(r)
    rows = [{"model": m, "domain": d, "mode": mo, "n": len(rs), **{k: stats([r[k] for r in rs]) for k in METRICS}}
            for (m, d, mo), rs in sorted(groups.items())]

    def mean(m, d, mo, k):
        vals = [r[k] for r in groups.get((m, d, mo), [])]
        return sum(vals) / len(vals) if vals and None not in vals else None  # None until every seed has the metric

    multiples = []
    for m in MODELS.values():
        for d in ("materials", "compbio", "aiml"):
            row = {"model": m, "domain": d}
            for k in ("R_replicated", "R_frontier", "mean_best_gain", "cost_usd"):
                a, b = mean(m, d, "team", k), mean(m, d, "solo", k)
                row[f"{k}_team_over_solo"] = round(a / b, 3) if a is not None and b else None
            multiples.append(row)

    out = {"benchmark": "ResearchCommunityBench", "dataset": "researchcommunitybench/main@v1 (verifier of v2)",
           "conditions": "60 minutes, effort high, up to 50 live researchers; Sol and Opus 1 seed, Luna and Haiku 3 seeds",
           "reverified": sum(r["source"] == "reverify" for r in tasks), "scored": len(tasks), "infra_failed": failed,
           "rows": rows, "team_over_solo": multiples, "tasks": tasks}
    json.dump(out, open(os.path.join(ROOT, "results/main_summary.json"), "w"), indent=1)
    print(f"wrote results/main_summary.json: {len(rows)} rows, {len(tasks)} tasks "
          f"({out['reverified']} re-verified), {len(failed)} infra failures")


if __name__ == "__main__":
    main()
