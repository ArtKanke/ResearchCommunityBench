"""Build site/data/results.json, the one file the results site reads.

Inputs (read only):
  results/main_summary.json   main run, per model x domain x mode rows (mean, se over seeds), when it exists
  results/main/*/*.task.json  main run, per-task Valkyrie results; used (as preliminary) until main_summary.json exists
  results/p10v2_pilot.json    pilot v2: 10 min, 3 seeds, per-task rows
  results/p10v3_pilot.json    pilot v3: 10 min, 1 seed, per-task rows

Output schema ("rcb-results/1"):
  datasets: [{id, label, status: main|pilot, minutes, seeds, rows: [row]}]
  row: {model, domain, mode: team|solo, n, m: {metric: {mean, se}}}

main_summary.json is read tolerantly: a list of rows (or {"rows": [...]}) where each metric is
either {"mean": x, "se": y}, a pair of keys "<metric>_mean" / "<metric>_se", or a bare number.

  python3 site/build_data.py            (from republic/)
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "site", "data", "results.json")

# metric key in the site -> key in the per-task pilot rows
METRICS = {
    "R": "R_replicated",
    "C": "C_collaborative",
    "submitted": "submitted_credited",
    "not_replicated": "not_replicated",
    "not_reproducible": "not_reproducible",
    "fabrication": "fabrication",
    "duplicates": "duplicates",
    "cost": "cost_usd",
    "agents": "agents_total",
    "agents_peak": "agents_peak",
    "depth": "max_depth",
    "messages": "messages",
    "calls": "model_calls",
    "experiments": "experiments",
    "consumed": "accepted_consumed_by_others",
    "built_on_failed": "built_on_failed_work",
    "replicated_failed": "replications_of_failed_work",
    "entropy": "dataset_entropy",
    "session_share": "session_share",
    "frontier": "R_frontier",
    "datasets_advanced": "datasets_advanced",
    "best_gain": "mean_best_gain",
}

MODELS = {
    "gpt-6.1-sol": ("GPT-6.1 Sol", "OpenAI"), "sol": ("GPT-6.1 Sol", "OpenAI"),
    "gpt-6-astra": ("GPT-6 Astra", "OpenAI"), "astra": ("GPT-6 Astra", "OpenAI"),
    "gpt-6-luna": ("GPT-6 Luna", "OpenAI"), "luna": ("GPT-6 Luna", "OpenAI"), "6-luna": ("GPT-6 Luna", "OpenAI"),
    "6.1-sol": ("GPT-6.1 Sol", "OpenAI"), "opus": ("Claude Opus 5.5", "Anthropic"), "haiku": ("Claude Haiku 5.5", "Anthropic"),
    "claude-opus-5-5": ("Claude Opus 5.5", "Anthropic"), "opus-5-5": ("Claude Opus 5.5", "Anthropic"),
    "claude-haiku-5-5": ("Claude Haiku 5.5", "Anthropic"), "haiku-5-5": ("Claude Haiku 5.5", "Anthropic"),
}
DOMAINS = {"materials": "Materials science", "compbio": "Computational biology", "aiml": "AI/ML on GPU"}


def model_key(m: str) -> str:
    m = m.split("/")[-1].lower()
    name = MODELS.get(m, (m, ""))[0]
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def parse_task(task: str):
    m = re.match(r"([a-z]+)-(solo-)?s(\d+)", task)
    return (m.group(1), "solo" if m.group(2) else "team", int(m.group(3))) if m else (None, None, None)


def mean_se(xs):
    n = len(xs)
    if not n:
        return None
    mu = sum(xs) / n
    se = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1) / n) if n > 1 else None
    return {"mean": round(mu, 4), "se": None if se is None else round(se, 4)}


def from_pilot(path, ds_id, label, minutes):
    rows = json.load(open(path))
    groups = defaultdict(list)
    for r in rows:
        if r.get("status") != "FINISHED":
            continue
        dom, mode, seed = parse_task(r["task"])
        if dom:
            groups[(model_key(r["model"]), dom, mode)].append(r)
    out = []
    for (mk, dom, mode), rs in sorted(groups.items()):
        m = {k: mean_se([float(r[src]) for r in rs if src in r]) for k, src in METRICS.items()}
        out.append({"model": mk, "domain": dom, "mode": mode, "n": len(rs), "m": {k: v for k, v in m.items() if v}})
    seeds = max((r["n"] for r in out), default=0)
    return {"id": ds_id, "label": label, "status": "pilot", "minutes": minutes, "seeds": seeds,
            "source": os.path.relpath(path, ROOT), "rows": out}


def domain_key(d: str) -> str:
    d = str(d).lower()
    return "materials" if "mat" in d else "compbio" if "bio" in d else "aiml" if ("ml" in d or "ai" in d or "gpu" in d) else d


def _metric(row, key):
    v = row.get(key)
    if isinstance(v, dict) and "mean" in v:
        return {"mean": v["mean"], "se": v.get("se")}
    if isinstance(v, (int, float)):
        return {"mean": v, "se": row.get(key + "_se")}
    if key + "_mean" in row:
        return {"mean": row[key + "_mean"], "se": row.get(key + "_se")}
    return None


ALIASES = {"session_share": ["session_used"]}


def from_main(path):
    raw = json.load(open(path))
    rows = raw.get("rows", raw) if isinstance(raw, dict) else raw
    out = []
    for r in rows:
        mode = "solo" if str(r.get("mode", "")).lower() in ("solo", "single") else "team"
        m = {}
        for k, src in METRICS.items():
            v = _metric(r, src) or _metric(r, k) or next((_metric(r, x) for x in ALIASES.get(k, []) if _metric(r, x)), None)
            if v and v["mean"] is not None:
                m[k] = v
        out.append({"model": model_key(r["model"]), "domain": domain_key(r["domain"]), "mode": mode,
                    "n": r.get("n", r.get("seeds", 1)), "m": m})
    ds = {"id": "main", "label": "Main run", "status": "main", "minutes": 60, "preliminary": False,
          "seeds": max((r["n"] for r in out), default=0), "source": "results/main_summary.json", "rows": out}
    if isinstance(raw, dict):
        ds["runs"] = []
        for t in raw.get("tasks", []):
            run = {"model": model_key(t["model"]), "domain": domain_key(t["domain"]), "mode": "solo" if t.get("mode") == "solo" else "team", "seed": t.get("seed")}
            for k, src in METRICS.items():
                for name in [src, k, *ALIASES.get(k, [])]:
                    if t.get(name) is not None:
                        run[k] = t[name]
                        break
            ds["runs"].append(run)
        ds["errored"] = [f"{model_key(x['model'])}/{x['task']}" for x in raw.get("infra_failed", [])]
        ds["infra_failed"] = raw.get("infra_failed", [])
        ds["conditions"] = raw.get("conditions")
    return ds


def from_main_tasks(folder, ds_id="main", label="Main run", minutes=60):
    """Per-task results of a run set, aggregated like the pilot. session_share is agent time / session length, capped at 1."""
    rows, with_verdicts = [], 0
    tasks = sorted(glob.glob(os.path.join(folder, "*", "*.task.json")))
    # re-verified numbers replace Valkyrie's only once every task has them, so one run set never mixes the two
    rescored = all(os.path.exists(f.replace(".task.json", ".verdicts.json")) for f in tasks)
    for f in tasks:
        d = json.load(open(f))
        r = dict((d.get("evaluation_result") or {}).get("rewards") or {})
        vf = f.replace(".task.json", ".verdicts.json")
        if rescored and os.path.exists(vf):  # per-claim re-verification: its rewards are the final numbers
            r.update(json.load(open(vf)).get("rewards") or {})
            with_verdicts += 1
        elif d.get("status") != "FINISHED" or not r:
            continue
        dur = (d.get("task_breakdown") or {}).get("agent_run_duration")
        if dur:
            r["session_share"] = min(1.0, dur / (minutes * 60))
        r.update(model=os.path.basename(os.path.dirname(f)), task=d["task_id"], status="FINISHED")
        rows.append(r)
    tmp = os.path.join(os.path.dirname(OUT), ".main_tasks.json")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    json.dump(rows, open(tmp, "w"))
    ds = from_pilot(tmp, ds_id, label, minutes)
    os.remove(tmp)
    n_tasks = len(glob.glob(os.path.join(folder, "*", "*.task.json")))
    rel = os.path.relpath(folder, ROOT)
    ds.update(status="main", preliminary=with_verdicts < n_tasks, verdicts=with_verdicts, tasks=n_tasks,
              source=f"{rel}/*/*.verdicts.json" if with_verdicts == n_tasks else f"{rel}/*/*.task.json, with verdicts where re-verified")
    ds["runs"] = []
    for r in rows:
        dom, mode, seed = parse_task(r["task"])
        ds["runs"].append({"model": model_key(r["model"]), "domain": dom, "mode": mode, "seed": seed,
                           **{k: r.get(src) for k, src in METRICS.items() if r.get(src) is not None}})
    ds["errored"] = [f"{os.path.basename(os.path.dirname(f))}/{os.path.basename(f)[:-10]}" for f in sorted(glob.glob(os.path.join(folder, "*", "*.task.json")))
                     if json.load(open(f)).get("status") != "FINISHED" and not os.path.exists(f.replace(".task.json", ".verdicts.json"))]
    return ds


def main():
    datasets = []
    main_path = os.path.join(RES, "main_summary.json")
    if os.path.exists(main_path):
        datasets.append(from_main(main_path))
    elif glob.glob(os.path.join(RES, "main", "*", "*.task.json")):
        datasets.append(from_main_tasks(os.path.join(RES, "main")))
    # 6-hour runs (Luna and Haiku, one seed): Valkyrie's rewards, replaced by per-claim verdicts when they exist
    if glob.glob(os.path.join(RES, "long", "*", "*.task.json")):
        datasets.append(from_main_tasks(os.path.join(RES, "long"), "long", "6-hour runs", 360))
    # pilots stay in results/ for reference
    models = {}
    for d in datasets:
        for r in d["rows"]:
            for raw, (name, lab) in MODELS.items():
                if model_key(raw) == r["model"]:
                    models[r["model"]] = {"key": r["model"], "name": name, "lab": lab}
    doc = {"format": "rcb-results/1", "generated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
           "models": list(models.values()), "domains": [{"key": k, "name": v} for k, v in DOMAINS.items()],
           "datasets": datasets}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(doc, open(OUT, "w"), indent=1)
    print(f"wrote {os.path.relpath(OUT, ROOT)}: " + ", ".join(f"{d['id']} ({len(d['rows'])} rows)" for d in datasets))


if __name__ == "__main__":
    main()
