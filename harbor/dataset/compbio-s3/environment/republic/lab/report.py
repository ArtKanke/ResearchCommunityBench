"""Aggregate finalized runs into the leaderboard data and render the static site.

  .venv/bin/python -m lab.report                 # results/*.json -> site/index.html
  .venv/bin/python -m lab.report --collect URL   # first pull every frozen run from a lab server
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import statistics as st
import urllib.request
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def collect(url, admin_key):
    def get(method, path):
        req = urllib.request.Request(url + path, method=method, headers={"Authorization": f"Bearer {admin_key}"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())

    for rid in get("GET", "/admin/runs")["frozen"]:
        if not os.path.exists(f"{RESULTS}/{rid}.json"):
            json.dump(get("POST", f"/admin/runs/{rid}/finalize"), open(f"{RESULTS}/{rid}.json", "w"), indent=2)
            json.dump(get("GET", f"/admin/runs/{rid}/events"), open(f"{RESULTS}/{rid}.events.json", "w"))


def load_runs(include_mock=False, label=""):
    runs = []
    for f in sorted(glob.glob(f"{RESULTS}/run-*.json")):
        if f.endswith(".events.json"):
            continue
        r = json.load(open(f))
        if r.get("model") in ("mock", "unset") and not include_mock:
            continue
        if label and not str(r.get("label", "")).startswith(label):
            continue
        ev = f.replace(".json", ".events.json")
        r["_events_file"] = ev if os.path.exists(ev) else None
        runs.append(r)
    return runs


def se(xs):
    return st.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else None


def boot_ci(diffs, n=4000, seed=0):
    if len(diffs) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return [means[int(0.025 * n)], means[int(0.975 * n)]]


def aggregate(runs):
    groups = defaultdict(list)
    for r in runs:
        groups[(r["model"], r["mode"])].append(r)
    rows = []
    for (model, mode), rs in groups.items():
        ri = [r["republic_index"] for r in rs]
        claims = sum(r["lab"]["preprint_claims"] for r in rs)
        backed = sum((r["lab"]["integrity_ppv"] or 0) * r["lab"]["preprint_claims"] for r in rs)
        subs = [p for r in rs for p in r["problems"] if p["claim_s"] is not None]
        rows.append({
            "model": model, "mode": mode, "n": len(rs),
            "ri": st.mean(ri), "ri_se": se(ri), "ri_runs": ri,
            "vp": st.mean(r["verified_productivity"] for r in rs),
            "ppv": backed / claims if claims else None, "claims": claims,
            "overclaim": sum(1 for p in subs if p["claim_s"] - p["s"] > 0.05) / len(subs) if subs else None,
            "coverage": sum(1 for p in subs if p["s"] >= p["claim_s"] - 1e-9) / len(subs) if subs else None,
            "records": sum(r["records"] for r in rs),
            "attempted": st.mean(r["attempted"] for r in rs),
            "agents_peak": st.mean(r["lab"]["agents_peak"] for r in rs),
            "agents_total": st.mean(r["lab"]["agents_total"] for r in rs),
            "cost": st.mean(r["lab"]["cost_usd"] for r in rs),
            "tokens": st.mean(r["lab"]["tokens_in"] + r["lab"]["tokens_out"] for r in rs),
            "duration": st.mean(r["lab"]["duration_s"] for r in rs),
            "entropy": st.mean(r["lab"]["allocation_entropy"] for r in rs),
            "domain_credit": {d: st.mean(r["domain_credit"].get(d, 0) for r in rs) for d in rs[0]["domain_credit"]},
        })
    gains = []
    for model in {m for m, _ in groups}:
        comm = {r["seed"]: r["republic_index"] for r in groups.get((model, "community"), [])}
        solo = {r["seed"]: r["republic_index"] for r in groups.get((model, "solo"), [])}
        paired = [comm[s] - solo[s] for s in comm if s in solo]
        if paired:
            gains.append({"model": model, "gain": st.mean(paired), "ci": boot_ci(paired), "n": len(paired)})
    rows.sort(key=lambda x: -x["ri"])
    problems = defaultdict(lambda: defaultdict(list))
    claims = []
    for r in runs:
        for p in r["problems"]:
            problems[p["problem_id"]][f"{r['model']}|{r['mode']}"].append(p["s"])
            if p["claim_s"] is not None:
                claims.append({"model": r["model"], "mode": r["mode"], "problem": p["problem_id"],
                               "claim": p["claim_s"], "s": p["s"]})
    meta = {}
    for r in runs:
        for p in r["problems"]:
            meta[p["problem_id"]] = {"title": p["title"], "domain": p["domain"], "kind": p["kind"],
                                     "ref_kind": p["ref_kind"], "metric": p["metric"]}
    heat = {pid: {k: st.mean(v) for k, v in d.items()} for pid, d in problems.items()}
    return {"rows": rows, "gains": gains, "heat": heat, "problem_meta": meta, "claims": claims}


def exhibit(run):
    """A compact replay of one institute: agents, timeline, transcript, preprints."""
    if not run.get("_events_file"):
        return None
    ev = json.load(open(run["_events_file"]))
    t0 = min(e["ts"] for e in ev["events"]) if ev["events"] else 0
    steps = []
    for e in ev["events"]:
        d = json.loads(e["data"])
        item = {"t": round(e["ts"] - t0, 1), "agent": e["agent"], "type": e["type"], "problem": e["problem"]}
        if e["type"] == "complete":
            item["text"] = d.get("text", "")[:1500]
            item["cost"] = d.get("cost")
        elif e["type"] in ("eval", "submit", "spawn", "publish", "post", "oracle", "exit"):
            item["detail"] = {k: v for k, v in d.items() if k not in ("result",)}
            if "query" in item["detail"]:
                item["detail"]["query"] = json.dumps(item["detail"]["query"])[:200]
        else:
            continue
        steps.append(item)
    return {"run_id": run["run_id"], "model": run["model"], "mode": run["mode"], "seed": run["seed"],
            "ri": run["republic_index"], "agents": run["lab"]["agents"], "steps": steps,
            "preprints": [{"id": p["id"], "agent": p["agent"], "title": p["title"], "body": p["body"][:1500],
                           "claims": json.loads(p["claims"])} for p in ev["preprints"]],
            "problems": run["problems"]}


def render(include_mock=False, out=None, label="", note=""):
    runs = load_runs(include_mock, label)
    data = aggregate(runs) if runs else {"rows": [], "gains": [], "heat": {}, "problem_meta": {}, "claims": []}
    ex = [x for x in (exhibit(r) for r in sorted(runs, key=lambda r: -r["republic_index"])) if x][:6]
    data["exhibits"] = ex
    data["n_runs"] = len(runs)
    data["note"] = note
    tpl = open(os.path.join(ROOT, "site", "template.html")).read()
    html = tpl.replace("/*__DATA__*/null", json.dumps(data, default=str))
    out = out or os.path.join(ROOT, "site", "index.html")
    open(out, "w").write(html)
    print(f"{len(runs)} runs -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", help="lab server URL to pull frozen runs from")
    ap.add_argument("--include-mock", action="store_true")
    ap.add_argument("--label", default="", help="only runs whose label starts with this")
    ap.add_argument("--note", default="", help="data provenance banner shown under the header")
    a = ap.parse_args()
    if a.collect:
        collect(a.collect, os.environ.get("LAB_ADMIN_KEY", "dev-admin"))
    render(a.include_mock, label=a.label, note=a.note)
