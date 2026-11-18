"""Merge per-claim verdicts into the main-run replays. Safe to rerun as more verdict files land.

Reads results/main/<model>/<task>.verdicts.json and updates site/replays/main-<model>-<domain>-<team|solo>-<seed>.json:
artifact status and reason, frontier and hidden-data fields, relation verified flags, and verdict events after
the session so the replay viewer can play verification. Then rebuilds site/replays/index.json fields.

  python3 site/apply_verdicts.py [--runs long]   (from republic/)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPLAYS = os.path.join(ROOT, "site", "replays")
MODEL = {"luna": "gpt-6-luna", "6-luna": "gpt-6-luna", "6.1-sol": "gpt-6-1-sol", "opus": "claude-opus-5-5", "haiku": "claude-haiku-5-5"}
STATUS = {"accepted": "accepted", "rejected": "rejected", "duplicate": "superseded",
          "pending auditor": "pending", "not credited": "uncredited"}


RUNS = "main"


def apply(vpath: str) -> str | None:
    folder = os.path.basename(os.path.dirname(vpath))
    task = os.path.basename(vpath)[: -len(".verdicts.json")]
    m = re.match(r"([a-z]+)-(solo-)?s(\d+)$", task)
    if folder not in MODEL or not m:
        return None
    dom, solo, seed = m.groups()
    rpath = os.path.join(REPLAYS, f"{RUNS}-{MODEL[folder]}-{dom}-{'solo' if solo else 'team'}-{seed}.json")
    if not os.path.exists(rpath):
        return None
    V = json.load(open(vpath))
    R = json.load(open(rpath))
    by = {c["id"]: c for c in V.get("claims", [])}
    for a in R["artifacts"]:
        v = by.get(a["id"])
        if not v:
            continue
        a["status"] = STATUS.get(v.get("status"), "pending")
        a["reason"] = f"{v.get('status')}: {v.get('reason', '')}"
        a["frontier"] = bool(v.get("frontier"))
        a["hidden"] = v.get("hidden")
        a["checks"] = v.get("checks", a.get("checks", {}))
        vr = {(x["target"], x["relation"]): x for x in v.get("relations", [])}
        for r in a.get("relations", []):
            x = vr.get((r["target"], r["relation"]))
            if x:
                r["verified"], r["reason"] = bool(x.get("verified")), x.get("reason", "")
    dur = R["meta"].get("duration") or 0
    R["events"] = [e for e in R["events"] if e.get("type") != "verdict"]
    judged = [a for a in sorted(R["artifacts"], key=lambda a: a["t"]) if a["id"] in by]
    for i, a in enumerate(judged):
        R["events"].append({"t": round(dur + 3 + i * 2, 2), "type": "verdict", "id": a["id"], "status": a["status"], "agent": a["agent"]})
    s = V.get("summary", {})
    R["meta"].update(verdicts="per-claim", verified=s.get("R_replicated"), submitted=s.get("submitted_credited"),
                     frontier=s.get("R_frontier"), collaborative=s.get("C_collaborative"))
    R["meta"]["note"] = (f"{'6-hour run' if RUNS == 'long' else 'Main run, 60 minutes'}. The verifier accepted {s.get('R_replicated')} of {s.get('submitted_credited')} credited claims; "
                         f"{s.get('R_frontier')} set a new best held-out result on their dataset. Verdicts from the re-run of the official verifier.")
    R["scores"] = V.get("rewards", R.get("scores"))
    json.dump(R, open(rpath, "w"), separators=(",", ":"))
    return os.path.basename(rpath)


def infra_failed() -> dict:
    """Tasks reported as infrastructure failures in results/main_summary.json: (model folder, task) -> reason."""
    path = os.path.join(ROOT, "results", "main_summary.json")
    if not os.path.exists(path):
        return {}
    folders = {v: k for k, v in MODEL.items()}
    out = {}
    for x in json.load(open(path)).get("infra_failed", []):
        key = x["model"].replace("gpt-6.1-sol", "gpt-6-1-sol")
        out[(folders.get(key, x["model"]), x["task"])] = x.get("reason", "infrastructure failure")
    return out


def main():
    global RUNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="main", choices=["main", "long"])
    RUNS = ap.parse_args().runs
    failed = infra_failed() if RUNS == "main" else {}
    done = []
    for p in sorted(glob.glob(os.path.join(ROOT, "results", RUNS, "*", "*.verdicts.json"))):
        folder, task = os.path.basename(os.path.dirname(p)), os.path.basename(p)[: -len(".verdicts.json")]
        if (folder, task) in failed:
            continue
        x = apply(p)
        if x:
            done.append(x)
    for (folder, task), reason in failed.items():  # keep the replay, label it, and keep it out of the results
        m = re.match(r"([a-z]+)-(solo-)?s(\d+)$", task)
        rpath = os.path.join(REPLAYS, f"main-{MODEL.get(folder, folder)}-{m.group(1)}-{'solo' if m.group(2) else 'team'}-{m.group(3)}.json") if m else ""
        if os.path.exists(rpath):
            R = json.load(open(rpath))
            R["meta"].update(verdicts="infra-failed", verified=None, submitted=None, infra_failed=reason,
                             note=f"Infrastructure failure: {reason}. This run is excluded from all results.")
            json.dump(R, open(rpath, "w"), separators=(",", ":"))
    idx_path = os.path.join(REPLAYS, "index.json")
    idx = json.load(open(idx_path))
    for e in idx:
        if e["file"] in done:
            meta = json.load(open(os.path.join(REPLAYS, e["file"])))["meta"]
            e.update(verified=meta.get("verified"), verdicts="per-claim")
    json.dump(idx, open(idx_path, "w"), indent=1)
    print(f"applied verdicts to {len(done)} replays")


if __name__ == "__main__":
    main()
