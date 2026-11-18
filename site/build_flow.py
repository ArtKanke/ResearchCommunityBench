"""Build site/data/flow.json from the main-run replays in site/replays/main-*.json.

Per run: cumulative claims and build-ons per minute (for the productivity curves) and lineage statistics
(claims that build on another researcher's claim, the longest chain of claims each building on the last).
A "build-on" is a declared relation (extends, replicates, corrects, refutes) to an earlier claim; "cites" does
not count. Declared is not verified: the verifier's ablation decides which build-ons were real (C).

  python3 site/build_flow.py            (from republic/)
"""

from __future__ import annotations

import glob
import json
import os
from functools import lru_cache

SITE = os.path.dirname(os.path.abspath(__file__))
CREDITED = {"finding", "null_result", "method", "hypothesis"}
BUILD = {"extends", "replicates", "corrects", "refutes"}
MINUTES = 70


def run_stats(path: str, minutes: int = MINUTES) -> dict:
    R = json.load(open(path))
    m = R["meta"]
    arts = {a["id"]: a for a in R["artifacts"]}
    claims = [a for a in R["artifacts"] if a["kind"] in CREDITED]
    edges = []  # (source claim, dependent claim, relation, across researchers)
    for a in R["artifacts"]:
        for r in a.get("relations", []):
            t = arts.get(r["target"])
            if t is not None and r["relation"] in BUILD:
                edges.append((t["id"], a["id"], r["relation"], t["agent"] != a["agent"]))
    parents = {}
    for s, d, _, _ in edges:
        parents.setdefault(d, []).append(s)

    @lru_cache(maxsize=None)
    def depth(i, seen=()):
        if i in seen:
            return 0
        return 1 + max((depth(p, seen + (i,)) for p in parents.get(i, [])), default=0)

    cum = lambda ts: [sum(1 for t in ts if t <= k * 60) for k in range(minutes + 1)]
    cross = [e for e in edges if e[3]]
    verified_known = m.get("verdicts") == "per-claim"
    vcross = [e for e in edges if e[3] and any(r.get("verified") for r in arts[e[1]].get("relations", []) if r["target"] == e[0])]
    return {
        "file": os.path.basename(path), "model": m.get("model"), "domain": m.get("domain"), "mode": m.get("mode"),
        "seed": m.get("seed"), "minutes": round(m.get("duration", 0) / 60, 1),
        "researchers": len(R["agents"]), "roles": [a.get("role", "") for a in R["agents"] if a["id"] != "founder"], "claims": len(claims), "verified": m.get("verified"), "submitted": m.get("submitted"),
        "buildons": len(edges), "cross": len(cross),
        "claims_building_on_colleague": len({d for _, d, _, x in edges if x}),
        "longest_chain": max((depth(c["id"]) for c in claims), default=0),
        "by_relation": {r: sum(1 for e in cross if e[2] == r) for r in sorted(BUILD)},
        "curve_claims": cum([c["t"] for c in claims]),
        "curve_verified": cum([c["t"] for c in claims if c["status"] == "accepted"]) if verified_known else None,
        "curve_cross": cum([arts[d]["t"] for _, d, _, x in edges if x]),
        "verdicts": verified_known, "infra_failed": bool(m.get("infra_failed")),
        "recruited": len(R["agents"]) - 1,
        "session": m.get("session_minutes", 60),
        # when the founder (lead agent) ended its own session, in minutes; None if it ran to the clock
        "founder_end": next((round(a["t_end"] / 60, 1) for a in R["agents"] if a["id"] == "founder" and a.get("t_end")), None),
        # confirmed collaborations by kind: credited claims whose cross-researcher relation the verifier confirmed
        "c_by_rel": {rel: len({a["id"] for a in claims for r in a.get("relations", []) if r.get("verified") and r["relation"] == rel
                               and arts.get(r["target"]) and arts[r["target"]]["agent"] != a["agent"]}) for rel in sorted(BUILD)} if verified_known else None, "frontier": m.get("frontier"), "cross_verified": len(vcross) if verified_known else None,
        "accepted": sum(1 for c in claims if c["status"] == "accepted") if verified_known else None,
    }


def funnel(runs: str = "main") -> dict | None:
    """Cross-researcher build-ons in community runs, from the verdicts (read only): claimed, both sides verified,
    testable by ablation (a consumed data table to shuffle), confirmed; and challenges (corrects/refutes)."""
    root = os.path.dirname(SITE)
    files = glob.glob(os.path.join(root, "results", runs, "*", "*.verdicts.json"))
    if not files:
        return None
    summary = os.path.join(root, "results", "main_summary.json")
    bad = {(x["model"], x["task"]) for x in json.load(open(summary)).get("infra_failed", [])} if os.path.exists(summary) else set()
    f = dict(runs=0, claimed=0, both_verified=0, testable=0, confirmed=0, challenges=0, challenges_confirmed=0)
    for path in files:
        d = json.load(open(path))
        if "solo" in d["task"] or any(m.endswith(d["model"]) and t == d["task"] for m, t in bad):
            continue
        f["runs"] += 1
        status = {c["id"]: c["status"] for c in d["claims"]}
        for c in d["claims"]:
            for r in c.get("relations", []):
                if not r.get("cross_agent"):
                    continue
                if r["relation"] == "extends":
                    f["claimed"] += 1
                    if c["status"] == "accepted" and status.get(r["target"]) == "accepted":
                        f["both_verified"] += 1
                        f["testable"] += r.get("ablation_material") is not None
                        f["confirmed"] += bool(r.get("verified"))
                elif r["relation"] in ("corrects", "refutes"):
                    f["challenges"] += 1
                    f["challenges_confirmed"] += bool(r.get("verified"))
    return f


def main():
    runs = [run_stats(p) for p in sorted(glob.glob(os.path.join(SITE, "replays", "main-*.json")))]
    out = os.path.join(SITE, "data", "flow.json")
    json.dump({"format": "rcb-flow/1", "minutes": MINUTES, "runs": runs, "funnel": funnel()}, open(out, "w"), separators=(",", ":"))
    print(f"wrote site/data/flow.json: {len(runs)} runs")
    # the 6-hour runs: same statistics on a 370-minute clock
    long = [run_stats(p, 370) for p in sorted(glob.glob(os.path.join(SITE, "replays", "long-*.json")))]
    if long:
        out = os.path.join(SITE, "data", "flow_long.json")
        json.dump({"format": "rcb-flow/1", "minutes": 370, "session": 360, "runs": long, "funnel": funnel("long")},
                  open(out, "w"), separators=(",", ":"))
        print(f"wrote site/data/flow_long.json: {len(long)} runs")


if __name__ == "__main__":
    main()
