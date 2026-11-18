"""Per-claim verdicts for the site: results/<runs>/<model>/<task>.verdicts.json from the re-verification records.

  .venv/bin/python scripts/export_verdicts.py [--runs long]   (safe to rerun as more re-verifications finish)
"""

import argparse
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEEP_HIDDEN = ("metric", "n", "a", "b", "diff", "ci95", "ci90", "bounds", "min_effect", "pass")


def claim(a):
    return {"id": a["id"], "kind": a["kind"], "agent": a["agent"], "title": a["title"], "status": a["status"],
            "reason": a.get("reason", ""), "frontier": a.get("frontier"), "fabrication": a.get("fabrication", False),
            "checks": a.get("checks", {}), "duplicate_of": a.get("duplicate_of"),
            "hidden": {k: v for k, v in (a.get("hidden") or {}).items() if k in KEEP_HIDDEN} or None,
            "relations": [{"target": v["target"], "relation": v["relation"], "cross_agent": v["cross_agent"],
                           "verified": v["verified"], "reason": v.get("reason", ""),
                           "ablation_material": (v.get("ablation") or {}).get("material")} for v in a.get("relations", [])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="main", choices=["main", "long"])
    runs = ap.parse_args().runs
    n = 0
    for f in sorted(glob.glob(os.path.join(ROOT, f"results/{runs}/*/reverify/*.json"))):
        d = json.load(open(f))
        v, s = d["verification"], d["verification"]["summary"]
        out = {"run_id": v["run_id"], "model": d["model"], "task": d["task"], "domain": v["domain"],
               "source": "re-verification on Modal in the task's own verifier image (scripts/reverify_modal.py)",
               "summary": {k: s[k] for k in ("R_replicated", "R_frontier", "C_collaborative", "submitted_credited",
                                             "duplicates", "best_gain", "rejected")},
               "rewards": d["rewards"], "claims": [claim(a) for a in v["artifacts"]]}
        json.dump(out, open(os.path.join(ROOT, f"results/{runs}/{d['model']}/{d['task']}.verdicts.json"), "w"), indent=1, default=str)
        n += 1
    print(f"wrote {n} verdict files")


if __name__ == "__main__":
    main()
