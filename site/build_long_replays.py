"""Replays of the 6-hour runs: site/replays/long-<model>-<domain>-<team|solo>-1.json from the downloaded stores.

Claim statuses stay "pending" until per-claim verdicts exist (results/long/<model>/<task>.verdicts.json, merged by
apply_verdicts.py --runs long); the run totals come from Valkyrie's verifier (results/long/<model>/<task>.task.json).

  python3 site/build_long_replays.py            (from republic/)
"""

from __future__ import annotations

import glob
import json
import os
import re
import sqlite3
import sys

SITE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SITE)
sys.path.insert(0, ROOT)
from lab.replay import from_db, write_index  # noqa: E402

MODEL = {"luna": ("gpt-6-luna", "GPT-6 Luna"), "haiku": ("claude-haiku-5-5", "Claude Haiku 5.5")}
DOMAIN = {"aiml": "Machine learning", "compbio": "Protein biology", "materials": "Materials science"}


def main():
    n = 0
    for store in sorted(glob.glob(os.path.join(ROOT, "results/long/*/outputs/*/artifacts/lab/store/store/lab.sqlite"))):
        parts = store.split(os.sep)
        folder, task = parts[-8], parts[-6]
        m = re.match(r"([a-z]+)-(solo-)?s(\d+)$", task)
        if folder not in MODEL or not m:
            continue
        dom, solo, seed = m.group(1), bool(m.group(2)), int(m.group(3))
        key, name = MODEL[folder]
        rid = sqlite3.connect(store).execute("select id from runs").fetchone()[0]
        R = from_db(store, rid)
        rw = json.load(open(os.path.join(ROOT, f"results/long/{folder}/{task}.task.json")))["evaluation_result"]["rewards"]
        R["meta"].update(model=key, model_name=name, domain=dom, mode="solo" if solo else "community", seed=seed,
                         benchmark="long", session_minutes=360, verdicts="pending",
                         verified=rw.get("R_replicated"), submitted=rw.get("submitted_credited"),
                         frontier=rw.get("R_frontier"), collaborative=rw.get("C_collaborative"),
                         title=f"{name} · {DOMAIN[dom]} · {'solo' if solo else 'team'} · 6 hours",
                         note=(f"6-hour run. The verifier accepted {rw.get('R_replicated'):.0f} of {rw.get('submitted_credited'):.0f} "
                               "credited claims. Per-claim verdicts are pending, so claims show as awaiting verification."))
        out = os.path.join(SITE, "replays", f"long-{key}-{dom}-{'solo' if solo else 'team'}-{seed}.json")
        json.dump(R, open(out, "w"), separators=(",", ":"))
        n += 1
        print(f"{os.path.basename(out)}: {len(R['agents'])} agents, {len(R['artifacts'])} artifacts, {R['meta']['duration'] / 60:.0f} min")
    write_index(os.path.join(SITE, "replays"))
    print(f"wrote {n} replays")


if __name__ == "__main__":
    main()
