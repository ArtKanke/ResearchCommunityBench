"""Shrink the community's store before Valkyrie copies it to the verifier (512 MB artifact limit).

Verification needs the content of: test code and test inputs, evidence code and evidence inputs (re-execution
regenerates outputs), and anything a relation says was used. Output files are referenced by hash in the
records (provenance, consumption checks), so their content can go. Run as root after the session:
  python -m lab.prune /lab/store
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys


def needed(db_path: str) -> set[str]:
    db = sqlite3.connect(db_path)
    keep: set[str] = set()
    execs = {r[0]: json.loads(r[1]) for r in db.execute("select id, record from execs")}
    for (body,) in db.execute("select body from artifacts"):
        a = json.loads(body)
        t = a.get("test") or {}
        keep |= {t.get("code")} | set((t.get("inputs") or {}).values()) | set(a.get("code") or [])
        for rel in a.get("relations") or []:
            keep |= set(rel.get("used") or [])
        for eid in a.get("evidence") or []:
            e = execs.get(eid)
            if e:
                keep |= {e.get("code")} | set((e.get("inputs") or {}).values())
    keep.discard(None)
    return keep


LIMIT = int(os.environ.get("LAB_STORE_LIMIT_MB", "470")) * 1_000_000  # Valkyrie copies at most 512 MB to the verifier


def within_limit(store: str, keep: set[str]) -> set[str]:
    """If the needed files exceed the limit, keep them in submission order until it is reached; later
    contributions then fail verification with a missing-file reason (stated in lab guide beforehand)."""
    blobs = os.path.join(store, "blobs")
    size = lambda b: os.path.getsize(os.path.join(blobs, b[:2], b)) if os.path.exists(os.path.join(blobs, b[:2], b)) else 0
    if sum(size(b) for b in keep) <= LIMIT:
        return keep
    db = sqlite3.connect(os.path.join(store, "lab.sqlite"))
    execs = {r[0]: json.loads(r[1]) for r in db.execute("select id, record from execs")}
    kept, total = set(), 0
    for (body,) in db.execute("select body from artifacts order by ts"):
        a = json.loads(body)
        t = a.get("test") or {}
        need = {t.get("code")} | set((t.get("inputs") or {}).values()) | set(a.get("code") or [])
        for rel in a.get("relations") or []:
            need |= set(rel.get("used") or [])
        for eid in a.get("evidence") or []:
            e = execs.get(eid) or {}
            need |= {e.get("code")} | set((e.get("inputs") or {}).values())
        need.discard(None)
        extra = sum(size(b) for b in need - kept)
        if total + extra > LIMIT:
            break
        kept |= need
        total += extra
    return kept


def prune(store: str) -> tuple[int, int]:
    keep = within_limit(store, needed(os.path.join(store, "lab.sqlite")))
    removed = freed = 0
    blobs = os.path.join(store, "blobs")
    for dirpath, _, files in os.walk(blobs):
        for fn in files:
            if fn not in keep:
                p = os.path.join(dirpath, fn)
                freed += os.path.getsize(p)
                os.remove(p)
                removed += 1
    return removed, freed


if __name__ == "__main__":
    store = sys.argv[1] if len(sys.argv) > 1 else "/lab/store"
    n, b = prune(store)
    print(f"pruned {n} unneeded files ({b / 1e6:.1f} MB) from {store}")
