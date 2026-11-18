"""Replay files: one JSON per run that the community viewer (site/replay.html) plays back.

Format "replay/1":
  meta       {id, title, source: real|synthetic, benchmark, model, domain, mode, duration, note}
  agents     [{id, role, parent, t_born, t_end, brief}]
  artifacts  [{id, kind, title, agent, t, status, reason, claim, evidence, checks, relations}]
             status: accepted | rejected | pending | superseded
             relations: [{target, relation: cites|replicates|extends|corrects|refutes, verified}]
  events     [{t, type, agent, ...}] sorted by t (seconds from run start)
             spawn{child} message{text, channel} thought{text} exec{label, ok} artifact{id}
             verdict{id, status} exit{} cost{usd}
  scores     whatever the run's finalize produced (optional)

Sources: a lab-server database (any run, v0 board or v1 open research) or the result files
written by scripts/dry_run.py. Synthetic replays come from scripts/synth_replay.py and always
carry source="synthetic"; the viewer labels them on every frame.

  python -m lab.replay --results results/run-XXXX.json --out site/replays/NAME.json
  python -m lab.replay --db results/lab.sqlite --run run-XXXX --out site/replays/NAME.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3

FORMAT = "replay/1"


def _thought(text: str) -> str:
    m = re.search(r"THOUGHT:\s*(.+?)(?:\n```|$)", text or "", re.S)
    return (m.group(1) if m else (text or "")).strip()[:600]


def _agents(spawns, exits, founder_t=0.0, roles=None):
    agents = {"founder": {"id": "founder", "role": "founder", "parent": None, "t_born": founder_t, "t_end": None, "brief": ""}}
    for s in spawns:
        agents[s["child"]] = {"id": s["child"], "role": s.get("role", "agent"), "parent": s["agent"],
                              "t_born": s["t"], "t_end": None, "brief": s.get("brief", "")[:2000]}
    for e in exits:
        if e["agent"] in agents:
            agents[e["agent"]]["t_end"] = e["t"]
    for aid, role in (roles or {}).items():
        if aid in agents and role:
            agents[aid]["role"] = role
    return list(agents.values())


def from_events(raw_events, preprints=(), run=None, label="", source="real"):
    """Shared core: lab-server event rows (dicts with type, agent, problem, data, ts)."""
    raw = sorted(raw_events, key=lambda e: e["ts"])
    t0 = raw[0]["ts"] if raw else 0.0
    events, spawns, exits = [], [], []
    evals = {}
    subs = []
    for e in raw:
        d = json.loads(e["data"]) if isinstance(e.get("data"), str) else (e.get("data") or {})
        t, a, typ = round(e["ts"] - t0, 2), e["agent"], e["type"]
        if typ == "spawn":
            spawns.append({"t": t, "agent": a, **d})
            events.append({"t": t, "type": "spawn", "agent": a, "child": d["child"], "role": d.get("role", "")})
        elif typ == "exit":
            exits.append({"t": t, "agent": a})
            events.append({"t": t, "type": "exit", "agent": a, "note": d.get("note", "")[:300]})
        elif typ == "post":
            events.append({"t": t, "type": "message", "agent": a, "channel": d.get("channel", "general"), "text": d.get("text", "")[:1500]})
        elif typ == "complete":
            events.append({"t": t, "type": "thought", "agent": a, "text": _thought(d.get("text", ""))})
            events.append({"t": t, "type": "cost", "agent": a, "usd": d.get("cost", 0.0)})
        elif typ == "oracle":
            events.append({"t": t, "type": "exec", "agent": a, "label": f"experiment on {e['problem']}", "ok": True})
        elif typ == "eval":
            evals[d.get("eval_id")] = d.get("value")
            events.append({"t": t, "type": "exec", "agent": a, "label": f"evaluate {e['problem']} = {d.get('value')}", "ok": True})
        elif typ == "exec":
            events.append({"t": t, "type": "exec", "agent": a, "label": f"lab exec ({d.get('seconds', 0)}s)",
                           "ok": d.get("exit_code") == 0, "id": d.get("exec_id")})
        elif typ == "submit":
            subs.append({"t": t, "agent": a, "problem": e["problem"], **d})
        elif typ == "artifact":
            events.append({"t": t, "type": "artifact", "agent": a, "id": d["artifact_id"]})
        elif typ == "publish":
            events.append({"t": t, "type": "artifact", "agent": a, "id": d.get("preprint") or d.get("preprint_id")})

    artifacts = []
    for p in preprints:  # v0 preprints: integrity = every claim matches the eval log
        claims = json.loads(p["claims"]) if isinstance(p["claims"], str) else (p["claims"] or [])
        cites = json.loads(p["cites"]) if isinstance(p.get("cites"), str) else (p.get("cites") or [])
        bad = [c for c in claims if c.get("eval_id") not in evals or abs(float(evals[c["eval_id"]]) - float(c["value"])) > 1e-9 * max(1, abs(float(c["value"])))]
        t = round(p["ts"] - t0, 2)
        artifacts.append({"id": p["id"], "kind": "preprint", "title": p["title"], "agent": p["agent"], "t": t,
                          "summary": (p.get("body") or "")[:3000],
                          "status": "rejected" if bad else ("accepted" if claims else "pending"),
                          "reason": f"{len(bad)} claim(s) not in the eval log" if bad else f"{len(claims)} claim(s) match the eval log",
                          "claim": {"quantities": claims}, "evidence": {}, "checks": {"claims_in_log": not bad},
                          "relations": [{"target": c, "relation": "cites", "verified": False} for c in cites]})
        if not any(ev.get("type") == "artifact" and ev.get("id") == p["id"] for ev in events):
            events.append({"t": t, "type": "artifact", "agent": p["agent"], "id": p["id"]})

    if subs:  # v0 submissions: last one per problem counts; status from the v0 claim rule
        scored = {pr["problem_id"]: pr for pr in (run or {}).get("problems", [])}
        last = {}
        for s in subs:
            last[s["problem"]] = s["submission_id"]
        for s in subs:
            pr = scored.get(s["problem"], {})
            final = last[s["problem"]] == s["submission_id"]
            c = pr.get("c")
            status = "superseded" if not final else ("accepted" if (c or 0) > 0 else "rejected")
            reason = ("a later submission replaced it" if not final else
                      f"score s={pr.get('s', 0):.2f}, credit {c:.2f} under the v0 claim rule" if c is not None else "unscored")
            artifacts.append({"id": s["submission_id"], "kind": "submission", "title": f"{s['problem']}: {pr.get('title', '')}",
                              "agent": s["agent"], "t": s["t"], "status": status, "reason": reason,
                              "claim": {"problem": s["problem"], "claimed": s.get("claim"), "metric": pr.get("metric")},
                              "evidence": {}, "checks": {}, "relations": []})
            events.append({"t": s["t"], "type": "artifact", "agent": s["agent"], "id": s["submission_id"]})

    events.sort(key=lambda e: e["t"])
    roles = {a["id"]: a.get("role") for a in (run or {}).get("lab", {}).get("agents", [])}
    meta = {"id": (run or {}).get("run_id", ""), "title": label or (run or {}).get("label", ""), "source": source,
            "benchmark": "v0-board" if subs or preprints else "v1", "model": (run or {}).get("model", ""),
            "domain": (run or {}).get("domain", "all five sciences" if subs else ""), "mode": (run or {}).get("mode", ""),
            "duration": events[-1]["t"] if events else 0, "note": ""}
    scores = {k: (run or {}).get(k) for k in ("republic_index", "verified_productivity", "domain_credit") if (run or {}).get(k) is not None}
    if run and run.get("lab"):
        scores["cost_usd"] = run["lab"].get("cost_usd")
    return {"format": FORMAT, "meta": meta, "agents": _agents(spawns, exits, roles=roles),
            "artifacts": sorted(artifacts, key=lambda x: x["t"]), "events": events, "scores": scores}


def from_results(run_json: str, label: str = "") -> dict:
    run = json.load(open(run_json))
    ev = json.load(open(run_json.replace(".json", ".events.json")))
    rep = from_events(ev["events"], ev.get("preprints", []), run, label)
    return rep


def from_db(db_path: str, rid: str) -> dict:
    """Any run in a lab-server database, including v1 artifacts from the research API."""
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    run = dict(db.execute("select * from runs where id=?", (rid,)).fetchone())
    events = [dict(r) for r in db.execute("select * from events where run=? order by id", (rid,))]
    preprints = [dict(r) for r in db.execute("select * from preprints where run=?", (rid,))]
    rep = from_events(events, preprints, {"run_id": rid, "label": run["label"], "model": run["model"],
                                          "mode": run["mode"], "domain": run.get("domain", "")})
    if db.execute("select name from sqlite_master where name='artifacts'").fetchone():
        t0 = min(e["ts"] for e in events) if events else 0
        for r in db.execute("select * from artifacts where run=? order by ts", (rid,)):
            body, prov = json.loads(r["body"]), json.loads(r["provenance"])
            rels = [{"target": d, "relation": "extends", "verified": False} for d in body.get("depends_on", [])]
            rels += [{"target": x["target"], "relation": x["relation"], "verified": False} for x in body.get("relations", [])]
            rep["artifacts"].append({
                "id": r["id"], "kind": body["type"] if "type" in body else body.get("kind"), "title": body["title"],
                "agent": r["agent"], "t": round(r["ts"] - t0, 2), "summary": body.get("summary", ""),
                "status": "pending" if prov["ok"] else "rejected",
                "reason": "awaiting verification" if prov["ok"] else "; ".join(prov["why"]),
                "claim": body.get("claim") or {}, "evidence": {"execs": body.get("evidence", [])},
                "checks": prov["checks"], "relations": rels})
        rep["artifacts"].sort(key=lambda x: x["t"])
        rep["meta"]["benchmark"] = "v1"
        rep["meta"]["domain"] = run.get("domain") or ""
        ver = None
        if db.execute("select name from sqlite_master where name='verifications'").fetchone():
            row = db.execute("select body from verifications where run=?", (rid,)).fetchone()
            ver = json.loads(row["body"]) if row else None
        if ver:  # post-run verdicts replace the live statuses
            dur = rep["meta"]["duration"]
            by = {x["id"]: x for x in ver["artifacts"]}
            for i, a in enumerate(rep["artifacts"]):
                v = by.get(a["id"])
                if not v:
                    continue
                st = v.get("status", "pending")
                a["status"] = {"accepted": "accepted", "rejected": "rejected", "duplicate": "superseded",
                               "not credited": "uncredited", "pending auditor": "pending"}.get(st, "pending")
                a["reason"] = f"{st}: {v.get('reason', '')}"
                a["checks"] = v.get("checks", {})
                a["tier"] = v.get("tier")
                a["hidden"] = v.get("hidden")
                vr = {(x["target"], x["relation"]): x for x in v.get("relations", [])}
                for r in a["relations"]:
                    x = vr.get((r["target"], r["relation"]))
                    if x:
                        r["verified"], r["reason"] = x.get("verified", False), x.get("reason", "")
                rep["events"].append({"t": dur + 3 + i * 2, "type": "verdict", "id": a["id"], "status": a["status"], "agent": a["agent"]})
            rep["scores"] = ver.get("summary", {})
    return rep


def write_index(folder: str) -> list[dict]:
    """site/replays/index.json: every replay in the folder, real runs first."""
    entries = []
    for fn in sorted(os.listdir(folder)):
        if fn.endswith(".json") and fn != "index.json":
            m = json.load(open(os.path.join(folder, fn)))["meta"]
            entries.append({"file": fn, "title": m["title"], "source": m["source"], "benchmark": m.get("benchmark", "")})
    entries.sort(key=lambda e: (e["source"] != "real", e["benchmark"] != "v1", e["file"]))
    json.dump(entries, open(os.path.join(folder, "index.json"), "w"), indent=1)
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results")
    ap.add_argument("--db")
    ap.add_argument("--run")
    ap.add_argument("--label", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rep = from_results(a.results, a.label) if a.results else from_db(a.db, a.run)
    if a.note:
        rep["meta"]["note"] = a.note
    if a.label:
        rep["meta"]["title"] = a.label
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1)
    write_index(os.path.dirname(a.out) or ".")
    print(f"{a.out}: {len(rep['agents'])} agents, {len(rep['artifacts'])} artifacts, {len(rep['events'])} events, "
          f"{rep['meta']['duration']:.0f}s, source={rep['meta']['source']}")


if __name__ == "__main__":
    main()
