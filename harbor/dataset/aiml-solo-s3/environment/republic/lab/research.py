"""Open-ended research API: blobs, server-run executions, and schema-checked artifacts.

Mounted on the lab server app (imported at the end of server.py). Domain environments live in
data/domains/<domain>/{public,eval,hidden}: agents' execs see public/ at /data; eval/ and
hidden/ are used only by lab/verify.py after the run.
"""

from __future__ import annotations

import base64
import json
import os
import time

from fastapi import Depends, HTTPException
from pydantic import BaseModel, ValidationError

from . import executor
from .artifacts import Artifact, check_live
from .server import Ctx, app, ctx, log, new_id, open_run, q, q1

DOMAINS_DIR = os.environ.get("LAB_DOMAINS", "data/domains")
# GPU environments: at most LAB_GPU_SLOTS experiments use the GPU at once; others queue (allocation is the community's problem)
import threading  # noqa: E402

_gpu = threading.BoundedSemaphore(int(os.environ["LAB_GPU_SLOTS"])) if os.environ.get("LAB_GPU_SLOTS") else None
# CPU admission: at most LAB_EXEC_SLOTS experiments run at once (each limited to LAB_EXEC_THREADS threads); others queue
_cpu = threading.BoundedSemaphore(int(os.environ["LAB_EXEC_SLOTS"])) if os.environ.get("LAB_EXEC_SLOTS") else None
MAX_BLOB = 20_000_000

q("create table if not exists execs (id text primary key, run text, agent text, record text, ts real)")
q("create table if not exists artifacts (id text primary key, run text, agent text, type text, title text, "
  "body text, provenance text, ts real)")


def domain_dir(run, part="public") -> str | None:
    dom = run.get("domain") if isinstance(run, dict) else run["domain"]
    d = os.path.join(DOMAINS_DIR, dom or "", part)
    return d if dom and os.path.isdir(d) else None


def run_execs(rid) -> dict[str, dict]:
    return {r["id"]: {**json.loads(r["record"]), "agent": r["agent"], "ts": r["ts"], "id": r["id"]}
            for r in q("select * from execs where run=?", rid)}


def run_artifacts(rid) -> dict[str, dict]:
    return {r["id"]: {**json.loads(r["body"]), "id": r["id"], "agent": r["agent"], "ts": r["ts"],
                      "live": json.loads(r["provenance"])}
            for r in q("select * from artifacts where run=? order by ts", rid)}


class BlobReq(BaseModel):
    b64: str


@app.post("/blobs")
def upload_blob(req: BlobReq, c: Ctx = Depends(ctx)):
    open_run(c)
    data = base64.b64decode(req.b64)
    if len(data) > MAX_BLOB:
        raise HTTPException(413, f"blob over {MAX_BLOB} bytes")
    sha = executor.put_blob(data)
    log(c.run["id"], c.agent, "blob", sha=sha, bytes=len(data))
    return {"blob": sha, "bytes": len(data)}


@app.get("/blobs/{sha}")
def download_blob(sha: str, c: Ctx = Depends(ctx)):
    try:
        return {"blob": sha, "b64": base64.b64encode(executor.get_blob(sha)).decode()}
    except KeyError:
        raise HTTPException(404, "no such blob")


class ExecReq(BaseModel):
    code: str
    cmd: str
    inputs: dict[str, str] = {}
    timeout: int = 900


@app.post("/exec")
def execute(req: ExecReq, c: Ctx = Depends(ctx)):
    run = open_run(c)
    import re
    if re.search(r"(^|[\s=;&|'\"(])/work(/|\b)", req.cmd):
        raise HTTPException(422, "lab exec runs a fresh copy of the directory you passed, so evidence can be re-run later: "
                                 "refer to files relative to it (e.g. `python run.py`), not /work/...; pass other files with --input")
    left = int(run["deadline"] - time.time())
    t_queue = time.time()
    try:
        if _cpu:
            _cpu.acquire()
        if _gpu:
            _gpu.acquire()
        left = int(run["deadline"] - time.time())
        rec = executor.run(req.code, req.cmd, req.inputs, domain_dir(dict(run)), timeout=max(1, min(req.timeout, left)))
    except KeyError as e:
        raise HTTPException(404, f"no such blob: {e}")
    finally:
        if _gpu:
            _gpu.release()
        if _cpu:
            _cpu.release()
    rec["queued_s"] = round(time.time() - t_queue - rec.get("seconds", 0), 1)
    eid = new_id("exec")
    q("insert into execs (id, run, agent, record, ts) values (?,?,?,?,?)", eid, c.run["id"], c.agent, json.dumps(rec), time.time())
    log(c.run["id"], c.agent, "exec", exec_id=eid, code=req.code, inputs=req.inputs, cmd=req.cmd[:300],
        exit_code=rec["exit_code"], seconds=rec["seconds"], metrics=rec["metrics"])
    return {"exec_id": eid, **rec}


@app.get("/execs")
def list_execs(mine: int = 1, c: Ctx = Depends(ctx)):
    rows = q("select id, agent, record, ts from execs where run=? order by ts", c.run["id"])
    out = []
    for r in rows:
        if mine and r["agent"] != c.agent:
            continue
        rec = json.loads(r["record"])
        out.append({"exec_id": r["id"], "agent": r["agent"], "cmd": rec["cmd"][:120], "code": rec["code"],
                    "exit_code": rec["exit_code"], "metrics": rec.get("metrics"), "outputs": rec.get("outputs")})
    return {"execs": out[-100:]}


@app.get("/exec/{eid}")
def read_exec(eid: str, c: Ctx = Depends(ctx)):
    r = q1("select * from execs where id=? and run=?", eid, c.run["id"])
    if not r:
        raise HTTPException(404, "no such exec in this run")
    return {"exec_id": eid, "agent": r["agent"], **json.loads(r["record"])}


@app.get("/schema")
def schema(c: Ctx = Depends(ctx)):
    return Artifact.model_json_schema()


@app.get("/datasets")
def datasets(c: Ctx = Depends(ctx)):
    d = domain_dir(c.run)
    if not d:
        return {"domain": c.run.get("domain"), "datasets": []}
    out = []
    for name in sorted(os.listdir(d)):
        card = os.path.join(d, name, "README.md")
        out.append({"id": name, "path": f"/data/{name}", "card": open(card).read()[:2000] if os.path.exists(card) else ""})
    return {"domain": c.run.get("domain"), "datasets": out}


@app.post("/artifacts")
def submit_artifact(body: dict, c: Ctx = Depends(ctx)):
    open_run(c)
    try:
        art = Artifact.model_validate(body)
    except ValidationError as e:
        raise HTTPException(422, json.loads(e.json(include_url=False)))
    rid = c.run["id"]
    live = check_live(art, run_execs(rid), run_artifacts(rid), time.time())
    aid = new_id(art.kind.split("_")[0][:6])
    q("insert into artifacts (id, run, agent, type, title, body, provenance, ts) values (?,?,?,?,?,?,?,?)",
      aid, rid, c.agent, art.kind, art.title, art.model_dump_json(), json.dumps(live), time.time())
    log(rid, c.agent, "artifact", artifact_id=aid, kind=art.kind, ok=live["ok"],
        relations=[r.model_dump() for r in art.relations])
    return {"artifact_id": aid, "live_checks": live,
            "note": "Live checks only. Re-execution, hidden-data verification and the auditor run after the run ends."}


@app.get("/artifacts")
def list_artifacts(kind: str = "", qtext: str = "", limit: int = 50, c: Ctx = Depends(ctx)):
    arts = list(run_artifacts(c.run["id"]).values())[::-1]
    if kind:
        arts = [a for a in arts if a["kind"] == kind]
    if qtext:
        arts = [a for a in arts if qtext.lower() in (a["title"] + " " + a.get("summary", "")).lower()]
    return {"artifacts": [{k: a[k] for k in ("id", "kind", "title", "agent")} | {"live_ok": a["live"]["ok"]}
                          for a in arts[:limit]]}


@app.get("/artifacts/{aid}")
def read_artifact(aid: str, c: Ctx = Depends(ctx)):
    a = run_artifacts(c.run["id"]).get(aid)
    if not a:
        raise HTTPException(404, "no such artifact in this run")
    log(c.run["id"], c.agent, "read_artifact", artifact_id=aid)
    return a
