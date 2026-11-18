"""Lab server: the only thing a sandbox can reach. It owns the model meter, the spawn
registry, the board, the preprint server, every oracle and evaluator, and every log.
Agents never see hidden data, secret seeds or provider keys.

Run:  LAB_ADMIN_KEY=... uvicorn lab.server:app --port 8100
"""

from __future__ import annotations

import io
import json
import math
import os
import secrets
import sqlite3
import tarfile
import threading
import time
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

from . import models, problems
from .charter import charter, open_charter
from .problems.base import InvalidSubmission, Problem
from .scoring import index, problem_result, scaled

DB_PATH = os.environ.get("LAB_DB", "lab.sqlite")
ADMIN_KEY = os.environ.get("LAB_ADMIN_KEY", "dev-admin")
PUBLIC_URL = os.environ.get("LAB_PUBLIC_URL", "http://host.docker.internal:8100")

SCHEMA = """
create table if not exists runs (id text primary key, token text, seed int, mode text, model text,
  budget_usd real, spent_usd real default 0, created real, deadline real, max_live int, board text,
  label text, frozen int default 0, domain text default '');
create table if not exists agents (id text, run text, parent text, role text, brief text,
  status text, created real, ended real, depth int, tin int default 0, tout int default 0, cost real default 0, calls int default 0, primary key (run, id));
create table if not exists events (id integer primary key, run text, agent text, type text, problem text, data text, ts real);
create table if not exists evals (id text primary key, run text, agent text, problem text, value real, ts real);
create table if not exists submissions (id text primary key, run text, agent text, problem text, payload text, claim real, ts real);
create table if not exists preprints (id text primary key, run text, agent text, title text, body text, claims text, cites text, ts real);
create table if not exists messages (id integer primary key, run text, agent text, channel text, text text, ts real);
create table if not exists usage (run text, problem text, kind text, n int, primary key (run, problem, kind));
"""

_lock = threading.RLock()
_db = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)
_db.row_factory = sqlite3.Row
_db.executescript(SCHEMA)
try:  # databases from before open-ended runs
    _db.execute("alter table runs add column domain text default ''")
except sqlite3.OperationalError:
    pass


def q(sql, *args):
    with _lock:
        return _db.execute(sql, args).fetchall()


def q1(sql, *args):
    rows = q(sql, *args)
    return rows[0] if rows else None


def log(run, agent, type_, problem=None, **data):
    q("insert into events (run, agent, type, problem, data, ts) values (?,?,?,?,?,?)",
      run, agent, type_, problem, json.dumps(data, default=str), time.time())


def new_id(prefix):
    return f"{prefix}-{secrets.token_hex(4)}"


app = FastAPI(title="Republic lab server")


@app.on_event("startup")
async def _more_threads():
    """Sync endpoints run in a thread pool; long experiments must not starve model calls and messages."""
    import anyio.to_thread

    anyio.to_thread.current_default_thread_limiter().total_tokens = 400
if os.environ.get("LAB_CORS"):  # lets the local replay viewer poll a running lab (live mode)
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(CORSMiddleware, allow_origins=os.environ["LAB_CORS"].split(","), allow_methods=["GET"],
                       allow_headers=["Authorization"])


# ----------------------------------------------------------------------------- auth

def admin(authorization: str = Header("")):
    if authorization != f"Bearer {ADMIN_KEY}":
        raise HTTPException(401, "admin key required")


class Ctx(BaseModel):
    run: dict
    agent: str


def ctx(authorization: str = Header(""), x_agent: str = Header("")) -> Ctx:
    token = authorization.removeprefix("Bearer ")
    run = q1("select * from runs where token = ?", token)
    if not run:
        raise HTTPException(401, "bad run token")
    a = q1("select * from agents where id = ? and run = ?", x_agent, run["id"])
    if not a:
        raise HTTPException(401, "unknown agent id")
    if a["status"] != "alive":
        raise HTTPException(409, f"agent {x_agent} is {a['status']}")
    return Ctx(run=dict(run), agent=x_agent)


def open_run(c: Ctx):
    r = q1("select * from runs where id = ?", c.run["id"])
    if r["frozen"]:
        raise HTTPException(409, "run is over")
    if time.time() > r["deadline"]:
        raise HTTPException(409, "deadline passed")
    return r


def board(run) -> list:
    return [problems.get(pid, run["seed"]) for pid in json.loads(run["board"])]


def problem_of(run, pid):
    if pid not in json.loads(run["board"]):
        raise HTTPException(404, f"no problem {pid} on this board")
    return problems.get(pid, run["seed"])


def log_scored(p) -> bool:
    return type(p).score_from_oracle_log is not Problem.score_from_oracle_log


def run_oracle_log(rid, pid):
    return [json.loads(e["data"]) for e in q(
        "select data from events where run=? and problem=? and type='oracle' order by id", rid, pid)]


def charge_usage(run_id, pid, kind, n, limit):
    with _lock:
        row = q1("select n from usage where run=? and problem=? and kind=?", run_id, pid, kind)
        used = row["n"] if row else 0
        if used + n > limit:
            raise HTTPException(429, f"{kind} budget for {pid} exhausted ({used}/{limit})")
        q("insert into usage (run, problem, kind, n) values (?,?,?,?) on conflict(run, problem, kind) "
          "do update set n = n + excluded.n", run_id, pid, kind, n)
        return used + n


# ----------------------------------------------------------------------------- admin

class RunSpec(BaseModel):
    seed: int = 1
    mode: str = "community"  # community | solo
    model: str = "mock"
    budget_usd: float = 30.0
    minutes: float = 120.0
    max_live: int = 12
    board: list[str] | None = None
    label: str = ""
    domain: str = ""  # open-ended research environment (data/domains/<domain>)


@app.post("/admin/runs", dependencies=[Depends(admin)])
def create_run(spec: RunSpec):
    rid, token = new_id("run"), secrets.token_urlsafe(24)
    pids = spec.board if spec.board is not None or spec.domain else [p for p in problems.DEFAULT_BOARD if p in problems.available()]
    pids = pids or []
    for pid in pids:  # warm caches (builder anchors) before the clock starts
        problems.get(pid, spec.seed).anchors()
    now = time.time()
    q("insert into runs (id, token, seed, mode, model, budget_usd, created, deadline, max_live, board, label, domain) "
      "values (?,?,?,?,?,?,?,?,?,?,?,?)", rid, token, spec.seed, spec.mode, spec.model, spec.budget_usd,
      now, now + 60 * spec.minutes, spec.max_live, json.dumps(pids), spec.label, spec.domain)
    q("insert into agents (id, run, parent, role, brief, status, created, depth) values (?,?,?,?,?,?,?,?)",
      "founder", rid, None, "founder", "", "alive", now, 0)
    log(rid, "founder", "run_created", **spec.model_dump())
    return {"run_id": rid, "token": token, "board": pids}


@app.post("/admin/runs/{rid}/start", dependencies=[Depends(admin)])
def start_clock(rid: str, minutes: float | None = None):
    """Restart the wall clock when the agent actually starts (setup can take a while)."""
    r = q1("select * from runs where id = ?", rid)
    mins = minutes or (r["deadline"] - r["created"]) / 60
    q("update runs set created = ?, deadline = ? where id = ?", time.time(), time.time() + 60 * mins, rid)
    return {"ok": True}


@app.get("/admin/runs/{rid}/bundle", dependencies=[Depends(admin)])
def bundle(rid: str):
    """tar.gz of everything the sandbox gets: charter, statements, public files, run.json."""
    from fastapi.responses import Response

    r = q1("select * from runs where id = ?", rid)
    if not r:
        raise HTTPException(404)
    b = board(r)
    minutes = round((r["deadline"] - r["created"]) / 60)
    files = {
        "charter.md": (open_charter(r["mode"], r["domain"], r["budget_usd"], minutes, r["max_live"]) if r["domain"]
                       else charter(r["mode"], b, r["budget_usd"], minutes, r["max_live"])),
        "run.json": json.dumps({"server": PUBLIC_URL, "run_id": rid, "token": r["token"], "mode": r["mode"],
                                "model": r["model"], "agent_id": "founder"}),
    }
    if r["domain"]:  # public data copy for free exploration; evidence still needs lab exec
        from .research import domain_dir

        pub = domain_dir(dict(r))
        for dirpath, _, fns in os.walk(pub or ""):
            for fn in fns:
                path = os.path.join(dirpath, fn)
                files["data/" + os.path.relpath(path, pub)] = open(path, "rb").read()
    for p in b:
        files[f"problems/{p.info.id}/statement.md"] = p.statement()
        for name, data in p.public_files().items():
            files[f"problems/{p.info.id}/{name}"] = data
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            data = data.encode() if isinstance(data, str) else data
            info = tarfile.TarInfo(name)
            info.size, info.mtime = len(data), int(time.time())
            tar.addfile(info, io.BytesIO(data))
    return Response(buf.getvalue(), media_type="application/gzip")


@app.post("/admin/runs/{rid}/finalize", dependencies=[Depends(admin)])
def finalize(rid: str, token: str = ""):
    r = q1("select * from runs where id = ?", rid)
    if not r:
        raise HTTPException(404)
    if token and token != r["token"]:  # run.json came back from the sandbox; don't trust it blindly
        raise HTTPException(403, "run id and token do not match")
    q("update runs set frozen = 1 where id = ?", rid)
    q("update agents set status = 'stopped', ended = ? where run = ? and status = 'alive'", time.time(), rid)
    if r["domain"]:  # open-ended research run: verification replaces problem scoring
        from . import verify as v

        res = v.verify_run(rid)
        store_verification(rid, res)
        return {"run_id": rid, "label": r["label"], "model": r["model"], "mode": r["mode"], "seed": r["seed"],
                "domain": r["domain"], **res, "lab": lab_stats(rid)}
    results = []
    for p in board(r):
        pid, a = p.info.id, p.anchors()
        sub = q1("select * from submissions where run=? and problem=? order by ts desc limit 1", rid, pid)
        raw, claim, attempted = None, None, sub is not None
        if log_scored(p):  # e.g. C1: the score is the run's own oracle log; the submission carries the claim
            oracle_log = run_oracle_log(rid, pid)
            raw = p.score_from_oracle_log(oracle_log) if oracle_log else None
            attempted = attempted or bool(oracle_log)
            claim = sub["claim"] if sub else None
        elif sub:
            try:
                raw = p.verify(json.loads(sub["payload"]))
            except InvalidSubmission:
                raw = None
            claim = sub["claim"]
        pr = problem_result(pid, p.info.domain, raw, claim, a, attempted=attempted)
        devs = [scaled(e["value"], a) for e in q("select value from evals where run=? and problem=?", rid, pid)]
        dev_best = max(devs) if devs and p.info.kind != "exact" else None
        results.append({**pr.__dict__, "title": p.info.title, "kind": p.info.kind,
                        "anchors": a.__dict__, "metric": p.info.metric, "dev_best_s": dev_best,
                        "generalization_gap": None if dev_best is None else dev_best - pr.s,
                        "submissions": q1("select count(*) n from submissions where run=? and problem=?", rid, pid)["n"]})
    from .scoring import ProblemResult

    idx = index([ProblemResult(**{k: v for k, v in x.items() if k in ProblemResult.__dataclass_fields__})
                 for x in results])
    out = {"run_id": rid, "label": r["label"], "model": r["model"], "mode": r["mode"], "seed": r["seed"],
           **idx, "problems": results, "lab": lab_stats(rid)}
    log(rid, None, "finalized", republic_index=idx["republic_index"])
    return out


def lab_stats(rid):
    r = q1("select * from runs where id = ?", rid)
    agents = [dict(a) for a in q("select * from agents where run = ?", rid)]
    pts = sorted([(a["created"], 1) for a in agents] + [(a["ended"] or r["deadline"], -1) for a in agents])
    live = peak = 0
    for _, d in pts:
        live += d
        peak = max(peak, live)
    claims = [c for p in q("select claims from preprints where run = ?", rid) for c in json.loads(p["claims"])]
    backed = 0
    for c in claims:
        e = q1("select * from evals where id = ? and run = ?", str(c.get("eval_id")), rid)
        if e and e["problem"] == c.get("problem") and _close(e["value"], c.get("value")):
            backed += 1
    effort = {}
    for e in q("select agent, problem, count(*) n from events where run=? and problem is not null "
               "and type in ('oracle','eval','submit') group by agent, problem", rid):
        effort.setdefault(e["problem"], 0)
        effort[e["problem"]] += e["n"]
    tot = sum(effort.values()) or 1
    shares = [v / tot for v in effort.values()]
    n_board = len(json.loads(r["board"]))
    entropy = -sum(s * math.log(s) for s in shares if s > 0) / math.log(n_board) if n_board > 1 else 0
    return {
        "agents_total": len(agents),
        "agents_peak": peak,
        "max_depth": max(a["depth"] for a in agents),
        "cost_usd": round(sum(a["cost"] for a in agents), 4),
        "tokens_in": sum(a["tin"] for a in agents),
        "tokens_out": sum(a["tout"] for a in agents),
        "model_calls": sum(a["calls"] for a in agents),
        "preprints": q1("select count(*) n from preprints where run = ?", rid)["n"],
        "messages": q1("select count(*) n from messages where run = ?", rid)["n"],
        "preprint_claims": len(claims),
        "integrity_ppv": backed / len(claims) if claims else None,
        "effort_by_problem": effort,
        "allocation_entropy": round(entropy, 3),
        "duration_s": round(min(time.time(), r["deadline"]) - r["created"], 1),
        "agents": [{k: a[k] for k in ("id", "parent", "role", "depth", "status", "cost", "calls", "created", "ended")}
                   for a in agents],
    }


def _close(a, b):
    try:
        return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(a)))
    except (TypeError, ValueError):
        return False


@app.get("/admin/runs", dependencies=[Depends(admin)])
def list_runs():
    rows = q("select id, label, model, mode, seed, frozen, spent_usd from runs order by created")
    return {"runs": [dict(r) for r in rows], "frozen": [r["id"] for r in rows if r["frozen"]]}


@app.get("/admin/runs/{rid}/events", dependencies=[Depends(admin)])
def events(rid: str):
    return {"events": [dict(e) for e in q("select * from events where run = ? order by id", rid)],
            "preprints": [dict(p) for p in q("select * from preprints where run = ? order by ts", rid)],
            "messages": [dict(m) for m in q("select * from messages where run = ? order by id", rid)]}


# ----------------------------------------------------------------------------- agent API

class CompleteReq(BaseModel):
    messages: list[dict]
    max_tokens: int = 16000  # room for adaptive thinking plus the reply


@app.post("/v1/complete")
def complete(req: CompleteReq, c: Ctx = Depends(ctx)):
    r = open_run(c)
    if r["spent_usd"] >= r["budget_usd"]:
        raise HTTPException(402, "budget exhausted")
    if r["model"] == "unset":
        raise HTTPException(409, "no model set for this run")
    t0 = time.time()
    try:
        out = models.complete(r["model"], req.messages, req.max_tokens)
    except Exception as e:  # provider errors are logged, not fatal to the run
        log(r["id"], c.agent, "model_error", error=str(e)[:500])
        raise HTTPException(502, f"model error: {str(e)[:200]}")
    with _lock:
        q("update runs set spent_usd = spent_usd + ? where id = ?", out.cost_usd, r["id"])
        q("update agents set tin = tin + ?, tout = tout + ?, cost = cost + ?, calls = calls + 1 where id = ? and run = ?",
          out.input_tokens, out.output_tokens, out.cost_usd, c.agent, r["id"])
    log(r["id"], c.agent, "complete", tin=out.input_tokens, tout=out.output_tokens, cached=out.cached_tokens, cost=out.cost_usd,
        secs=round(time.time() - t0, 2), text=out.text[:4000])
    return {"text": out.text, "cost_usd": out.cost_usd}


class ModelReq(BaseModel):
    model: str


@app.post("/run/model")
def set_model(req: ModelReq, c: Ctx = Depends(ctx)):
    """The harness passes Valkyrie's --model at boot. It can be set once, before any call."""
    r = open_run(c)
    if c.agent != "founder" or r["model"] not in ("unset", req.model):
        raise HTTPException(409, f"model already set to {r['model']}")
    q("update runs set model = ? where id = ?", req.model, r["id"])
    log(r["id"], c.agent, "model_set", model=req.model)
    return {"model": req.model}


@app.get("/budget")
def budget(c: Ctx = Depends(ctx)):
    r = q1("select * from runs where id = ?", c.run["id"])
    used = {f"{u['problem']}.{u['kind']}": u["n"] for u in q("select * from usage where run = ?", r["id"])}
    out = {"spent_usd": round(r["spent_usd"], 4), "seconds_left": max(0, round(r["deadline"] - time.time())), "usage": used}
    if not r["domain"]:  # v0 board runs had a real budget; open research runs only have a hidden safety ceiling
        out["budget_usd"] = r["budget_usd"]
    return out


class SpawnReq(BaseModel):
    role: str
    brief: str


@app.post("/agents/spawn")
def spawn(req: SpawnReq, c: Ctx = Depends(ctx)):
    r = open_run(c)
    if r["mode"] == "solo":
        raise HTTPException(403, "solo run: recruiting is disabled")
    if r["mode"] == "fixed":  # PI + exactly three specialists, no deeper spawning
        n = q1("select count(*) n from agents where run = ? and id != 'founder'", r["id"])["n"]
        if c.agent != "founder" or n >= 3:
            raise HTTPException(403, "fixed-structure run: only the PI spawns, and at most three specialists")
    with _lock:
        live = q1("select count(*) n from agents where run = ? and status = 'alive'", r["id"])["n"]
        if live >= r["max_live"]:
            raise HTTPException(429, f"{live} agents alive; the cap is {r['max_live']}")
        parent = q1("select depth from agents where id = ? and run = ?", c.agent, r["id"])
        aid = new_id("agent")
        q("insert into agents (id, run, parent, role, brief, status, created, depth) values (?,?,?,?,?,?,?,?)",
          aid, r["id"], c.agent, req.role[:200], req.brief[:20000], "alive", time.time(), parent["depth"] + 1)
    log(r["id"], c.agent, "spawn", child=aid, role=req.role[:200], brief=req.brief[:2000])
    return {"agent_id": aid}


@app.post("/agents/exit")
def agent_exit(note: dict | None = None, c: Ctx = Depends(ctx)):
    q("update agents set status = 'done', ended = ? where id = ? and run = ?", time.time(), c.agent, c.run["id"])
    log(c.run["id"], c.agent, "exit", note=(note or {}).get("note", "")[:2000])
    return {"ok": True}


@app.get("/agents")
def list_agents(c: Ctx = Depends(ctx)):
    rows = q("select id, parent, role, status, depth, cost from agents where run = ? order by created", c.run["id"])
    return {"agents": [dict(a) for a in rows]}


@app.get("/agents/{aid}/brief")
def agent_brief(aid: str, c: Ctx = Depends(ctx)):
    a = q1("select role, brief, parent from agents where id = ? and run = ?", aid, c.run["id"])
    if not a:
        raise HTTPException(404)
    return dict(a)


@app.get("/problems")
def list_problems(c: Ctx = Depends(ctx)):
    out = []
    for p in board(c.run):
        best = q1("select claim, ts from submissions where run=? and problem=? order by ts desc limit 1", c.run["id"], p.info.id)
        out.append({"id": p.info.id, "domain": p.info.domain, "title": p.info.title, "kind": p.info.kind,
                    "metric": p.info.metric, "oracle_budget": p.info.oracle.budget, "dev_budget": p.info.dev_budget,
                    "submitted": bool(best)})
    return {"problems": out}


@app.post("/problems/{pid}/oracle")
def oracle(pid: str, query: dict, c: Ctx = Depends(ctx)):
    r = open_run(c)
    p = problem_of(r, pid)
    if not p.info.oracle.budget:
        raise HTTPException(400, f"{pid} has no oracle")
    used = q1("select n from usage where run=? and problem=? and kind='oracle'", r["id"], pid)
    if used and used["n"] >= p.info.oracle.budget:
        raise HTTPException(429, f"oracle budget for {pid} exhausted")
    try:
        out = p.oracle(query)
    except InvalidSubmission as e:
        raise HTTPException(400, str(e))
    cost = int(out.pop("_cost", 1)) if isinstance(out, dict) else 1
    n = charge_usage(r["id"], pid, "oracle", cost, p.info.oracle.budget)
    log(r["id"], c.agent, "oracle", pid, query=query, result=out)
    return {**out, "budget_used": n, "budget": p.info.oracle.budget}


@app.post("/problems/{pid}/eval")
def dev_eval(pid: str, submission: dict, c: Ctx = Depends(ctx)):
    r = open_run(c)
    p = problem_of(r, pid)
    if p.info.kind != "exact":
        if not p.info.dev_budget:
            raise HTTPException(400, f"{pid} has no dev evaluations; use your own validation")
        charge_usage(r["id"], pid, "dev_eval", 1, p.info.dev_budget)
    try:
        if log_scored(p):  # score of this run's own oracle log so far, never another run's
            log_ = run_oracle_log(r["id"], pid)
            value = p.score_from_oracle_log(log_) if log_ else 0.0
        else:
            value = p.dev_eval(submission) if p.info.kind != "exact" else p.verify(submission)
    except InvalidSubmission as e:
        raise HTTPException(400, str(e))
    eid = new_id("eval")
    q("insert into evals (id, run, agent, problem, value, ts) values (?,?,?,?,?,?)", eid, r["id"], c.agent, pid, value, time.time())
    log(r["id"], c.agent, "eval", pid, eval_id=eid, value=value)
    return {"eval_id": eid, "value": value, "split": "exact" if p.info.kind == "exact" else "dev"}


class SubmitReq(BaseModel):
    submission: dict
    claim: float


@app.post("/problems/{pid}/submit")
def submit(pid: str, req: SubmitReq, c: Ctx = Depends(ctx)):
    r = open_run(c)
    p = problem_of(r, pid)
    if not math.isfinite(req.claim):
        raise HTTPException(400, "claim must be a finite number in the problem's raw metric")
    n = q1("select count(*) n from submissions where run=? and problem=?", r["id"], pid)["n"]
    if n >= p.info.max_submissions:
        raise HTTPException(429, f"{pid} allows {p.info.max_submissions} submissions; the last one counts")
    try:  # validate format only; the hidden score is never returned
        if p.info.kind == "exact":
            p.verify(req.submission)
        else:
            _validate_shape(p, req.submission)
    except InvalidSubmission as e:
        raise HTTPException(400, f"invalid submission: {e}")
    sid = new_id("sub")
    q("insert into submissions (id, run, agent, problem, payload, claim, ts) values (?,?,?,?,?,?,?)",
      sid, r["id"], c.agent, pid, json.dumps(req.submission), req.claim, time.time())
    log(r["id"], c.agent, "submit", pid, submission_id=sid, claim=req.claim)
    return {"submission_id": sid, "submissions_used": n + 1, "max": p.info.max_submissions}


def _validate_shape(p, submission):
    try:
        p.verify(submission)
    except InvalidSubmission:
        raise
    except Exception as e:
        raise InvalidSubmission(str(e))


class Preprint(BaseModel):
    title: str
    body: str
    claims: list[dict] = []
    cites: list[str] = []


@app.post("/preprints")
def publish(pp: Preprint, c: Ctx = Depends(ctx)):
    r = open_run(c)
    pid = new_id("pp")
    q("insert into preprints (id, run, agent, title, body, claims, cites, ts) values (?,?,?,?,?,?,?,?)",
      pid, r["id"], c.agent, pp.title[:300], pp.body[:50000], json.dumps(pp.claims[:50]), json.dumps(pp.cites[:50]), time.time())
    log(r["id"], c.agent, "publish", preprint=pid, title=pp.title[:300], claims=pp.claims[:50], cites=pp.cites[:50])
    return {"preprint_id": pid}


@app.get("/preprints")
def search(qtext: str = "", limit: int = 20, c: Ctx = Depends(ctx)):
    rows = q("select id, agent, title, body, claims, ts from preprints where run = ? order by ts desc", c.run["id"])
    terms = [t for t in qtext.lower().split() if t]

    def score(row):
        text = (row["title"] + " " + row["body"]).lower()
        return sum(text.count(t) for t in terms) if terms else 1

    ranked = sorted((r for r in rows if score(r) > 0), key=score, reverse=True)[:limit]
    return {"preprints": [{"id": r["id"], "agent": r["agent"], "title": r["title"],
                           "abstract": r["body"][:400], "claims": json.loads(r["claims"])} for r in ranked]}


@app.get("/preprints/{ppid}")
def read_preprint(ppid: str, c: Ctx = Depends(ctx)):
    row = q1("select * from preprints where id = ? and run = ?", ppid, c.run["id"])
    if not row:
        raise HTTPException(404)
    log(c.run["id"], c.agent, "read", preprint=ppid)
    return {**dict(row), "claims": json.loads(row["claims"]), "cites": json.loads(row["cites"])}


class Post(BaseModel):
    text: str
    channel: str = "general"


@app.post("/board")
def post(m: Post, c: Ctx = Depends(ctx)):
    r = open_run(c)
    q("insert into messages (run, agent, channel, text, ts) values (?,?,?,?,?)", r["id"], c.agent, m.channel[:50], m.text[:5000], time.time())
    log(r["id"], c.agent, "post", channel=m.channel[:50], text=m.text[:1000])
    return {"ok": True}


@app.get("/board")
def inbox(since: int = 0, channel: str = "", c: Ctx = Depends(ctx)):
    rows = q("select * from messages where run = ? and id > ? order by id", c.run["id"], since)
    rows = [dict(m) for m in rows if not channel or m["channel"] == channel]
    return {"messages": rows[-100:], "last_id": rows[-1]["id"] if rows else since}


@app.get("/health")
def health():
    return {"ok": True, "problems": problems.available()}


from . import research  # noqa: E402,F401  (registers the open-ended research routes)


@app.post("/admin/runs/{rid}/verify", dependencies=[Depends(admin)])
def verify(rid: str):
    """Post-run verification of an open-research run (re-execution, hidden data, relations)."""
    from . import verify as v

    if not q1("select id from runs where id = ?", rid):
        raise HTTPException(404)
    q("update runs set frozen = 1 where id = ?", rid)
    res = v.verify_run(rid)
    store_verification(rid, res)
    return res


def store_verification(rid, res):
    q("create table if not exists verifications (run text primary key, body text, ts real)")
    q("insert or replace into verifications (run, body, ts) values (?,?,?)", rid, json.dumps(res, default=str), time.time())
    log(rid, None, "verified", **res["summary"])


@app.get("/admin/runs/{rid}/replay", dependencies=[Depends(admin)])
def replay_json(rid: str):
    """The run as a viewer replay, live while it runs (site/replay.html?live=...)."""
    from .replay import from_db

    return from_db(DB_PATH, rid)
