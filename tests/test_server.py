"""Lab server lifecycle: open a run, act as agents, finalize, check scores and integrity."""

import os
import tempfile

os.environ["LAB_DB"] = os.path.join(tempfile.mkdtemp(), "t.sqlite")
os.environ["LAB_ADMIN_KEY"] = "k"

from fastapi.testclient import TestClient  # noqa: E402

from lab.server import app  # noqa: E402

c = TestClient(app)
ADMIN = {"Authorization": "Bearer k"}


def open_run(mode="community", board=("M1", "C1", "P2")):
    r = c.post("/admin/runs", json={"seed": 1, "mode": mode, "model": "mock", "board": list(board), "minutes": 5}, headers=ADMIN)
    assert r.status_code == 200, r.text
    run = r.json()
    return run["run_id"], {"Authorization": f"Bearer {run['token']}", "X-Agent": "founder"}


def test_full_lifecycle():
    rid, h = open_run()
    assert c.post("/v1/complete", json={"messages": [{"role": "system", "content": "You are the **founder**"}]}, headers=h).status_code == 200
    child = c.post("/agents/spawn", json={"role": "chemist", "brief": "C1"}, headers=h).json()["agent_id"]
    hc = {**h, "X-Agent": child}
    o = c.post("/problems/C1/oracle", json={"smiles": ["CCO", "c1ccccc1", "CC(=O)O"]}, headers=hc).json()
    assert o["budget_used"] == 3  # per-molecule charging via _cost
    ev = c.post("/problems/C1/eval", json={}, headers=hc).json()
    assert ev["value"] > 0
    e1 = c.post("/problems/M1/eval", json={"heights": [1.0] * 10}, headers=h).json()
    c.post("/preprints", json={"title": "ok", "body": "", "claims": [{"problem": "M1", "eval_id": e1["eval_id"], "value": e1["value"]}]}, headers=h)
    c.post("/preprints", json={"title": "bad", "body": "", "claims": [{"problem": "M1", "eval_id": e1["eval_id"], "value": 1.4}]}, headers=h)
    assert c.post("/problems/M1/submit", json={"submission": {"heights": [1.0] * 10}, "claim": 2.0}, headers=h).status_code == 200
    assert c.post("/problems/C1/submit", json={"submission": {}, "claim": 0.1}, headers=hc).status_code == 200
    res = c.post(f"/admin/runs/{rid}/finalize", headers=ADMIN).json()
    by = {p["problem_id"]: p for p in res["problems"]}
    assert by["C1"]["raw"] == ev["value"] and by["C1"]["attempted"]
    assert by["M1"]["raw"] == 2.0 and by["P2"]["attempted"] is False
    assert res["lab"]["agents_total"] == 2 and res["lab"]["integrity_ppv"] == 0.5
    assert c.post("/problems/M1/eval", json={"heights": [1.0]}, headers=h).status_code == 409  # frozen


def test_solo_cannot_spawn_and_runs_are_isolated():
    rid, h = open_run("solo")
    assert c.post("/agents/spawn", json={"role": "x", "brief": ""}, headers=h).status_code == 403
    rid2, h2 = open_run("solo")
    c.post("/problems/C1/oracle", json={"smiles": ["CCO"]}, headers=h2)
    assert c.post("/problems/C1/eval", json={}, headers=h).json()["value"] == 0.0  # not run 2's log
    assert c.post(f"/admin/runs/{rid}/finalize", params={"token": "wrong"}, headers=ADMIN).status_code == 403


def test_fixed_arm_allows_three_specialists_only():
    rid, h = open_run("fixed")
    kids = [c.post("/agents/spawn", json={"role": f"s{i}", "brief": ""}, headers=h) for i in range(4)]
    assert [k.status_code for k in kids] == [200, 200, 200, 403]
    hc = {**h, "X-Agent": kids[0].json()["agent_id"]}
    assert c.post("/agents/spawn", json={"role": "x", "brief": ""}, headers=hc).status_code == 403
