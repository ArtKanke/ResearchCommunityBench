"""Replay export: a v1 research run in the lab database becomes a viewer-ready replay."""

from tests.test_research import ADMIN, PRODUCE, c, finding, put, setup  # first: sets the test env

from lab import replay  # noqa: E402
from lab.server import DB_PATH  # noqa: E402


def test_replay_from_db_has_agents_execs_and_artifacts():
    rid, ha, hb = setup()
    code = put(ha, PRODUCE)
    ex = c.post("/exec", json={"code": code, "cmd": "python run.py"}, headers=ha).json()
    good = c.post("/artifacts", json=finding(0.4213, [ex["exec_id"]], code), headers=ha).json()
    r = c.post("/artifacts", json=finding(0.9, [ex["exec_id"]], code, [{"target": good["artifact_id"], "relation": "extends"}]), headers=hb)
    assert r.status_code == 200, r.text
    rep = replay.from_db(DB_PATH, rid)
    assert rep["format"] == "replay/1" and rep["meta"]["source"] == "real" and rep["meta"]["benchmark"] == "v1"
    assert {a["id"] for a in rep["agents"]} >= {"founder"} and len(rep["agents"]) == 2
    kinds = {e["type"] for e in rep["events"]}
    assert {"spawn", "exec", "artifact"} <= kinds
    arts = {a["id"]: a for a in rep["artifacts"]}
    assert arts[good["artifact_id"]]["status"] == "pending"
    bad = [a for a in rep["artifacts"] if a["id"] != good["artifact_id"]][0]
    assert bad["status"] == "rejected" and bad["relations"][0]["target"] == good["artifact_id"]
    assert ADMIN  # admin header shared with the research tests
