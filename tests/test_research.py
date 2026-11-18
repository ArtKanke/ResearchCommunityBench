"""Open-ended research API (schema v1): server-run execs, live checks, fabrication."""

import base64
import io
import os
import tarfile
import tempfile

os.environ.setdefault("LAB_DB", os.path.join(tempfile.mkdtemp(), "t.sqlite"))
os.environ.setdefault("LAB_ADMIN_KEY", "k")

from fastapi.testclient import TestClient  # noqa: E402

from lab import executor  # noqa: E402
from lab.server import ADMIN_KEY, app  # noqa: E402

executor.EXECUTOR = "local"
executor.BLOB_DIR = tempfile.mkdtemp()
c = TestClient(app)
ADMIN = {"Authorization": f"Bearer {ADMIN_KEY}"}


def tar(files: dict[str, str]) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        for name, text in files.items():
            data = text.encode()
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))
    return base64.b64encode(buf.getvalue()).decode()


def setup(domain="", minutes=5):
    r = c.post("/admin/runs", json={"seed": 1, "model": "mock", "minutes": minutes, "domain": domain,
                                    **({} if domain else {"board": ["M1"]})}, headers=ADMIN).json()
    h = {"Authorization": f"Bearer {r['token']}", "X-Agent": "founder"}
    b = c.post("/agents/spawn", json={"role": "analyst", "brief": ""}, headers=h).json()["agent_id"]
    return r["run_id"], h, {**h, "X-Agent": b}


def put(h, files):
    return c.post("/blobs", json={"b64": tar(files)}, headers=h).json()["blob"]


def run(h, code, inputs=None):
    r = c.post("/exec", json={"code": code, "cmd": "python run.py", "inputs": inputs or {}}, headers=h).json()
    assert r["exit_code"] == 0, r["stdout"]
    return r


PRODUCE = {"run.py": "import json, os\nopen(os.environ['OUT_DIR'] + '/table.csv', 'w').write('x,y\\n1,2\\n')\n"
                     "json.dump({'value': 0.4213, 'n': 2}, open(os.environ['OUT_DIR'] + '/metrics.json', 'w'))\n"}


def finding(value, evidence, test_code, relations=(), profile="custom"):
    return {"kind": "finding", "title": "A measured effect",
            "claim": {"statement": "The effect on the toy dataset is about 0.42.", "datasets": ["toy"],
                      "quantities": [{"name": "value", "value": value}]},
            "test": {"profile": profile, "code": test_code, "cmd": "python run.py"},
            "evidence": evidence, "relations": list(relations)}


def test_live_checks_catch_fabrication_and_unconsumed_dependencies():
    _, ha, hb = setup()
    code = put(ha, PRODUCE)
    ex = run(ha, code)
    assert ex["metrics"]["value"] == 0.4213 and set(ex["outputs"]) == {"metrics.json", "table.csv"}
    good = c.post("/artifacts", json=finding(0.421, [ex["exec_id"]], code), headers=ha).json()
    assert good["live_checks"]["ok"], good  # rounding to 3 significant figures is fine
    fake = c.post("/artifacts", json=finding(0.61, [ex["exec_id"]], code), headers=ha).json()
    assert not fake["live_checks"]["ok"] and fake["live_checks"]["fabrication"]
    # B says it used A's table but no evidence exec of B consumed it
    exb = run(hb, put(hb, {"run.py": PRODUCE["run.py"].replace("table.csv", "mine.csv")}))
    rel = {"target": good["artifact_id"], "relation": "extends", "used": [ex["outputs"]["table.csv"]]}
    bad = c.post("/artifacts", json=finding(0.4213, [exb["exec_id"]], exb["code"], [rel]), headers=hb).json()
    assert not bad["live_checks"]["checks"]["used_inputs_consumed"]


def test_schema_rules():
    _, ha, _ = setup()
    code = put(ha, PRODUCE)
    assert c.post("/artifacts", json=finding(0.42, [], code), headers=ha).status_code == 422  # finding needs evidence
    assert c.post("/artifacts", json=finding(0.5, [], code) | {"kind": "hypothesis"}, headers=ha).status_code == 200
    null = finding(0.42, ["exec-x"], code) | {"kind": "null_result"}
    assert c.post("/artifacts", json=null, headers=ha).status_code == 422  # needs equivalence + prespec
    pred = finding(0.42, ["exec-x"], code, profile="prediction")
    assert c.post("/artifacts", json=pred, headers=ha).status_code == 422  # needs a dataset
    assert c.post("/artifacts", json={"kind": "note", "title": "Plan for today"}, headers=ha).json()["live_checks"]["ok"]


def test_failed_exec_is_not_evidence():
    _, ha, _ = setup()
    code = put(ha, {"run.py": "raise SystemExit(3)\n"})
    ex = c.post("/exec", json={"code": code, "cmd": "python run.py"}, headers=ha).json()
    assert ex["exit_code"] == 3
    r = c.post("/artifacts", json=finding(0.42, [ex["exec_id"]], code), headers=ha).json()
    assert not r["live_checks"]["checks"]["evidence_succeeded"]


def test_miscited_evidence_is_not_fabrication():
    _, ha, _ = setup()
    code = put(ha, PRODUCE)
    first = run(ha, code)  # records value 0.4213
    other = run(ha, put(ha, {"run.py": PRODUCE["run.py"].replace("0.4213", "0.9")}))
    r = c.post("/artifacts", json=finding(0.4213, [other["exec_id"]], code), headers=ha).json()["live_checks"]
    assert not r["ok"] and not r["fabrication"] and "mis-cited evidence" in r["why"][0] and first["exec_id"] in r["why"][0]
    r = c.post("/artifacts", json=finding(0.777, [other["exec_id"]], code), headers=ha).json()["live_checks"]
    assert r["fabrication"]
