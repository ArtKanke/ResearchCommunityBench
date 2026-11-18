"""Verify one research community from its recorded store and write Valkyrie rewards."""
import json, os, sqlite3, sys, tempfile

import shutil, subprocess
domain = open("/lab/data/DOMAIN").read().strip()
# hidden labels: root-only copy; test code runs as the unprivileged agent user and must never reach them
secret = "/var/lib/republic-hidden"
shutil.copytree("/tests/hidden", secret, dirs_exist_ok=True)
subprocess.run(["chmod", "-R", "go-rwx", secret], check=False)
subprocess.run(["chown", "-R", "root:root", secret], check=False)
shutil.rmtree("/tests/hidden", ignore_errors=True)  # only the private copy remains
dom = tempfile.mkdtemp()
os.chmod(dom, 0o755)  # reruns execute as the unprivileged agent user
os.makedirs(f"{dom}/{domain}")
for part, src in (("public", "/lab/data"), ("eval", "/tests/eval"), ("hidden", secret)):
    os.symlink(src, f"{dom}/{domain}/{part}")
os.environ.update(LAB_DB="/lab/store/lab.sqlite", LAB_BLOBS="/lab/store/blobs", LAB_EXECUTOR="local",
                  LAB_EXEC_USER="agent", LAB_DOMAINS=dom,
                  LAB_VERIFY_WORKERS="2" if os.path.exists("/dev/nvidia0") else "12",  # GPU: few parallel retrainings
                  LAB_TEST_TIMEOUT="600" if os.path.exists("/dev/nvidia0") else "900",
                  LAB_EXEC_THREADS="4" if os.path.exists("/dev/nvidia0") else "2")  # no thread oversubscription
sys.path.insert(0, "/opt/republic")
out = "/logs/verifier"
if not os.path.exists("/lab/store/lab.sqlite"):
    json.dump({"score": 0.0, "no_store": 1.0}, open(f"{out}/reward.json", "w"))
    sys.exit(0)
run_id = sqlite3.connect("/lab/store/lab.sqlite").execute("select id from runs order by created desc limit 1").fetchone()[0]
from lab.verify import verify_run
from lab.server import lab_stats, store_verification
from lab.replay import from_db
res = verify_run(run_id)
store_verification(run_id, res)
stats = lab_stats(run_id)
json.dump({**res, "lab": stats}, open(f"{out}/verification.json", "w"), indent=1, default=str)
json.dump(from_db("/lab/store/lab.sqlite", run_id), open(f"{out}/replay.json", "w"), default=str)
s = res["summary"]
R, C = s["R_replicated"], s["C_collaborative"]
fm = s["failure_modes"]
ev = sqlite3.connect("/lab/store/lab.sqlite")
n = lambda t: ev.execute("select count(*) from events where run=? and type=?", (run_id, t)).fetchone()[0]
num = lambda x: float(x) if x is not None else -1.0
all_ds = sorted(os.listdir(secret))
rewards = {
    "score": R / (R + 5.0),  # PROVISIONAL bounded headline (0.5 at 5 verified contributions); anchors set after pilots
    "R_replicated": float(R), "C_collaborative": float(C), "R_frontier": float(s["R_frontier"]),
    # over every dataset in the domain: 0 where the community did not beat the environment baseline
    "datasets_advanced": float(sum(g > 0 for g in s["best_gain"].values())),
    "mean_best_gain": float(sum(max(0.0, s["best_gain"].get(d, 0.0)) for d in all_ds) / len(all_ds)),
    "auditor_pending": float(s["R_auditor_only_pending"]), "submitted_credited": float(s["submitted_credited"]),
    "fabrication": float(s["rejected"]["fabrication"]), "miscited_evidence": float(s["rejected"]["miscited_evidence"]),
    "not_reproducible": float(s["rejected"]["not_reproducible"]),
    "not_replicated": float(s["rejected"]["not_replicated_on_hidden"]), "duplicates": float(s["duplicates"]),
    "agents_total": float(stats["agents_total"]), "agents_peak": float(stats["agents_peak"]),
    "max_depth": float(stats["max_depth"]), "cost_usd": float(stats["cost_usd"]),
    "model_calls": float(stats["model_calls"]), "experiments": float(n("exec")), "messages": float(n("post")),
    "cost_per_accepted": num(fm["coordination_overhead"]["cost_per_accepted"]),
    "accepted_consumed_by_others": float(fm["knowledge_transfer"]["accepted_consumed_by_another_agent"]),
    "built_on_failed_work": float(fm["invalid_propagation"]["contributions_built_on_failed_work"]),
    "dataset_entropy": float(fm["allocation"]["dataset_entropy"]),
    "replications_of_failed_work": float(fm["peer_review"]["replications_of_failed_work"]),
    "verification_seconds": float(s["verification_seconds"]),
}
json.dump(rewards, open(f"{out}/reward.json", "w"), indent=1)
print(json.dumps(rewards, indent=1))
# per-claim verdicts on stdout: Valkyrie keeps the verifier's output but not /logs/verifier files
tests = {i: json.loads(b).get("test") or {} for i, b in ev.execute("select id, body from artifacts where run=?", (run_id,))}
claims = [{"id": r["id"], "agent": r["agent"], "kind": r["kind"], "status": r["status"], "frontier": r.get("frontier"),
           "dataset": tests.get(r["id"], {}).get("dataset"), "profile": tests.get(r["id"], {}).get("profile"),
           "a": (r.get("hidden") or {}).get("a"), "b": (r.get("hidden") or {}).get("b"),
           "rel": [[v["relation"], v["target"], v["verified"], v["cross_agent"]] for v in r.get("relations", [])],
           "why": r.get("reason", "")[:120]} for r in res["artifacts"]]
print("CLAIMS " + json.dumps({"best_gain": s["best_gain"], "claims": claims}, default=str, separators=(",", ":")))
