"""Package the benchmark as Harbor tasks for Valkyrie, one dataset per domain and mode.

  .venv/bin/python scripts/build_harbor.py --domains materials,compbio,aiml
  valkyrie dataset push republic/aiml ./harbor/republic-aiml --tag v3

One task = one research community (or solo researcher) started from scratch in one domain, with
its own random hidden split (seed). Layout per task:
  instruction.md            the charter (no research question)
  task.toml                 agent phase with model APIs only; separate no-network verifier
  environment/              image: public data, the lab toolkit (server, CLI, bootstrap loop), workbench
  tests/                    hidden inputs + labels and the verifier; only the verifier sandbox gets them
The agent bundle (harbor/agents/republic-kernel) starts the in-sandbox lab server and the founder.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lab.charter import guide, open_charter  # noqa: E402
from scripts.build_domains import build_aiml, build_compbio  # noqa: E402
from scripts.build_materials import main as build_materials  # noqa: E402

DOMAINS = {
    "materials": {"build": build_materials, "cpus": 16, "memory_mb": 32768, "gpus": 0,
                  "compute": "16 CPUs, 32 GB RAM, no GPU"},
    "compbio": {"build": build_compbio, "cpus": 16, "memory_mb": 32768, "gpus": 0,
                "compute": "16 CPUs, 32 GB RAM, no GPU"},
    "aiml": {"build": build_aiml, "cpus": 8, "memory_mb": 32768, "gpus": 1, "gpu_type": "a10",
             "compute": "8 CPUs, 32 GB RAM, 1 NVIDIA A10 GPU (24 GB), shared by every researcher through lab exec",
             "test_note": ("\n  Tests get 10 minutes each: write your evidence run so it saves the trained weights to "
                           "`$OUT_DIR`, and pass that blob as `test.inputs`, so the test only loads the weights and "
                           "predicts.")},
}

TASK_TOML = """# ResearchCommunityBench: a research community ({mode}) in {domain}, seed {seed}.
artifacts = ["/lab/store"]

[task]
name = "researchcommunitybench/{domain}{suffix}-s{seed}"
description = "Open-ended research community: verified research output and collaboration"
keywords = ["research-agents", "multi-agent", "science", "verification"]

[metadata]
domain = "{domain}"
mode = "{mode}"
seed = {seed}
minutes = {minutes}

[environment]
build_timeout_sec = 3600
cpus = {cpus}
memory_mb = {memory_mb}
network_mode = "public"{gpu}

[agent]
timeout_sec = {agent_timeout}
network_mode = "allowlist"
allowed_hosts = ["api.anthropic.com", "api.openai.com"]

[verifier]
timeout_sec = {verifier_timeout}
environment_mode = "separate"
network_mode = "no-network"

[verifier.environment]
cpus = {verifier_cpus}
memory_mb = 32768
network_mode = "no-network"{gpu}
"""

APP = """RUN useradd -m -s /bin/bash agent
COPY republic /opt/republic
COPY data /lab/data
RUN mkdir -p /lab/store /lab/logs /lab/cursors /work \\
    && chown -R agent:agent /work /lab/logs /lab/cursors && chmod 700 /lab/store \\
    && ln -sf /opt/republic/agent/lab /usr/local/bin/lab && chmod 755 /opt/republic/agent/lab
ENV PYTHONPATH=/opt/republic LAB_DIR=/lab WORK_DIR=/work{env}
WORKDIR /work
"""

LAB_PIP = 'fastapi "uvicorn[standard]" "pydantic>=2.10" "httpx>=0.28" litellm "anthropic==2.8.0" pyarrow'

DOCKERFILE_CPU = """FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends procps curl git build-essential xxd \\
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy scipy scikit-learn pandas lightgbm sympy networkx """ + LAB_PIP + "\n"

DOCKERFILE_GPU = """FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends procps curl git xxd && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade torchvision==0.20.1 numpy scipy scikit-learn pandas "brotli>=1.1" """ + LAB_PIP + """
COPY prepare_cifar.py /tmp/prepare_cifar.py
RUN python /tmp/prepare_cifar.py /lab/data/cifar10 && rm /tmp/prepare_cifar.py
"""

PREPARE_CIFAR = '''"""Download CIFAR-10 at image build time and keep only the training batches (test labels never enter the image)."""
import io, os, pickle, sys, tarfile, urllib.request
import numpy as np
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
data = urllib.request.urlopen("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", timeout=600).read()
tf = tarfile.open(fileobj=io.BytesIO(data))
xs, ys = [], []
for i in range(1, 6):
    d = pickle.load(tf.extractfile(f"cifar-10-batches-py/data_batch_{i}"), encoding="bytes")
    xs.append(d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)); ys.append(np.array(d[b"labels"]))
x, y = np.concatenate(xs), np.concatenate(ys)
np.savez_compressed(f"{out}/train.npz", x=x, y=y, id=np.array([f"cifar-tr-{i:05d}" for i in range(len(x))]))
print("cifar10 train", x.shape)
'''

TEST_SH = """#!/bin/bash
# Separate verifier: /tests holds the hidden data; /lab/store is the community's record (copied artifact).
set -uo pipefail
mkdir -p /logs/verifier
cd /opt/republic && python /tests/verify_task.py
"""

VERIFY_TASK = '''"""Verify one research community from its recorded store and write Valkyrie rewards."""
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
'''


def copy_toolkit(dst):
    for d in ("lab", "agent"):
        shutil.copytree(os.path.join(ROOT, d), os.path.join(dst, d), ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def card_line(path):
    lines = [x.strip() for x in open(path).read().splitlines() if x.strip() and not x.startswith("#")]
    return lines[0][:220] if lines else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="materials,compbio,aiml")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--minutes", type=int, default=120)
    ap.add_argument("--max-live", type=int, default=50)
    ap.add_argument("--modes", default="community,solo")
    ap.add_argument("--out", default=os.path.join(ROOT, "harbor"))
    ap.add_argument("--combined", default="", help="also copy every task into OUT/COMBINED (one dataset for all conditions)")
    a = ap.parse_args()
    for domain in a.domains.split(","):
        cfg = DOMAINS[domain]
        for mode in a.modes.split(","):
            suffix = "" if mode == "community" else "-solo"
            ds_dir = os.path.join(a.out, f"republic-{domain}{suffix}")
            shutil.rmtree(ds_dir, ignore_errors=True)
            for seed in [int(s) for s in a.seeds.split(",")]:
                task = os.path.join(ds_dir, f"{domain}{suffix}-s{seed}")
                env, tests = os.path.join(task, "environment"), os.path.join(task, "tests")
                os.makedirs(env)
                os.makedirs(tests)
                env_seed = 1 if domain == "aiml" else seed  # aiml: fixed official test sets, one shared image
                tmp = os.path.join(ROOT, "harbor", ".build", f"{domain}-s{env_seed}")
                if not os.path.exists(tmp):
                    cfg["build"](tmp, 20261114 + env_seed)
                ignore = shutil.ignore_patterns("train.npz") if domain == "aiml" else None  # CIFAR is fetched at image build
                shutil.copytree(os.path.join(tmp, "public"), os.path.join(env, "data"), ignore=ignore)
                open(os.path.join(env, "data", "DOMAIN"), "w").write(domain)
                open(os.path.join(env, "data", "GUIDE.md"), "w").write(guide("community", cfg.get("test_note", "")))
                shutil.copytree(os.path.join(tmp, "eval"), os.path.join(tests, "eval"))
                shutil.copytree(os.path.join(tmp, "hidden"), os.path.join(tests, "hidden"))
                copy_toolkit(os.path.join(env, "republic"))
                if cfg["gpus"]:
                    open(os.path.join(env, "prepare_cifar.py"), "w").write(PREPARE_CIFAR)
                    docker = DOCKERFILE_GPU + APP.format(env=f" LAB_GPU_SLOTS={cfg['gpus']}")
                else:
                    docker = DOCKERFILE_CPU + APP.format(env="")
                open(os.path.join(env, "Dockerfile"), "w").write(docker)
                datasets = [(d, card_line(os.path.join(tmp, "public", d, "README.md"))) for d in sorted(os.listdir(os.path.join(tmp, "public")))
                            if d != "baselines" and os.path.exists(os.path.join(tmp, "public", d, "README.md"))]
                open(os.path.join(task, "instruction.md"), "w").write(
                    open_charter(mode, domain, None, a.minutes, a.max_live, datasets, cfg["compute"], cfg.get("test_note", "")))
                gpu = f'\ngpus = {cfg["gpus"]}\ngpu_types = ["{cfg["gpu_type"]}"]' if cfg["gpus"] else ""
                open(os.path.join(task, "task.toml"), "w").write(TASK_TOML.format(
                    mode=mode, domain=domain, suffix=suffix, seed=seed, minutes=a.minutes, cpus=cfg["cpus"],
                    memory_mb=cfg["memory_mb"], gpu=gpu, agent_timeout=a.minutes * 60 + 600, verifier_timeout=max(10800, a.minutes * 100),
                    verifier_cpus=8 if cfg["gpus"] else 16))
                open(os.path.join(tests, "test.sh"), "w").write(TEST_SH)
                os.chmod(os.path.join(tests, "test.sh"), 0o755)
                open(os.path.join(tests, "verify_task.py"), "w").write(VERIFY_TASK)
                print("wrote", os.path.relpath(task, ROOT))
                if a.combined:
                    dst = os.path.join(a.out, a.combined, os.path.basename(task))
                    shutil.rmtree(dst, ignore_errors=True)
                    shutil.copytree(task, dst)


if __name__ == "__main__":
    main()
