"""Re-run the task verifier on Modal for downloaded main-run stores (Valkyrie keeps no state to regrade these runs).

Each task runs in its own task image (built from the dataset's environment/Dockerfile), with no network, the
task's tests/ (hidden labels) and the community's store mounted, the same way Valkyrie's separate verifier runs it.
Writes results/<runs>/<model>/reverify/<task>.json with the rewards, the per-claim verdicts and the full verification.

  MODAL_PROFILE=artkanke .venv/bin/python scripts/reverify_modal.py [--models haiku,opus] [--tasks compbio-solo-s3]
  # 6-hour runs, detached: sandboxes write to a Modal volume, so the laptop can sleep; collect later
  MODAL_PROFILE=artkanke .venv/bin/python scripts/reverify_modal.py --runs long --models haiku,luna --detach
  MODAL_PROFILE=artkanke .venv/bin/python scripts/reverify_modal.py --runs long --models haiku,luna --collect
"""

import argparse
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = os.path.join(ROOT, "harbor/main2/republic-main")
RUNS = "main"
TASK_DIRS = {"main": "harbor/main2/republic-main", "long": "harbor/long/republic-long"}
CMD = ("mkdir -p /lab/store && cp -a /in/store/. /lab/store/ && cp -a /in/tests /tests && bash /tests/test.sh > /tmp/out.txt 2>&1;"
       " echo \"EXIT $?\"; grep '^CLAIMS ' /tmp/out.txt | head -1 | cut -c1-8; tail -c 3000 /tmp/out.txt | grep -v '^CLAIMS' | tail -40 >&2;"
       " echo '===REWARD'; cat /logs/verifier/reward.json; echo '===VERIFICATION'; cat /logs/verifier/verification.json")

app = modal.App.lookup("rcb-reverify", create_if_missing=True)
VOL = "rcb-reverify-out"
_images: dict[str, modal.Image] = {}


def image(task):
    env = os.path.join(TASKS, task, "environment")
    if task not in _images:
        _images[task] = modal.Image.from_dockerfile(os.path.join(env, "Dockerfile"), context_dir=env)
    return _images[task]


def parse(stdout):
    return (json.loads(stdout.split("===REWARD")[1].split("===VERIFICATION")[0]),
            json.loads(stdout.split("===VERIFICATION")[1]))


def one(model, task, detach=False):
    store = os.path.join(ROOT, f"results/{RUNS}/{model}/outputs/{task}/artifacts/lab/store/store")
    out = os.path.join(ROOT, f"results/{RUNS}/{model}/reverify/{task}.json")
    if os.path.exists(out):
        return model, task, "cached"
    if not os.path.exists(os.path.join(store, "lab.sqlite")):
        return model, task, "no store"
    gpu = task.startswith("aiml")
    img = (image(task).add_local_dir(store, "/in/store", copy=False)
           .add_local_dir(os.path.join(TASKS, task, "tests"), "/in/tests", copy=False))
    t0 = time.time()
    if detach:  # the sandbox keeps its own output on a volume; nothing here waits for it
        cmd = f"mkdir -p /out/{RUNS}/{model}; ({CMD}) > /out/{RUNS}/{model}/{task}.txt 2>&1; sync"
        sb = modal.Sandbox.create("bash", "-c", cmd, image=img, app=app, gpu="a10" if gpu else None,
                                  cpu=8 if gpu else 16, memory=32768, timeout=10 * 3600, block_network=True,
                                  volumes={"/out": modal.Volume.from_name(VOL, create_if_missing=True)})
        return model, task, f"launched {sb.object_id}"
    sb = modal.Sandbox.create("bash", "-c", CMD, image=img, app=app, gpu="a10" if gpu else None,
                              cpu=8 if gpu else 16, memory=32768, timeout=4 * 3600, block_network=True)
    sb.wait()
    stdout, stderr = sb.stdout.read(), sb.stderr.read()
    try:
        reward, verification = parse(stdout)
    except (IndexError, ValueError):
        return model, task, f"FAILED after {time.time() - t0:.0f}s: {stdout[:300]} {stderr[-800:]}"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"model": model, "task": task, "seconds": round(time.time() - t0), "rewards": reward,
               "verification": verification}, open(out, "w"), default=str)
    return model, task, f"ok {time.time() - t0:.0f}s R={reward.get('R_replicated')} F={reward.get('R_frontier')}"


def collect(model, task):
    out = os.path.join(ROOT, f"results/{RUNS}/{model}/reverify/{task}.json")
    if os.path.exists(out):
        return model, task, "cached"
    try:
        stdout = b"".join(modal.Volume.from_name(VOL).read_file(f"{RUNS}/{model}/{task}.txt")).decode()
        reward, verification = parse(stdout)
    except Exception as e:  # not written yet (still running) or the verifier failed
        return model, task, f"not ready: {str(e)[:120]}"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"model": model, "task": task, "rewards": reward, "verification": verification}, open(out, "w"), default=str)
    return model, task, f"ok R={reward.get('R_replicated')} F={reward.get('R_frontier')}"


def main():
    global TASKS, RUNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="main", choices=sorted(TASK_DIRS))
    ap.add_argument("--detach", action="store_true", help="launch and return; results land on the volume")
    ap.add_argument("--collect", action="store_true", help="fetch finished detached results from the volume")
    ap.add_argument("--models", default="haiku,opus,6.1-sol,6-luna")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    RUNS, TASKS = a.runs, os.path.join(ROOT, TASK_DIRS[a.runs])
    jobs = []
    for m in a.models.split(","):
        for d in sorted(glob.glob(os.path.join(ROOT, f"results/{RUNS}/{m}/outputs/*-s[0-9]"))):
            t = os.path.basename(d)
            if not a.tasks or t in a.tasks.split(","):
                jobs.append((m, t))
    print(f"{len(jobs)} tasks", flush=True)
    with ThreadPoolExecutor(a.workers) as pool:
        for m, t, msg in pool.map(lambda j: collect(*j) if a.collect else one(*j, detach=a.detach), jobs):
            print(f"{m:8} {t:18} {msg}", flush=True)


if __name__ == "__main__":
    main()
