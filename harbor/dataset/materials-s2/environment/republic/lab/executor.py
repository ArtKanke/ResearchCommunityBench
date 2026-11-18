"""Server-run executions (`lab exec`): the only results that count as evidence.

The server, not the agent, runs the code, so every number in a claim can be traced to a
recorded execution: code hash, input hashes, output hashes, exit code. Agents can still
experiment freely in their own sandbox; only witnessed runs back a claim.

Layout inside an execution:  /work  the code (a blob, extracted)
                             /data  the run's public datasets (read-only)
                             /in/NAME  input blobs (other agents' outputs, models, ...)
                             /out   everything written here is stored; metrics.json is parsed

Executors (LAB_EXECUTOR): docker (default, no network) | local (tests and dev only: runs on
the host) | modal (TODO: Modal Sandbox, for the hosted lab server).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

BLOB_DIR = os.environ.get("LAB_BLOBS", "results/blobs")
EXECUTOR = os.environ.get("LAB_EXECUTOR", "docker")
IMAGE = os.environ.get("LAB_EXEC_IMAGE", "republic-sandbox:dev")
MAX_OUT_BYTES = 50_000_000


def put_blob(data: bytes) -> str:
    sha = hashlib.sha256(data).hexdigest()
    path = os.path.join(BLOB_DIR, sha[:2], sha)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.{__import__('threading').get_ident()}.tmp"  # concurrent uploads of one blob
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    return sha


def get_blob(sha: str) -> bytes:
    if len(sha) != 64 or not all(ch in "0123456789abcdef" for ch in sha):
        raise KeyError(sha)
    path = os.path.join(BLOB_DIR, sha[:2], sha)
    if not os.path.exists(path):
        raise KeyError(sha)
    with open(path, "rb") as f:
        return f.read()


def _extract(blob: bytes, dest: str):
    """Code blobs are tar(.gz) archives; anything else is a single file named main."""
    os.makedirs(dest, exist_ok=True)
    try:
        with tarfile.open(fileobj=io.BytesIO(blob)) as t:
            t.extractall(dest, filter="data")
    except tarfile.ReadError:
        with open(os.path.join(dest, "main"), "wb") as f:
            f.write(blob)


def run(code: str, cmd: str, inputs: dict[str, str], data_dir: str | None, timeout: int = 900,
        eval_dir: str | None = None, input_files: dict[str, bytes] | None = None) -> dict:
    """Run `cmd` on code blob `code`. Returns the exec record (without id/run/agent).
    `eval_dir` (verification only) mounts hidden inputs at /eval; `input_files` overrides /in files
    with raw bytes (the evaluator's ablations)."""
    root = tempfile.mkdtemp(prefix="exec-")
    os.chmod(root, 0o777)
    work, inp, out = (os.path.join(root, d) for d in ("work", "in", "out"))
    try:
        _extract(get_blob(code), work)
        os.makedirs(inp)
        os.makedirs(out)
        for d in (work, inp, out):
            os.chmod(d, 0o777)
            for dirpath, dirnames, filenames in os.walk(d):
                for x in dirnames + filenames:
                    os.chmod(os.path.join(dirpath, x), 0o777)
        for name, sha in inputs.items():
            with open(os.path.join(inp, os.path.basename(name)), "wb") as f:
                f.write((input_files or {}).get(name) or get_blob(sha))
        t0 = time.time()
        if EXECUTOR == "local":
            path = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
            env = {**os.environ, "PATH": path, "DATA_DIR": os.path.abspath(data_dir) if data_dir else "",
                   "EVAL_DIR": os.path.abspath(eval_dir) if eval_dir else "", "IN_DIR": inp, "OUT_DIR": out, "WORK_DIR": work}
            p = _sub(["bash", "-c", cmd], work, env, timeout)
        elif EXECUTOR == "docker":
            vols = ["-v", f"{work}:/work", "-v", f"{inp}:/in:ro", "-v", f"{out}:/out"]
            if data_dir:
                vols += ["-v", f"{os.path.abspath(data_dir)}:/data:ro"]
            if eval_dir:
                vols += ["-v", f"{os.path.abspath(eval_dir)}:/eval:ro"]
            p = _sub(["docker", "run", "--rm", "--network", "none", *vols, "-w", "/work",
                      "-e", "DATA_DIR=/data", "-e", f"EVAL_DIR={'/eval' if eval_dir else ''}", "-e", "IN_DIR=/in", "-e", "OUT_DIR=/out",
                      IMAGE, "bash", "-c", cmd], None, None, timeout)
        else:
            raise RuntimeError(f"executor {EXECUTOR} not available")
        outputs, total = {}, 0
        for dirpath, _, files in os.walk(out):
            for fn in files:
                path = os.path.join(dirpath, fn)
                total += os.path.getsize(path)
                if total > MAX_OUT_BYTES:
                    break
                with open(path, "rb") as f:
                    outputs[os.path.relpath(path, out)] = put_blob(f.read())
        metrics = None
        if "metrics.json" in outputs:
            try:
                metrics = json.loads(get_blob(outputs["metrics.json"]))
            except ValueError:
                metrics = None
        return {"code": code, "cmd": cmd, "inputs": inputs, "outputs": outputs, "metrics": metrics,
                "exit_code": p["exit_code"], "stdout": p["stdout"][-4000:], "seconds": round(time.time() - t0, 2)}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _demote():
    """Agent code never runs as root (it could otherwise read the server's keys or rewrite the store)."""
    user = os.environ.get("LAB_EXEC_USER")
    if not user or os.getuid() != 0:
        return None
    import pwd
    pw = pwd.getpwnam(user)

    def demote():
        os.setgid(pw.pw_gid)
        os.setuid(pw.pw_uid)
    return demote


SECRET_ENV = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LAB_ADMIN_KEY")


THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT")


def _sub(argv, cwd, env, timeout):
    if env is not None:
        env = {k: v for k, v in env.items() if k not in SECRET_ENV}
        if os.environ.get("LAB_EXEC_THREADS"):  # parallel experiments must not each grab every core
            env.update({v: os.environ["LAB_EXEC_THREADS"] for v in THREAD_VARS})
    try:
        r = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout, preexec_fn=_demote())
        return {"exit_code": r.returncode, "stdout": (r.stdout or "") + (r.stderr or "")}
    except subprocess.TimeoutExpired as e:
        return {"exit_code": 124, "stdout": f"timeout after {timeout}s\n" + str(e.stdout or "")}
