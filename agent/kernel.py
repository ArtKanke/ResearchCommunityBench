#!/usr/bin/env python3
"""The identical lab kernel every model runs on: a bash-only agent loop (mini-swe-agent
style). Each turn the model returns a THOUGHT and one ```bash block; the kernel runs it and
returns the output. All lab tools are `lab ...` commands. Standard library only."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

LAB = os.environ.get("LAB_DIR", "/lab")
WORK = os.environ.get("WORK_DIR", "/work")
MAX_OBS = 6000
MAX_HISTORY_CHARS = 150_000
CMD_TIMEOUT = 900

PROTOCOL = """## How you act

Each reply: a short THOUGHT, then exactly one fenced bash block:

THOUGHT: what you are doing and why.
```bash
<one or more shell commands>
```

It runs in the shared work directory and you get its output plus the time left in the session. A block is
stopped after {timeout}s; run long jobs in the background and check on them later. `lab wait` pauses without
using the model until a message or a new contribution arrives.
"""


def run_cfg():
    with open(os.path.join(LAB, "run.json")) as f:
        return json.load(f)


def call(cfg, agent, method, path, body=None, timeout=900):
    req = urllib.request.Request(cfg["server"].rstrip("/") + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {cfg['token']}", "X-Agent": agent,
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("detail", "")
        except Exception:
            detail = ""
        return e.code, {"detail": detail}
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:  # never crash the loop on a socket error
        return 599, {"detail": f"connection problem: {e}"}


def system_prompt(cfg, agent):
    charter = open(os.path.join(LAB, "charter.md")).read()
    if agent == "founder":
        who = "You are the founder of this research community."
        brief = ""
    else:
        _, b = call(cfg, agent, "GET", f"/agents/{agent}/brief")
        who = f"You are researcher `{agent}` in this research community. Your role: {b.get('role', '')} (recruited by `{b.get('parent')}`)."
        brief = f"\nYour brief: {b.get('brief', '')}\n"
    return f"{who}\n\n{charter}\n{brief}\n{PROTOCOL.format(timeout=CMD_TIMEOUT)}"


def parse(text):
    blocks = re.findall(r"```(?:bash|sh)?\s*\n(.*?)```", text, re.S)
    return blocks[0].strip() if blocks else None


def _drop_privileges():
    """Agents' commands never run as root: they cannot read the harness's API key or edit the lab's logs."""
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


def execute(cmd, agent):
    env = {k: v for k, v in os.environ.items() if k not in SECRET_ENV}
    env["LAB_AGENT_ID"] = agent
    if os.getuid() == 0 and os.environ.get("LAB_EXEC_USER"):
        env["HOME"] = "/home/" + os.environ["LAB_EXEC_USER"]
    try:
        p = subprocess.run(["bash", "-lc", cmd], cwd=WORK, env=env, capture_output=True, text=True, errors="replace",
                           timeout=CMD_TIMEOUT, preexec_fn=_drop_privileges())
        out = (p.stdout + ("\n[stderr]\n" + p.stderr if p.stderr.strip() else "")).strip()
        out = f"[exit {p.returncode}]\n{out}"
    except subprocess.TimeoutExpired as e:
        out = f"[timed out after {CMD_TIMEOUT}s]\n{(e.stdout or '')[-2000:] if isinstance(e.stdout, str) else ''}"
    if len(out) > MAX_OBS:
        out = out[: MAX_OBS // 2] + f"\n... [{len(out) - MAX_OBS} chars truncated] ...\n" + out[-MAX_OBS // 2:]
    return out


def trim(messages):
    while sum(len(m["content"]) for m in messages) > MAX_HISTORY_CHARS and len(messages) > 4:
        del messages[2:4]
    return messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-id", default=os.environ.get("LAB_AGENT_ID", "founder"))
    ap.add_argument("--max-steps", type=int, default=3000)  # 400 capped busy researchers at ~2.5 h of a 6 h session
    args = ap.parse_args()
    agent = args.agent_id
    os.makedirs(WORK, exist_ok=True)
    cfg = run_cfg()
    messages = [{"role": "system", "content": system_prompt(cfg, agent)},
                {"role": "user", "content": "Begin." if agent == "founder" else "Begin your brief."}]
    failures = 0
    for step in range(args.max_steps):
        status, out = call(cfg, agent, "POST", "/v1/complete", {"messages": trim(messages)})
        if status in (402, 409):
            print(f"[{agent}] stopping: {out.get('detail')}", flush=True)
            break
        if status != 200:
            print(f"[{agent}] model call failed ({status}): {str(out.get('detail'))[:300]}", flush=True)
            failures += 1
            if failures > 5:
                break
            time.sleep(min(60, 2**failures))
            continue
        failures = 0
        text = out["text"]
        messages.append({"role": "assistant", "content": text})
        cmd = parse(text)
        if cmd is None:
            messages.append({"role": "user", "content": "FORMAT ERROR: reply with a THOUGHT and exactly one ```bash block."})
            continue
        obs = execute(cmd, agent)
        print(f"[{agent}] step {step}: {cmd[:200]!r}\n{obs[:500]}", flush=True)
        st, b = call(cfg, agent, "GET", "/budget")  # also tells us whether `lab done` really ran (409 once exited)
        if st == 409:
            return
        clock = f"\n[time left in the session: {b.get('seconds_left', 0) // 60} min {b.get('seconds_left', 0) % 60} s]" if st == 200 else ""
        messages.append({"role": "user", "content": f"OBSERVATION:\n{obs}{clock}"})
    call(cfg, agent, "POST", "/agents/exit", {"note": "kernel loop ended"})


if __name__ == "__main__":
    sys.exit(main())
