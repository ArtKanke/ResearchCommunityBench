"""Model backends behind the metering proxy. Provider keys live only on the lab server.

Model strings:  mock                      scripted policy, for tests (free)
                codex[:MODEL]             local `codex exec` (ChatGPT subscription), for dev runs
                claude[:MODEL]            local `claude -p` (Claude subscription), for dev runs
                litellm:PROVIDER/MODEL    real APIs through LiteLLM, keys from the server env
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass

# USD per million tokens: (input, output) or (input, output, cache_read, cache_write).
# Secondary-source prices (Nov 2026); verify on the provider pages before reporting. Override with LAB_PRICES.
PRICES = {
    "anthropic/claude-opus-5-5": (4.0, 20.0, 0.20, 5.0),
    "anthropic/claude-sonnet-5-5": (3.0, 15.0, 0.30, 3.75),
    "anthropic/claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
    "openai/gpt-6-astra": (10.0, 50.0, 1.0, 10.0),
    "openai/gpt-6.1-sol": (2.0, 10.0, 0.20, 2.0),
    "openai/gpt-6-luna": (0.25, 2.0, 0.025, 0.25),
    "codex": (1.25, 10.0),  # nominal, so budgets bite during dev runs
    "claude": (3.0, 15.0),  # nominal
    "mock": (0.0, 0.0),
}
PRICES.update({k: tuple(v) for k, v in json.loads(os.environ.get("LAB_PRICES", "{}")).items()})


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cached_tokens: int = 0


def _price(model: str) -> tuple[float, ...]:
    key = model.split(":", 1)[-1]
    return PRICES.get(key) or PRICES.get(model.split(":", 1)[0]) or (5.0, 25.0)


def _flatten(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        parts.append(f"### {m['role'].upper()}\n{m['content']}")
    parts.append("### ASSISTANT\n(Reply as the assistant above would: one THOUGHT and one bash block.)")
    return "\n\n".join(parts)


EFFORT = os.environ.get("LAB_EFFORT", "medium")  # same nominal reasoning effort for every model in a run
_clients: dict = {}


def _anthropic(name: str, messages: list[dict], max_tokens: int) -> Completion:
    """Claude through the official SDK: adaptive thinking (the default), explicit effort, prompt caching.
    No server-side fallback to another model: a run measures one model, so refusals are reported as errors."""
    import anthropic

    client = _clients.setdefault("anthropic", anthropic.Anthropic(max_retries=4))
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    convo = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages if m["role"] != "system"]
    if convo:
        convo[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    r = client.messages.create(model=name, max_tokens=max_tokens, messages=convo,
                               system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                               output_config={"effort": EFFORT})
    u = r.usage
    cached, written, fresh = u.cache_read_input_tokens or 0, u.cache_creation_input_tokens or 0, u.input_tokens or 0
    p = _price("anthropic/" + name)
    pread, pwrite = (p[2], p[3]) if len(p) > 2 else (p[0], p[0])
    cost = (fresh * p[0] + cached * pread + written * pwrite + u.output_tokens * p[1]) / 1e6
    if r.stop_reason == "refusal":
        cat = getattr(getattr(r, "stop_details", None), "category", None)
        raise RuntimeError(f"refusal ({cat}); cost {cost:.4f}")
    text = "".join(b.text for b in r.content if b.type == "text")
    return Completion(text, fresh + cached + written, u.output_tokens, cost, cached)


def complete(model: str, messages: list[dict], max_tokens: int = 16000) -> Completion:
    kind = model.split(":", 1)[0]
    if kind == "mock":
        from .mock_policy import respond

        text = respond(messages)
        n_in = sum(len(m["content"]) for m in messages) // 4
        return Completion(text, n_in, len(text) // 4, 0.0)

    if kind == "codex":
        prompt = _flatten(messages)
        with tempfile.NamedTemporaryFile("r", suffix=".txt") as out:
            cmd = ["codex", "exec", "--skip-git-repo-check", "-s", "read-only", "--ephemeral",
                   "-c", 'model_reasoning_effort="low"', "--output-last-message", out.name, "-"]
            if ":" in model:
                cmd[2:2] = ["-m", model.split(":", 1)[1]]
            subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=600, cwd=tempfile.gettempdir())
            text = open(out.name).read().strip()
        n_in, n_out = len(prompt) // 4, len(text) // 4
        pin, pout = _price("codex")
        return Completion(text, n_in, n_out, (n_in * pin + n_out * pout) / 1e6)

    if kind == "claude":
        # Local Claude Code CLI on the user's subscription; tools off so it answers as a plain model.
        # A real system prompt plus a task message (a flattened "continue this transcript" prompt is refused).
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system").replace("THOUGHT", "PLAN")
        steps = []
        for m in messages:
            if m["role"] == "assistant":
                steps.append("YOUR PREVIOUS STEP:\n" + m["content"].replace("THOUGHT", "PLAN"))
            elif m["role"] == "user":
                steps.append("RESULT:\n" + m["content"])
        prompt = ("Here is the work log of this session so far.\n\n" + "\n\n".join(steps) +
                  "\n\nWrite your next step: one short PLAN line, then exactly one ```bash block.")
        cmd = ["claude", "-p", "--output-format", "json", "--tools", "", "--no-session-persistence", "--system-prompt", system]
        if ":" in model:
            cmd += ["--model", model.split(":", 1)[1]]
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=600, cwd=tempfile.gettempdir())
        d = json.loads(r.stdout or "{}")
        if d.get("is_error"):
            raise RuntimeError(f"claude CLI: {d.get('result')}")
        text = (d.get("result") or "").strip()
        n_in, n_out = len(prompt) // 4, len(text) // 4
        pin, pout = _price("claude")
        return Completion(text, n_in, n_out, (n_in * pin + n_out * pout) / 1e6)

    if kind == "litellm" and model.split(":", 1)[1].startswith("anthropic/"):
        return _anthropic(model.split(":", 1)[1].split("/", 1)[1], messages, max_tokens)

    if kind == "litellm":
        import litellm

        name = model.split(":", 1)[1]
        msgs = [dict(m) for m in messages]
        if name.startswith("anthropic/"):  # prompt caching: system prompt and the conversation so far
            for i in {0, len(msgs) - 1}:
                msgs[i]["content"] = [{"type": "text", "text": msgs[i]["content"], "cache_control": {"type": "ephemeral"}}]
        extra = {"reasoning_effort": EFFORT} if name.startswith("openai/") else {}
        r = litellm.completion(model=name, messages=msgs, max_tokens=max_tokens, num_retries=4, **extra)  # same retries as Claude
        u = r.usage
        cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or getattr(u, "cache_read_input_tokens", 0) or 0
        written = getattr(u, "cache_creation_input_tokens", 0) or 0
        fresh = max(0, u.prompt_tokens - cached - written)
        p = _price(name)
        pin, pout = p[0], p[1]
        pread, pwrite = (p[2], p[3]) if len(p) > 2 else (pin, pin)
        cost = (fresh * pin + cached * pread + written * pwrite + u.completion_tokens * pout) / 1e6
        return Completion(r.choices[0].message.content or "", u.prompt_tokens, u.completion_tokens, cost, cached)

    raise ValueError(f"unknown model backend: {model}")
