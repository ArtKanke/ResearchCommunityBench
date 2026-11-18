"""Scripted stand-in for a model, so the whole pipeline can be tested for free.
It exercises spawn, eval, submit (with claims), publish (with a backed claim), oracle,
the board, and done."""

FOUNDER = [
    "lab problems && lab budget",
    "python3 -c \"import json; json.dump({'heights': [1.0]*40}, open('m1.json','w'))\" && lab eval M1 m1.json",
    "E=$(lab eval M1 m1.json | python3 -c \"import sys,json; d=json.load(sys.stdin); print(d['eval_id']+':'+repr(d['value']))\") && "
    "lab publish --title 'Constant baseline for M1' --body 'A flat step function gives C=2.' --claim M1:$E",
    "lab submit M1 m1.json --claim 2.0",
    "lab spawn --role 'physics experimentalist' --brief 'Run two oracle experiments on P1 and post what you see.' || echo solo",
    "lab oracle P1 '{\"y0\": [1.0, 0.0], \"t_end\": 3}' | head -c 300",
    "lab post 'founder: M1 baseline submitted' && lab inbox",
    "sleep 5; lab agents",
    "lab done 'mock founder finished'",
]

WORKER = [
    "lab inbox && lab search",
    "lab oracle P1 '{\"y0\": [0.5, 0.5], \"t_end\": 5}' | head -c 300",
    "lab post 'experimentalist: P1 trajectories look damped and nonlinear'",
    "lab done 'mock worker finished'",
]


def respond(messages):
    system = messages[0]["content"]
    script = FOUNDER if "You are the **founder**" in system else WORKER
    step = sum(1 for m in messages if m["role"] == "assistant")
    cmd = script[min(step, len(script) - 1)]
    return f"THOUGHT: scripted step {step}.\n```bash\n{cmd}\n```"
