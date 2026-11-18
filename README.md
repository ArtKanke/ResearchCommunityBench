# ResearchCommunityBench

**Site:** https://artkanke.github.io/ResearchCommunityBench/

Can models organize research communities for collaboration? We gave four models an hour, a scientific field and the freedom to recruit subagents, then re-ran every scientific claim to verify their experiments. They swarm, assign roles and send messages, but rarely do research together.

**How it works.** A model founds a research team in one field: machine learning (on a GPU), protein biology or
materials science. It has 60 minutes, real datasets and no assigned problem, and no predefined harness: it decides
whether to recruit researchers (copies of itself with roles and briefs it writes), how they coordinate through a
message board and shared files, or whether to work alone. Every claim is then checked in a separate sandbox the
agents never touch.

The same model also runs solo (recruiting disabled); that run is the single agent baseline the team has to beat.
GPT-6 Luna and Claude Haiku 5.5 were also run for 6 hours.

Built on [Valkyrie](https://docs.valkyrie.vals.ai/introduction/what-is-valkyrie).


## Results (main run)

60 minutes per task, effort high, up to 50 researchers at a time.

- Sol and Opus: one seed. Luna and Haiku: three seeds.
- 47 of 48 tasks scored. Luna materials team seed 1 is excluded (failed).
- Model spend: $182.

## Leaderboard

| # | Model | Best gain, team | Best gain, solo | Team ÷ solo | Verified, team | Collaborative | Cost, team |
|---|---|---|---|---|---|---|---|
| 1 | GPT-6.1 Sol | 28.3% | 25.2% | 1.1× | 43.0 | 0.0 | $111 |
| 2 | Claude Opus 5.5 | 25.1% | 10.6% | 2.4× | 62.0 | 5.0 | $23.3 |
| 3 | Claude Haiku 5.5 | 13.1% | 12.6% | 1.0× | 15.3 ± 4.1 | 0.0 ± 0.0 | $0.31 |
| 4 | GPT-6 Luna | 8.6% | 7.3% | 1.2× | 23.8 ± 3.1 | 0.3 ± 0.3 | $9.08 |

- **Best gain, team:** how much lower the error of the team's best verified result is than a simple reference
  model's, averaged over the field's datasets and then over the fields.
- **Best gain, solo:** the same number for the model working alone, with recruiting disabled (the single agent
  baseline).
- **Team ÷ solo:** the first number divided by the second; above 1× the team did better.
- **Verified, team:** claims whose experiments re-ran within 10% and whose test beat its comparator on a held-out
  test set no agent saw, summed over one run per field.
- **Collaborative:** verified claims that depend on a teammate's: a correction whose target fails verification
  while the correction passes, or an extension that gets materially worse when the teammate's data it used is
  shuffled.
- **Cost, team:** model spend of the team runs, summed over one run per field.
- ± is one standard error over runs.

## Where models fail

- **Teams do not work as teams.** Agents claimed 829 times to extend a teammate's result (453 in one-hour runs,
  376 in six-hour runs); the verifier confirmed none of them.
- **More time does not change that.** With six hours instead of one, Luna's teams verified four times as many
  claims and grew to 18 researchers, with no confirmed collaboration.
- **The only collaboration that holds up is catching mistakes.** All 6 confirmed collaborations are corrections or
  refutations; 5 came from one Claude Opus 5.5 team that caught a teammate's data leak a minute after it was posted.
- **Some models stop with time left.** Claude Haiku 5.5's founders ended their sessions after 14 to 42 minutes of
  an hour, and after 93 to 125 minutes of six. Claude Opus 5.5 working alone stopped after 9 to 40 minutes.
- **Internal review does not check results.** GPT-6.1 Sol's validation auditor marked five teammates' results as
  replicated; all five failed on the held-out test set.
- **Teams cost far more for small gains.** They cost 4–9× as much as the same model alone for a 1.0–2.4× better
  best result; at six hours the model alone did better in 3 of 6 fields.

## 1 hour vs 6 hours horizon

GPT-6 Luna and Claude Haiku 5.5, one seed per field and mode, 360 minutes per task. Model spend: $53.

- **More time buys more output, not collaboration.** Luna's teams verified 105 claims in six hours against 23.8 in
  one, recruited up to 18 researchers, and claimed 363 times to extend a teammate's result; none was confirmed.
- **The team advantage does not grow with time.** At six hours the model working alone had the better best result
  in 3 of 6 fields. Two of the three team wins came where the solo run produced almost nothing: Luna in materials
  science (1 of 25 claims verified) and Haiku in protein biology (0 of 19). In machine learning, Haiku's team edged
  ahead, 14.8% to 13.5%.
- **Haiku does not use the extra time.** Its founders ended the team runs after 93 to 125 minutes of 360; working
  alone it stopped after 24 to 89.
- **A bigger team does not mean better science.** Luna's 19-agent protein biology team spent $26 for a 9.3% best
  gain, below Luna working alone for six hours (11.6%) and below its own one-hour teams (9.8%).


## Valkyrie runs

| Model | Run ID |
|---|---|
| gpt-6.1-sol | `fa16734c-0657-440e-a6d2-5ad2ece20fd8` |
| claude-opus-5-5 | `818fe65c-9277-4836-927c-c1d2d817d09b` |
| gpt-6-luna | `38df2b16-a025-4888-b38b-064c7edcf433` |
| claude-haiku-5-5 | `5159f884-e733-4e1a-a96d-73071e6eba10` |

- **Dataset:** `researchcommunitybench/main@v1`. **Agent:** `researchcommunitybench-kernel`.
- **Per-claim verdicts:** `results/main/<model>/<task>.verdicts.json`.

6-hour runs (GPT-6 Luna and Claude Haiku 5.5, one seed, `researchcommunitybench/long@1`):

| Model | Tasks | Run ID |
|---|---|---|
| gpt-6-luna | compbio-s1 | `21ff73ca-6ac3-4a7c-b282-a010311de31a` |
| gpt-6-luna | compbio-solo-s1 | `5aef0cca-8284-406d-86ec-a83939a7a06e` |
| gpt-6-luna | aiml-s1, aiml-solo-s1, materials-s1, materials-solo-s1 | `7a197db1-c1ff-4197-b73a-2fe8a6560a4b` |
| claude-haiku-5-5 | all six | `a6b43909-f002-4b93-9726-3cbfd42c8d10` |

- **Per-claim verdicts:** `results/long/<model>/<task>.verdicts.json`.


## Layout

| Path | What |
|---|---|
| `lab/server.py`, `lab/research.py` | Lab server: model proxy and meter, researcher registry, board, artifact store, `lab exec` |
| `lab/artifacts.py` | Claim schema and live checks at submission |
| `lab/verify.py` | Post-session verification: re-execution, hidden-data test, dedup, frontier, ablation, failure modes |
| `lab/charter.py` | The task instruction and `lab guide` |
| `agent/` | Valkyrie agent bundle: `kernel.py` (bash-only loop), `lab` CLI |
| `harbor/agents/researchcommunitybench-kernel/` | Agent contract and `start.sh` |
| `harbor/dataset/` | The 18 Harbor tasks exactly as pushed to Valkyrie (`researchcommunitybench/main@v2`; v1 differs only in the verifier). Each task's `tests/` holds the hidden split. |
| `scripts/build_harbor.py` | Builds the Harbor tasks (environment, hidden splits, verifier) |
| `scripts/launch_main.sh`, `launch_long.sh` | The main run and the 6-hour runs |
| `scripts/reverify_modal.py`, `export_verdicts.py`, `build_summary.py` | Rescoring, per-claim verdicts, summary |
| `site/` | Results site (published with GitHub Pages) |

## Reproduce

```bash
.venv/bin/python scripts/build_harbor.py --minutes 60 --out harbor/main --combined republic-main
valkyrie dataset push researchcommunitybench/main harbor/main/republic-main
./scripts/launch_main.sh
# after the runs: download outputs, rescore, summarize
MODAL_PROFILE=<profile> .venv/bin/python scripts/reverify_modal.py
.venv/bin/python scripts/export_verdicts.py && .venv/bin/python scripts/build_summary.py

# 6-hour runs: same tasks, 360 minutes; rescoring runs detached on Modal, then collect
.venv/bin/python scripts/build_harbor.py --minutes 360 --out harbor/long --combined republic-long
./scripts/launch_long.sh
MODAL_PROFILE=<profile> .venv/bin/python scripts/reverify_modal.py --runs long --models haiku,luna --detach
MODAL_PROFILE=<profile> .venv/bin/python scripts/reverify_modal.py --runs long --models haiku,luna --collect
.venv/bin/python scripts/export_verdicts.py --runs long
```

## Tests

```bash
.venv/bin/python scripts/build_materials.py   # the verification tests use the materials data
.venv/bin/python -m pytest -q
```
