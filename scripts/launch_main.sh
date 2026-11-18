#!/bin/bash
# MAIN RUN. 60 minutes, effort high, up to 50 live researchers, new prompt, dataset researchcommunitybench/main@v1.
# Sol and Opus: 1 run per condition (seed 1: 3 domains x team/solo = 6 tasks).
# Luna and Haiku: 3 runs per condition (seeds 1-3 = 18 tasks). All tasks start at once.
# Ceilings are per team task (solo tasks get 35%); they only stop a runaway run.
OUT="$(cd "$(dirname "$0")/.." && pwd)/results/main_runs.txt"; : > $OUT
S1="aiml-s1,aiml-solo-s1,compbio-s1,compbio-solo-s1,materials-s1,materials-solo-s1"
launch() {  # model key secret ceiling task-ids
  local ids=""; [ -n "$5" ] && ids="--task-ids $5"
  out=$(valkyrie run start --agent researchcommunitybench-kernel --dataset researchcommunitybench/main@v1 --model $1 --provider modal --concurrency 18 \
        --label main-v1 -s $2 $3 -k minutes 60 -k max_live 50 -k ceiling_usd $4 -k effort high $ids 2>&1)
  R=$(echo "$out" | grep -o "Run ID: *[a-f0-9-]*" | awk '{print $3}'); [ -z "$R" ] && echo "$out" | tail -3
  echo "$(basename $1) $R" | tee -a $OUT
}
launch openai/gpt-6.1-sol OPENAI_API_KEY openai-api-key 100 "$S1"
launch anthropic/claude-opus-5-5 ANTHROPIC_API_KEY anthropic-api-key 30 "$S1"
launch openai/gpt-6-luna OPENAI_API_KEY openai-api-key 10 ""
launch anthropic/claude-haiku-5-5 ANTHROPIC_API_KEY anthropic-api-key 3 ""
