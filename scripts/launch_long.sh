#!/bin/bash
# LONG RUN. 360 minutes, otherwise as the main run (effort high, up to 50 live researchers), dataset
# researchcommunitybench/long@1: seed 1 only, kernel step cap 3000, verifier timeout 10 h.
# Ceilings are per team task (solo tasks get 35%); worst case Luna $88, Haiku $24.
OUT="$(cd "$(dirname "$0")/.." && pwd)/results/long_runs.txt"; : > $OUT
launch() {  # model key secret ceiling task-ids
  out=$(valkyrie run start --agent researchcommunitybench-kernel --dataset researchcommunitybench/long@1 --model $1 --provider modal --concurrency 6 \
        --label long-v1 -s $2 $3 -k minutes 360 -k max_live 50 -k ceiling_usd $4 -k effort high --task-ids $5 2>&1)
  R=$(echo "$out" | grep -o "Run ID: *[a-f0-9-]*" | awk '{print $3}'); [ -z "$R" ] && echo "$out" | tail -3
  echo "$(basename $1) $5 $R" | tee -a $OUT
}
launch openai/gpt-6-luna OPENAI_API_KEY openai-api-key 45 "compbio-s1"
launch openai/gpt-6-luna OPENAI_API_KEY openai-api-key 30 "compbio-solo-s1"
launch openai/gpt-6-luna OPENAI_API_KEY openai-api-key 12 "aiml-s1,aiml-solo-s1,materials-s1,materials-solo-s1"
launch anthropic/claude-haiku-5-5 ANTHROPIC_API_KEY anthropic-api-key 6 "aiml-s1,aiml-solo-s1,compbio-s1,compbio-solo-s1,materials-s1,materials-solo-s1"
