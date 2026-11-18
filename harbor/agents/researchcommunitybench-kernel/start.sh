#!/bin/bash
# ResearchCommunityBench bootstrap harness: start the in-sandbox lab server (model proxy + meter, researcher registry,
# board, artifact store, lab exec), open the community, and run the founder. Every model starts here
# and may replace any of it except the server.
set -uo pipefail
MODEL="$1"; MODE="$2"; MINUTES="$3"; MAX_LIVE="$4"; CEILING="$5"; PROBLEM="$6"; TASK="$7"; EFFORT="${8:-medium}"
export LAB_EFFORT="$EFFORT"
export LAB_DB=/lab/store/lab.sqlite LAB_BLOBS=/lab/store/blobs LAB_EXECUTOR=local LAB_EXEC_USER=agent
export LAB_DOMAINS=/lab/domains LAB_PUBLIC_URL=http://127.0.0.1:8100 PYTHONPATH=/opt/republic
export LAB_ADMIN_KEY="$(head -c 16 /dev/urandom | xxd -p)"
export LAB_EXEC_THREADS=2 LAB_EXEC_SLOTS=$(( $(nproc) / 2 > 1 ? $(nproc) / 2 : 1 ))  # experiments share CPUs fairly
DOMAIN="$(cat /lab/data/DOMAIN 2>/dev/null || echo materials)"
if grep -q "This is a solo run" "$PROBLEM" 2>/dev/null; then MODE=solo; else MODE=community; fi  # the task decides
if [ "$MODE" = solo ]; then CEILING=$(python -c "print(round($CEILING * 0.35, 2))"); fi  # solo safety ceiling
mkdir -p /lab/store /lab/domains/$DOMAIN /lab/logs && ln -sfn /lab/data /lab/domains/$DOMAIN/public
chown -R agent:agent /lab/logs /lab/cursors /work 2>/dev/null || true
cd /opt/republic
python -m uvicorn lab.server:app --host 127.0.0.1 --port 8100 > /lab/logs/server.log 2>&1 &
for i in $(seq 1 60); do curl -sf http://127.0.0.1:8100/health >/dev/null && break; sleep 1; done
ADMIN=(-H "Authorization: Bearer $LAB_ADMIN_KEY" -H "Content-Type: application/json")
case "$MODEL" in */*) BACKEND="litellm:$MODEL" ;; *) BACKEND="$MODEL" ;; esac  # mock / codex for local tests
RUN=$(curl -sf -X POST http://127.0.0.1:8100/admin/runs "${ADMIN[@]}" -d "{\"model\": \"$BACKEND\", \"mode\": \"$MODE\", \
  \"minutes\": $MINUTES, \"max_live\": $MAX_LIVE, \"budget_usd\": $CEILING, \"domain\": \"$DOMAIN\", \"label\": \"$TASK\"}")
RID=$(echo "$RUN" | python -c "import sys,json;print(json.load(sys.stdin)['run_id'])")
TOKEN=$(echo "$RUN" | python -c "import sys,json;print(json.load(sys.stdin)['token'])")
echo "{\"server\": \"http://127.0.0.1:8100\", \"run_id\": \"$RID\", \"token\": \"$TOKEN\", \"mode\": \"$MODE\", \"model\": \"$MODEL\", \"agent_id\": \"founder\"}" > /lab/run.json
cp "$PROBLEM" /lab/charter.md
chmod 644 /lab/run.json /lab/charter.md
curl -sf -X POST "http://127.0.0.1:8100/admin/runs/$RID/start" "${ADMIN[@]}" > /dev/null
echo "community $RID ($MODE, $MODEL, effort $EFFORT, $MINUTES min, up to $MAX_LIVE agents) for $TASK"
python /opt/republic/agent/lab boot
pkill -f "kernel.py --agent-id" 2>/dev/null || true
cp /lab/logs/*.log /lab/store/ 2>/dev/null || true
python -m lab.prune /lab/store || true  # keep only what verification needs (Valkyrie artifact limit 512 MB)
echo "community $RID finished"
