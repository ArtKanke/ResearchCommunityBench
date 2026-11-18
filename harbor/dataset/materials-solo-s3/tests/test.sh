#!/bin/bash
# Separate verifier: /tests holds the hidden data; /lab/store is the community's record (copied artifact).
set -uo pipefail
mkdir -p /logs/verifier
cd /opt/republic && python /tests/verify_task.py
