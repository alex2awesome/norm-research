#!/usr/bin/env bash
set -euo pipefail

# Compatibility entry point retained for the already-approved launcher.  The
# v2 queue is completion-aware and implementation-hash-pinned; the retired v1
# PID-bound behavior must not be used after terminal-chunk retries.
exec bash scripts/tools/silver_match_v3/run_content_gepa_label_handoff_queue_v2.sh
