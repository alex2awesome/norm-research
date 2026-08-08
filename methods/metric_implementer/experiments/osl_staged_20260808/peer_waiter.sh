#!/bin/bash
while pgrep -f "executor magistral-24b" >/dev/null; do sleep 60; done
sleep 120
cd /lfs/skampere3/0/alexspan/outputs/osl_multi
exec bash gen_panel_lane.sh 6 peer_review qwen3-1.7b qwen3-4b qwen3-8b qwen3-14b qwen3-32b
