import sys, runpy
sys.argv = [sys.argv[0], *sys.argv[1:], "--thinking", "off"]
runpy.run_path("/lfs/skampere3/0/alexspan/norm-research/methods/metric_seam/pilot/qwen_thinking_score_sk3.py", run_name="__main__")
