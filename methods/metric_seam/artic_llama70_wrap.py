import os, runpy, sys
os.environ["SEAM_MAX_TOKENS"]="256"
runpy.run_path("/lfs/skampere3/0/alexspan/norm-research/methods/metric_seam/pilot/llama_score_sk3.py", run_name="__main__")
