import os, runpy, sys
os.environ["SEAM_MAX_TOKENS"]="256"
runpy.run_path("/lfs/skampere3/0/alexspan/norm-research/outputs/metric_seam_pilot/v1/gemma_score_v1.py", run_name="__main__")
