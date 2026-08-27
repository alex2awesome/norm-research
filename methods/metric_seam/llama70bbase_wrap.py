import sys, runpy
sys.argv = [sys.argv[0], *sys.argv[1:], "--model-dir", "/lfs/skampere3/0/shared_hf_cache/models--meta-llama--Llama-3.1-70B",
            "--gpu-mem-util", "0.93"]
runpy.run_path("/lfs/skampere3/0/alexspan/norm-research/methods/metric_seam/llama_base_score_sk3.py", run_name="__main__")
