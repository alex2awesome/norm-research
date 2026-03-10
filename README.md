# Goals and Overview
This research aims to measure the articulability gap by comparing dense models trained on human feedback data against articulated models that rely on explicit, human-interpretable criteria. We are trying to understand when learned latent objectives outperform explicit, structured objectives, and when the reverse is true. The core idea is to benchmark performance, sample efficiency, and generalization across multiple tasks while holding data access and evaluation protocols as constant as possible.

# Usage
Dense training is orchestrated through the sweep script, which runs `methods/dense/train_reward_model.py` over multiple data fractions.

```bash
./train_sweep.sh datasets/creative-writing/LitBench-Train.csv.gz runs/sweep_01 --use-optuna --optuna_trials 20
```

You can also run a single training job directly:

```bash
python methods/dense/train_reward_model.py \
  --data_path datasets/creative-writing/LitBench-Train.csv.gz \
  --output_dir runs/single_run
```

# File structure
- `methods/dense/` contains dense model training code and FSDP configs.
- `methods/autometrics/` contains articulated/metric-based modeling code.
- `scripts/` contains runnable utilities (e.g., `scripts/run_autometrics_vllm.py`).
- `datasets/` contains task data organized by domain.
- `runs/` contains training outputs and logs.
- `notebooks/` contains exploratory analysis and data prep notebooks.

## Model Training

| Model | Press Releases | Legal Outcome Prediction | Code Review | Creative Writing | Grant Funding | Humor | Peer Review |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense: Llama-3.3-8b (10%–100%) | ✓ |  |  |  |  |  |  |
| Dense: Llama-3.3-70b (10%–100%) |  |  |  |  |  |  |  |
| Dense: Phi 4 |  |  |  |  |  |  |  |
| Dense: GLM 4.6 (?) |  |  |  |  |  |  |  |
| Articulable: Autometrics ++ |  |  |  |  |  |  |  |
| Articulable: EM Algorithm |  |  |  |  |  |  |  |
| Articulable: Google Taxonomy Generation |  |  |  |  |  |  |  |
| Articulable: Sheldon's Bandit Algorithm |  |  |  |  |  |  |  |
| Articulable: Human-derived Metrics found online |  |  |  |  |  |  |  |

## Dataset Collection
- Coding ❓
- Legal Outcome Prediction 🟨
- Grants (Rio, OGrants, NIH) 🟨
- Press Releases ✅
- City Council Discussions ❌
- Creative Writing ✅
- Humor ✅
- Wikipedia Editorial Decisions ❌
- Patent examiner decisions 🟨
- News homepages 🟨
