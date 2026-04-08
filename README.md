# Goals and Overview
This research aims to measure the articulability gap by comparing dense models trained on human feedback data against articulated models that rely on explicit, human-interpretable criteria. We are trying to understand when learned latent objectives outperform explicit, structured objectives, and when the reverse is true. The core idea is to benchmark performance, sample efficiency, and generalization across multiple tasks while holding data access and evaluation protocols as constant as possible.

<p align="center">
  <img src="assets/reward-scaling-laws.png" alt="Reward Scaling Laws" width="600"/>
  <br/>
  <em>We hypothesize that task types have distinct scaling behaviors: Creative Tasks (diffuse, community-originated norms) require the most data, Codified Tasks (articulated rules) converge faster, and Auditable Tasks (objectively verifiable outcomes) require the least. By sweeping across data fractions we can characterize these scaling curves and identify diminishing-returns thresholds for each domain.</em>
</p>

<p align="center">
  <img src="scripts/assets/preference-modeling-tasks.png" alt="Preference Modeling Tasks" width="600"/>
  <br/>
  <em>Dense reward models learn a latent objective end-to-end from preference data, while articulable approaches decompose the objective into explicit, interpretable metrics. We compare these paradigms across tasks to understand when dense expressiveness justifies its opacity and when structured approaches can match or exceed it.</em>
</p>

# Usage
Dense training is orchestrated through the sweep script, which runs `methods/dense/train_reward_model.py` across a range of training data fractions. The sweep iterates over 10%, 20%, 30%, ..., 90%, 100% of the training split to characterize scaling laws and measure asymptotic performance for each task. The underlying data is partitioned into a fixed 80/10/10 train/eval/test split (persisted on disk to ensure reproducibility across runs), and the `--train_subset_percentage` flag controls how much of the training partition is used in each run. This lets us plot learning curves and identify the data requirements for each domain.

Optionally, each fraction can use [Optuna](https://optuna.org/) for hyperparameter search (learning rate, batch size, etc.), with the eval split used for trial selection and the held-out test split reserved for final evaluation.

```bash
# Run a full scaling sweep with Optuna hyperparameter tuning
./train_sweep.sh datasets/creative-writing/LitBench-Train.csv.gz runs/sweep_01 --use-optuna --optuna_trials 20
```

You can also run a single training job directly at a specific data fraction:

```bash
python methods/dense/train_reward_model.py \
  --data_path datasets/creative-writing/LitBench-Train.csv.gz \
  --train_subset_percentage 0.5 \
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

We train reward models using two families of approaches across all task domains. **Dense models** are end-to-end neural reward models (fine-tuned LLMs) that learn a scalar preference score directly from human judgment data. We train these at multiple scales (8B and 70B parameters) and across all data fractions to establish an empirical performance ceiling for each task. This ceiling represents the best achievable accuracy given the noise inherent in human annotations, and serves as the upper bound against which articulable methods are compared.

**Articulable models** decompose the reward signal into explicit, human-readable metrics and combine them (via learned weights, bandit algorithms, or EM-based latent variable models) to produce a final score. These approaches are more interpretable and auditable, but may sacrifice expressiveness. The gap between the dense ceiling and articulable performance on a given task is what we call the **articulability gap** -- it quantifies how much of human preference can be captured by explicit criteria versus latent features that resist articulation.

The table below tracks training progress across model types and datasets.

<p align="center">
  <img src="assets/articulability-gap.png" alt="Articulability Gap" width="600"/>
  <br/>
  <em>Dense performance serves as an upper bound reflecting each task's inherent noise; the articulability gap is the difference between this ceiling and the best articulable model. A small gap means explicit metrics suffice, while a large gap indicates preferences that only end-to-end models can capture.</em>
</p>

<table>
  <thead>
    <tr>
      <th>Approach</th>
      <th>Model</th>
      <th>Press Releases</th>
      <th>Legal Outcome Prediction</th>
      <th>Code Review</th>
      <th>Creative Writing</th>
      <th>Grant Funding</th>
      <th>Humor</th>
      <th>Peer Review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4" style="writing-mode: vertical-rl; text-orientation: mixed;"><strong>Dense</strong></td>
      <td>Llama-3.3-8b (10%–100%)</td>
      <td>✓</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Llama-3.3-70b (10%–100%)</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Phi 4</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>GLM 4.6 (?)</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="5" style="writing-mode: vertical-rl; text-orientation: mixed;"><strong>Articulable</strong></td>
      <td>Autometrics ++</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>EM Algorithm</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Google Taxonomy Generation</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sheldon's Bandit Algorithm</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Human-derived Metrics found online</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>

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
