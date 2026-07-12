"""Run manifest for scaled E7: which datasets × metrics × judge-tiers × budget caps × seeds.

A manifest is plain data (dataclass, serializable) so a run is fully specified and reproducible.
``load_metrics`` and ``load_corpus`` turn manifest entries into MetricArtifacts + (texts, ids).
Metric sources are pluggable: trial ladders (built-in), online-rubric JSON (datasets/*/online-
rubrics), and registered code programs. Dataset paths/columns come from the project's canonical
locations ([[reference_clean_datasets_per_task]], [[reference_v2_task_datasets]]); the inventory
sweep (2026-06-12) fills the registry below with real paths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .artifact import MetricArtifact
from .config import REPO_ROOT

# ---- judge-capability tiers (the scaling axis), sk3-hostable open models -------------------
# Broad cached ladder, weakest->strongest, multi-family (Llama / Qwen / Gemma) — all verified
# present in /lfs/skampere3/0/shared_hf_cache (2026-06-13). The 1B/3B rungs are LOW-theta ANCHORS
# for the capability/scaling curve; the 06-12 pilot found 1B / small-gemma degenerate on hard tasks
# under HARD-LABEL scoring, so (a) continuous fidelity (U3) partially rescues them and (b) we filter
# degenerate (tier x task) cells in analysis, not a priori. Models load ONE AT A TIME (scale.py
# resident singleton) -> ONE GPU, sequential ([[feedback_gpu_usage]], [[feedback_minimize_gpus]]).
# All verified COMPLETE in the shared cache (config.json + safetensors, no .incomplete blobs,
# 2026-06-13). gemma-3-12b-it and Qwen3-Next-80B-A3B had no resolvable config.json -> dropped.
# Run vLLM with HF_HUB_OFFLINE=1 so it loads from cache and never re-downloads (a stale .incomplete
# blob in the shared cache triggers a PermissionError otherwise).
# Verified loadable on sk3 2026-06-13 via local-snapshot-path resolution
# (vllm_backend._resolve_model_path) under HF_HUB_OFFLINE. Multi-family 3B->47B(MoE) capability
# ladder, ordered weakest->strongest-ish. Run ONE per process (`scale measure --tier`).
DEFAULT_TIERS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-8B",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]
# Too new for the sk3 transformers 4.57 (qwen3_5 / gemma4 / olmo3 / qwen3_next archs); add when the
# env is upgraded. The long table appends, so new tiers extend the same dataset.
EXPAND_TIERS = [
    "Qwen/Qwen3.6-27B", "google/gemma-4-31B-it", "Qwen/Qwen2.5-Coder-32B-Instruct",
    "allenai/Olmo-3-1125-32B", "Qwen/Qwen3-Next-80B-A3B-Instruct",
]
# Strong reference / reviser / reconstructor (resident strong model; sk3-only -> open, not Sonnet).
# Used only in the SEARCH (GEPA self-improve) phase. Strongest verified-loadable model.
DEFAULT_STRONG = "mistralai/Mixtral-8x7B-Instruct-v0.1"


@dataclass
class DatasetEntry:
    task: str                 # config preset name (code-review / creative-writing / law / ...)
    name: str                 # short dataset id for the long table
    path: str                 # jsonl(.gz)/parquet/csv on disk (sk3 or laptop)
    text_column: str
    id_column: Optional[str] = None
    label_column: Optional[str] = None     # evaluation-only; never seen by the optimizer
    metric_source: str = "trial"           # "trial" | "online_rubrics" | "code_programs"
    metric_glob: Optional[str] = None       # for online_rubrics/code_programs sources
    max_metric_files: Optional[int] = None  # bound the online-rubrics bank size
    max_metrics: Optional[int] = None


@dataclass
class RunManifest:
    run_id: str
    datasets: List[DatasetEntry]
    tiers: List[str] = field(default_factory=lambda: list(DEFAULT_TIERS))
    strong_model: str = DEFAULT_STRONG
    token_caps: List[int] = field(default_factory=lambda: [120, 1000])
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    gepa_rounds: int = 2
    passes: int = 3                     # test-retest replicates (pilot lesson: >=3)
    n_items: int = 60                   # >= 60 per the pilot's top protocol fix
    n_oracle_items: int = 16

    def to_json(self) -> dict:
        return {"run_id": self.run_id, "tiers": self.tiers, "strong_model": self.strong_model,
                "token_caps": self.token_caps, "seeds": self.seeds,
                "gepa_rounds": self.gepa_rounds, "passes": self.passes,
                "n_items": self.n_items, "n_oracle_items": self.n_oracle_items,
                "datasets": [d.__dict__ for d in self.datasets]}


# ---- metric loaders ------------------------------------------------------------------------

def _trial_metrics(task: str) -> List[Tuple[MetricArtifact, MetricArtifact]]:
    if task == "creative-writing":
        from .trial.trial_metrics_cw import trial_metrics_cw
        return trial_metrics_cw()
    if task == "law":
        from .trial.trial_metrics_law import trial_metrics_law
        return trial_metrics_law()
    from .trial.trial_metrics import trial_metrics
    return trial_metrics()


def _online_rubric_metrics(glob_pat: str, task: str,
                           max_files: Optional[int] = None,
                           max_metrics: Optional[int] = None) -> List[MetricArtifact]:
    """Load metrics parsed from datasets/<task>/online-rubrics/gpt-parsed/. The inventory
    sweep (2026-06-12) confirmed the schema: each JSON has ``extracted.rubrics_metrics`` = a
    list of {name, description, guidance}. ~67K such files exist across 11 tasks. Seed body =
    ``guidance`` (the scoring instruction) prefixed with the description for context.

    A metric is usable as a seed only if it reads like an evaluation criterion; we keep ones
    with non-trivial guidance and a name, and dedup by (name, guidance-hash). Caller bounds the
    count via max_files / max_metrics (the bank size is a manifest knob)."""
    out, seen = [], set()
    files = sorted(Path(REPO_ROOT).glob(glob_pat))
    if max_files:
        files = files[:max_files]
    for p in files:
        try:
            obj = json.loads(p.read_text())
        except Exception:
            continue
        extracted = (obj.get("extracted") or {}) if isinstance(obj, dict) else {}
        breadth = extracted.get("subtask_breadth")   # very_narrow..very_broad (specificity class)
        recs = (extracted.get("rubrics_metrics")
                if isinstance(obj, dict) else None)
        if recs is None:
            recs = obj if isinstance(obj, list) else (
                obj.get("rubrics_metrics") or obj.get("metrics") or obj.get("rubrics") or [obj])
        for i, r in enumerate(recs or []):
            if not isinstance(r, dict):
                continue
            name = (r.get("name") or r.get("title") or r.get("criterion") or "").strip()
            desc = (r.get("description") or r.get("definition") or "").strip()
            guidance = (r.get("guidance") or r.get("rubric") or "").strip()
            body = (f"{desc}\n\nScoring guidance: {guidance}".strip()
                    if guidance else desc)
            if not name or len(body) < 40:
                continue
            key = (name.lower(), hash(guidance[:200]))
            if key in seen:
                continue
            seen.add(key)
            mid = f"{p.stem}_{i}".replace(" ", "_").replace("/", "_")[:72]
            out.append(MetricArtifact(metric_id=mid, kind="prompt", body=body,
                                      name=name[:120], description=desc or name,
                                      invariances=["blank_lines"],
                                      meta={"subtask_breadth": breadth} if breadth else {}))
            if max_metrics and len(out) >= max_metrics:
                return out
    return out


def load_metrics(entry: DatasetEntry) -> List[MetricArtifact]:
    """Prompt-kind seed artifacts for a dataset entry (code-kind handled separately)."""
    if entry.metric_source == "trial":
        return [p for p, c in _trial_metrics(entry.task)]
    if entry.metric_source == "online_rubrics" and entry.metric_glob:
        return _online_rubric_metrics(entry.metric_glob, entry.task,
                                      max_files=entry.max_metric_files,
                                      max_metrics=entry.max_metrics)
    raise ValueError(f"unknown metric_source {entry.metric_source!r}")


def _load_subsampled_df(entry: DatasetEntry, n: int, seed: int = 7):
    """Load the corpus file for `entry` and subsample to `n` rows with the GIVEN seed — the SAME
    subsample `load_corpus` uses, so any column sliced from this df (text / id / LABEL) stays aligned
    with `load_corpus`'s text/ids. The shared spine that lets the §12.3 value census pull the labels
    for the EXACT probe shard the behavior census scored (GV4)."""
    import numpy as np
    p = Path(entry.path)
    name = p.name
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif name.endswith(".jsonl.gz") or name.endswith(".json.gz"):
        df = pd.read_json(p, lines=name.endswith(".jsonl.gz"), compression="gzip")
    elif name.endswith(".csv.gz"):
        df = pd.read_csv(p, compression="gzip")
    elif name.endswith(".tsv.gz"):
        df = pd.read_csv(p, compression="gzip", sep="\t")
    elif p.suffix in (".jsonl", ".json"):
        df = pd.read_json(p, lines=p.suffix == ".jsonl")
    elif p.suffix == ".gz":             # bare .gz: assume jsonl.gz (pilot pools)
        df = pd.read_json(p, lines=True, compression="gzip")
    else:
        df = pd.read_csv(p)
    rng = np.random.default_rng(seed)
    if n and len(df) > n:
        df = df.iloc[rng.choice(len(df), size=n, replace=False)].reset_index(drop=True)
    return df


def load_corpus(entry: DatasetEntry, n: int, seed: int = 7) -> Tuple[List[str], List[str]]:
    df = _load_subsampled_df(entry, n, seed)
    texts = df[entry.text_column].astype(str).tolist()
    ids = (df[entry.id_column].astype(str).tolist() if entry.id_column and
           entry.id_column in df.columns else [str(i) for i in range(len(df))])
    return texts, ids


def load_corpus_labels(entry: DatasetEntry, n: int, seed: int = 7) -> Tuple[List[str], List[int], List[str]]:
    """Same subsample as `load_corpus` (so slicing matches `_load_texts`), PLUS the binarized label
    column — for the §12.3 value census (supervised; GV1). Returns (texts, labels_binary, ids).

    Binary {0,1} labels pass through; multi-valued/continuous labels are split at the median (a 50/50
    base rate) so I(Y;·) is meaningful. Raises if the task has no label column or it is constant."""
    df = _load_subsampled_df(entry, n, seed)
    texts = df[entry.text_column].astype(str).tolist()
    ids = (df[entry.id_column].astype(str).tolist() if entry.id_column and
           entry.id_column in df.columns else [str(i) for i in range(len(df))])
    if not entry.label_column or entry.label_column not in df.columns:
        raise ValueError(f"entry {entry.task!r} has no usable label_column "
                         f"(need one for the value census; got {entry.label_column!r})")
    lab = pd.to_numeric(df[entry.label_column], errors="coerce")
    vals = lab.dropna().unique()
    if len(vals) <= 1:
        raise ValueError(f"label {entry.label_column!r} on {entry.task!r} is constant — "
                         f"no signal for a value census")
    if set(int(v) for v in vals) <= {0, 1}:
        labels = lab.fillna(0).astype(int).tolist()
    else:                                              # continuous/multi-valued → median split
        thr = float(lab.median())
        labels = (lab > thr).astype(int).tolist()
    return texts, labels, ids


# ---- the 06-12 pilot manifest (three trial ladders), extended by the inventory sweep -------

def pilot_manifest(run_id: str = "scale_pilot") -> RunManifest:
    R = str(REPO_ROOT)
    mi = f"{R}/methods/metric_implementer/trial"
    return RunManifest(
        run_id=run_id,
        datasets=[
            DatasetEntry("law", "title_vii", f"{mi}/pool_law.jsonl.gz",
                         text_column="facts", id_column="doc_id",
                         label_column="binary_label"),
            DatasetEntry("creative-writing", "writingprompts",
                         f"{mi}/pool_creative_writing.jsonl.gz", text_column="text"),
            DatasetEntry("code-review", "competitive_code",
                         f"{mi}/pool_competitive_code.jsonl.gz", text_column="code",
                         id_column="candidate_id"),
        ],
        tiers=DEFAULT_TIERS, token_caps=[120, 1000], seeds=[0, 1, 2],
        gepa_rounds=2, passes=3, n_items=60, n_oracle_items=16)


# ---- the full scale-out manifest: real labeled datasets + online-rubric metric banks -------
# Paths/columns confirmed by the 2026-06-12 inventory sweep. Text col = "text" except legal
# ("facts"); label = "judgement" except legal ("binary_label"). creative_writing + code_review
# canonical files are sk3-only (run there); press_releases locally corrupted (omitted until
# rebuilt). Metric banks drawn from datasets/<task>/online-rubrics/gpt-parsed/gpt-5-mini.

def _rubric_glob(task_dir: str) -> str:
    return f"datasets/{task_dir}/online-rubrics/gpt-parsed/gpt-5-mini/*.json"


# (config_task, name, rel_path, text_col, id_col, label_col, rubric_task_dir)
_FULL_DATASETS = [
    ("law", "title_vii", "datasets/legal-outcome-prediction/title_vii_balanced_v2.jsonl",
     "facts", None, "binary_label", "legal-outcome-prediction"),
    ("law", "flsa", "datasets/legal-outcome-prediction/flsa_fullpool_v3.jsonl",
     "facts", None, "binary_label", "legal-outcome-prediction"),
    ("law", "ss_disability", "datasets/legal-outcome-prediction/ss_disability_balanced_v2.jsonl",
     "facts", None, "binary_label", "legal-outcome-prediction"),
    ("peer-review", "peer_review", "datasets/peer-review/splits/train.csv.gz",
     "text", "paper_id", "judgement", "peer-review"),
    ("news-homepages", "news_homepages",
     "datasets/news-homepages/homepage_newsworthiness_topic_balanced_groupsplit.csv.gz",
     "text", None, "judgement", "news-homepages"),
    ("math", "math_se", "datasets/math/stackexchange/math_se_modeling.csv.gz",
     "text", None, "judgement", "math/stackexchange"),
    ("humor", "reddit_humor", "datasets/humor/reddit_humor_modeling_dedup.csv.gz",
     "text", None, "judgement", "humor"),
    ("patents", "patents", "datasets/patents/patents_final_outcome_balanced.csv.gz",
     "text", None, "judgement", "patents"),
    ("notice-and-comment", "notice_and_comment",
     "datasets/notice-and-comment/notice_and_comment.csv.gz",
     "text", None, "judgement", "notice-and-comment"),
    # creative_writing: WritingPrompts modeling corpus (96K rows, balanced 50/50). Rubric bank =
    # 7,699 gpt-parsed online-rubrics from CW authorities (Aristotle/Booker/Atwood/narratology).
    # NB the bank is git-untracked -> must be rsynced to sk3 explicitly (see reference_metric_banks).
    ("creative-writing", "creative_writing",
     "datasets/creative-writing/writingprompts_modeling_clean.csv.gz",
     "text", None, "judgement", "creative-writing"),
    # press_releases: deconfounded corpus (2026-06-25 rebuild; publisher/topic confounds removed —
    # see project_press_release_results). Replaces the corrupted pre-rebuild file omitted above.
    ("press-releases", "press_releases",
     "datasets/press-releases/press_release_deconfounded.parquet",
     "text", "id", "judgement", "press-releases"),
    # math-stackexchange: key-bridge alias of "math" (same corpus) so the task name matches the
    # R3 hierarchy files (math-stackexchange_general_r3_expanded.json) for the domain sweeps.
    ("math-stackexchange", "math_se_bridge",
     "datasets/math/stackexchange/math_se_modeling.csv.gz",
     "text", None, "judgement", "math/stackexchange"),
    # ---- wave-2 domain sweeps (2026-07-05): keys must equal the hierarchy task names ----
    # legal-outcome-prediction: bridge alias of "law"; probes = title_vii facts (balanced slice).
    ("legal-outcome-prediction", "law_bridge",
     "datasets/legal-outcome-prediction/title_vii_balanced_v2.jsonl",
     "facts", None, "binary_label", "legal-outcome-prediction"),
    # grant-funding: open-source-grants labeled pool — SMALL (210 docs, median 29K chars,
    # truncated to max_text_chars=4000 at scoring). label 'status' is non-numeric and unused
    # (reconstruction-only) -> None. Sweeps use --gepa-reserve 10 so probes = texts[10:210].
    ("grant-funding", "grants_os",
     "datasets/grant-funding/open-source-grants/processed/grants_labeled.csv.gz",
     "full_text", None, None, "grant-funding"),
    # code-review: canonical dense-training corpus (sk3-only file; run sweeps there).
    ("code-review", "code_review_dense",
     "datasets/code-review/code_review_dense_4096tok.csv.gz",
     "text", "paper_id", "judgement", "code-review"),
]


def full_manifest(run_id: str = "scale_full", metrics_per_task: int = 40,
                  metric_files_cap: int = 400, n_items: int = 60) -> RunManifest:
    """Scale-out manifest: 9 locally-available labeled datasets × ~``metrics_per_task``
    online-rubric metrics each (bounded by ``metric_files_cap`` files scanned) × the tier
    ladder. Several config_tasks repeat (3 legal slices) — each is its own dataset entry with
    its own metric bank. Tune the caps up for the real run."""
    ds = []
    for ctask, name, path, tcol, idcol, lcol, rdir in _FULL_DATASETS:
        ds.append(DatasetEntry(
            task=ctask,           # each task now has a correct config preset (config.py)
            name=name, path=str(REPO_ROOT / path), text_column=tcol, id_column=idcol,
            label_column=lcol, metric_source="online_rubrics",
            metric_glob=_rubric_glob(rdir), max_metric_files=metric_files_cap,
            max_metrics=metrics_per_task))
    return RunManifest(run_id=run_id, datasets=ds, tiers=DEFAULT_TIERS,
                       token_caps=[120, 1000], seeds=[0], gepa_rounds=2, passes=3,
                       n_items=n_items, n_oracle_items=16)
