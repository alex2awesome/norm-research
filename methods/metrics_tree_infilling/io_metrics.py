"""Inputs and materialization (spec §1).

Loads the explicit metrics and the labeled corpus, then materializes every metric's
*level* over the whole corpus into a design matrix ``X`` (within-node model features) plus a
frame of partitioning covariates ``z``.

Where the metrics live
----------------------
Most explicit metrics in this repo are **LLM-judge rubrics** extracted from real-world
guideline documents, stored per task at::

    datasets/<task>/online-rubrics/{gpt-parsed,claude-parsed}/**/*.json
        -> json["extracted"]["rubrics_metrics"] = [{name, description, guidance}, ...]

Code-based metrics (deterministic ``score(text)->float``) live in module directories such as
``methods/existing_metrics_runner/coded/metrics/`` and are loaded by import (mirroring that
method's runner). Both kinds are normalized to :class:`MetricSpec`.

Materialization is decoupled from any specific LLM backend: judge metrics are scored through
an injected ``judge_scorer`` callable (see :func:`make_vllm_judge_scorer`, which reuses
``LLMClient`` from ``verification_library``) or supplied as ``precomputed`` levels (e.g. from
the v2 cells DB), so the rest of the method runs without a live model.
"""

from __future__ import annotations

import glob
import hashlib
import importlib.util
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

# judge_scorer(metrics, texts) -> (levels (n, M) float w/ NaN where N/A, applicable (n, M) bool)
JudgeScorer = Callable[[List["MetricSpec"], List[str]], Tuple[np.ndarray, np.ndarray]]


# --------------------------------------------------------------------------------------
# Metric specs
# --------------------------------------------------------------------------------------

@dataclass
class MetricSpec:
    """One explicit metric: a code scorer or an LLM-judge rubric."""

    metric_id: str
    name: str
    description: str
    kind: str                                   # "code" | "judge"
    guidance: str = ""                          # judge: the rubric body to apply
    code_fn: Optional[Callable[[str], Optional[float]]] = None   # code: score(text)->float|None
    # Partitioning role: "feature" -> within-node model (X) only; "context" -> splitting
    # covariate (z) only; "both" -> X and z. Discovered features default to "feature" so the
    # tree never splits on their raw value (which would fragment regions); their coefficient
    # instability across the context covariates still drives splits (standard MOB practice).
    role: str = "both"

    @property
    def rubric_text(self) -> str:
        return self.guidance or self.description


def _stable_id(prefix: str, *parts: str) -> str:
    h = hashlib.sha256("||".join(parts).encode()).hexdigest()[:12]
    return f"{prefix}_{h}"


def load_rubric_metrics(
    task: str,
    repo_root: Path = REPO_ROOT,
    parsers: Tuple[str, ...] = ("gpt-parsed", "claude-parsed"),
    dedup: bool = True,
    limit: Optional[int] = None,
) -> List[MetricSpec]:
    """Load LLM-judge rubric metrics from ``datasets/<task>/online-rubrics/*-parsed/``."""
    base = repo_root / "datasets" / task / "online-rubrics"
    specs: List[MetricSpec] = []
    seen: set = set()
    for parser in parsers:
        for fp in sorted(glob.glob(str(base / parser / "**" / "*.json"), recursive=True)):
            try:
                doc = json.load(open(fp))
            except Exception:
                continue
            extracted = doc.get("extracted") if isinstance(doc, dict) else None
            if not isinstance(extracted, dict):
                continue
            for rm in extracted.get("rubrics_metrics", []) or []:
                name = (rm.get("name") or "").strip()
                desc = (rm.get("description") or "").strip()
                guidance = (rm.get("guidance") or "").strip()
                if not name:
                    continue
                key = (name.lower(), desc.lower())
                if dedup and key in seen:
                    continue
                seen.add(key)
                specs.append(MetricSpec(
                    metric_id=_stable_id("j", name, desc),
                    name=name, description=desc, kind="judge", guidance=guidance,
                ))
    if limit is not None:
        specs = specs[:limit]
    return specs


def load_rubric_metrics_from_dir(rubrics_dir: str | Path) -> List[MetricSpec]:
    """Load judge rubric metrics from a custom directory of ``{"extracted":{"rubrics_metrics":[...]}}``
    JSON files (e.g. distilled rubrics exported by ``metric_implementer.export``). Mirrors
    :func:`load_rubric_metrics` over an arbitrary path instead of ``datasets/<task>/online-rubrics``.
    """
    rubrics_dir = Path(rubrics_dir)
    specs: List[MetricSpec] = []
    seen: set = set()
    for fp in sorted(glob.glob(str(rubrics_dir / "**" / "*.json"), recursive=True)):
        try:
            doc = json.load(open(fp))
        except Exception:
            continue
        extracted = doc.get("extracted") if isinstance(doc, dict) else None
        if not isinstance(extracted, dict):
            continue
        for rm in extracted.get("rubrics_metrics", []) or []:
            name = (rm.get("name") or "").strip()
            desc = (rm.get("description") or "").strip()
            guidance = (rm.get("guidance") or "").strip()
            if not name:
                continue
            key = (name.lower(), desc.lower())
            if key in seen:
                continue
            seen.add(key)
            specs.append(MetricSpec(
                metric_id=_stable_id("j", name, desc),
                name=name, description=desc, kind="judge", guidance=guidance,
            ))
    return specs


def load_code_metrics(metrics_dir: str | Path) -> List[MetricSpec]:
    """Import every ``*.py`` in ``metrics_dir`` exposing ``score(text)->float|None``.

    Mirrors ``methods/existing_metrics_runner/coded/runner.py``'s discover-and-call convention; an
    optional module-level ``applies(text)->bool`` gates scoring (N/A when False).
    """
    metrics_dir = Path(metrics_dir)
    specs: List[MetricSpec] = []
    for fp in sorted(metrics_dir.glob("*.py")):
        if fp.name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(f"_infill_metric_{fp.stem}", fp)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            continue
        if not hasattr(mod, "score"):
            continue
        name = getattr(mod, "ASPECT_NAME", fp.stem)
        desc = (mod.__doc__ or name).strip().splitlines()[0]
        applies = getattr(mod, "applies", None)
        score = mod.score

        def make_fn(score=score, applies=applies):
            def fn(text: str) -> Optional[float]:
                if applies is not None and not applies(text):
                    return None
                try:
                    v = score(text)
                except Exception:
                    return None
                return None if v is None else float(v)
            return fn

        specs.append(MetricSpec(
            metric_id=_stable_id("c", fp.stem, name),
            name=name, description=desc, kind="code", code_fn=make_fn(),
        ))
    return specs


# --------------------------------------------------------------------------------------
# Corpus loading + honest split
# --------------------------------------------------------------------------------------

def load_items(split_path: str | Path, cfg) -> pd.DataFrame:
    """Load the labeled corpus from a split directory or a single CSV.

    Supports the repo convention: a directory with ``{train,eval,test}.{csv.gz,csv}`` (all
    concatenated), or a single ``.csv``/``.csv.gz`` file. Coerces the label to {0,1} and
    drops rows missing id/text/label.
    """
    p = Path(split_path)
    frames: List[pd.DataFrame] = []
    if p.is_dir():
        for split in ("train", "eval", "test"):
            for ext in (".csv.gz", ".csv"):
                fp = p / f"{split}{ext}"
                if fp.exists():
                    frames.append(pd.read_csv(fp, low_memory=False))
                    break
        if not frames:
            raise FileNotFoundError(f"No train/eval/test splits under {p}")
    else:
        frames.append(pd.read_csv(p, low_memory=False))

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=[cfg.id_column, cfg.text_column, cfg.label_column]).copy()
    df[cfg.label_column] = _coerce_binary(df[cfg.label_column])
    df = df.dropna(subset=[cfg.label_column]).reset_index(drop=True)
    df[cfg.label_column] = df[cfg.label_column].astype(int)
    return df


def _coerce_binary(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(float)
    out = pd.to_numeric(s, errors="coerce")
    if out.notna().any():
        u = set(np.unique(out.dropna()))
        if u <= {0.0, 1.0}:
            return out
        return (out > out.median()).astype(float)
    mapping = {"yes": 1, "accept": 1, "accepted": 1, "true": 1, "1": 1,
               "no": 0, "reject": 0, "rejected": 0, "false": 0, "0": 0}
    return s.astype(str).str.strip().str.lower().map(mapping)


def discover_test_split(df: pd.DataFrame, cfg) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Seeded honest split (spec §1). Every keep/drop decision is made on the test side."""
    rng = np.random.default_rng(cfg.random_seed)
    idx = rng.permutation(len(df))
    n_disc = int(round(cfg.discover_fraction * len(df)))
    d = df.iloc[idx[:n_disc]].reset_index(drop=True)
    t = df.iloc[idx[n_disc:]].reset_index(drop=True)
    return d, t


def _group_keys(df: pd.DataFrame, cfg) -> np.ndarray:
    """Per-row group key: the id_column value when grouping is requested, else a unique per-row
    key (so each row is its own group == ordinary row-random splitting)."""
    if getattr(cfg, "group_split_by_id", False):
        return df[cfg.id_column].astype(str).to_numpy()
    return np.arange(len(df))  # unique -> no rows are forced together


def three_way_split(
    df: pd.DataFrame, cfg,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Seeded discover / guard / test split (spec §1, 2026-06-10 next-steps plan).

    - **discover**: the tree is fit here (within-node GLMs + instability tests).
    - **guard** (select): gap nodes are flagged and every keep/drop decision is made here.
    - **test** (confirm): touched ONLY to materialize features for the final power read; no
      decision ever consults it, so the reported test AUC is a genuine holdout.

    Fractions: test = cfg.test_fraction, guard = cfg.guard_fraction, discover = the rest. If
    ``cfg.group_split_by_id`` is set, all rows sharing an ``id_column`` value land in one split
    (prevents a comparison pair or shared-source rows from straddling train/test).
    """
    rng = np.random.default_rng(cfg.random_seed)
    keys = _group_keys(df, cfg)
    uniq_keys = np.array(sorted(set(keys.tolist())))
    rng.shuffle(uniq_keys)
    n = len(df)
    n_test = int(round(getattr(cfg, "test_fraction", 0.30) * n))
    n_guard = int(round(getattr(cfg, "guard_fraction", 0.14) * n))
    # assign whole groups until each split reaches its target row count
    test_keys, guard_keys, disc_keys = set(), set(), set()
    test_n = guard_n = 0
    for k in uniq_keys:
        size = int((keys == k).sum())
        if test_n < n_test:
            test_keys.add(k); test_n += size
        elif guard_n < n_guard:
            guard_keys.add(k); guard_n += size
        else:
            disc_keys.add(k)
    disc = df[np.isin(keys, list(disc_keys))].reset_index(drop=True)
    guard = df[np.isin(keys, list(guard_keys))].reset_index(drop=True)
    test = df[np.isin(keys, list(test_keys))].reset_index(drop=True)
    # guard against a degenerate split (too-small input): fall back to row-random 3-way
    if len(disc) == 0 or len(guard) == 0 or len(test) == 0:
        idx2 = rng.permutation(len(df))
        nt = max(1, int(round(getattr(cfg, "test_fraction", 0.30) * len(df))))
        ng = max(1, int(round(getattr(cfg, "guard_fraction", 0.14) * len(df))))
        test = df.iloc[idx2[:nt]].reset_index(drop=True)
        guard = df.iloc[idx2[nt:nt + ng]].reset_index(drop=True)
        disc = df.iloc[idx2[nt + ng:]].reset_index(drop=True)
    return disc, guard, test


# --------------------------------------------------------------------------------------
# Materialization
# --------------------------------------------------------------------------------------

@dataclass
class ScoreMatrix:
    """Materialized metric levels over a set of items."""

    levels: np.ndarray          # (n, M) float, NaN where not applicable
    applicable: np.ndarray      # (n, M) bool
    metric_ids: List[str]
    metric_names: List[str]
    roles: List[str] = field(default_factory=list)   # per-metric "feature"|"context"|"both"


def materialize(
    metrics: List[MetricSpec],
    df: pd.DataFrame,
    cfg,
    judge_scorer: Optional[JudgeScorer] = None,
    precomputed: Optional[Dict[Tuple[str, str], float]] = None,
) -> ScoreMatrix:
    """Score every metric over every item -> levels matrix.

    ``precomputed`` maps ``(metric_id, item_id) -> level`` (e.g. from the cells DB) and is
    consulted first for judge metrics; remaining judge metrics are scored via ``judge_scorer``.
    """
    texts = df[cfg.text_column].astype(str).tolist()
    ids = df[cfg.id_column].astype(str).tolist()
    n, M = len(df), len(metrics)
    levels = np.full((n, M), np.nan)
    applicable = np.zeros((n, M), dtype=bool)

    code_cols = [j for j, m in enumerate(metrics) if m.kind == "code"]
    judge_cols = [j for j, m in enumerate(metrics) if m.kind == "judge"]

    for j in code_cols:
        fn = metrics[j].code_fn
        for i, txt in enumerate(texts):
            v = fn(txt)
            if v is not None and np.isfinite(v):
                levels[i, j] = v
                applicable[i, j] = True

    # judge metrics: precomputed first, then live scorer for the remainder
    remaining = list(judge_cols)
    if precomputed:
        still: List[int] = []
        for j in judge_cols:
            mid = metrics[j].metric_id
            got_any = False
            for i, iid in enumerate(ids):
                key = (mid, iid)
                if key in precomputed:
                    levels[i, j] = precomputed[key]
                    applicable[i, j] = not np.isnan(precomputed[key])
                    got_any = True
            if not got_any:
                still.append(j)
        remaining = still

    if remaining:
        if judge_scorer is None:
            raise ValueError(
                f"{len(remaining)} judge metrics need scoring but no judge_scorer/precomputed "
                "was provided. Pass make_vllm_judge_scorer(cfg) or precomputed levels."
            )
        sub = [metrics[j] for j in remaining]
        lv, ap = judge_scorer(sub, texts)
        for col, j in enumerate(remaining):
            levels[:, j] = lv[:, col]
            applicable[:, j] = ap[:, col]

    return ScoreMatrix(
        levels=levels, applicable=applicable,
        metric_ids=[m.metric_id for m in metrics],
        metric_names=[m.name for m in metrics],
        roles=[m.role for m in metrics],
    )


# --------------------------------------------------------------------------------------
# Design matrix (X) + partitioning covariates (z), with NA handling
# --------------------------------------------------------------------------------------

@dataclass
class DesignSpec:
    """Frozen column layout so discover and test rows share the same features."""

    metric_ids: List[str]
    metric_names: List[str]
    na_metric_cols: List[int]            # metric columns that carry a missingness indicator
    impute_values: np.ndarray            # (M,) median level per metric (from discover)
    extra_z_numeric: List[str]
    extra_z_categorical: List[str]
    x_metric_cols: List[int]             # metrics in the within-node model X (role feature|both)
    z_metric_cols: List[int]             # metrics offered as splitting covariates (role context|both)


def make_design(
    sm: ScoreMatrix, df: pd.DataFrame, cfg, spec: Optional[DesignSpec] = None,
) -> Tuple[np.ndarray, List[str], Dict[str, Tuple[np.ndarray, str]], DesignSpec]:
    """Build the within-node feature matrix ``X`` and the ``z`` frame.

    Metric membership follows each metric's ``role`` (see :class:`MetricSpec`): ``feature``
    -> X only, ``context`` -> z only, ``both`` -> X and z. ``X`` also carries a 0/1
    missingness indicator for any X-metric that is ever N/A; ``z`` carries each splitting
    metric's level plus ``text_length`` and configured extra item columns. Splitting on a
    metric's raw value is what fragments regions, so discovered features are kept out of z.
    """
    n, M = sm.levels.shape
    roles = sm.roles if sm.roles else ["both"] * M
    if spec is None:
        impute = np.nanmedian(np.where(sm.applicable, sm.levels, np.nan), axis=0)
        impute = np.where(np.isfinite(impute), impute, 0.5)
        na_cols = [j for j in range(M) if (~sm.applicable[:, j]).any()]
        extra_num, extra_cat = [], []
        for c in cfg.extra_z_columns:
            if c in df.columns:
                (extra_num if pd.api.types.is_numeric_dtype(df[c]) else extra_cat).append(c)
        x_cols = [j for j in range(M) if roles[j] in ("feature", "both")]
        if getattr(cfg, "curated_z_only", False):
            z_cols = [j for j in range(M) if roles[j] == "context"]
        else:
            z_cols = [j for j in range(M) if roles[j] in ("context", "both")]
        spec = DesignSpec(sm.metric_ids, sm.metric_names, na_cols, impute,
                          extra_num, extra_cat, x_cols, z_cols)

    # imputed levels
    levels = sm.levels.copy()
    for j in range(M):
        col = levels[:, j]
        col[~np.isfinite(col)] = spec.impute_values[j]
        levels[:, j] = col

    # within-node features X (role feature|both) + their missingness indicators
    X_cols, feature_names = [], []
    for j in spec.x_metric_cols:
        X_cols.append(levels[:, j])
        feature_names.append(spec.metric_names[j])
    for j in spec.na_metric_cols:
        if j not in spec.x_metric_cols:
            continue
        X_cols.append((~sm.applicable[:, j]).astype(float))
        feature_names.append(f"{spec.metric_names[j]}__NA")

    # splitting covariates z (role context|both) + their missingness indicators
    z: Dict[str, Tuple[np.ndarray, str]] = {}
    for j in spec.z_metric_cols:
        z[spec.metric_names[j]] = (levels[:, j], "numeric")
    for j in spec.na_metric_cols:
        if j in spec.z_metric_cols:
            z[f"{spec.metric_names[j]}__NA"] = ((~sm.applicable[:, j]).astype(float), "categorical")

    if getattr(cfg, "include_text_length_in_z", True):
        z["text_length"] = (df[cfg.text_column].astype(str).str.len().to_numpy(float), "numeric")
    for c in spec.extra_z_numeric:
        z[c] = (pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy(float), "numeric")
    for c in spec.extra_z_categorical:
        z[c] = (df[c].astype(str).to_numpy(), "categorical")

    X = np.column_stack(X_cols) if X_cols else np.zeros((n, 0))
    return X, feature_names, z, spec


# --------------------------------------------------------------------------------------
# Default vLLM/LLM judge scorer (reuses verification_library.LLMClient)
# --------------------------------------------------------------------------------------

# Shared judge instruction. The pre-2026-07-05 wording ("decide if the criterion is applicable
# ... and if so give a score") let executors conflate NOT-APPLICABLE with FAILS-THE-CRITERION:
# Llama-70B marked binary rubrics applicable=false whenever the answer was NO, censoring exactly
# the informative negative verdicts as N/A (caught by the planted-bank calibration's fidelity
# anchors — AUC vs code truth collapsed to ~0.5). GLM-5.2 did not conflate; the wording below
# pins the semantics for every executor.
_JUDGE_PROMPT_HEADER = (
    "Score the TEXT on each criterion. Mark applicable=false ONLY when the criterion cannot be "
    "meaningfully evaluated on this kind of text at all (wrong genre or text type). If it can "
    "be evaluated, set applicable=true and give score 1 if the text satisfies the criterion "
    "and 0 if it does not (fractional scores allowed for partial satisfaction). A text that "
    "FAILS a criterion is still applicable — failing means score 0, NOT applicable=false. "
    "Return ONLY a JSON array of objects "
    '{"index": int, "applicable": bool, "score": number}.'
)


def make_vllm_judge_scorer(cfg) -> JudgeScorer:
    """A judge scorer backed by ``LLMClient`` (anthropic or openai-compatible/vLLM).

    One prompt per item scores all requested metrics at once (returns a JSON array), keeping
    the call count at N and the batch wide. Levels are in [0,1]; N/A when the judge marks a
    metric inapplicable. Swap for the ``metric_tree`` ``score_ternary_subset`` path on sk3 if
    you want the exact v2 judge formatting.
    """
    import asyncio

    from verification_library.client import LLMClient

    if cfg.materialize_backend == "vllm_offline":
        return make_offline_vllm_judge_scorer(cfg)
    if cfg.materialize_backend == "anthropic":
        client = LLMClient.from_anthropic(model=cfg.materialize_model, concurrency=cfg.llm_concurrency)
    else:
        client = LLMClient.from_openai_compatible(
            model=cfg.materialize_model,
            base_url=cfg.openai_base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY"),
            concurrency=cfg.llm_concurrency,
        )

    def scorer(metrics: List[MetricSpec], texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        rubric_block = "\n".join(
            f"{k}. {m.name}: {m.rubric_text or m.description}" for k, m in enumerate(metrics)
        )
        max_chars = cfg.max_text_tokens * 4
        prompts = [
            f"{_JUDGE_PROMPT_HEADER}\n\nCRITERIA:\n{rubric_block}\n\nTEXT:\n{t[:max_chars]}"
            for t in texts
        ]
        cache_path = Path(cfg.cache_dir or Path(cfg.output_dir) / "judge_cache") / "judge.jsonl"
        responses = asyncio.run(client.generate_batch(
            prompts, max_tokens=1500, temperature=0.0,
            cache_path=cache_path, cache_key_fn=lambda i: hashlib.sha256(prompts[i].encode()).hexdigest()[:16],
        ))
        n, M = len(texts), len(metrics)
        levels = np.full((n, M), np.nan)
        applicable = np.zeros((n, M), dtype=bool)
        for i, resp in enumerate(responses):
            for obj in _parse_json_array(resp):
                j = obj.get("index")
                if not isinstance(j, int) or not (0 <= j < M):
                    continue
                if obj.get("applicable", True) and obj.get("score") is not None:
                    levels[i, j] = float(np.clip(obj["score"], 0.0, 1.0))
                    applicable[i, j] = True
        return levels, applicable

    return scorer


# One offline engine per model path, shared by the judge scorer AND the local proposer —
# a second vLLM engine on the same GPU would OOM, and the proposer's one-prompt calls ride
# the resident judge engine for free (also removes the external-quota failure mode that
# voided the 2026-07-05 overnight: z.ai rate limits exhausted every proposer call).
_OFFLINE_ENGINES: Dict[str, object] = {}


def _get_offline_engine(cfg):
    key = str(cfg.materialize_model)
    if key not in _OFFLINE_ENGINES:
        from vllm import LLM
        _OFFLINE_ENGINES[key] = LLM(
            model=cfg.materialize_model,
            max_model_len=int(getattr(cfg, "vllm_max_model_len", 8192)),
            gpu_memory_utilization=float(getattr(cfg, "vllm_gpu_mem_util", 0.93)),
        )
    return _OFFLINE_ENGINES[key]


def make_offline_vllm_proposer(cfg):
    """Proposer backed by the SAME resident offline engine as the judge (one prompt -> text).

    Used when ``cfg.proposer_backend == "vllm_offline"``: the articulator and the executor
    are then the same model — the fully executor-closed certificate (articulability BY E FOR
    E), with no external API in the loop.

    The prompt is CLAMPED to fit the context window. The RESIDUAL arm assembles WRONG/RIGHT
    text excerpts (contrast_max_chars per example x many) and routinely blew the 8192-token
    window (2026-07-06: it failed EVERY round -> the residual arm was silently disabled across
    the overnight matrix). We keep the head+tail of an over-long prompt (the instruction/return
    format lives at both ends) rather than drop the request.
    """
    from vllm import SamplingParams
    out_tokens = 900
    sp = SamplingParams(temperature=float(cfg.llm_temperature), max_tokens=out_tokens)
    max_len = int(getattr(cfg, "vllm_max_model_len", 16384))
    # rough 3.3 chars/token; leave room for output + chat template overhead
    char_budget = int((max_len - out_tokens - 256) * 3.3)

    def _clamp(p: str) -> str:
        if len(p) <= char_budget:
            return p
        head = int(char_budget * 0.6)
        tail = char_budget - head
        return p[:head] + "\n\n...[excerpts truncated to fit context]...\n\n" + p[-tail:]

    def proposer(prompt: str) -> Optional[str]:
        llm = _get_offline_engine(cfg)
        outs = llm.chat([[{"role": "user", "content": _clamp(prompt)}]], sp)
        return outs[0].outputs[0].text if outs and outs[0].outputs else None

    return proposer


def make_offline_vllm_judge_scorer(cfg) -> JudgeScorer:
    """In-process vLLM OFFLINE-batch judge (``materialize_backend == "vllm_offline"``).

    Large-scale scoring never goes through an HTTP server: one ``llm.chat`` call per scorer
    invocation carries the full text batch (hundreds to thousands of prompts). Same prompt
    protocol + JSONL cache as :func:`make_vllm_judge_scorer`, so cached anthropic runs and
    offline runs interoperate per prompt-hash. The engine loads lazily on first call (imports
    stay cheap for --help / dry runs) and is shared with the local proposer.
    """
    def _engine():
        from vllm import SamplingParams
        return _get_offline_engine(cfg), SamplingParams(temperature=0.0, max_tokens=1500)

    def scorer(metrics: List[MetricSpec], texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        rubric_block = "\n".join(
            f"{k}. {m.name}: {m.rubric_text or m.description}" for k, m in enumerate(metrics)
        )
        max_chars = cfg.max_text_tokens * 4
        prompts = [
            f"{_JUDGE_PROMPT_HEADER}\n\nCRITERIA:\n{rubric_block}\n\nTEXT:\n{t[:max_chars]}"
            for t in texts
        ]
        keys = [hashlib.sha256(p.encode()).hexdigest()[:16] for p in prompts]
        cache_path = Path(cfg.cache_dir or Path(cfg.output_dir) / "judge_cache") / "judge.jsonl"
        cache: Dict[str, str] = {}
        if cache_path.exists():
            with open(cache_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        cache[rec["key"]] = rec["response"]
                    except Exception:
                        continue
        miss_idx = [i for i, k in enumerate(keys) if k not in cache]
        if miss_idx:
            llm, sp = _engine()
            convs = [[{"role": "user", "content": prompts[i]}] for i in miss_idx]
            outs = llm.chat(convs, sp)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "a") as f:
                for i, o in zip(miss_idx, outs):
                    resp = o.outputs[0].text if o.outputs else ""
                    cache[keys[i]] = resp
                    f.write(json.dumps({"key": keys[i], "response": resp}) + "\n")
        n, M = len(texts), len(metrics)
        levels = np.full((n, M), np.nan)
        applicable = np.zeros((n, M), dtype=bool)
        for i, k in enumerate(keys):
            for obj in _parse_json_array(cache.get(k)):
                j = obj.get("index")
                if not isinstance(j, int) or not (0 <= j < M):
                    continue
                if obj.get("applicable", True) and obj.get("score") is not None:
                    levels[i, j] = float(np.clip(obj["score"], 0.0, 1.0))
                    applicable[i, j] = True
        return levels, applicable

    return scorer


def _parse_json_array(resp: Optional[str]) -> List[dict]:
    if not resp:
        return []
    s = resp.strip()
    lo, hi = s.find("["), s.rfind("]")
    if lo == -1 or hi == -1 or hi <= lo:
        return []
    try:
        arr = json.loads(s[lo:hi + 1])
        return [o for o in arr if isinstance(o, dict)]
    except Exception:
        return []
