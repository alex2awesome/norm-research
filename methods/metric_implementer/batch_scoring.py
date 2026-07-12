"""Unified long-table batch scoring — the persisted output of the scaled E7 system.

The user's spec (2026-06-12): batch-process metrics with vLLM (run through batches of
metrics × datapoints) and keep an output keyed by *(judge, item, (prompt, iteration))* plus the
GEPA operator labels (INIT / CLARIFY / MECHANIZE / FEWSHOT+ / ANCHOR / EDGE / PRUNE / DECOMPOSE)
that produced each prompt, saving every prompt and iteration.

This module is the score-event sink. One row per (judge_model × metric × prompt-version × item ×
pass). The prompt VERSION (its full body + operator + round lineage) lives in the Registry; the
long table references it by ``version_id`` + ``prompt_hash`` and carries the operator/round/
iteration inline so the table is self-describing for analysis without a registry join.

``batch_score`` is the primitive the orchestrator calls: it scores one (artifact, version) over
many items × passes through a resident judge backend (one vLLM model, already loaded), parses the
{score, applicable} contract, appends long rows via ``LongTableWriter``, and returns the score
matrix for the scorecard/optimizer. Batching ACROSS metrics is achieved by the orchestrator
keeping the model resident and calling ``batch_score`` for every metric/candidate under it — and,
in mega-batch mode, by ``batch_score_many`` which unions all (metric × item × pass) prompts into a
SINGLE ``generate_batch`` call (thousands of prompts, per [[feedback_vllm_batch_size]]).

Persistence is parquet (append-as-new-rowgroup via per-flush part files), never gzip-append
([[feedback_no_long_running_gzip_append]]); flushed every ``flush_every`` rows so an overnight run
checkpoints ([[feedback_vllm_safe_run_config]]).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .artifact import MetricArtifact
from .backends import parse_json_obj
from .judges import judge_prompts

# The long-table schema (the output we agreed on, extended with GEPA lineage).
LONGTABLE_COLUMNS = [
    "run_id", "dataset", "task",
    "judge_model",                  # the capability-axis tier that produced this score
    "metric_id", "version_id",      # prompt identity (-> Registry version record)
    "operator",                     # GEPA op that MINTED this prompt: INIT/CLARIFY/MECHANIZE/...
    "round",                        # GEPA round (the "iteration" axis)
    "token_cap",                    # budget cell this prompt was produced under
    "pass",                         # scoring replicate (test-retest)
    "item_id", "score", "applicable",
    "prompt_hash",                  # sha256 of the prompt body (cross-check vs registry)
    "ts",
]


@dataclass
class LongTableWriter:
    """Append-only long-table sink. Buffers rows, flushes to numbered parquet parts so an
    interrupted run loses at most ``flush_every`` rows and never corrupts a stream."""
    out_dir: Path
    run_id: str
    flush_every: int = 2000
    _buf: List[dict] = field(default_factory=list)
    _part: int = 0
    n_written: int = 0

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def add(self, rows: List[dict]) -> None:
        self._buf.extend(rows)
        if len(self._buf) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        df = pd.DataFrame(self._buf, columns=LONGTABLE_COLUMNS)
        path = self.out_dir / f"{self.run_id}__part{self._part:04d}.parquet"
        df.to_parquet(path, index=False)
        self.n_written += len(self._buf)
        self._part += 1
        self._buf = []

    @staticmethod
    def load(out_dir, run_id: Optional[str] = None) -> pd.DataFrame:
        out_dir = Path(out_dir)
        pat = f"{run_id}__part*.parquet" if run_id else "*__part*.parquet"
        parts = sorted(out_dir.glob(pat))
        if not parts:
            return pd.DataFrame(columns=LONGTABLE_COLUMNS)
        return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def _parse_score(raw: str) -> Tuple[float, bool]:
    obj = parse_json_obj(raw)
    if not obj or "score" not in obj:
        return float("nan"), False
    try:
        s = float(obj["score"])
    except (TypeError, ValueError):
        return float("nan"), False
    if not (0.0 <= s <= 1.0):
        return float("nan"), False
    if obj.get("applicable") is False:
        return float("nan"), False
    return s, True


@dataclass
class ScoreJob:
    """One (metric-version × corpus) unit to score. The orchestrator builds a list of these
    across ALL metrics, then mega-batches them through one resident model."""
    artifact: MetricArtifact
    version_id: str
    texts: List[str]
    item_ids: List[str]
    operator: str = "INIT"
    round: int = 0
    token_cap: Optional[int] = None
    dataset: str = ""
    task: str = ""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _rows_for(job: ScoreJob, judge_model: str, run_id: str, passes: int,
              score_mat: np.ndarray, appl_mat: np.ndarray) -> List[dict]:
    rows = []
    for p in range(passes):
        for k, iid in enumerate(job.item_ids):
            rows.append({
                "run_id": run_id, "dataset": job.dataset, "task": job.task,
                "judge_model": judge_model, "metric_id": job.artifact.metric_id,
                "version_id": job.version_id, "operator": job.operator,
                "round": job.round, "token_cap": job.token_cap, "pass": p,
                "item_id": iid, "score": float(score_mat[p, k]),
                "applicable": bool(appl_mat[p, k]),
                "prompt_hash": job.artifact.body_hash, "ts": _now(),
            })
    return rows


def batch_score_many(jobs: List[ScoreJob], judge_backend, cfg, *, run_id: str,
                     passes: int = 1, writer: Optional[LongTableWriter] = None,
                     max_tokens: int = 80) -> Dict[str, np.ndarray]:
    """MEGA-BATCH: union every (job × item × pass) prompt into ONE generate_batch call on the
    resident model, then demux. Returns {version_id: score_mat[passes, n_items]} and appends
    long rows. This is the throughput primitive — thousands of prompts per flush.

    All jobs are scored by the SAME judge_backend (one resident tier). Different tiers = call
    again with a different backend (the orchestrator's outer loop)."""
    system, template = judge_prompts(cfg)
    prompts, index = [], []     # index[i] = (job_idx, pass, item_k)
    for ji, job in enumerate(jobs):
        for p in range(passes):
            for k, t in enumerate(job.texts):
                prompts.append(template.format(rubric=job.artifact.body,
                                               text=t[: cfg.max_text_chars]))
                index.append((ji, p, k))
    raws = judge_backend.generate_batch(
        prompts, system=system, max_tokens=max_tokens,
        validate=lambda s: _parse_score(s)[1] or parse_json_obj(s) is not None)
    # allocate per-job matrices
    out = {job.version_id: (np.full((passes, len(job.texts)), np.nan),
                            np.zeros((passes, len(job.texts)), dtype=bool))
           for job in jobs}
    for (ji, p, k), raw in zip(index, raws):
        vid = jobs[ji].version_id
        s, ap = _parse_score(raw)
        out[vid][0][p, k] = s
        out[vid][1][p, k] = ap
    if writer is not None:
        for job in jobs:
            sm, am = out[job.version_id]
            writer.add(_rows_for(job, judge_backend.model, run_id, passes, sm, am))
    return {vid: sm for vid, (sm, am) in out.items()}, \
           {vid: am for vid, (sm, am) in out.items()}


_YESNO_TEMPLATE = (
    "You are evaluating a text on ONE specific criterion.\n\n"
    "Criterion:\n{rubric}\n\n"
    "Text:\n{text}\n\n"
    "Does the text satisfy the criterion? Answer with exactly one word: YES or NO."
)


# ---- empty-rubric baseline (conditional V-information) -------------------------------------
# The H_V(Y|X) baseline: an executor reading NO criteria, scoring an item from its unconditioned
# prior alone. This is a NO-RUBRIC CONTROL, deliberately NOT a target metric M — it exists so
# analysis can subtract it from any rubric to get the rubric's MARGINAL usable information
# (conditional V-info / REV, Chen et al. 2023) and split substitutive (baseline already high) vs
# complementary (rubric unlocks more than its standalone value) tacit knowledge. It is never used
# as the target M in any recovery/certificate (those use a behavioral bank-metric or full-rubric
# verdict). Rides the same continuous/sampled/discrete paths.
VACUOUS_BASELINE_ID = "__vacuous_baseline__"


def vacuous_baseline_artifact(task: str) -> MetricArtifact:
    noun = (task or "text").replace("_", " ").strip() or "text"
    body = ("Overall, is this a strong example of its kind — NO specific criteria are provided.\n"
            f"Considering this {noun} as a whole, use only your own unconditioned judgment; "
            "do not apply any particular checklist. Is it a strong example? Answer YES or NO.")
    return MetricArtifact(metric_id=VACUOUS_BASELINE_ID, kind="prompt", body=body,
                          name="vacuous no-rubric baseline",
                          description="empty-rubric H_V(Y|X) baseline (control, not a target M) for conditional V-information")


def score_continuous_many(jobs: List[ScoreJob], judge_backend, cfg, *, run_id: str,
                          passes: int = 1, writer: Optional[LongTableWriter] = None
                          ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """CONTINUOUS-fidelity scoring (U3): score = P(YES) from 1-token logprobs via
    ``judge_backend.score_binary`` — bypasses the JSON-format failure that degenerated the discrete
    MEASURE on weak judges. Same long-table schema; ``score`` in [0,1], ``applicable`` = score not
    nan. The right responses for the capability-axis IRT (irt_model.py)."""
    sys_tmpl = getattr(cfg, "judge_system_continuous", None)
    prompts, index = [], []
    for ji, job in enumerate(jobs):
        for p in range(passes):
            for k, t in enumerate(job.texts):
                prompts.append(_YESNO_TEMPLATE.format(rubric=job.artifact.body,
                                                      text=t[: cfg.max_text_chars]))
                index.append((ji, p, k))
    scores = judge_backend.score_binary(prompts, system=sys_tmpl, pos="YES", neg="NO")
    out = {job.version_id: (np.full((passes, len(job.texts)), np.nan),
                            np.zeros((passes, len(job.texts)), dtype=bool)) for job in jobs}
    for (ji, p, k), sc in zip(index, scores):
        vid = jobs[ji].version_id
        ok = sc is not None and not (isinstance(sc, float) and np.isnan(sc))
        out[vid][0][p, k] = sc if ok else np.nan
        out[vid][1][p, k] = ok
    if writer is not None:
        for job in jobs:
            sm, am = out[job.version_id]
            writer.add(_rows_for(job, judge_backend.model, run_id, passes, sm, am))
    return ({vid: sm for vid, (sm, am) in out.items()},
            {vid: am for vid, (sm, am) in out.items()})


def score_sampled_many(jobs: List[ScoreJob], judge_backend, cfg, *, run_id: str,
                       passes: int = 5, temperature: float = 0.7,
                       writer: Optional[LongTableWriter] = None
                       ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """NO-ABSTENTION, multi-pass SAMPLED scoring (procedure change 2026-06-14): force a YES/NO
    verdict (no 'not applicable' option — non-applicability shows up as a low score / low item-z),
    SAMPLE ``passes`` times at temperature>0 with a distinct seed per pass, record per-pass 0/1.
    Pass-rate over passes = empirical response P_ij (the Beta-IRT input); across-pass agreement =
    consistency. Runs over every version (include_history) × every tier (orchestrator loop)."""
    base, index = [], []
    for ji, job in enumerate(jobs):
        for k, t in enumerate(job.texts):
            base.append(_YESNO_TEMPLATE.format(rubric=job.artifact.body, text=t[: cfg.max_text_chars]))
            index.append((ji, k))
    out = {job.version_id: (np.full((passes, len(job.texts)), np.nan),
                            np.zeros((passes, len(job.texts)), dtype=bool)) for job in jobs}
    for p in range(passes):
        outs = judge_backend.generate_batch(base, system=None, max_tokens=4,
                                            temperature=temperature, seed=p + 1)
        for (ji, k), o in zip(index, outs):
            t = (o or "").strip().lower()
            sc = (1.0 if t.startswith("yes") or t[:8].find("yes") >= 0 and t[:8].find("no") < 0
                  else (0.0 if t.startswith("no") or "no" in t[:8] else np.nan))
            vid = jobs[ji].version_id
            out[vid][0][p, k] = sc
            out[vid][1][p, k] = not (isinstance(sc, float) and np.isnan(sc))
    if writer is not None:
        for job in jobs:
            sm, am = out[job.version_id]
            writer.add(_rows_for(job, judge_backend.model, run_id, passes, sm, am))
    return ({vid: sm for vid, (sm, am) in out.items()},
            {vid: am for vid, (sm, am) in out.items()})


def batch_score(job: ScoreJob, judge_backend, cfg, *, run_id: str, passes: int = 1,
                writer: Optional[LongTableWriter] = None,
                max_tokens: int = 80) -> Tuple[np.ndarray, np.ndarray]:
    """Single-job convenience wrapper over ``batch_score_many``."""
    sm, am = batch_score_many([job], judge_backend, cfg, run_id=run_id, passes=passes,
                              writer=writer, max_tokens=max_tokens)
    return sm[job.version_id], am[job.version_id]
