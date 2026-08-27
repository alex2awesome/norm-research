"""Per-metric judge test-retest reliability — the disambiguation instrument.

A measured gain of ~0 has two very different causes:
  (a) the metric genuinely does not predict the label (a real null — the articulability result), or
  (b) the executor cannot APPLY the rubric consistently, so its scores are noise that attenuates
      any real signal toward 0 (a measurement floor, NOT a result).

Nothing in the acceptance gate separates these. This module scores a sample of items for ONE
metric TWICE, independently (temperature > 0, distinct cache-busting salts), and reports the
test-retest agreement. Low reliability caps the recoverable gain: a metric the judge applies at
Spearman ρ can transmit at most ~ρ of its true signal, so a ~0 gain under low ρ is uninformative.

Reliability is a property of (metric × executor), exactly like the certificate (A*_E is
executor-indexed). Report it per accepted metric and per executor tier.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Sequence

import numpy as np

from .io_metrics import _JUDGE_PROMPT_HEADER, _get_offline_engine, _parse_json_array


def _one_metric_prompt(metric, text: str, max_chars: int, salt: str) -> str:
    rubric = metric.rubric_text or metric.description or metric.name
    # single-criterion scoring: isolates the metric (also removes the many-criteria-per-call
    # context dilution that the batch scorer incurs), so this is an UPPER bound on the batch
    # scorer's reliability — if it is low here it is at least as low in the real run.
    return (f"{_JUDGE_PROMPT_HEADER}\n\n(eval id {salt})\n\nCRITERIA:\n0. {metric.name}: {rubric}"
            f"\n\nTEXT:\n{text[:max_chars]}")


def _score_offline_once(cfg, metric, texts: Sequence[str], temperature: float, salt: str):
    from vllm import SamplingParams
    llm = _get_offline_engine(cfg)
    sp = SamplingParams(temperature=float(temperature), max_tokens=200, seed=None)
    max_chars = int(getattr(cfg, "max_text_tokens", 700)) * 4
    convs = [[{"role": "user", "content": _one_metric_prompt(metric, str(t), max_chars, salt)}]
             for t in texts]
    outs = llm.chat(convs, sp)
    return _extract(outs, len(texts))


def _score_anthropic_once(cfg, metric, texts, temperature, salt):
    import asyncio
    from verification_library.client import LLMClient
    client = LLMClient.from_anthropic(model=cfg.materialize_model, concurrency=cfg.llm_concurrency)
    max_chars = int(getattr(cfg, "max_text_tokens", 700)) * 4
    prompts = [_one_metric_prompt(metric, str(t), max_chars, salt) for t in texts]
    # no cache: reliability needs INDEPENDENT draws, so the salt differs per pass and we never
    # want a cache hit collapsing the two passes to one
    resp = asyncio.run(client.generate_batch(prompts, max_tokens=200, temperature=temperature,
                                             cache_path=None))
    return _extract_texts(resp, len(texts))


def _extract(outs, n):
    lv = np.full(n, np.nan); ap = np.zeros(n, bool)
    for i, o in enumerate(outs):
        txt = o.outputs[0].text if o and o.outputs else ""
        _fill(lv, ap, i, txt)
    return lv, ap


def _extract_texts(resps, n):
    lv = np.full(n, np.nan); ap = np.zeros(n, bool)
    for i, r in enumerate(resps):
        _fill(lv, ap, i, r or "")
    return lv, ap


def _fill(lv, ap, i, txt):
    for obj in _parse_json_array(txt):
        if obj.get("index") == 0 and obj.get("score") is not None:
            if obj.get("applicable", True):
                lv[i] = float(np.clip(obj["score"], 0.0, 1.0)); ap[i] = True
            return


def judge_test_retest(
    metric, texts: Sequence[str], cfg, n_sample: int = 48,
    temperature: float = 0.6, seed: int = 0,
) -> dict:
    """Score ``n_sample`` items for ``metric`` twice, independently, and report agreement.

    Returns:
      retest_spearman     — rank correlation of the two level readings on rows applicable in BOTH
      retest_pearson      — linear correlation (same rows)
      applicability_agree — fraction of rows with matching applicable/not-applicable verdicts
      binary_agree        — agreement of mean-split binarized verdicts (Cohen-style raw agreement)
      n_both_applicable   — support for the correlation
      attenuation_flag    — True when ρ < ``cfg.min_reliability`` (default 0.5): a ~0 gain here is
                            a MEASUREMENT FLOOR, not a null. The recoverable gain is capped by ρ.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(texts), size=min(n_sample, len(texts)), replace=False)
    sample = [str(texts[i]) for i in idx]
    backend = getattr(cfg, "materialize_backend", "anthropic")
    scorer = _score_offline_once if backend == "vllm_offline" else _score_anthropic_once
    lv1, ap1 = scorer(cfg, metric, sample, temperature, salt="passA")
    lv2, ap2 = scorer(cfg, metric, sample, temperature, salt="passB")

    both = ap1 & ap2 & np.isfinite(lv1) & np.isfinite(lv2)
    out = {
        "metric": metric.name, "n_sample": len(sample),
        "applicability_agree": float((ap1 == ap2).mean()),
        "mean_applicability": float((ap1.mean() + ap2.mean()) / 2),
        "n_both_applicable": int(both.sum()),
        "retest_spearman": float("nan"), "retest_pearson": float("nan"),
        "binary_agree": float("nan"),
    }
    if both.sum() >= 8 and np.std(lv1[both]) > 0 and np.std(lv2[both]) > 0:
        from scipy.stats import spearmanr
        out["retest_spearman"] = float(spearmanr(lv1[both], lv2[both]).correlation)
        out["retest_pearson"] = float(np.corrcoef(lv1[both], lv2[both])[0, 1])
        thr = np.nanmean(np.concatenate([lv1[both], lv2[both]]))
        out["binary_agree"] = float(((lv1[both] >= thr) == (lv2[both] >= thr)).mean())
    elif both.sum() >= 8:
        # zero within-pass variance but both applicable: the judge is DEGENERATE-constant on this
        # metric (no discrimination) — treat as perfectly reliable but uninformative
        out["retest_spearman"] = 1.0 if np.allclose(lv1[both], lv2[both]) else 0.0
        out["binary_agree"] = float((lv1[both] == lv2[both]).mean())
    rho = out["retest_spearman"]
    min_rel = float(getattr(cfg, "min_reliability", 0.5))
    out["attenuation_flag"] = bool(np.isfinite(rho) and rho < min_rel)
    return out
