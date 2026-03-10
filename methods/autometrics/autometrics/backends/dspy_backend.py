"""DSPy-based scoring backend – wraps the existing DSPy/OpenAI call patterns."""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import dspy

from autometrics.metrics.Metric import MetricResult
from autometrics.util.dspy_adapters import LenientJSONAdapter
from autometrics.util.text import DEFAULT_PROMPT_MAX_TOKENS, truncate_text_to_token_limit

from .llm_backend import LLMResponse, MultiMetricResponse


# ── DSPy Signatures (mirrors GeneratedLLMJudgeMetric & runner.py) ──

class _JudgeSignatureRefFree(dspy.Signature):
    """Given the task description, and an evaluation axis, rate the output text along the axis."""

    task_description: str = dspy.InputField(desc="Brief description of the underlying task.")
    axis: str = dspy.InputField(desc="The evaluation axis / rubric.")
    input_text: str = dspy.InputField(desc="The text that was input to the model.")
    output_text: str = dspy.InputField(desc="The text produced by the model (to rate).")
    score: int = dspy.OutputField(desc="A numerical score 1-5.")


class _JudgeSignatureRefBased(dspy.Signature):
    """Given the task description, and an evaluation axis, rate the output text using the reference text as guidance."""

    task_description: str = dspy.InputField(desc="Brief description of the underlying task.")
    axis: str = dspy.InputField(desc="The evaluation axis / rubric.")
    input_text: str = dspy.InputField(desc="The text that was input to the model.")
    reference_text: str = dspy.InputField(desc="The reference text to compare against.")
    output_text: str = dspy.InputField(desc="The text produced by the model (to rate).")
    score: int = dspy.OutputField(desc="A numerical score 1-5.")


class _MultiMetricSignature(dspy.Signature):
    """Score multiple metrics at once. Return a JSON object mapping metric names to numeric scores."""

    task_description: str = dspy.InputField(desc="Brief description of the underlying task.")
    rubrics: str = dspy.InputField(desc="Metric names with rubrics. Return scores for each metric name.")
    input_text: str = dspy.InputField(desc="Input text.")
    output_text: str = dspy.InputField(desc="Output text.")
    scores_json: str = dspy.OutputField(desc="JSON object mapping metric names to numeric scores.")


# ── Helpers ──

def _parse_scores_json(raw: str) -> Dict[str, float]:
    """Reuse the runner's JSON-parsing logic."""
    import ast
    import json

    text = (raw or "").strip()
    if not text:
        return {}
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return {str(k): float(v) for k, v in obj.items()}
    except Exception:
        pass
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return {str(k): float(v) for k, v in obj.items()}
    except Exception:
        pass
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, dict):
            return {str(k): float(v) for k, v in obj.items()}
    except Exception:
        pass
    return {}


def _call_judge(
    module: dspy.Module,
    lm: dspy.LM,
    task_description: str,
    axis: str,
    input_text: str,
    output_text: str,
    reference_text: Optional[str] = None,
) -> LLMResponse:
    """Single DSPy judge call with retry logic (extracted from GeneratedLLMJudgeMetric._call_llm)."""
    model_name = getattr(lm, "model", "gpt-3.5-turbo")
    input_text = truncate_text_to_token_limit(str(input_text or ""), DEFAULT_PROMPT_MAX_TOKENS, model_name)
    output_text = truncate_text_to_token_limit(str(output_text or ""), DEFAULT_PROMPT_MAX_TOKENS, model_name)

    def _invoke(td: str):
        with dspy.settings.context(lm=lm, adapter=LenientJSONAdapter()):
            if reference_text is not None:
                return module(
                    task_description=td, axis=axis,
                    input_text=input_text, reference_text=reference_text,
                    output_text=output_text, lm=lm,
                )
            else:
                return module(
                    task_description=td, axis=axis,
                    input_text=input_text, output_text=output_text, lm=lm,
                )

    retries = 3
    while True:
        try:
            pred = _invoke(task_description)
            try:
                score_val = float(pred.score)
            except Exception:
                try:
                    score_val = float(str(pred.score).strip())
                except Exception:
                    score_val = 0.0
            reasoning = getattr(pred, "reasoning", "")
            return LLMResponse(score=score_val, reasoning=reasoning)
        except Exception as e:
            msg = str(e)
            if "Adapter JSONAdapter failed to parse" in msg and "LM Response:" in msg:
                try:
                    lm_resp = msg.split("LM Response:", 1)[1]
                    m = re.search(r"Score\s*:\s*([-+]?\d*\.?\d+)", lm_resp)
                    score_val = float(m.group(1)) if m else 0.0
                    reasoning = ""
                    if "Reasoning:" in lm_resp:
                        reasoning = lm_resp.split("Reasoning:", 1)[1].split("Score:", 1)[0].strip()
                    return LLMResponse(score=score_val, reasoning=reasoning)
                except Exception:
                    pass
            is_rate_limit = any(tok in msg for tok in (
                "RateLimitError", "Rate limit", "rate limit", "429",
                "Too Many Requests", "rate_limit_exceeded", "quota",
            ))
            if is_rate_limit and retries > 0:
                retries -= 1
                time.sleep(min(30.0, 5.0))
                continue
            raise


# ── DSPyBackend ──

class DSPyBackend:
    """Scoring backend using DSPy (existing OpenAI/LiteLLM call path)."""

    def __init__(
        self,
        judge_llm: dspy.LM,
        max_workers: int = 32,
    ):
        self.judge_llm = judge_llm
        self.max_workers = max_workers
        self._ref_free_module = dspy.ChainOfThought(_JudgeSignatureRefFree)
        self._ref_based_module = dspy.ChainOfThought(_JudgeSignatureRefBased)
        self._multi_module = dspy.ChainOfThought(_MultiMetricSignature)

    def score_single_metric_batch(
        self,
        task_description: str,
        axis: str,
        inputs: List[str],
        outputs: List[str],
        references: Optional[List[Optional[str]]] = None,
    ) -> List[LLMResponse]:
        n = len(inputs)
        refs = references or [None] * n
        is_ref_based = any(r is not None for r in refs)
        module = self._ref_based_module if is_ref_based else self._ref_free_module

        if self.max_workers <= 1 or n <= 1:
            return [
                _call_judge(module, self.judge_llm, task_description, axis, inp, out, ref)
                for inp, out, ref in zip(inputs, outputs, refs)
            ]

        results: List[Optional[LLMResponse]] = [None] * n
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(
                    _call_judge, module, self.judge_llm,
                    task_description, axis, inp, out, ref,
                ): idx
                for idx, (inp, out, ref) in enumerate(zip(inputs, outputs, refs))
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception:
                    results[idx] = LLMResponse(score=0.0)
        return results  # type: ignore[return-value]

    def score_multi_metric_batch(
        self,
        task_description: str,
        rubrics_text: str,
        metric_names: List[str],
        inputs: List[str],
        outputs: List[str],
    ) -> List[MultiMetricResponse]:
        n = len(inputs)

        def _score_one(inp: str, out: str) -> MultiMetricResponse:
            with dspy.settings.context(lm=self.judge_llm):
                pred = self._multi_module(
                    task_description=task_description,
                    rubrics=rubrics_text,
                    input_text=inp,
                    output_text=out,
                )
            scores = _parse_scores_json(getattr(pred, "scores_json", ""))
            for name in metric_names:
                if name not in scores:
                    scores[name] = 0.0
            return MultiMetricResponse(scores=scores, raw_text=getattr(pred, "scores_json", ""))

        if self.max_workers <= 1 or n <= 1:
            return [_score_one(inp, out) for inp, out in zip(inputs, outputs)]

        results: List[Optional[MultiMetricResponse]] = [None] * n
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(_score_one, inp, out): idx
                for idx, (inp, out) in enumerate(zip(inputs, outputs))
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception:
                    results[idx] = MultiMetricResponse()
        return results  # type: ignore[return-value]

    def generate_metrics(
        self,
        task_description: str,
        positive_examples: str,
        negative_examples: str,
        current_metrics: str,
        contrastive_pairs: str,
        num_metrics: int,
        num_rubrics: int,
    ) -> str:
        """Delegate to ContrastiveRubricProposer-style DSPy call."""
        from autometrics.generator.ContrastiveRubricProposer import (
            ContrastiveRubricSignature,
        )

        module = dspy.ChainOfThought(ContrastiveRubricSignature)
        with dspy.settings.context(lm=self.judge_llm):
            prediction = module(
                task_description=task_description,
                positive_examples=positive_examples or "None",
                negative_examples=negative_examples or "None",
                current_metrics=current_metrics or "None",
                contrastive_pairs=contrastive_pairs or "None",
                num_metrics=int(num_metrics),
                num_rubrics=int(num_rubrics),
            )
        return getattr(prediction, "metrics_json", "")

    def check_dedup(
        self,
        existing_metrics: str,
        candidate_metric: str,
    ) -> str:
        """Delegate to _DedupSignature-style DSPy call."""

        class _DedupSig(dspy.Signature):
            """Check if a candidate metric is distinct from existing metrics."""
            existing_metrics: str = dspy.InputField(desc="Existing metrics with rubrics.")
            candidate_metric: str = dspy.InputField(desc="Candidate metric name and rubric.")
            verdict: str = dspy.OutputField(desc="Return 'distinct' or 'duplicate: <metric name>'.")

        with dspy.settings.context(lm=self.judge_llm):
            prediction = dspy.Predict(_DedupSig)(
                existing_metrics=existing_metrics,
                candidate_metric=candidate_metric,
            )
        return str(getattr(prediction, "verdict", "distinct")).strip()

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 2048,
    ) -> str:
        """Generic text generation via the judge LLM."""
        with dspy.settings.context(lm=self.judge_llm):
            response = self.judge_llm(prompt, max_tokens=max_tokens)
        if isinstance(response, list) and response:
            return str(response[0])
        return str(response)
