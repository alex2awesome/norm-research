"""VLLM offline batch-inference scoring backend.

Uses DSPy's adapter system for prompt rendering (so prompts are identical
to what DSPy would produce), but executes them via ``vllm.LLM.generate()``
for true offline batch throughput.

Usage::

    from autometrics.backends import create_backend

    backend = create_backend("vllm", model_name_or_path="meta-llama/Llama-3.1-8B-Instruct")
    # or pass a pre-initialised vllm.LLM:
    backend = create_backend("vllm", llm=my_vllm_llm)
"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any, Dict, List, Optional

import dspy
from dspy.adapters.chat_adapter import ChatAdapter

from .llm_backend import LLMResponse, MultiMetricResponse

logger = logging.getLogger("autometrics.backends.vllm")


# ── DSPy Signatures (reused for prompt rendering only) ──

class _JudgeSignatureRefFree(dspy.Signature):
    """Given the task description and an evaluation axis with a detailed rubric, carefully rate the output text. Read the rubric level descriptions closely and assign the score whose description best matches the text. Provide step-by-step reasoning before your score."""
    task_description: str = dspy.InputField(desc="Brief description of the underlying task.")
    axis: str = dspy.InputField(desc="The evaluation axis with a detailed rubric. Each score level (1-5) has a specific description — match the text to the most appropriate level.")
    input_text: str = dspy.InputField(desc="The text that was input to the model to produce the output text.")
    output_text: str = dspy.InputField(desc="The text that was produced by the model (this is the text that we want to rate).")
    score: int = dspy.OutputField(desc="A numerical score 1-5 based on the rubric level descriptions. Must be a single integer.")


class _JudgeSignatureRefBased(dspy.Signature):
    """Given the task description and an evaluation axis with a detailed rubric, carefully rate the output text using the reference text as guidance. Read the rubric level descriptions closely and assign the score whose description best matches the text. Provide step-by-step reasoning before your score."""
    task_description: str = dspy.InputField(desc="Brief description of the underlying task.")
    axis: str = dspy.InputField(desc="The evaluation axis with a detailed rubric. Each score level (1-5) has a specific description — match the text to the most appropriate level.")
    input_text: str = dspy.InputField(desc="The text that was input to the model to produce the output text.")
    reference_text: str = dspy.InputField(desc="The reference text to compare against.")
    output_text: str = dspy.InputField(desc="The text that was produced by the model (this is the text that we want to rate).")
    score: int = dspy.OutputField(desc="A numerical score 1-5 based on the rubric level descriptions. Must be a single integer.")


class _MultiMetricSignature(dspy.Signature):
    """Score the output text on multiple metrics at once. For each metric, read its rubric carefully and assign the score (1-5) whose level description best matches the text. Provide brief reasoning for each metric before the scores."""
    task_description: str = dspy.InputField(desc="Brief description of the underlying task.")
    rubrics: str = dspy.InputField(desc="Metric names with detailed rubrics. Each metric has score-level descriptions (1-5). Assign the score whose description best matches the text.")
    input_text: str = dspy.InputField(desc="Input text.")
    output_text: str = dspy.InputField(desc="Output text to evaluate.")
    scores_json: str = dspy.OutputField(desc="JSON object mapping metric names to numeric scores (1-5). Example: {\"Clarity\": 4, \"Relevance\": 3}")


class _SingleTextMultiMetricSignature(dspy.Signature):
    """Score the text on multiple metrics at once. For each metric, read its rubric carefully and assign the score (1-5) whose level description best matches the text. Provide brief reasoning for each metric before the scores."""
    task_description: str = dspy.InputField(desc="Brief description of the underlying task.")
    rubrics: str = dspy.InputField(desc="Metric names with detailed rubrics. Each metric has score-level descriptions (1-5). Assign the score whose description best matches the text.")
    text: str = dspy.InputField(desc="The text to evaluate.")
    scores_json: str = dspy.OutputField(desc="JSON object mapping metric names to numeric scores (1-5). Example: {\"Clarity\": 4, \"Relevance\": 3}")


class _GenerateMetricsSignature(dspy.Signature):
    """Propose metrics and rubrics that distinguish positive vs negative examples. Metrics must capture substantive, content-level distinctions (e.g. evidence quality, domain relevance, specificity of claims) rather than surface-level features (e.g. text length, formatting, readability, word choice). Every proposed metric must plausibly distinguish between items a domain expert would rate differently. Each metric must have a DETAILED rubric with specific, descriptive scoring criteria for each level (1-5). The rubric must explain exactly what distinguishes each score level so a human annotator could reliably apply it."""
    task_description: str = dspy.InputField(desc="Brief description of the task.")
    positive_examples: str = dspy.InputField(desc="Positive examples (k), formatted for readability.")
    negative_examples: str = dspy.InputField(desc="Negative examples (k), formatted for readability.")
    current_metrics: str = dspy.InputField(desc="Existing metrics with rubrics (if any).")
    contrastive_pairs: str = dspy.InputField(desc="Matched pairs or hard examples with current scores.")
    num_metrics: int = dspy.InputField(desc="Number of single-dimension metrics to propose.")
    num_rubrics: int = dspy.InputField(desc="Number of holistic rubrics to propose.")
    metrics_json: str = dspy.OutputField(
        desc=(
            "Return JSON only. Provide a JSON list of metric objects. Each object must have keys: "
            "{name, rubric, scale}. scale is 'ordinal' or 'binary'. "
            "rubric MUST be a detailed dict mapping score levels to multi-sentence descriptions. "
            "For ordinal metrics use keys '1' through '5' where each value is 2-3 sentences "
            "explaining exactly what that score means with concrete, observable criteria. "
            "Example: {'1': 'The text completely lacks X. There is no evidence of Y. "
            "The reader cannot determine Z.', '2': 'The text shows minimal X...', ...}. "
            "For binary metrics use keys 'yes'/'no' with detailed descriptions. "
            "Metric names should be specific and descriptive (not generic like 'Quality Score'). "
            "Use plain ASCII text."
        )
    )


class _DedupSignature(dspy.Signature):
    """Check if a candidate metric is distinct from existing metrics."""
    existing_metrics: str = dspy.InputField(desc="Existing metrics with rubrics.")
    candidate_metric: str = dspy.InputField(desc="Candidate metric name and rubric.")
    verdict: str = dspy.OutputField(desc="Return 'distinct' or 'duplicate: <metric name>'.")


class _SelfCritiqueSignature(dspy.Signature):
    """Critique a proposed evaluation metric. Determine if it is substantive and would meaningfully distinguish between high- and low-quality items based on genuine content differences, NOT surface-level features like text length, formatting, readability, or writing style."""
    task_description: str = dspy.InputField(desc="Brief description of the task.")
    metric_name: str = dspy.InputField(desc="Name of the proposed metric.")
    metric_rubric: str = dspy.InputField(desc="The metric's full rubric description.")
    verdict: str = dspy.OutputField(
        desc="Return 'substantive' if the metric captures a genuine content-level "
             "distinction, or 'superficial: <reason>' if it is surface-level or trivial."
    )


# ── Pre-build extended signatures (ChainOfThought adds reasoning) ──

def _cot_signature(sig_cls: type) -> type:
    """Return the extended signature that ChainOfThought would use."""
    module = dspy.ChainOfThought(sig_cls)
    return module.predict.signature


_COT_JUDGE_REF_FREE = _cot_signature(_JudgeSignatureRefFree)
_COT_JUDGE_REF_BASED = _cot_signature(_JudgeSignatureRefBased)
_COT_MULTI_METRIC = _cot_signature(_MultiMetricSignature)
_COT_SINGLE_TEXT_MULTI_METRIC = _cot_signature(_SingleTextMultiMetricSignature)
_COT_GENERATE_METRICS = _cot_signature(_GenerateMetricsSignature)
# Dedup uses Predict (no CoT), so no extension needed


# ── Output parsers ──

def _parse_score_from_text(text: str) -> tuple[float, str]:
    """Extract a numeric score and reasoning from raw LLM text."""
    reasoning = ""
    score = 0.0

    m = re.search(r"Score\s*:\s*([-+]?\d+(?:\.\d+)?)", text)
    if m:
        score = float(m.group(1))
        idx = text.find(m.group(0))
        if idx > 0:
            reasoning = text[:idx].strip()
    else:
        nums = re.findall(r"([-+]?\d+(?:\.\d+)?)", text)
        if nums:
            score = float(nums[-1])
        reasoning = text.strip()

    return score, reasoning


def _parse_multi_metric_from_text(text: str) -> Dict[str, float]:
    """Extract a JSON dict of metric scores from raw LLM text."""
    m = re.search(r"Scores?\s*Json?\s*:\s*(\{[^}]+\})", text, re.IGNORECASE | re.DOTALL)
    blob = m.group(1) if m else text

    for candidate in [blob, text]:
        try:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start >= 0 and end > start:
                obj = json.loads(candidate[start : end + 1])
                if isinstance(obj, dict):
                    return {str(k): float(v) for k, v in obj.items()}
        except Exception:
            pass

    try:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            obj = ast.literal_eval(text[start : end + 1])
            if isinstance(obj, dict):
                return {str(k): float(v) for k, v in obj.items()}
    except Exception:
        pass

    return {}


def _truncate(text: str, max_chars: int = 12000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + "..."


# ── VLLM Backend ──

class VLLMOfflineBackend:
    """Offline (local) VLLM scoring backend.

    Uses DSPy's ``ChatAdapter.format()`` to render prompts identically to
    what DSPy would produce, then executes them via ``vllm.LLM.generate()``
    in a single batch for maximum throughput.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model id or local path.  Ignored if *llm* is provided.
    llm : optional
        Pre-initialised ``vllm.LLM`` instance.
    sampling_params : optional
        Pre-built ``vllm.SamplingParams``.  If *None* a sensible default
        (temperature=0, max_tokens=1024) is used.
    max_prompt_chars : int
        Character limit applied to each input/output field before prompt
        assembly (default 12 000).
    **vllm_kwargs
        Extra keyword arguments forwarded to ``vllm.LLM()``.
    """

    def __init__(
        self,
        model_name_or_path: str = "",
        *,
        llm: Any = None,
        sampling_params: Any = None,
        max_prompt_chars: int = 12_000,
        **vllm_kwargs: Any,
    ):
        from vllm import LLM, SamplingParams  # type: ignore[import-untyped]

        if llm is not None:
            self.llm = llm
        else:
            if not model_name_or_path:
                raise ValueError("Provide model_name_or_path or a pre-built vllm.LLM via llm=")
            self.llm = LLM(model=model_name_or_path, **vllm_kwargs)

        self.sampling_params = sampling_params or SamplingParams(
            temperature=0, max_tokens=1024,
        )
        self.max_prompt_chars = max_prompt_chars
        self._tokenizer = self.llm.get_tokenizer()
        self._adapter = ChatAdapter()

    # ── Prompt rendering ──

    def _render_prompt(self, signature: type, inputs: dict) -> str:
        """Use DSPy's adapter to format, then apply the model's chat template."""
        messages = self._adapter.format(signature, demos=[], inputs=inputs)
        # Use the tokenizer's chat template to convert messages -> string
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                pass
        # Fallback: concatenate
        return "\n\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        )

    # ── Public API ──

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
        sig = _COT_JUDGE_REF_BASED if is_ref_based else _COT_JUDGE_REF_FREE

        prompts: List[str] = []
        for inp, out, ref in zip(inputs, outputs, refs):
            input_dict: Dict[str, Any] = {
                "task_description": _truncate(str(task_description or ""), self.max_prompt_chars),
                "axis": str(axis or ""),
                "input_text": _truncate(str(inp or ""), self.max_prompt_chars),
                "output_text": _truncate(str(out or ""), self.max_prompt_chars),
            }
            if is_ref_based:
                input_dict["reference_text"] = _truncate(str(ref or ""), self.max_prompt_chars)
            prompts.append(self._render_prompt(sig, input_dict))

        vllm_outputs = self.llm.generate(prompts, self.sampling_params)

        results: List[LLMResponse] = []
        for vout in vllm_outputs:
            text = vout.outputs[0].text if vout.outputs else ""
            score, reasoning = _parse_score_from_text(text)
            results.append(LLMResponse(score=score, reasoning=reasoning, raw_text=text))
        return results

    def score_multi_metric_batch(
        self,
        task_description: str,
        rubrics_text: str,
        metric_names: List[str],
        inputs: List[str],
        outputs: List[str],
    ) -> List[MultiMetricResponse]:
        prompts: List[str] = []
        for inp, out in zip(inputs, outputs):
            input_dict = {
                "task_description": _truncate(str(task_description or ""), self.max_prompt_chars),
                "rubrics": str(rubrics_text or ""),
                "input_text": _truncate(str(inp or ""), self.max_prompt_chars),
                "output_text": _truncate(str(out or ""), self.max_prompt_chars),
            }
            prompts.append(self._render_prompt(_COT_MULTI_METRIC, input_dict))

        vllm_outputs = self.llm.generate(prompts, self.sampling_params)

        results: List[MultiMetricResponse] = []
        for vout in vllm_outputs:
            text = vout.outputs[0].text if vout.outputs else ""
            scores = _parse_multi_metric_from_text(text)
            for name in metric_names:
                if name not in scores:
                    scores[name] = 0.0
            results.append(MultiMetricResponse(scores=scores, raw_text=text))
        return results

    def score_single_text_batch(
        self,
        task_description: str,
        rubrics_text: str,
        metric_names: List[str],
        texts: List[str],
    ) -> List[MultiMetricResponse]:
        """Score texts on multiple metrics — single-text variant.

        Unlike ``score_multi_metric_batch`` which takes separate inputs/outputs
        (duplicating the text when they are identical), this method includes the
        text only once in the prompt, nearly halving prompt size.
        """
        prompts: List[str] = []
        for txt in texts:
            input_dict = {
                "task_description": _truncate(str(task_description or ""), self.max_prompt_chars),
                "rubrics": str(rubrics_text or ""),
                "text": _truncate(str(txt or ""), self.max_prompt_chars),
            }
            prompts.append(self._render_prompt(_COT_SINGLE_TEXT_MULTI_METRIC, input_dict))

        vllm_outputs = self.llm.generate(prompts, self.sampling_params)

        results: List[MultiMetricResponse] = []
        for vout in vllm_outputs:
            text = vout.outputs[0].text if vout.outputs else ""
            scores = _parse_multi_metric_from_text(text)
            for name in metric_names:
                if name not in scores:
                    scores[name] = 0.0
            results.append(MultiMetricResponse(scores=scores, raw_text=text))
        return results

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
        from vllm import SamplingParams  # type: ignore[import-untyped]

        input_dict = {
            "task_description": _truncate(str(task_description or ""), self.max_prompt_chars),
            "positive_examples": _truncate(str(positive_examples or "None"), self.max_prompt_chars),
            "negative_examples": _truncate(str(negative_examples or "None"), self.max_prompt_chars),
            "current_metrics": _truncate(str(current_metrics or "None"), self.max_prompt_chars),
            "contrastive_pairs": _truncate(str(contrastive_pairs or "None"), self.max_prompt_chars),
            "num_metrics": int(num_metrics),
            "num_rubrics": int(num_rubrics),
        }
        prompt = self._render_prompt(_COT_GENERATE_METRICS, input_dict)
        gen_params = SamplingParams(
            temperature=self.sampling_params.temperature,
            max_tokens=max(8192, self.sampling_params.max_tokens),
        )
        vllm_outputs = self.llm.generate([prompt], gen_params)
        raw = vllm_outputs[0].outputs[0].text if vllm_outputs and vllm_outputs[0].outputs else ""
        logger.info("generate_metrics raw output (%d chars):\n%s", len(raw), raw)
        return raw

    def check_dedup(
        self,
        existing_metrics: str,
        candidate_metric: str,
    ) -> str:
        from vllm import SamplingParams  # type: ignore[import-untyped]

        input_dict = {
            "existing_metrics": str(existing_metrics or ""),
            "candidate_metric": str(candidate_metric or ""),
        }
        # Dedup uses Predict (no CoT), so use the raw signature
        prompt = self._render_prompt(_DedupSignature, input_dict)
        dedup_params = SamplingParams(temperature=0, max_tokens=64)
        vllm_outputs = self.llm.generate([prompt], dedup_params)
        return vllm_outputs[0].outputs[0].text.strip() if vllm_outputs and vllm_outputs[0].outputs else "distinct"

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 2048,
    ) -> str:
        """Generic text generation from a plain-text prompt."""
        from vllm import SamplingParams  # type: ignore[import-untyped]

        messages = [{"role": "user", "content": prompt}]
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                formatted = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                formatted = f"USER: {prompt}\n\nASSISTANT:"
        else:
            formatted = f"USER: {prompt}\n\nASSISTANT:"

        params = SamplingParams(temperature=0, max_tokens=max_tokens)
        vllm_outputs = self.llm.generate([formatted], params)
        return vllm_outputs[0].outputs[0].text.strip() if vllm_outputs and vllm_outputs[0].outputs else ""

    def check_dedup_batch(
        self,
        existing_metrics: str,
        candidate_metrics: List[str],
    ) -> List[str]:
        """Batch dedup: check multiple candidates at once."""
        from vllm import SamplingParams  # type: ignore[import-untyped]

        prompts = []
        for cand in candidate_metrics:
            input_dict = {
                "existing_metrics": str(existing_metrics or ""),
                "candidate_metric": str(cand or ""),
            }
            prompts.append(self._render_prompt(_DedupSignature, input_dict))
        dedup_params = SamplingParams(temperature=0, max_tokens=64)
        vllm_outputs = self.llm.generate(prompts, dedup_params)
        return [
            vout.outputs[0].text.strip() if vout.outputs else "distinct"
            for vout in vllm_outputs
        ]

    def critique_metrics_batch(
        self,
        task_description: str,
        candidates: List[Dict[str, str]],
    ) -> List[str]:
        """Batch self-critique: check if each candidate metric is substantive."""
        from vllm import SamplingParams  # type: ignore[import-untyped]

        _COT_CRITIQUE = _cot_signature(_SelfCritiqueSignature)
        prompts = []
        for cand in candidates:
            input_dict = {
                "task_description": _truncate(str(task_description or ""), self.max_prompt_chars),
                "metric_name": str(cand.get("name", "")),
                "metric_rubric": str(cand.get("rubric", "")),
            }
            prompts.append(self._render_prompt(_COT_CRITIQUE, input_dict))
        params = SamplingParams(temperature=0, max_tokens=256)
        vllm_outputs = self.llm.generate(prompts, params)
        return [
            vout.outputs[0].text.strip() if vout.outputs else "substantive"
            for vout in vllm_outputs
        ]
