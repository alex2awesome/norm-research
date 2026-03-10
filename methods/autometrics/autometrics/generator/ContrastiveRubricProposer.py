from __future__ import annotations

import json
import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional

import dspy

logger = logging.getLogger("autometrics.generator.ContrastiveRubricProposer")


class ContrastiveRubricSignature(dspy.Signature):
    """Propose metrics and rubrics that distinguish positive vs negative examples. Each metric must have a DETAILED rubric with specific, descriptive scoring criteria for each level (1-5). The rubric must explain exactly what distinguishes each score level so a human annotator could reliably apply it."""

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


def _sanitize_metric_name(text: str, max_words: int = 8, max_len: int = 80) -> str:
    raw = str(text or "")
    first_line = raw.splitlines()[0].strip() if raw else ""
    normalized = unicodedata.normalize("NFKD", first_line).encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if normalized:
        normalized = " ".join(normalized.split(" ")[:max_words])
    token = re.sub(r"[^A-Za-z0-9_]+", "_", normalized)
    token = re.sub(r"_+", "_", token).strip("_")
    if not token:
        token = "Metric"
    if token[0].isdigit():
        token = f"M_{token}"
    return token[:max_len]


def _extract_json_blob(text: str) -> str | None:
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        return stripped
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return None


def _parse_metrics_json(text: str) -> List[Dict[str, Any]]:
    # Strip DSPy field markers (e.g. [[ ## reasoning ## ]], [[ ## metrics_json ## ]])
    # and extract only the metrics_json field content if present
    if "[[ ## metrics_json ## ]]" in text:
        text = text.split("[[ ## metrics_json ## ]]", 1)[1]
        # Strip trailing markers like [[ ## completed ## ]]
        if "[[ ##" in text:
            text = text[:text.index("[[ ##")]
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    blob = _extract_json_blob(text)
    if not blob:
        return []
    try:
        data = json.loads(blob)
    except Exception:
        return []
    if isinstance(data, dict):
        data = data.get("metrics") or data.get("candidates") or []
    if not isinstance(data, list):
        return []
    cleaned: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("title") or "Metric"
        rubric = item.get("rubric")
        scale = item.get("scale") or item.get("type") or "ordinal"
        cleaned.append(
            {
                "name": _sanitize_metric_name(str(name)),
                "rubric": rubric,
                "scale": str(scale).strip().lower(),
            }
        )
    return cleaned


class ContrastiveRubricProposer:
    """LLM proposer for contrastive metrics and rubrics (JSON output)."""

    def __init__(self, generator_llm: dspy.LM, seed: Optional[int] = None, scoring_backend: Optional[Any] = None):
        self.generator_llm = generator_llm
        self.seed = seed
        self.scoring_backend = scoring_backend
        self._module = dspy.ChainOfThought(ContrastiveRubricSignature)

    def propose(
        self,
        task_description: str,
        positive_examples: str,
        negative_examples: str,
        current_metrics: str = "",
        contrastive_pairs: str = "",
        num_metrics: int = 5,
        num_rubrics: int = 5,
    ) -> List[Dict[str, Any]]:
        if self.scoring_backend is not None:
            raw_json = self.scoring_backend.generate_metrics(
                task_description=task_description,
                positive_examples=positive_examples or "None",
                negative_examples=negative_examples or "None",
                current_metrics=current_metrics or "None",
                contrastive_pairs=contrastive_pairs or "None",
                num_metrics=int(num_metrics),
                num_rubrics=int(num_rubrics),
            )
            parsed = _parse_metrics_json(raw_json)
            logger.info("Backend generate_metrics returned %d chars, parsed %d metrics", len(raw_json), len(parsed))
            if not parsed:
                logger.warning("Failed to parse metrics from backend output: %s", raw_json[:1000])
            return parsed
        with dspy.settings.context(lm=self.generator_llm):
            prediction = self._module(
                task_description=task_description,
                positive_examples=positive_examples,
                negative_examples=negative_examples,
                current_metrics=current_metrics or "None",
                contrastive_pairs=contrastive_pairs or "None",
                num_metrics=int(num_metrics),
                num_rubrics=int(num_rubrics),
            )
        return _parse_metrics_json(getattr(prediction, "metrics_json", ""))
