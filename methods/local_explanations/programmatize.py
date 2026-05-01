"""Attempt to convert STaR rationales (z₊/z₋) into verifiable programs.

For each extracted rationale, asks an LLM whether it can be checked
programmatically. If yes, generates the program + input schema.
If no, the rationale stays as a thick rubric criterion.

This implements the fluid boundary between verification and articulation:
each z gets classified as thin (programmable) or thick (needs LLM judgment)
based on whether program synthesis succeeds empirically.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ProgrammatizationResult:
    """Result of attempting to convert a rationale into a program."""

    rationale: str  # original z₊ or z₋ text
    classification: str  # "thin", "thick", or "failed"
    program_code: Optional[str] = None  # if thin: the program
    input_schema: Optional[List[dict]] = None  # if thin: what inputs it needs
    rubric: Optional[Dict[str, str]] = None  # if thick: YES/NO rubric
    confidence: float = 0.0  # LLM's confidence it can be programmatized
    reasoning: str = ""  # why it classified this way


ATTEMPT_PROMPT = """\
An expert evaluating a {text_type} identified this criterion:

"{rationale}"

This criterion was used to argue {polarity} ({label_desc}).

Here is the actual text where this criterion was identified (use it to \
understand what concrete textual patterns correspond to this criterion):

--- TEXT ---
{example_text}
--- END TEXT ---
{schema_section}{library_section}
**Question**: Can this criterion be checked programmatically — using only \
Python standard library (re, string, collections, statistics, math) — \
without needing to "understand" the text?

Think about this carefully. Some criteria SEEM like they need understanding \
but can actually be approximated by structural checks (counting patterns, \
measuring lengths, checking formats). Others genuinely require reading \
comprehension.

Respond with JSON:

```json
{{
  "can_programmatize": true or false,
  "confidence": 0.0 to 1.0,
  "reasoning": "Why this can/cannot be checked programmatically",
  "program": {{
    "inputs": [
      {{
        "name": "field_name",
        "type": "int|float|bool|str",
        "extraction": "regex|program|llm",
        "description": "what this field is",
        "pattern_or_code_or_prompt": "extraction detail"
      }}
    ],
    "code": "def predict(extracted: dict) -> float:\\n    ..."
  }},
  "rubric": {{
    "yes": "What constitutes YES (if this needs LLM judgment instead)",
    "no": "What constitutes NO"
  }}
}}
```

If can_programmatize is true, fill in "program" (with inputs + code). \
If false, fill in "rubric" (with YES/NO descriptions). \
Include both fields either way — the unused one can have null values.

Return ONLY the JSON wrapped in ```json ... ``` markers."""


BATCH_CLASSIFY_PROMPT = """\
You are classifying evaluation criteria for a {text_type} into two categories:

**THIN (programmable)**: Can be checked with Python standard library. \
Structural, countable, pattern-matchable. Examples: word count thresholds, \
presence of citations, formatting compliance, numeric patterns.

**THICK (needs judgment)**: Requires reading comprehension, domain expertise, \
or subjective assessment. Examples: argument quality, novelty, significance, \
writing clarity.

For each criterion below, classify it and briefly explain why.

Criteria:
{criteria_list}

Respond with a JSON list:
```json
[
  {{
    "criterion": "the original text",
    "classification": "thin" or "thick",
    "reasoning": "brief explanation",
    "confidence": 0.0 to 1.0
  }}
]
```

Return ONLY the JSON list wrapped in ```json ... ``` markers."""


def _extract_json(response: str):
    pattern = r"```json\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    text = match.group(1).strip() if match else response.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


async def attempt_programmatization(
    rationale: str,
    polarity: str,
    text_type: str,
    positive_label: str,
    negative_label: str,
    generate_fn: Callable,
    example_text: str = "",
    existing_schema: Optional[List[dict]] = None,
    library_code: Optional[str] = None,
) -> ProgrammatizationResult:
    """Try to convert a single rationale into a program.

    Args:
        rationale: the z₊ or z₋ text
        polarity: "for" or "against"
        text_type: e.g. "scientific paper"
        positive_label: e.g. "accepted"
        negative_label: e.g. "rejected"
        generate_fn: async LLM callable (prompt -> response)
        example_text: the actual text this rationale was extracted from
        existing_schema: input fields already available for reuse
        library_code: existing utility library code

    Returns:
        ProgrammatizationResult with classification and either program or rubric
    """
    label_desc = positive_label if polarity == "for" else negative_label

    schema_section = ""
    if existing_schema:
        schema_section = (
            "\nThese input fields are already available (you can reuse them):\n"
            f"```json\n{json.dumps(existing_schema, indent=2)}\n```\n\n"
        )

    library_section = ""
    if library_code:
        library_section = (
            "\nThese utility functions are available:\n"
            f"```python\n{library_code}\n```\n\n"
        )

    prompt = ATTEMPT_PROMPT.format(
        text_type=text_type,
        rationale=rationale,
        polarity=polarity,
        label_desc=label_desc,
        example_text=example_text[:2000] if example_text else "(no example provided)",
        schema_section=schema_section,
        library_section=library_section,
    )

    response = await generate_fn(prompt)
    parsed = _extract_json(response)

    if parsed is None:
        return ProgrammatizationResult(
            rationale=rationale,
            classification="failed",
            reasoning="Could not parse response",
        )

    can_prog = parsed.get("can_programmatize", False)
    confidence = parsed.get("confidence", 0.0)
    reasoning = parsed.get("reasoning", "")

    if can_prog and parsed.get("program"):
        prog = parsed["program"]
        return ProgrammatizationResult(
            rationale=rationale,
            classification="thin",
            program_code=prog.get("code", ""),
            input_schema=prog.get("inputs", []),
            confidence=confidence,
            reasoning=reasoning,
        )
    else:
        rubric = parsed.get("rubric", {})
        if not rubric or not rubric.get("yes"):
            rubric = {"yes": rationale, "no": f"Does not exhibit: {rationale}"}
        return ProgrammatizationResult(
            rationale=rationale,
            classification="thick",
            rubric=rubric,
            confidence=confidence,
            reasoning=reasoning,
        )


async def batch_classify_thin_thick(
    rationales: List[str],
    text_type: str,
    generate_fn: Callable,
    batch_size: int = 50,
) -> List[Tuple[str, str, float]]:
    """Quick classification of rationales as thin/thick without generating programs.

    Faster than full programmatization — good for initial triage.

    Returns: list of (rationale, classification, confidence)
    """
    results = []
    for i in range(0, len(rationales), batch_size):
        batch = rationales[i:i + batch_size]
        criteria_list = "\n".join(f"{j+1}. {r}" for j, r in enumerate(batch))
        prompt = BATCH_CLASSIFY_PROMPT.format(
            text_type=text_type,
            criteria_list=criteria_list,
        )
        response = await generate_fn(prompt)
        parsed = _extract_json(response)
        if parsed and isinstance(parsed, list):
            for item in parsed:
                results.append((
                    item.get("criterion", ""),
                    item.get("classification", "thick"),
                    item.get("confidence", 0.5),
                ))
        else:
            # Fallback: classify everything as thick
            for r in batch:
                results.append((r, "thick", 0.0))
    return results


async def programmatize_weighted_features(
    weighted_features: List[Tuple[str, float]],
    text_type: str,
    positive_label: str,
    negative_label: str,
    generate_fn: Callable,
    feature_polarity: Optional[Dict[str, str]] = None,
    output_dir: Optional[str] = None,
    max_programs: int = 50,
) -> Tuple[List[ProgrammatizationResult], Dict[str, any]]:
    """Full pipeline: classify weighted features, programmatize thin ones.

    Two-phase:
    1. Quick batch classification (cheap, all features)
    2. Full programmatization attempt (expensive, thin candidates only)

    Args:
        weighted_features: (feature_text, weight) from STaR
        text_type, positive_label, negative_label: task context
        generate_fn: async LLM callable
        feature_polarity: optional map of feature → "for"/"against"
        output_dir: save results for caching
        max_programs: limit full programmatization to top-N thin candidates

    Returns:
        (results, summary_stats)
    """
    out_path = Path(output_dir) if output_dir else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)

    # Check cache
    cache_path = out_path / "programmatization_results.json" if out_path else None
    if cache_path and cache_path.exists():
        logger.info("Loading cached programmatization results from %s", cache_path)
        with open(cache_path) as f:
            cached = json.load(f)
        results = [ProgrammatizationResult(**r) for r in cached["results"]]
        return results, cached["summary"]

    # Deduplicate
    unique_features = list(set(f for f, _ in weighted_features))
    logger.info("Programmatizing %d unique features", len(unique_features))

    # Phase 1: Quick batch classification
    logger.info("Phase 1: batch thin/thick classification")
    classifications = await batch_classify_thin_thick(
        unique_features, text_type, generate_fn,
    )
    class_map = {c[0]: (c[1], c[2]) for c in classifications}

    thin_candidates = [
        (feat, class_map.get(feat, ("thick", 0.0))[1])
        for feat in unique_features
        if class_map.get(feat, ("thick", 0.0))[0] == "thin"
    ]
    thin_candidates.sort(key=lambda x: x[1], reverse=True)
    thin_candidates = thin_candidates[:max_programs]

    logger.info(
        "Phase 1 result: %d thin candidates, %d thick (of %d total)",
        len(thin_candidates),
        len(unique_features) - len(thin_candidates),
        len(unique_features),
    )

    # Phase 2: Full programmatization of thin candidates
    logger.info("Phase 2: programmatizing %d thin candidates", len(thin_candidates))
    results = []

    for feat, conf in thin_candidates:
        polarity = (feature_polarity or {}).get(feat, "for")
        result = await attempt_programmatization(
            rationale=feat,
            polarity=polarity,
            text_type=text_type,
            positive_label=positive_label,
            negative_label=negative_label,
            generate_fn=generate_fn,
        )
        results.append(result)

    # Add thick features (no program attempt needed)
    for feat in unique_features:
        if class_map.get(feat, ("thick", 0.0))[0] != "thin":
            results.append(ProgrammatizationResult(
                rationale=feat,
                classification="thick",
                rubric={"yes": feat, "no": f"Does not exhibit: {feat}"},
                confidence=class_map.get(feat, ("thick", 0.0))[1],
                reasoning="Classified as thick in batch classification",
            ))

    # Summary
    n_thin = sum(1 for r in results if r.classification == "thin")
    n_thick = sum(1 for r in results if r.classification == "thick")
    n_failed = sum(1 for r in results if r.classification == "failed")

    # Count extraction methods across thin programs
    extraction_counts = {"regex": 0, "program": 0, "llm": 0}
    for r in results:
        if r.input_schema:
            for field in r.input_schema:
                method = field.get("extraction", "llm")
                extraction_counts[method] = extraction_counts.get(method, 0) + 1

    summary = {
        "total_features": len(unique_features),
        "thin": n_thin,
        "thick": n_thick,
        "failed": n_failed,
        "thin_fraction": n_thin / len(results) if results else 0,
        "extraction_methods": extraction_counts,
    }

    logger.info(
        "Programmatization: %d thin (%.0f%%), %d thick, %d failed",
        n_thin, summary["thin_fraction"] * 100, n_thick, n_failed,
    )

    # Cache
    if cache_path:
        cache_data = {
            "results": [
                {
                    "rationale": r.rationale,
                    "classification": r.classification,
                    "program_code": r.program_code,
                    "input_schema": r.input_schema,
                    "rubric": r.rubric,
                    "confidence": r.confidence,
                    "reasoning": r.reasoning,
                }
                for r in results
            ],
            "summary": summary,
        }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f, indent=2)
        logger.info("Saved programmatization results to %s", cache_path)

    return results, summary
