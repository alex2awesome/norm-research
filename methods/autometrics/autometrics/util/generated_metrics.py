from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional


_STRING_LITERAL_RE = r"(?:r?'''[\s\S]*?'''|r?\"\"\"[\s\S]*?\"\"\"|r?'(?:\\.|[^'])*'|r?\"(?:\\.|[^\"])*\")"


def _safe_token(text: str) -> str:
    return str(text or "").replace(" ", "_").replace("/", "_")


def _extract_literal(text: str, key: str) -> Optional[str]:
    pattern = rf"{key}\s*=\s*({_STRING_LITERAL_RE})"
    match = re.search(pattern, text, re.S)
    if not match:
        return None
    raw = match.group(1)
    try:
        return ast.literal_eval(raw)
    except Exception:
        return None


def _extract_docstring(text: str) -> Optional[str]:
    match = re.search(r"class\s+\w+\s*\([^)]*\):\s*\n\s*(r?\"\"\"[\s\S]*?\"\"\")", text, re.S)
    if not match:
        match = re.search(r"class\s+\w+\s*\([^)]*\):\s*\n\s*(r?'''[\s\S]*?''')", text, re.S)
    if not match:
        return None
    try:
        return ast.literal_eval(match.group(1))
    except Exception:
        return None


def _detect_reference_based(text: str) -> bool:
    return "GeneratedRefBased" in text or "reference-based" in text.lower()


def list_generated_metrics_for_task(
    dataset_name: str,
    target_measure: str,
    seed: int = 42,
    generated_metrics_dir: str = "generated_metrics",
) -> List[Dict[str, str]]:
    """List generated metric files for a dataset/target/seed and parse key fields."""
    safe_dataset = _safe_token(dataset_name)
    safe_measure = _safe_token(target_measure)
    base = Path(generated_metrics_dir)
    pattern = f"**/generated_metrics/{safe_dataset}/{safe_measure}/seed_{seed}/*/*.py"
    files = sorted(base.glob(pattern))

    entries: List[Dict[str, str]] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        generator_type = path.parent.name
        name = _extract_literal(text, "name") or path.stem
        description = _extract_literal(text, "description") or ""
        axis = _extract_literal(text, "axis") or ""
        task_description = _extract_literal(text, "task_description") or ""
        docstring = _extract_docstring(text) or ""
        is_reference_based = _detect_reference_based(text)

        prompt_parts = []
        if task_description:
            prompt_parts.append(f"Task: {task_description}")
        if axis:
            prompt_parts.append(f"Axis/Rubric: {axis}")
        else:
            prompt_parts.append(f"Description: {description}")
        prompt_parts.append("Input text: <input_text>")
        if is_reference_based:
            prompt_parts.append("Reference text: <reference_text>")
        prompt_parts.append("Output text: <output_text>")

        entries.append(
            {
                "name": str(name),
                "generator": generator_type,
                "path": str(path),
                "description": str(description),
                "axis": str(axis),
                "task_description": str(task_description),
                "prompt_template": "\n".join(prompt_parts),
                "metric_card": str(docstring),
            }
        )

    return entries


def render_generated_metrics_report(entries: Iterable[Dict[str, str]], max_docstring_chars: int = 800) -> str:
    """Return a markdown report for generated metrics."""
    blocks: List[str] = []
    for entry in entries:
        name = entry.get("name", "Metric")
        generator = entry.get("generator", "")
        path = entry.get("path", "")
        description = entry.get("description", "")
        axis = entry.get("axis", "")
        task_description = entry.get("task_description", "")
        prompt_template = entry.get("prompt_template", "")
        metric_card = entry.get("metric_card", "")

        metric_card_excerpt = metric_card.strip()
        if max_docstring_chars and len(metric_card_excerpt) > max_docstring_chars:
            metric_card_excerpt = metric_card_excerpt[: max_docstring_chars - 3] + "..."

        blocks.append(
            "\n".join(
                [
                    f"## {name} ({generator})",
                    f"**Path:** `{path}`",
                    "",
                    f"**Description:** {description}" if description else "**Description:** (none)",
                    f"**Axis/Rubric:** {axis}" if axis else "**Axis/Rubric:** (none)",
                    f"**Task Description:** {task_description}" if task_description else "**Task Description:** (none)",
                    "",
                    "**Prompt Template:**",
                    "```text",
                    prompt_template.strip(),
                    "```",
                    "",
                    "**Metric Card (excerpt):**",
                    "```text",
                    metric_card_excerpt.strip() if metric_card_excerpt else "(none)",
                    "```",
                ]
            )
        )

    return "\n\n".join(blocks)
