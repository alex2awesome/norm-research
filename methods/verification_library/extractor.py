"""Extraction layer: pull structured fields from raw text.

Each field in the extraction schema has a method:
- "regex": extract via pattern match
- "program": extract via stdlib function
- "llm": extract via LLM prompt (thick — requires understanding)

The schema evolves during library refactoring. LLM fields may get
thinned to regex/program as the library matures.
"""

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .sandbox import execute_program_single, validate_source

logger = logging.getLogger(__name__)


def extract_field_regex(text: str, pattern: str, field_type: str) -> Any:
    """Extract a field value using a regex pattern."""
    match = re.search(pattern, text)
    if match is None:
        return _default_for_type(field_type)
    raw = match.group(1) if match.lastindex else match.group(0)
    return _cast(raw, field_type)


def extract_field_program(text: str, code: str, field_type: str) -> Any:
    """Extract a field by running a stdlib program."""
    # Wrap the extraction code as a predict function that returns the field
    wrapper = f"""
{code}

def predict(text):
    result = extract(text)
    # Encode as float for sandbox protocol
    import json
    return float(json.dumps(result) is not None and result is not None)

# Override: we actually need the raw value, not float
import sys, json
texts = json.loads(sys.stdin.read())
results = []
for t in texts:
    try:
        results.append(extract(t))
    except Exception:
        results.append(None)
json.dump(results, sys.stdout)
"""
    # Use sandbox but with custom driver
    ok, err = validate_source(code)
    if not ok:
        logger.warning("Extraction program failed validation: %s", err)
        return _default_for_type(field_type)

    import subprocess
    import tempfile
    from pathlib import Path

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix="verlib_ext_") as f:
            f.write(wrapper)
            tmp_path = f.name

        result = subprocess.run(
            ["python3", "-u", tmp_path],
            input=json.dumps([text]),
            capture_output=True, text=True, timeout=5.0,
        )
        Path(tmp_path).unlink(missing_ok=True)

        if result.returncode != 0:
            return _default_for_type(field_type)

        values = json.loads(result.stdout)
        return values[0] if values else _default_for_type(field_type)

    except Exception:
        return _default_for_type(field_type)


def extract_field_llm(text: str, prompt: str, field_type: str, generate_fn: Callable) -> Any:
    """Extract a field by asking an LLM."""
    full_prompt = f"""Given this text, answer the following question. \
Return ONLY the answer, nothing else.

Text: {text[:3000]}

Question: {prompt}

Answer:"""
    response = generate_fn(full_prompt)
    return _cast(response.strip(), field_type)


def extract_all_fields(
    text: str,
    schema: List[dict],
    generate_fn: Optional[Callable] = None,
) -> dict:
    """Extract all fields from text according to the schema.

    Args:
        text: raw input text
        schema: list of field definitions (name, type, extraction, ...)
        generate_fn: LLM callable for "llm" extraction fields

    Returns:
        dict mapping field names to extracted values
    """
    extracted = {}
    for field in schema:
        name = field["name"]
        ftype = field.get("type", "str")
        method = field.get("extraction", "llm")

        try:
            if method == "regex":
                pattern = field.get("pattern", "")
                extracted[name] = extract_field_regex(text, pattern, ftype)
            elif method == "program":
                code = field.get("code", "")
                extracted[name] = extract_field_program(text, code, ftype)
            elif method == "llm":
                if generate_fn is None:
                    logger.warning("No generate_fn for LLM field %s, using default", name)
                    extracted[name] = _default_for_type(ftype)
                else:
                    prompt = field.get("prompt", f"What is the {name}?")
                    extracted[name] = extract_field_llm(text, prompt, ftype, generate_fn)
            else:
                extracted[name] = _default_for_type(ftype)
        except Exception as e:
            logger.warning("Failed to extract field %s: %s", name, e)
            extracted[name] = _default_for_type(ftype)

    return extracted


def compute_schema_thickness(schema: List[dict]) -> dict:
    """Measure how thick/thin the extraction schema is.

    Returns breakdown of extraction methods and thickness ratio.
    """
    counts = {"regex": 0, "program": 0, "llm": 0}
    for field in schema:
        method = field.get("extraction", "llm")
        counts[method] = counts.get(method, 0) + 1
    total = sum(counts.values())
    return {
        "counts": counts,
        "total_fields": total,
        "thin_fraction": (counts["regex"] + counts["program"]) / total if total else 0,
        "thick_fraction": counts["llm"] / total if total else 0,
    }


def merge_schemas(schema_a: List[dict], schema_b: List[dict]) -> List[dict]:
    """Naive merge: deduplicate by name, prefer thinner extraction method."""
    by_name = {}
    for field in schema_a + schema_b:
        name = field["name"]
        if name not in by_name:
            by_name[name] = field
        else:
            # Prefer thinner extraction
            existing = by_name[name]
            rank = {"regex": 0, "program": 1, "llm": 2}
            if rank.get(field["extraction"], 3) < rank.get(existing["extraction"], 3):
                by_name[name] = field
    return list(by_name.values())


def _default_for_type(field_type: str) -> Any:
    defaults = {
        "int": 0,
        "float": 0.0,
        "bool": False,
        "str": "",
        "list": [],
    }
    return defaults.get(field_type, None)


def _cast(raw: str, field_type: str) -> Any:
    """Cast a raw string value to the target type."""
    try:
        if field_type == "int":
            # Extract first number
            nums = re.findall(r"\d+", str(raw))
            return int(nums[0]) if nums else 0
        elif field_type == "float":
            nums = re.findall(r"\d+\.?\d*", str(raw))
            return float(nums[0]) if nums else 0.0
        elif field_type == "bool":
            return str(raw).lower().strip() in ("true", "yes", "1", "y")
        elif field_type == "list":
            if isinstance(raw, list):
                return raw
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return [raw] if raw else []
        else:
            return str(raw)
    except (ValueError, TypeError, IndexError):
        return _default_for_type(field_type)
