"""Sandboxed execution of LLM-generated Python programs."""

import ast
import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

ALLOWED_MODULES = frozenset({
    "re", "string", "collections", "statistics", "math", "itertools",
    "functools", "operator", "textwrap", "unicodedata", "difflib",
    "hashlib",
})

FORBIDDEN_PATTERNS = [
    r"\bimport\s+os\b",
    r"\bimport\s+sys\b",
    r"\bimport\s+subprocess\b",
    r"\bimport\s+socket\b",
    r"\bimport\s+http\b",
    r"\bimport\s+urllib\b",
    r"\bimport\s+shutil\b",
    r"\bimport\s+pathlib\b",
    r"\bimport\s+pickle\b",
    r"\bimport\s+ctypes\b",
    r"\bimport\s+signal\b",
    r"\b__import__\s*\(",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bcompile\s*\(",
    r"\bglobals\s*\(",
    r"\bbreakpoint\s*\(",
    r"\bopen\s*\(",
]


def validate_source(source: str) -> Tuple[bool, str]:
    """Static check: reject forbidden patterns before execution."""
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, source):
            return False, f"Forbidden pattern: {pattern}"

    # Check syntax
    try:
        ast.parse(source)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    # Check imports
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name.split(".")[0]
                if mod not in ALLOWED_MODULES:
                    return False, f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mod = node.module.split(".")[0]
                if mod not in ALLOWED_MODULES and mod != "lib":
                    return False, f"Forbidden import: from {node.module}"

    return True, ""


_DRIVER_TEMPLATE = """\
import sys
import json

{library_code}

{predict_code}

texts = json.loads(sys.stdin.read())
results = []
for text in texts:
    try:
        score = predict(text)
        score = max(0.0, min(1.0, float(score)))
        results.append(score)
    except Exception:
        results.append(None)
json.dump(results, sys.stdout)
"""


def execute_programs_batch(
    source: str,
    texts: List[str],
    timeout: float = 10.0,
    library_code: Optional[str] = None,
) -> List[Tuple[Optional[float], Optional[str]]]:
    """Execute predict() on multiple texts in a single subprocess.

    Returns list of (score, error_message) tuples.
    """
    # Static validation
    ok, err = validate_source(source)
    if not ok:
        return [(None, err)] * len(texts)

    if library_code:
        ok_lib, err_lib = validate_source(library_code)
        if not ok_lib:
            return [(None, f"Library validation failed: {err_lib}")] * len(texts)

    # Build driver script
    driver = _DRIVER_TEMPLATE.format(
        library_code=library_code or "# no library",
        predict_code=source,
    )

    # Write to temp file and execute
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="verlib_"
        ) as f:
            f.write(driver)
            tmp_path = f.name

        input_json = json.dumps(texts)
        result = subprocess.run(
            ["python3", "-u", tmp_path],
            input=input_json,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            err_msg = result.stderr[:500] if result.stderr else "Non-zero exit"
            return [(None, err_msg)] * len(texts)

        try:
            scores = json.loads(result.stdout)
        except json.JSONDecodeError:
            return [(None, f"Invalid JSON output: {result.stdout[:200]}")] * len(texts)

        results = []
        for s in scores:
            if s is None:
                results.append((None, "Runtime error"))
            else:
                results.append((float(s), None))
        return results

    except subprocess.TimeoutExpired:
        return [(None, f"Timeout after {timeout}s")] * len(texts)
    except Exception as e:
        return [(None, str(e))] * len(texts)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def execute_program_single(
    source: str,
    text: str,
    timeout: float = 5.0,
    library_code: Optional[str] = None,
) -> Tuple[Optional[float], Optional[str]]:
    """Execute predict() on a single text. Convenience wrapper."""
    results = execute_programs_batch(source, [text], timeout, library_code)
    return results[0]


# =========================================================================
# V2: Extraction-aware execution
# =========================================================================

_DRIVER_TEMPLATE_V2 = """\
import sys
import json

{library_code}

{predict_code}

items = json.loads(sys.stdin.read())
results = []
for extracted in items:
    try:
        score = predict(extracted)
        score = max(0.0, min(1.0, float(score)))
        results.append(score)
    except Exception:
        results.append(None)
json.dump(results, sys.stdout)
"""


def execute_programs_batch_v2(
    source: str,
    extracted_dicts: List[dict],
    timeout: float = 10.0,
    library_code: Optional[str] = None,
) -> List[Tuple[Optional[float], Optional[str]]]:
    """Execute predict(extracted: dict) on pre-extracted inputs.

    V2: programs take dicts of extracted fields, not raw text.
    """
    ok, err = validate_source(source)
    if not ok:
        return [(None, err)] * len(extracted_dicts)

    if library_code:
        ok_lib, err_lib = validate_source(library_code)
        if not ok_lib:
            return [(None, f"Library validation failed: {err_lib}")] * len(extracted_dicts)

    driver = _DRIVER_TEMPLATE_V2.format(
        library_code=library_code or "# no library",
        predict_code=source,
    )

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="verlib_v2_"
        ) as f:
            f.write(driver)
            tmp_path = f.name

        input_json = json.dumps(extracted_dicts)
        result = subprocess.run(
            ["python3", "-u", tmp_path],
            input=input_json,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            err_msg = result.stderr[:500] if result.stderr else "Non-zero exit"
            return [(None, err_msg)] * len(extracted_dicts)

        try:
            scores = json.loads(result.stdout)
        except json.JSONDecodeError:
            return [(None, f"Invalid JSON output: {result.stdout[:200]}")] * len(extracted_dicts)

        results = []
        for s in scores:
            if s is None:
                results.append((None, "Runtime error"))
            else:
                results.append((float(s), None))
        return results

    except subprocess.TimeoutExpired:
        return [(None, f"Timeout after {timeout}s")] * len(extracted_dicts)
    except Exception as e:
        return [(None, str(e))] * len(extracted_dicts)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
