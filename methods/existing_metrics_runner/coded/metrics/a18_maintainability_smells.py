"""a18: Maintainability smells — complexity and coupling beyond CCN.

This is a *complementary* metric to a0 (cyclomatic complexity). Where a0
captures branching-depth in the worst function, a18 captures three structural
"smells" that drive maintenance pain independently of CCN:

  1. Composite Maintainability Index (Python only).
     Halstead-volume-based composite from radon (MI = 171 - 5.2*ln(HV)
     - 0.23*CCN - 16.2*ln(LOC), scaled to 0..100). Covers vocabulary/operator
     richness and code volume that CCN ignores. We use the radon Python API
     directly: `mi_visit(source, multi=True)`.

  2. Worst-function parameter count (cross-language, via lizard).
     Many-parameter functions are a coupling smell (Fowler "Long Parameter
     List"). Threshold: 4 params is fine, 8+ is a strong smell.

  3. Worst-function NLOC (cross-language, via lizard).
     Long functions are a smell independent of branching — a 200-line
     straight-line function has CCN=1 but is still hard to maintain.

Combination. For Python files where MI is available we blend:
    score = 0.50 * mi_norm + 0.25 * param_norm + 0.25 * nloc_norm
For non-Python:
    score = 0.50 * param_norm + 0.50 * nloc_norm

Per-signal normalization (all to [0,1], 1 = healthy):
    mi_norm    = clip(mi/100, 0, 1)               # radon MI already 0..100
    param_norm = exp(-max(0, max_params - 4) / 4) # 4 params ~1, 8 ~0.37
    nloc_norm  = exp(-max(0, max_nloc - 30) / 50) # 30 LOC ~1, 80 ~0.37, 130 ~0.14

Weight rationale: MI is the strongest single signal (it incorporates volume,
operators, and CCN), so 0.5 in Python. Param/NLOC together get the other 0.5
because they capture distinct smells (coupling vs. length). For non-Python
we fall back to the language-agnostic pair with equal weight.

Difference vs. a0:
  - a0 = exp(-max_CCN / 10). Single signal, branching-only.
  - a18 = composite of MI + params + NLOC. CCN enters only indirectly via the
    radon MI formula in Python; non-Python uses *no* CCN at all. Correlation
    between the two should be moderate (both touch worst-function behavior)
    but they measure different smells.
"""
from __future__ import annotations

import math
import shutil
from typing import Optional

from ..sandbox import added_files_by_ext, have_tool, run, write_temp_files

ASPECT_ID = "a18"
ASPECT_NAME = "Maintainability smells: complexity and coupling"
TIER = 3
TOOLS = ["radon", "lizard"]
APPLIES_TO_LANGS = ["Python", "JavaScript", "TypeScript", "Java", "Go",
                    "C++", "C", "Ruby", "C#", "PHP"]
CLASSIFICATION = "THIN"

LIZARD_EXTS = [".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go",
               ".c", ".h", ".cpp", ".cc", ".cs", ".rb", ".php"]
PY_EXTS = [".py"]


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, LIZARD_EXTS))


def _mi_for_python(by_path) -> Optional[float]:
    """Return mean radon Maintainability Index over added Python sources, or None."""
    try:
        from radon.metrics import mi_visit
    except ImportError:
        return None
    vals = []
    for _path, src in by_path.items():
        if not src.strip():
            continue
        try:
            mi = mi_visit(src, multi=True)
        except Exception:
            # radon raises on syntax errors / partial files (common in diff hunks)
            continue
        if mi is None:
            continue
        # radon clamps to [0, 100] internally; double-check.
        vals.append(max(0.0, min(100.0, float(mi))))
    if not vals:
        return None
    return sum(vals) / len(vals)


def _lizard_worst(by_path) -> Optional[tuple]:
    """Return (max_params, max_nloc) over functions in added code, or None."""
    if not by_path:
        return None
    td = write_temp_files(by_path)
    try:
        # CSV cols: NLOC, CCN, token, param, length, location, file, func, ...
        rc, out, _ = run(["lizard", "--CCN", "1", "--csv", str(td)],
                         timeout=15.0)
        if rc < 0:
            return None
        max_params = 0
        max_nloc = 0
        n_fns = 0
        for line in out.splitlines():
            parts = line.split(",")  # REGEX_OK: tool_output (lizard CSV; we just split commas)
            if len(parts) < 5:
                continue
            try:
                nloc = int(parts[0])
                params = int(parts[3])
            except ValueError:
                continue
            n_fns += 1
            if params > max_params:
                max_params = params
            if nloc > max_nloc:
                max_nloc = nloc
        if n_fns == 0:
            return None
        return max_params, max_nloc
    finally:
        shutil.rmtree(td, ignore_errors=True)


def score(diff_text: str) -> Optional[float]:
    if not have_tool("lizard"):
        return None
    by_path_all = added_files_by_ext(diff_text, LIZARD_EXTS)
    if not by_path_all:
        return None

    # MI is Python-only.
    py_by_path = {p: t for p, t in by_path_all.items()
                  if any(p.lower().endswith(e) for e in PY_EXTS)}
    mi_mean = _mi_for_python(py_by_path) if py_by_path else None

    # params + NLOC cover everything lizard parses.
    worst = _lizard_worst(by_path_all)
    if worst is None and mi_mean is None:
        return None

    components = []
    weights = []

    if mi_mean is not None:
        mi_norm = max(0.0, min(1.0, mi_mean / 100.0))
        components.append(mi_norm)
        weights.append(0.50)

    if worst is not None:
        max_params, max_nloc = worst
        param_norm = math.exp(-max(0, max_params - 4) / 4.0)
        nloc_norm = math.exp(-max(0, max_nloc - 30) / 50.0)
        # Equal split of the structural-smell side.
        if mi_mean is not None:
            components += [param_norm, nloc_norm]
            weights += [0.25, 0.25]
        else:
            # Non-Python (or radon failed): each gets half.
            components += [param_norm, nloc_norm]
            weights += [0.50, 0.50]

    total_w = sum(weights)
    if total_w <= 0:
        return None
    return float(sum(c * w for c, w in zip(components, weights)) / total_w)
