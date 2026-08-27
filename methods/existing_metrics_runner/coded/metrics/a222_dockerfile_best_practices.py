"""a222: Dockerfile best practices and hygiene.

The norm: "Write efficient, predictable Dockerfiles: minimize layers by
consolidating RUNs, use WORKDIR, add HEALTHCHECK, avoid cache bloat
(apk/apt --no-cache and cleanup), and remove transient package lists and
superfluous metadata."

Measurement strategy: `hadolint` — the standard Dockerfile linter (a Haskell
binary, `brew install hadolint`). Hadolint has direct rules for nearly every
sub-clause of this norm; we restrict to a curated subset rather than counting
every DL/SC code so the score reflects the norm itself (efficiency / hygiene /
predictability) and is not polluted by orthogonal style choices (shell quoting
SC2086, label-schema policy, etc.).

Curated rule set ("norm-relevant", mapping to clauses of the norm):

  Minimize layers / consolidate RUNs:
    DL3059  Multiple consecutive `RUN` instructions. Consider consolidation.

  Cache / transient package-list bloat (apt/apk hygiene):
    DL3009  Delete the apt lists (/var/lib/apt/lists) after installing
    DL3019  Use the `--no-cache` switch in apk add
    DL3015  Avoid additional packages: --no-install-recommends
    DL3027  Do not use apt; use apt-get/apt-cache (apt is unstable)

  Predictability / pinning (no surprise updates):
    DL3007  Using `latest` is prone to errors. Pin tag.
    DL3008  Pin versions in apt-get install
    DL3018  Pin versions in apk add
    DL3013  Pin versions in pip install
    DL3016  Pin versions in npm install
    DL3022  COPY --from references should be pinned with digest

  Use WORKDIR / structural hygiene:
    DL3000  Use absolute WORKDIR
    DL3003  Use WORKDIR to switch dirs (don't `RUN cd ...`)

  Shell predictability (so multi-line RUNs behave):
    DL4006  Set the SHELL option -o pipefail before RUN with a pipe
    DL3001  For inter-process signals; for some commands, e.g. SSH

We deliberately IGNORE: DL3025 (JSON-form CMD/ENTRYPOINT — style), DL3020
(use COPY not ADD — style), DL3006 (use COPY instead of MAINTAINER —
deprecation), label-policy SC* codes, and most SC shellcheck codes (which
catch generic shell bugs, not Docker hygiene).

Score = exp(-relevant_violations_per_added_dockerfile_line * 4). Tuned so
0 viol/line = 1.0, 0.05 = 0.82, 0.25 = 0.37, 1.0 = 0.018. Dockerfiles are
short (median ~30 lines) so we use a gentler decay than a181/a226.

Applicability: ONLY Dockerfile-touching diffs. Detection by filename:
basename == "Dockerfile", starts with "Dockerfile." (Dockerfile.dev,
Dockerfile.prod), or ends with ".dockerfile" (rarer convention). The norm
explicitly addresses Dockerfile authoring — we do not try to lint
docker-compose YAMLs or shell scripts that happen to call docker.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, Optional

from ..sandbox import have_tool, parse_diff_added_by_file, run

ASPECT_ID = "a222"
ASPECT_NAME = "Dockerfile best practices and hygiene"
TIER = 3
TOOLS = ["hadolint"]
APPLIES_TO_LANGS = ["Dockerfile"]
CLASSIFICATION = "THIN"

# Rule codes that map to clauses of the norm. See module docstring for the
# clause-by-clause mapping. Anything not in this set is treated as orthogonal
# style and does NOT count toward the violation density.
RELEVANT_RULES = {
    # Minimize layers
    "DL3059",
    # Cache / package-list hygiene
    "DL3009", "DL3019", "DL3015", "DL3027",
    # Predictability / pinning
    "DL3007", "DL3008", "DL3018", "DL3013", "DL3016", "DL3022",
    # WORKDIR / structural
    "DL3000", "DL3003",
    # Shell predictability for multi-line RUNs
    "DL4006", "DL3001",
}


def _is_dockerfile_path(path: str) -> bool:
    """Return True iff `path` is a Dockerfile by conventional naming.

    Recognized forms:
        Dockerfile
        Dockerfile.<variant>     (Dockerfile.dev, Dockerfile.prod, ...)
        <anything>.dockerfile    (rarer; some repos use this)
        <anything>.Dockerfile    (rarer; case variant of the above)

    Path is matched on basename only so nested paths like
    `services/api/Dockerfile` are picked up.
    """
    base = os.path.basename(path)
    if not base:
        return False
    if base == "Dockerfile":
        return True
    if base.startswith("Dockerfile."):
        return True
    low = base.lower()
    if low.endswith(".dockerfile"):
        return True
    return False


def _added_dockerfiles(diff_text: str) -> Dict[str, str]:
    """Filter the diff to only added content from Dockerfile-like paths."""
    all_files = parse_diff_added_by_file(diff_text)
    return {p: t for p, t in all_files.items() if _is_dockerfile_path(p)}


def applies(diff_text: str) -> bool:
    """True iff the diff adds lines to at least one Dockerfile-like path.

    Cheap: just diff-header parsing + basename matching, no subprocess.
    Conservative: we abstain on PRs that only touch docker-compose.yml,
    shell scripts that call docker, or GitHub Actions that build images —
    those raise adjacent norms, not this one.
    """
    return bool(_added_dockerfiles(diff_text))


def _line_count(s: str) -> int:
    if not s:
        return 0
    return s.count("\n") + (0 if s.endswith("\n") else 1)


def _lint_one(content: str) -> Optional[int]:
    """Run hadolint over a single Dockerfile snippet.

    Returns the count of NORM-RELEVANT violations (intersection with
    RELEVANT_RULES). Returns None on tool/parse failure so the caller can
    skip this file rather than score it as 0 violations.

    --no-fail keeps exit code 0 even when issues exist, so stdout JSON is
    always meaningful.
    """
    rc, out, err = run(
        ["hadolint", "--format=json", "--no-fail", "-"],
        stdin=content,
        timeout=15.0,
    )
    if rc < 0:
        # subprocess timeout / tool-missing
        return None
    if rc != 0:
        # nonzero with --no-fail means hadolint itself errored, not just
        # that it found lints — abstain.
        return None
    out = out.strip()
    if not out:
        # Empty output sometimes means "nothing to report" on a degenerate
        # input. Treat as zero violations.
        return 0
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    n = 0
    for v in data:
        if not isinstance(v, dict):
            continue
        code = v.get("code", "")
        if code in RELEVANT_RULES:
            n += 1
    return n


def score(diff_text: str) -> Optional[float]:
    if not have_tool("hadolint"):
        return None
    by_path = _added_dockerfiles(diff_text)
    if not by_path:
        return None

    total_lines = 0
    total_violations = 0
    n_measurable = 0
    for _path, content in by_path.items():
        if not content.strip():
            continue
        n_v = _lint_one(content)
        if n_v is None:
            continue
        n_measurable += 1
        total_lines += _line_count(content)
        total_violations += n_v

    if n_measurable == 0 or total_lines == 0:
        return None

    density = total_violations / total_lines
    return float(math.exp(-density * 4.0))
