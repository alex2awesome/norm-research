"""a0: Control-flow clarity and low cognitive complexity.

We measure the cyclomatic complexity of code ADDED in the diff using lizard.
Lizard handles 20+ languages; we restrict to ones with reliable parsing.

Norm conformance: higher = lower complexity of added code = better.

Score = exp(-max_added_ccn / SCALE). SCALE=10 chosen so CCN=1 ~ 0.90, CCN=10
~ 0.37, CCN=25 ~ 0.08. We use the max function CCN (the worst function)
rather than the mean — long-tail behavior is what reviewers actually flag.
"""
from __future__ import annotations

import math
import shutil
from typing import Optional

from ..sandbox import added_files_by_ext, have_tool, run, write_temp_files

ASPECT_ID = "a0"
ASPECT_NAME = "Control-flow clarity and low cognitive complexity"
TIER = 3
TOOLS = ["lizard"]
APPLIES_TO_LANGS = ["Python", "JavaScript", "TypeScript", "Java", "Go", "C++",
                    "C", "Ruby", "C#", "PHP"]
CLASSIFICATION = "THIN"

LIZARD_EXTS = [".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go",
               ".c", ".h", ".cpp", ".cc", ".cs", ".rb", ".php"]


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, LIZARD_EXTS))


def score(diff_text: str) -> Optional[float]:
    if not have_tool("lizard"):
        return None
    by_path = added_files_by_ext(diff_text, LIZARD_EXTS)
    if not by_path:
        return None
    td = write_temp_files(by_path)
    try:
        # CSV cols: NLOC,CCN,token,param,length,location,file,func,longname,start,end
        rc, out, _ = run(["lizard", "--CCN", "1", "--csv", str(td)],
                         timeout=15.0)
        if rc < 0:
            return None
        max_ccn = 0
        n_fns = 0
        for line in out.splitlines():
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                ccn = int(parts[1])
            except ValueError:
                continue
            n_fns += 1
            if ccn > max_ccn:
                max_ccn = ccn
        if n_fns == 0:
            return None  # no parseable functions in added code
        return float(math.exp(-max_ccn / 10.0))
    finally:
        shutil.rmtree(td, ignore_errors=True)
