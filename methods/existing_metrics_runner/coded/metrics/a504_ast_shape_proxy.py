"""a504: AST shape proxy — keyword category proportions.

Language-agnostic surrogate for an AST shape vector. We count tokens in
three categories on the candidate code:

  DECL  — declarations: def / class / struct / template / int / auto /
          using / typedef / namespace / enum
  CTRL  — control flow: for / while / if / else / switch / case / return /
          break / continue / goto / try / catch / except / finally / yield
  EXPR  — expression-side operators: + - * / % == != <= >= && || << >> -> .

Score returns the entropy of the (DECL, CTRL, EXPR) proportion distribution,
normalized to [0,1]. A code blob heavy on one category gives low entropy;
balanced shape gives high entropy.

Univariate property of the CANDIDATE CODE ALONE. CLASSIFICATION: THIN.

Note: tree-sitter is available in this repo, but we use a regex tally
deliberately to keep the metric stable across both Python and C++ without
adding two parsers. The keyword sets are language-neutral surface tokens.
"""
from __future__ import annotations

import math
import re
from typing import Optional

from ..sandbox import added_files_by_ext

ASPECT_ID = "a504"
ASPECT_NAME = "AST shape proxy entropy"
TIER = 2
TOOLS = []
APPLIES_TO_LANGS = ["Python", "C", "C++"]
CLASSIFICATION = "THIN"

EXTS = [".py", ".pyi", ".c", ".h", ".cpp", ".cc", ".cxx", ".c++", ".hpp",
        ".hxx", ".hh"]

_STR_LIT = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')

DECL_KW = re.compile(
    r"\b(def|class|struct|template|typename|using|typedef|namespace|"
    r"enum|int|long|short|char|void|bool|float|double|unsigned|signed|"
    r"auto|const|static|public|private|protected|virtual|inline|extern)\b"
)
CTRL_KW = re.compile(
    r"\b(for|while|if|elif|else|switch|case|return|break|continue|goto|"
    r"try|catch|except|finally|yield|throw|raise|do)\b"
)
EXPR_OP = re.compile(
    r"==|!=|<=|>=|&&|\|\||<<|>>|->|::|"
    r"\+\+|--|"
    r"\+=|-=|\*=|/=|%=|"
    r"[\+\-\*/%=<>]"
)


def _strip(content: str) -> str:
    # strip C-style and # comments + string literals
    s = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
    out = []
    for ln in s.splitlines():
        ln2 = _STR_LIT.sub('""', ln)
        # remove // and # comments
        for marker in ("//", "#"):
            idx = ln2.find(marker)
            if idx >= 0:
                ln2 = ln2[:idx]
        out.append(ln2)
    return "\n".join(out)


def applies(diff_text: str) -> bool:
    return bool(added_files_by_ext(diff_text, EXTS))


def score(diff_text: str) -> Optional[float]:
    by_path = added_files_by_ext(diff_text, EXTS)
    if not by_path:
        return None
    text = "\n".join(_strip(c) for c in by_path.values())
    n_decl = len(DECL_KW.findall(text))
    n_ctrl = len(CTRL_KW.findall(text))
    n_expr = len(EXPR_OP.findall(text))
    total = n_decl + n_ctrl + n_expr
    if total == 0:
        return None
    ps = [n_decl / total, n_ctrl / total, n_expr / total]
    h = 0.0
    for p in ps:
        if p > 0:
            h -= p * math.log2(p)
    # max entropy over 3 cats = log2(3) ~ 1.585
    return float(max(0.0, min(1.0, h / math.log2(3))))
