"""SE ladder step 0: build per-slice bank inputs (2026-06-11).

For each of the four balanced slices (crse, so_python, so_js, so_sql):
  - load the balanced csv (labels, splits, strata — NEVER re-split),
  - join answer_id -> raw answer Body (CR.SE: HTML posts.parquet;
    SO: markdown answers parquet) to recover real newlines (the balanced
    `text` column is whitespace-collapsed and useless for AST/line metrics),
  - extract code blocks (CR.SE: <pre><code>; SO: fenced ``` blocks),
  - write outputs/v2_analysis/se_ladder/{slice}_input.parquet with
    row_id, question_id, answer_id, label, split, stratum, lang, code, body.

Run on sk3 from /lfs/skampere3/0/alexspan/norm-research.
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
OUT_DIR = REPO / "outputs/v2_analysis/se_ladder"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- CR.SE html
PRE_RE = re.compile(r"<pre[^>]*>(.*?)</pre>", re.DOTALL | re.IGNORECASE)
CODE_RE = re.compile(r"<code[^>]*>(.*?)</code>", re.DOTALL | re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")

LANG_TO_EXT = {
    "python": "py", "python-3.x": "py", "python-2.x": "py",
    "cpp": "cpp", "c++": "cpp", "c++11": "cpp", "c++14": "cpp",
    "c++17": "cpp", "c++20": "cpp",
    "c": "c", "java": "java",
    "javascript": "js", "node.js": "js", "jquery": "js",
    "typescript": "ts",
    "go": "go", "ruby": "rb", "rust": "rs",
    "c#": "cs", "csharp": "cs", "php": "php", "kotlin": "kt",
    "swift": "swift", "scala": "scala",
    "sql": "sql", "tsql": "sql", "plsql": "sql", "mysql": "sql",
    "sql-server": "sql", "postgresql": "sql", "sqlite": "sql",
    "oracle": "sql",
}


def crse_extract(body_html: str) -> tuple[str, str]:
    """(code, full_text). code = concat <pre> blocks; full_text = body with
    tags stripped but BLOCK STRUCTURE preserved (pre blocks kept verbatim)."""
    if not isinstance(body_html, str) or not body_html:
        return "", ""
    code_chunks = []
    for pb in PRE_RE.findall(body_html):
        m = CODE_RE.search(pb)
        inner = m.group(1) if m else pb
        code_chunks.append(html.unescape(TAG_RE.sub("", inner)))
    code = "\n\n".join(c for c in code_chunks if c.strip()).strip()
    # full text: replace <pre> with fenced markers so presentation metrics
    # see code blocks uniformly across CR.SE and SO
    def _fence(m):
        mm = CODE_RE.search(m.group(1))
        inner = mm.group(1) if mm else m.group(1)
        return "\n```\n" + html.unescape(TAG_RE.sub("", inner)) + "\n```\n"
    body = PRE_RE.sub(_fence, body_html)
    body = re.sub(r"<br\s*/?>", "\n", body, flags=re.IGNORECASE)
    body = re.sub(r"</(p|div|li|h[1-6]|blockquote)>", "\n", body,
                  flags=re.IGNORECASE)
    body = html.unescape(TAG_RE.sub("", body))
    return code, body.strip()


FENCE_RE = re.compile(r"^[ \t]*(```|~~~)", re.MULTILINE)


def md_extract_code(body_md: str) -> str:
    """Concat fenced code blocks from markdown (line-toggle parser)."""
    if not isinstance(body_md, str) or not body_md:
        return ""
    chunks, cur, in_block = [], [], False
    for line in body_md.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            if in_block:
                chunks.append("\n".join(cur))
                cur = []
                in_block = False
            else:
                in_block = True
            continue
        if in_block:
            cur.append(line)
    if cur:  # unterminated fence
        chunks.append("\n".join(cur))
    return "\n\n".join(c for c in chunks if c.strip()).strip()


def pick_lang_crse(tags_str) -> str:
    if not isinstance(tags_str, str):
        return "unknown"
    for t in tags_str.split("|"):
        if t in LANG_TO_EXT:
            return t
    return "unknown"


SLICES = {
    "crse": {
        "balanced": REPO / "datasets/code-review/crse_balanced_v2/crse_v2_propensity_balanced.csv.gz",
        "answers": REPO / "datasets/codereview_se/posts.parquet",
        "kind": "html",
    },
    "so_python": {
        "balanced": REPO / "datasets/stackoverflow_python/balanced/so_python_v2_propensity_balanced.csv.gz",
        "answers": REPO / "datasets/stackoverflow_python/so_python_answers.parquet",
        "kind": "md", "ext": "py",
    },
    "so_js": {
        "balanced": REPO / "datasets/stackoverflow_js/balanced/so_js_v2_propensity_balanced.csv.gz",
        "answers": REPO / "datasets/stackoverflow_js/so_js_answers.parquet",
        "kind": "md", "ext": "js",
    },
    "so_sql": {
        "balanced": REPO / "datasets/stackoverflow_sql/balanced/so_sql_v2_propensity_balanced.csv.gz",
        "answers": REPO / "datasets/stackoverflow_sql/so_sql_answers.parquet",
        "kind": "md", "ext": "sql",
    },
}


def main():
    for name, cfg in SLICES.items():
        out_path = OUT_DIR / f"{name}_input.parquet"
        if out_path.exists():
            print(f"[{name}] exists, skip", flush=True)
            continue
        print(f"[{name}] loading balanced ...", flush=True)
        bal = pd.read_csv(cfg["balanced"])
        bal["answer_id"] = bal["answer_id"].astype("int64")

        if cfg["kind"] == "html":
            posts = pd.read_parquet(cfg["answers"],
                                    columns=["Id", "Body"])
            posts = posts.rename(columns={"Id": "answer_id", "Body": "raw_body"})
        else:
            posts = pd.read_parquet(cfg["answers"], columns=["Id", "Body"])
            posts = posts.rename(columns={"Id": "answer_id", "Body": "raw_body"})
        posts["answer_id"] = posts["answer_id"].astype("int64")
        df = bal.merge(posts, on="answer_id", how="left")
        n_miss = df["raw_body"].isna().sum()
        print(f"[{name}] rows={len(df)} missing_body={n_miss}", flush=True)
        assert n_miss < 0.01 * len(df), f"{name}: too many missing bodies"

        if cfg["kind"] == "html":
            ex = df["raw_body"].apply(crse_extract)
            df["code"] = ex.apply(lambda x: x[0])
            df["body"] = ex.apply(lambda x: x[1])
            df["lang"] = df["question_tags"].apply(pick_lang_crse)
            df["ext"] = df["lang"].map(LANG_TO_EXT).fillna("txt")
        else:
            df["code"] = df["raw_body"].apply(md_extract_code)
            df["body"] = df["raw_body"].fillna("")
            df["lang"] = name.replace("so_", "")
            df["ext"] = cfg["ext"]

        if "stratum" not in df.columns:
            df["stratum"] = "all"
        out = df[["question_id", "answer_id", "judgement", "split",
                  "stratum", "lang", "ext", "code", "body"]].copy()
        out = out.rename(columns={"judgement": "label"})
        out["row_id"] = np.arange(len(out))
        out.to_parquet(out_path, index=False)
        code_rate = (out["code"].str.len() > 0).mean()
        print(f"[{name}] wrote {out_path}  n={len(out)} "
              f"code_rate={code_rate:.3f} "
              f"splits={out.split.value_counts().to_dict()}", flush=True)


if __name__ == "__main__":
    main()
