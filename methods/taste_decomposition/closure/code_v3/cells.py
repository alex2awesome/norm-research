#!/usr/bin/env python3
"""maps_hw_si/cells.py-compatible adapter for the single CODE cell, so the generic
harness / audit / arbiter / species modules copied from that batch run unchanged.

`load(cell)` returns the same dict cells_code.load() returns, plus `meta`.

TEXT RENDERING FOR PROPOSERS (recorded decision).  The v3 document is up to 24,000
characters and is dominated by the diff, while the dense model saw only the first 2,048
tokens of it -- i.e. title, description, most of the review comments and the head of the
diff.  A blind prefix would therefore show a proposer nothing but diff on PRs with no
description, and would not match what the dense model read.  The slice card instead
preserves the document's four sections at fixed budgets.

TOKENS, NOT CHARACTERS (accumulated ruling).  Every budget below is a TOKEN budget,
measured with `tiktoken` cl100k_base -- a reasonable common denominator for the
gpt-5.6-luna and GLM-5.2 legs that actually read these cards.  Character budgets are not
used anywhere in the renderer, because a character budget silently gives dense
minified/CJK code far more content than sparse code for the same nominal size:

    Title            60 tokens
    Description     220 tokens
    Review comments  first 8, 50 tokens each
    Diff head       300 tokens, with a one-line summary of what was cut

i.e. <= ~1,000 tokens per card, so a 40-row slice is ~40K tokens.  Deterministic and
label-blind.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cells_code as _CC                                     # noqa: E402

CELL_META = {
    "code_v3": dict(
        group_column="repository", item="pull request",
        corpus="pull requests submitted to open-source GitHub repositories, each shown "
               "as its title, its description, the inline review comments left on it, "
               "and its code diff",
        construct="how good the pull request is as a contribution to that repository",
        text_trunc=1100,   # TOKEN budget (cl100k), not characters -- see module docstring
        layer1="code_v3_enriched_layer1.json",
        bank="code_v3",
    )
}
CELLS = ["code_v3"]

SK3 = "sk3"
SK3_D = "/lfs/skampere3/0/alexspan/norm-research/datasets/code-review/dense_standard_v3"
CACHE = HERE / "slice_text_cache.jsonl"


# ------------------------------------------------------------ slice cards ---
TOK_TITLE, TOK_DESC, TOK_COMMENT, TOK_DIFF = 60, 220, 50, 300
_ENC = None


def _enc():
    global _ENC
    if _ENC is None:
        import tiktoken
        _ENC = tiktoken.get_encoding("cl100k_base")
    return _ENC


def ntok(s: str) -> int:
    return len(_enc().encode(str(s), disallowed_special=()))


def tcut(s: str, n: int, marker: str = " …") -> str:
    """Truncate to n TOKENS (not characters)."""
    e = _enc()
    ids = e.encode(str(s), disallowed_special=())
    return str(s) if len(ids) <= n else e.decode(ids[:n]).rstrip() + marker


def render_card(text: str) -> str:
    t = str(text)
    m = re.search(r"^Diff:\n", t, re.M)
    diff, head = (t[m.end():], t[:m.start()]) if m else ("", t)
    mc = re.search(r"^Review comments:\n", head, re.M)
    comments, head = (head[mc.end():], head[:mc.start()]) if mc else ("", head)
    md = re.search(r"^Description:", head, re.M)
    desc, head = (head[md.end():], head[:md.start()]) if md else ("", head)
    title = tcut(head.replace("Title:", "", 1).strip(), TOK_TITLE)
    desc = tcut(desc.strip(), TOK_DESC, " …[description truncated]")
    cb = [l for l in comments.split("\n") if l.startswith("- [")]
    shown = [tcut(c, TOK_COMMENT) for c in cb[:8]]
    more = f"\n  …[{len(cb) - 8} further review comments]" if len(cb) > 8 else ""
    nfiles = len(re.findall(r"^diff --git a/(\S+)", diff, re.M))
    nhunks = diff.count("\n@@")
    dhead = tcut(diff, TOK_DIFF, "")
    dtail = (f"\n…[diff truncated; {nfiles} file(s), {nhunks} hunk(s) in full]"
             if ntok(diff) > TOK_DIFF else "")
    return (f"Title: {title or '(unknown)'}\n\n"
            f"Description: {desc or '(none)'}\n\n"
            f"Review comments:\n" + ("\n".join(shown) + more if shown else "(none)") +
            f"\n\nDiff:\n{dhead}{dtail}")


def fetch_texts(row_ids):
    """Pull the v3 text for the given rows from sk3 and render slice cards.
    Cached on disk so a rerun of a round never re-fetches."""
    import json
    cache = {}
    if CACHE.exists():
        for l in open(CACHE):
            r = json.loads(l)
            cache[r["row_id"]] = r["card"]
    need = [r for r in row_ids if r not in cache]
    if need:
        script = f"""
import pandas as pd, json, sys
D = "{SK3_D}"
want = set(json.loads(sys.stdin.read()))
out = []
for sp in ("eval", "test"):
    d = pd.read_csv(f"{{D}}/split/{{sp}}.csv", usecols=["repo","pr_number","text"])
    d["row_id"] = d["repo"].astype(str) + "/" + d["pr_number"].astype(str)
    d = d[d["row_id"].isin(want)]
    for _, r in d.iterrows():
        out.append({{"row_id": r["row_id"], "text": str(r["text"])[:26000]}})
print(json.dumps(out))
"""
        p = subprocess.run(
            ["ssh", SK3, "export HOME=/lfs/skampere3/0/alexspan; "
                         "/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python -"],
            input=script.replace("sys.stdin.read()", repr(json.dumps(need))),
            capture_output=True, text=True, timeout=900)
        blob = [l for l in p.stdout.splitlines() if l.startswith("[")]
        assert blob, f"sk3 text fetch failed: {p.stderr[-800:]}"
        got = json.loads(blob[-1])
        with open(CACHE, "a") as fh:
            for r in got:
                card = render_card(r["text"])
                cache[r["row_id"]] = card
                fh.write(json.dumps({"row_id": r["row_id"], "card": card}) + "\n")
    missing = [r for r in row_ids if r not in cache]
    assert not missing, f"{len(missing)} rows had no text on sk3, e.g. {missing[:3]}"
    return [cache[r] for r in row_ids]


def load(cell="code_v3"):
    d = _CC.load()
    d["meta"] = CELL_META[cell]
    return d


if __name__ == "__main__":
    d = load()
    cards = fetch_texts(d["ids"][:3])
    for c in cards:
        print("=" * 70)
        print(c[:1200])
