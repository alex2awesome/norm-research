"""
extract_editorial_code.py

Extract solution code embedded in the prose `editorial_text` column of CF / AC
editorials in `competition_unified/editorials.parquet`.

Background (see datasets/competition_unified/README.md):
    `cf` (cf_blog, open-r1) and `ac` (organizer_blog) editorials have
    `editorial_code` == "" (empty). The solution code IS in `editorial_text`
    but it is (a) embedded without ``` fences or <pre> tags, and (b) almost
    always *tokenized* — one C++/Python token per line — because of how the
    upstream HTML→text scrape ran. Some AC blocks are also duplicated immediately
    after themselves (Copy-button artifact).

We DO NOT modify `editorials.parquet`. We write a new file
`editorials_code_extracted.parquet` with columns:
    editorial_id, platform, problem_id, canonical_pid,
    extracted_code, code_lang, n_code_blocks, extraction_confidence

Confidence buckets:
    confident — a single full-program block was found (e.g. has main()/return 0
                for C++, or `if __name__ == '__main__'` / def + final call for
                Python, ≥10 logical lines).
    ambiguous — multiple plausible blocks were found (e.g. C++ AND Python AND
                Java), OR a block was found but it's short / lacks a main / lacks
                a return — likely a snippet not a full program.
    none      — no `#include`, `def`, `import`, `public class`, etc. signal —
                the editorial is pure prose / pseudocode / math.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd

CODE_START_RE = re.compile(
    r"(?:^|\n)("
    r"#include"                              # C++
    r"|using\s+namespace\s+std"               # C++
    r"|int\s+main\b"                          # C++
    r"|int32_t\s+main\b"                      # C++
    r"|signed\s+main\b"                       # C++
    r"|public\s+(?:static\s+)?(?:final\s+)?class\s+\w+"  # Java
    r"|public\s+static\s+void\s+main"         # Java
    r"|import\s+java\."                       # Java
    r"|from\s+\w+(?:\.\w+)*\s+import\s+"      # Python
    r"|import\s+(?:sys|os|math|collections|heapq|bisect|itertools|functools|re|io)\b"  # Python
    r"|def\s+\w+\s*\("                        # Python
    r"|if\s+__name__\s*==\s*['\"]__main__['\"]"  # Python
    r")",
    re.MULTILINE,
)

# Tokens that look like single C++/Python tokens — used to decide whether a line
# is "code-ish" when scanning past the start anchor.
CODE_TOKEN_RE = re.compile(
    r"^[\s]*("
    r"[\{\}\[\]\(\);,]"                      # punctuation
    r"|[<>=!+\-*/%&|^~]+"                    # operators
    r"|\d+"                                  # integers
    r"|[A-Za-z_]\w*"                         # identifiers
    r"|\"[^\"]*\""                           # string literals
    r"|'[^']*'"                              # char literals
    r"|#\s*\w+"                              # preprocessor
    r"|//.*"                                 # cpp line comment
    r"|->"
    r"|::"
    r")[\s]*$"
)

# Prose detector: a line with multiple English words AND ending in . ! ? :, OR
# containing Japanese characters / CF/AC navigation markers.
PROSE_END_RE = re.compile(
    r"^(posted:|last update:|"
    r"Bonus|Bonus \d|Challenge:|Harder version|Try to solve|Refer to|"
    r"Submit|提出|この解説|Tutorial|Editorial|Code 2|Solution 2|"
    r"Implementation on \w+|実装例|解答例|サンプル|ソースコード)\b",
    re.IGNORECASE,
)

JAPANESE_RE = re.compile(r"[぀-ヿ一-鿿]")

# A line that is clearly natural-language prose: ≥6 words, contains spaces between
# letter-only tokens, ends in . / ! / ? and is NOT a code statement (no semicolon
# trailing).
def looks_like_prose(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if PROSE_END_RE.match(s):
        return True
    if JAPANESE_RE.search(s):
        return True
    # Has math markers like $$$ — prose
    if "$$$" in s or "\\(" in s or "\\[" in s:
        return True
    # Mostly English words?
    words = re.findall(r"[A-Za-z][A-Za-z\-']{2,}", s)
    if len(words) >= 6 and not any(ch in s for ch in (";", "{", "}", "(", ")", "=")):
        return True
    if len(words) >= 8 and s.endswith((".", "?", "!", ":")) and s.count(";") == 0:
        return True
    return False


# ---------------------------------------------------------------------------
# De-tokenization
# ---------------------------------------------------------------------------
_CPP_KEYWORDS = {
    "int", "long", "short", "char", "void", "bool", "double", "float", "auto",
    "const", "static", "struct", "class", "typedef", "namespace", "using",
    "if", "else", "for", "while", "do", "switch", "case", "default", "break",
    "continue", "return", "new", "delete", "try", "catch", "throw", "public",
    "private", "protected", "template", "typename", "this", "operator",
    "constexpr", "inline", "virtual", "override", "signed", "unsigned",
}

_PY_STATEMENT_STARTS = (
    "import ", "from ", "def ", "class ", "return", "if ", "elif ", "else",
    "for ", "while ", "try", "except", "finally", "with ", "yield", "raise",
    "pass", "break", "continue", "global", "nonlocal", "assert ", "async ",
    "await ", "print(", "@",
)


def _join_tokenized_block(lines: list[str], lang_hint: str = "auto") -> str:
    """Many CF/AC code blocks have one token per line. Join them.

    For C++-style code we insert breaks at `;`, `{`, `}`. For Python we
    insert breaks before common statement keywords. The output is a
    best-effort partially-reconstructed program — sufficient for embedding /
    code-similarity uses, but not guaranteed to compile.
    """
    pieces: list[str] = []
    for ln in lines:
        s = ln.rstrip("\r")
        t = s.strip()
        if t == "":
            pieces.append("\n")
            continue
        if t.startswith("#") and not t.startswith("#include<") and not t.startswith("#define("):
            pieces.append("\n" + t + " ")
            continue
        pieces.append(t + " ")

    joined = "".join(pieces)
    joined = re.sub(r"\s+", " ", joined)

    looks_cpp = ("#include" in joined) or ("int main" in joined) or ("std::" in joined) or ("cout" in joined) or ("using namespace" in joined)
    looks_py = (not looks_cpp) and any(tok in joined for tok in (" def ", " import ", "import ", "from ", "print(", "if __name__"))

    if looks_cpp:
        joined = re.sub(r"\s*;\s*", ";\n", joined)
        joined = re.sub(r"\s*\{\s*", " {\n", joined)
        joined = re.sub(r"\s*\}\s*", "\n}\n", joined)
        joined = re.sub(r"(?<!\n)\s*(#\s*(?:include|define|pragma|ifdef|ifndef|endif|if|else)\b)",
                        r"\n\1", joined)
    elif looks_py:
        py_starts = ("import", "from", "def", "class", "return", "if", "elif", "else",
                     "for", "while", "try", "except", "finally", "with", "yield",
                     "raise", "pass", "break", "continue", "print")
        py_pat = "|".join(rf"\b{k}\b" for k in py_starts)
        # Break before statement keywords if preceded by `)` `]` `"` `'` or alnum
        joined = re.sub(rf"([\)\]\"'\w]) ({py_pat})", r"\1\n\2", joined)
        # Always break top-level imports / defs / classes
        joined = re.sub(r"\s+(def\s+\w+\s*\()", r"\n\1", joined)
        joined = re.sub(r"\s+(class\s+\w+)", r"\n\1", joined)
        joined = re.sub(r"\s+(from\s+\w+(?:\.\w+)*\s+import\b)", r"\n\1", joined)
        joined = re.sub(r"(?<!\bfrom )(?<!\w)(import\s+\w+)", r"\n\1", joined)
        # After `:` followed by statement keyword, newline
        joined = re.sub(rf":\s+({py_pat})", r":\n\1", joined)

    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()


def _looks_tokenized(block: str) -> bool:
    """A block is 'tokenized' (one token per line) if a very high fraction of
    non-blank lines are SINGLE tokens — including isolated punctuation like
    `;`, `{`, `}`, `(`, `)`, `,`, single operators, or a single identifier.

    Carefully distinguish from properly-formatted code which has many short
    lines but they are full *statements* (containing operators, brackets, etc.).
    """
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if len(lines) < 10:
        return False
    # A "tokenized" line: short AND contains no whitespace inside (single token),
    # OR is just a single punctuation char/operator.
    single_token = 0
    for ln in lines:
        if " " in ln:
            # has internal whitespace → likely a real statement
            continue
        if len(ln) <= 30:
            single_token += 1
    return single_token / len(lines) > 0.7


def normalize_block(raw: str) -> str:
    """If the block is in token-per-line format, de-tokenize. Else return as-is."""
    if _looks_tokenized(raw):
        return _join_tokenized_block(raw.splitlines())
    return raw.strip()


# ---------------------------------------------------------------------------
# Block extraction
# ---------------------------------------------------------------------------
def _scan_block(text: str, start: int) -> tuple[int, int]:
    """From `start` in `text`, scan forward, return (start, end) char positions
    of the maximal contiguous code-ish region.
    """
    # Walk character-by-character (but really line-by-line) collecting lines
    # until we encounter clear prose or end of text.
    end = start
    n = len(text)
    while end < n:
        # find next newline
        nl = text.find("\n", end)
        if nl == -1:
            nl = n
        line = text[end:nl]
        # End conditions:
        if looks_like_prose(line):
            break
        # A line that's a typical multi-word prose marker
        s = line.strip()
        # Heuristic: 1 line of all-letters words separated by spaces, no
        # code symbols, ≥3 words.
        if s and not JAPANESE_RE.search(s):
            no_code = not any(c in s for c in "{};=()[]<>+*&|/\\")
            words = re.findall(r"[A-Za-z][A-Za-z\-']+", s)
            if no_code and len(words) >= 4 and len(words) == len(s.split()):
                # Stop — but only if next line is also prose-like, to avoid
                # stopping on an `auto x = y;` style line that wraps.
                next_nl = text.find("\n", nl + 1)
                if next_nl == -1:
                    next_nl = n
                next_line = text[nl + 1: next_nl].strip()
                if (not next_line or looks_like_prose(next_line) or
                        (not any(c in next_line for c in "{};=()[]<>+*&|/\\")
                         and len(re.findall(r"[A-Za-z][A-Za-z\-']+", next_line)) >= 3)):
                    break
        end = nl + 1
    return start, end


def _split_blocks(text: str) -> list[tuple[int, int]]:
    """Return list of (start, end) positions of all candidate code blocks."""
    spans: list[tuple[int, int]] = []
    pos = 0
    while True:
        m = CODE_START_RE.search(text, pos)
        if not m:
            break
        # Move back to start of the line containing the match
        start = m.start(1)
        # Walk back to previous newline
        line_start = text.rfind("\n", 0, start) + 1
        block_start = line_start
        block_end = _scan_block(text, block_start)[1]
        if block_end - block_start >= 30:  # avoid spurious tiny matches
            # avoid overlap with previous span
            if spans and block_start < spans[-1][1]:
                # extend previous span if this is contiguous code
                pass
            else:
                spans.append((block_start, block_end))
        pos = max(block_end, m.end())
    return spans


_TRAIL_PROSE_RE = re.compile(
    r"(?:\n|^)(?:posted:.*|last update:.*|提出|この解説.*|This editorial.*"
    r"|\d+\s+years?\s+ago.*)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _strip_trailing_prose(s: str) -> str:
    """Remove AC/CF trailing 'posted:', 'last update:', etc."""
    while True:
        m = _TRAIL_PROSE_RE.search(s)
        if not m:
            break
        s = s[:m.start()].rstrip()
    return s


def _norm_for_compare(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _dedupe_doubled(text: str) -> str:
    """AtCoder editorials commonly duplicate a code block immediately. If the
    second half of `text` exactly equals the first half (modulo trailing
    whitespace / trailing prose markers), return the first half.
    """
    s = _strip_trailing_prose(text.strip())
    n = len(s)
    if n < 60:
        return s

    # Strategy 1: find a candidate "start of second copy" anchor and check if
    # the substring from 0..anchor equals anchor..end (ignoring whitespace).
    starts = ("#include", "import ", "from ", "def ", "if __name__",
              "public class", "public static", "using namespace")
    first_anchor = None
    for tok in starts:
        if s.startswith(tok):
            first_anchor = tok
            break
    if first_anchor:
        # find second occurrence of the same anchor
        snd = s.find(first_anchor, len(first_anchor))
        while snd != -1:
            a = s[:snd].rstrip()
            b = s[snd:].rstrip()
            if _norm_for_compare(a) == _norm_for_compare(b):
                return a
            # try the next occurrence (in case there are 3 copies)
            snd = s.find(first_anchor, snd + len(first_anchor))

    # Strategy 2: midpoint exact comparison
    half = n // 2
    for off in range(-5, 6):
        cut = half + off
        if 0 < cut < n:
            a = s[:cut].strip()
            b = s[cut:].strip()
            if a and _norm_for_compare(a) == _norm_for_compare(b):
                return a

    return s


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------
def detect_lang(code: str) -> str:
    if not code:
        return "unknown"
    if "#include" in code or re.search(r"\bint\s+main\b", code) or "std::" in code or "cout" in code:
        return "cpp"
    if re.search(r"\bdef\s+\w+\s*\(", code) or "print(" in code or "import sys" in code:
        return "python"
    if "public class" in code or "public static void main" in code or "System.out" in code:
        return "java"
    return "unknown"


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------
def _is_full_program(code: str, lang: str) -> bool:
    if not code:
        return False
    n_lines = len([ln for ln in code.splitlines() if ln.strip()])
    if n_lines < 8:
        return False
    if lang == "cpp":
        return bool(re.search(r"\bint(?:32_t)?\s+main\b|\bsigned\s+main\b", code))
    if lang == "python":
        return ("if __name__" in code) or bool(re.search(r"\bdef\s+\w+\s*\(", code)) \
               or ("input(" in code) or ("sys.stdin" in code)
    if lang == "java":
        return "public static void main" in code
    return False


def score_extraction(blocks: list[str]) -> tuple[str, str, str]:
    """Returns (extracted_code, code_lang, confidence)."""
    if not blocks:
        return "", "unknown", "none"
    # Rank blocks: full programs first (by length), else longest.
    scored = []
    for b in blocks:
        lang = detect_lang(b)
        full = _is_full_program(b, lang)
        scored.append((full, len(b), b, lang))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_full, best_len, best_code, best_lang = scored[0]
    full_programs = [s for s in scored if s[0]]
    if best_full:
        if len(full_programs) >= 2:
            # Multiple full programs (likely multiple languages) — still confident
            # in the best one but flag.
            return best_code, best_lang, "confident_multi"
        return best_code, best_lang, "confident"
    # Not a full program but at least a long code-ish block
    if best_len >= 200:
        return best_code, best_lang, "ambiguous"
    return best_code, best_lang, "ambiguous_short"


# ---------------------------------------------------------------------------
# Per-row extraction
# ---------------------------------------------------------------------------
def extract_one(text: str) -> dict:
    if not text or not isinstance(text, str):
        return dict(extracted_code="", code_lang="unknown", n_code_blocks=0,
                    extraction_confidence="none")
    spans = _split_blocks(text)
    if not spans:
        return dict(extracted_code="", code_lang="unknown", n_code_blocks=0,
                    extraction_confidence="none")
    raw_blocks = [text[s:e] for s, e in spans]
    norm_blocks = [normalize_block(b) for b in raw_blocks]
    deduped = [_dedupe_doubled(b) for b in norm_blocks]
    # Drop blocks that collapse to nothing or are just one or two tokens.
    deduped = [b for b in deduped if len(b) >= 30 and len(b.splitlines()) >= 3]
    n_blocks = len(deduped)
    code, lang, conf = score_extraction(deduped)
    return dict(extracted_code=code, code_lang=lang, n_code_blocks=n_blocks,
                extraction_confidence=conf)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True,
                    help="Path to editorials.parquet")
    ap.add_argument("--out", dest="out", required=True,
                    help="Output parquet path")
    ap.add_argument("--platforms", default="cf,ac",
                    help="Comma-separated platforms to process (default cf,ac)")
    ap.add_argument("--sample", type=int, default=0,
                    help="If >0, only process that many rows (per platform) for dry-run")
    ap.add_argument("--dump-sample", default="",
                    help="If set, also write `{path}.examples.txt` showing 30 CF + 20 AC examples for manual review (use with --sample to keep it fast)")
    args = ap.parse_args()

    plats = [p.strip() for p in args.platforms.split(",") if p.strip()]
    print(f"Loading {args.inp} …", flush=True)
    df = pd.read_parquet(args.inp)
    print(f"Loaded {len(df):,} rows.", flush=True)
    sub = df[df["platform"].isin(plats)].copy()
    print(f"Processing {len(sub):,} rows across platforms {plats}.", flush=True)
    if args.sample > 0:
        sub = (sub.groupby("platform", group_keys=False)
                  .apply(lambda g: g.sample(min(args.sample, len(g)), random_state=0)))
        print(f"Sampled down to {len(sub):,} rows for dry-run.", flush=True)

    out_rows = []
    counts = {p: {"confident": 0, "confident_multi": 0, "ambiguous": 0,
                  "ambiguous_short": 0, "none": 0} for p in plats}
    for i, (idx, row) in enumerate(sub.iterrows()):
        if i % 1000 == 0:
            print(f"  … {i:,}/{len(sub):,}", flush=True)
        res = extract_one(row.get("editorial_text", "") or "")
        counts[row["platform"]][res["extraction_confidence"]] = \
            counts[row["platform"]].get(res["extraction_confidence"], 0) + 1
        out_rows.append({
            "editorial_id": row["editorial_id"],
            "platform": row["platform"],
            "problem_id": row["problem_id"],
            "canonical_pid": row.get("canonical_pid"),
            "source": row.get("source"),
            **res,
        })

    out_df = pd.DataFrame(out_rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"Wrote {len(out_df):,} rows to {out_path}", flush=True)
    print("\nCoverage by platform / confidence bucket:")
    for plat, c in counts.items():
        total = sum(c.values())
        print(f"  {plat}: total={total:,}")
        for k, v in c.items():
            pct = 100.0 * v / total if total else 0.0
            print(f"    {k:18s} {v:6,}  ({pct:5.1f}%)")

    if args.dump_sample:
        # Dump 30 CF + 20 AC examples to a text file for human review
        ex_path = str(out_path) + ".examples.txt"
        keep = {"cf": 30, "ac": 20}
        with open(ex_path, "w") as fh:
            sub_for_merge = sub.reset_index(drop=True).drop(columns=["code_lang"], errors="ignore")
            merged = sub_for_merge.merge(
                out_df[["editorial_id", "platform", "extracted_code", "code_lang",
                        "n_code_blocks", "extraction_confidence"]],
                on=["editorial_id", "platform"], how="left")
            for plat, k in keep.items():
                msk = merged["platform"] == plat
                sl = merged[msk].sample(min(k, int(msk.sum())), random_state=42)
                for _, row in sl.iterrows():
                    fh.write("=" * 100 + "\n")
                    fh.write(f"PLATFORM={plat} pid={row['problem_id']} title={row.get('problem_title','')} ed_id={row['editorial_id']}\n")
                    fh.write(f"confidence={row['extraction_confidence']} n_blocks={row['n_code_blocks']} lang={row['code_lang']}\n")
                    fh.write("--- editorial_text (first 800 chars) ---\n")
                    fh.write(((row['editorial_text'] or "")[:800]) + "\n")
                    fh.write("--- extracted_code (first 1500 chars) ---\n")
                    fh.write(((row['extracted_code'] or "")[:1500]) + "\n\n")
        print(f"Wrote inspection samples to {ex_path}", flush=True)


if __name__ == "__main__":
    main()
