#!/usr/bin/env python3
"""Migrate methods/metric_implementer_patents/metrics/p*.py into
runs/validity_full/v2/patents/codegen_claude/aXXX_v{0,1,2}_*.py.

Strategy:
- v1_structure : port the source verbatim, rewriting `from ..sandbox`
  imports to `from _patent_utils`, and wrapping the source's
  applies()+score() into a top-level `score(text: str) -> float`.
- v0_keyword   : slim regex/keyword variant per metric (handwritten map).
- v2_holistic  : combined variant — calls v1_structure logic then mixes
  with the v0 keyword signal via a saturating blend.

Numbering: p001->a190, p002->a191, ... p054->a243, p100->a244, ..., p110->a254.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research")
SRC = ROOT / "methods/metric_implementer_patents/metrics"
DST = ROOT / "runs/validity_full/v2/patents/codegen_claude"
ASPECTS_PATH = ROOT / "runs/validity_full/v2/patents/aspects.json"
EXEMPLARS_PATH = ROOT / "runs/validity_full/v2/patents/exemplars_new_aspects.json"
DROPPED_DIR = DST / "_dropped"

# Mapping: source-file-stem -> new aspect_id
SRC_FILES = sorted([p for p in SRC.glob("p*.py") if p.name != "__init__.py"])

# MPEP rules: p001..p054 -> a190..a243
# FLAN DAG  : p100..p110 -> a244..a254
def src_to_aid(stem: str) -> str:
    num = int(re.match(r"p(\d+)", stem).group(1))
    if num <= 54:
        return f"a{189 + num}"  # p001->a190 ... p054->a243
    else:  # 100..110
        return f"a{144 + num}"  # p100->a244 ... p110->a254

# Keyword cues used by v0_keyword files — must be tuned per-metric.
# Each entry: list of regex patterns to match in text.lower().
# Score = min(1.0, n_hits / target).
KEYWORD_CUES = {
    "p001": (["abstract", "claims", "description", "specification",
              "background", "summary", "drawings"], 5),
    "p002": (["abstract"], 1),  # special: length check is structural
    "p003": ([r"\bclaim\b"], 1),
    "p004": ([r"according to claim", r"of claim \d", r"as in claim",
              r"as claimed in claim"], 2),
    "p005": ([r"\(\s*\d+\s*\)"], 1),
    "p006": ([r"relates to", r"useful for", r"\bprovides\b", r"\benables\b",
              r"\bsolves\b"], 3),
    "p007": ([r"fig\.", r"figure", r"drawings?"], 3),
    "p008": ([r"background", r"prior art", r"problem"], 2),
    "p009": ([r"summary", r"according to (?:an|one) (?:embodiment|aspect)"], 2),
    "p010": ([r"\(\d+\)", r"reference"], 2),
    "p011": ([r"\bmethod\b", r"\bapparatus\b", r"\bsystem\b",
              r"\bdevice\b", r"\bcomposition\b", r"\bcompound\b",
              r"manufacture", r"process"], 3),
    "p012": ([r"abstract idea", r"mental", r"mathematical", r"algorithm",
              r"natural law", r"law of nature"], 2),
    "p013": ([r"practical application", r"improvement", r"technological",
              r"specific"], 2),
    "p014": ([r"inventive", r"more than", r"unconventional",
              r"ordered combination"], 2),
    "p015": ([r"useful", r"utility", r"\bbenefit\b", r"\badvantage\b"], 2),
    "p016": ([r"inherent", r"necessarily present", r"inevitable"], 1),
    "p017": ([r"configured to", r"adapted to", r"operable to",
              r"capable of"], 3),
    "p018": ([r"prior art", r"reference", r"anticipat"], 2),
    "p019": ([r"obvious", r"ksr", r"motivation", r"combination"], 2),
    "p020": ([r"per se", r"changes in size", r"order of step",
              r"obvious to try"], 1),
    "p021": ([r"commercial success", r"long felt", r"unexpected",
              r"skeptic", r"copying"], 1),
    "p022": ([r"working example", r"embodiment", r"enabling",
              r"experiment", r"\bdirection\b"], 3),
    "p023": ([r"embodiment", r"possess", r"description", r"support"], 3),
    "p024": ([r"example \d", r"working example", r"\bexample\s",
              r"\bexamples?\b"], 3),
    "p025": ([r"skilled in the art", r"ordinary skill", r"\bposita\b",
              r"person of ordinary skill"], 2),
    "p026": ([r"\bsaid\b", r"\bthe\s", r"\ba\s", r"\ban\s"], 2),
    "p027": ([r"\bsubstantially\b", r"\babout\b", r"\bapproximately\b",
              r"\bnearly\b", r"\bessentially\b"], 2),
    "p028": ([r"\d+\s*(?:to|-|–)\s*\d+", r"\bbetween\b", r"\brange\b"], 2),
    "p029": ([r"such as", r"for example", r"e\.g\.", r"including but not"], 1),
    "p030": ([r"configured to", r"adapted to", r"operable", r"capable of"], 2),
    "p031": ([r"selected from", r"group consisting of", r"\bmarkush\b"], 1),
    "p032": ([r"comprising", r"method", r"apparatus"], 2),
    "p033": ([r"\buse of\b", r"\bmethod of using\b"], 1),
    "p034": ([r"substantially as described", r"as shown",
              r"as herein described"], 1),
    "p035": ([r"fig\.", r"figure", r"as shown in"], 1),
    "p036": ([r"®", r"™", r"trademark"], 1),
    "p037": ([r"\bwherein\b.*\bis\b", r"=", r"variable", r"\bformula\b"], 2),
    "p038": ([r"\bmeans for\b", r"\bmodule\b", r"\bmechanism\b",
              r"\bunit\b"], 2),
    "p039": ([r"\bmeans for\b", r"\bconfigured to\b", r"\bfor\s+\w+ing\b"], 2),
    "p040": ([r"\bmeans\b.*\bstructure\b", r"corresponding structure",
              r"algorithm"], 1),
    "p041": ([r"algorithm", r"flow ?chart", r"step\s\d", r"pseudo-?code"], 2),
    "p042": ([r"\bcomprising\b", r"\bincluding\b", r"\bconsisting of\b",
              r"\bhaving\b"], 1),
    "p043": ([r"preamble", r"for\s+(?:V-?ing|the purpose)"], 1),
    "p044": ([r"\bwherein\b", r"\bwhereby\b", r"\bthereby\b"], 2),
    "p045": ([r"prior art", r"admission", r"known", r"conventional"], 2),
    "p046": ([r"fig\.", r"figure", r"drawing", r"as shown"], 3),
    "p047": ([r"fig\.\s*\d", r"figure\s*\d"], 3),
    "p048": ([r"\(\d+\)", r"reference (?:numeral|character|sign)"], 2),
    "p049": ([r"unity", r"restriction", r"single (?:general )?inventive",
              r"divisional"], 1),
    "p050": ([r"continuation", r"parent application", r"priority",
              r"continuation-in-part", r"\bcip\b"], 1),
    "p051": ([r"\bclaim\b", r"independent claim"], 1),
    "p052": ([r"abstract"], 1),
    "p053": ([r"method comprising", r"\bsteps?\b", r"\b\w+ing\b"], 2),
    "p054": ([r"title"], 1),
    # FLAN DAG
    "p100": ([r"\bclaim\b"], 1),
    "p101": ([r"\bclaim\b", r"according to claim"], 2),
    "p102": ([r"according to claim", r"of claim"], 1),
    "p103": ([r"of claim 1"], 1),
    "p104": ([r"any (?:one )?of claims", r"claims? \d+\s*(?:to|-|or|and)\s*\d+"], 1),
    "p105": ([r";", r"wherein", r"whereby"], 3),
    "p106": ([r";", r"wherein", r"comprising"], 3),
    "p107": ([r";", r"\bwherein\b", r"\bcomprising\b"], 4),
    "p108": ([r"any of claims", r"of claim"], 2),
    "p109": ([r"of claim \d", r"according to claim"], 2),
    "p110": ([r"\bcomprising\b", r";", r"\bwherein\b"], 4),
}


def make_v0_keyword(stem: str) -> str:
    cues, target = KEYWORD_CUES.get(stem, ([r"\bclaim\b"], 1))
    cues_repr = "[" + ", ".join(repr(c) for c in cues) + "]"
    return f'''import re
def score(text: str) -> float:
    try:
        t = text.lower()
        cues = {cues_repr}
        n = sum(1 for p in cues if re.search(p, t))
        return min(1.0, n / {float(target)})
    except Exception:
        return 0.5
'''


def make_v1_structure(stem: str, src_path: Path) -> str:
    """Port the source file as v1_structure.

    - Strip docstrings (keep short header comment).
    - Rewrite `from ..sandbox import X` -> `from _patent_utils import X`.
    - Rewrite ASPECT_ID/etc constants away.
    - Wrap applies()+score() into a top-level `score(text: str) -> float`
      that returns 0.5 when applies()==False or score()==None.
    """
    src = src_path.read_text()

    # Sanitize: remove top-level docstring (first triple-quoted block).
    src = re.sub(r'^"""[\s\S]*?"""\s*\n', "", src, count=1)
    # Remove `from __future__ import annotations`
    src = re.sub(r"^from __future__ import annotations\s*\n", "", src, flags=re.MULTILINE)
    # Remove typing-only imports leftover with empty lines if needed (we'll keep)
    # Rewrite sandbox import.
    src = src.replace("from ..sandbox import", "from _patent_utils import")
    # Strip metric metadata constants (ASPECT_ID etc). Keep helpers + applies + score.
    # Handle multi-line parenthesized values (e.g. THICK_REASON = (... \n ... \n ))
    META_NAMES = ("ASPECT_ID", "ASPECT_NAME", "MPEP_SECTION", "SOURCE",
                  "TIER", "TOOLS", "APPLIES_TO_LANGS", "CLASSIFICATION",
                  "THICK_REASON")
    out_lines = []
    paren_depth = 0  # net open-paren depth tracked across lines we're skipping
    for line in src.splitlines(keepends=True):
        if paren_depth > 0:
            # Track open/close on this line (ignoring quotes — naive, but
            # close-paren in string literal still keeps depth correct because
            # we also count opens inside the same string).
            paren_depth += (line.count("(") + line.count("[") + line.count("{")
                            - line.count(")") - line.count("]") - line.count("}"))
            if paren_depth <= 0:
                paren_depth = 0
            continue
        stripped = line.lstrip()
        is_meta = any(stripped.startswith(n + " =") or stripped.startswith(n + "=")
                       for n in META_NAMES)
        if is_meta:
            opens = line.count("(") + line.count("[") + line.count("{")
            closes = line.count(")") + line.count("]") + line.count("}")
            if opens > closes:
                paren_depth = opens - closes
            continue
        out_lines.append(line)
    src = "".join(out_lines)
    # Rename the source's `score` to `_score_impl` and its `applies` to
    # `_applies_impl`, then add our top-level wrapper.
    # First, replace function definitions.
    src = re.sub(r"^def\s+score\s*\(", "def _score_impl(", src,
                 count=1, flags=re.MULTILINE)
    src = re.sub(r"^def\s+applies\s*\(", "def _applies_impl(", src,
                 count=1, flags=re.MULTILINE)
    # In Python recursive calls inside the file may reference `applies`/`score`.
    # We'll do a more careful rename for body references.
    # Replace bare calls applies( and score( inside helper bodies.
    # (Only happens in a few files; safe to do globally since we renamed defs.)
    src = re.sub(r"\bscore\s*\(", "_score_impl(", src)
    src = re.sub(r"\bapplies\s*\(", "_applies_impl(", src)
    # But we just clobbered our own def line — re-fix.
    src = src.replace("def _score_impl_impl(", "def _score_impl(")
    src = src.replace("def _applies_impl_impl(", "def _applies_impl(")

    wrapper = '''

def score(text: str) -> float:
    try:
        if not _applies_impl(text):
            return 0.5
        v = _score_impl(text)
        if v is None:
            return 0.5
        return float(max(0.0, min(1.0, v)))
    except Exception:
        return 0.5
'''
    return src.rstrip() + "\n" + wrapper


def make_v2_holistic(stem: str) -> str:
    """A combined variant: invokes v1_structure score AND adds a
    keyword-density blend so the holistic version is smoother."""
    cues, target = KEYWORD_CUES.get(stem, ([r"\bclaim\b"], 1))
    cues_repr = "[" + ", ".join(repr(c) for c in cues) + "]"
    aid = src_to_aid(stem)
    # Note: import the v1_structure as a module.
    v1_mod = f"{aid}_v1_structure"
    return f'''import re, math, importlib
try:
    _v1 = importlib.import_module("{v1_mod}").score
except Exception:
    _v1 = None
def _sat(x, k=1.0):
    return 1.0 - math.exp(-x / max(1e-6, k))
def score(text: str) -> float:
    try:
        t = text.lower()
        cues = {cues_repr}
        kw_n = sum(1 for p in cues if re.search(p, t))
        kw_score = _sat(kw_n, {float(target)})
        try:
            struct_score = _v1(text) if _v1 else 0.5
        except Exception:
            struct_score = 0.5
        # 60% structural, 40% keyword saturation.
        blended = 0.6 * struct_score + 0.4 * kw_score
        return max(0.0, min(1.0, blended))
    except Exception:
        return 0.5
'''


def main():
    # 1. Move dropped aspects to _dropped/
    DROPPED_DIR.mkdir(parents=True, exist_ok=True)
    DROPPED_IDS = ["a124", "a133", "a160", "a173", "a176"]
    for aid in DROPPED_IDS:
        for f in DST.glob(f"{aid}_v*.py"):
            tgt = DROPPED_DIR / f.name
            if not tgt.exists():
                shutil.move(str(f), str(tgt))
                print(f"  moved {f.name} -> _dropped/")

    # 2. Generate new codegen files per source metric.
    new_aspect_rows = []
    for src_path in SRC_FILES:
        stem = src_path.stem.split("_")[0]   # 'p001_required_sections' -> 'p001'
        aid = src_to_aid(stem)
        # Write the three flavors.
        (DST / f"{aid}_v0_keyword.py").write_text(make_v0_keyword(stem))
        (DST / f"{aid}_v1_structure.py").write_text(make_v1_structure(stem, src_path))
        (DST / f"{aid}_v2_holistic.py").write_text(make_v2_holistic(stem))
        print(f"  wrote {aid}_v{{0,1,2}}_* (from {src_path.name})")
        # Pull metadata for aspects.json.
        src_text = src_path.read_text()
        name = re.search(r'ASPECT_NAME\s*=\s*"([^"]+)"', src_text)
        mpep = re.search(r'MPEP_SECTION\s*=\s*"([^"]+)"', src_text)
        clf  = re.search(r'CLASSIFICATION\s*=\s*"([^"]+)"', src_text)
        # First-paragraph docstring as description (best-effort).
        doc = re.search(r'"""([\s\S]*?)"""', src_text)
        desc = ""
        if doc:
            blob = doc.group(1).strip()
            # take first paragraph
            desc = blob.split("\n\n")[0].replace("\n", " ").strip()
        new_aspect_rows.append({
            "aspect_id": aid,
            "name": name.group(1) if name else f"Patent metric {aid}",
            "description": desc[:600],
            "bucket": "specific" if stem.startswith("p1") else "specific",
            "n_children": 0,
            "child_examples": [],
            "source_stem": stem,
            "mpep_section": mpep.group(1) if mpep else "",
            "classification": clf.group(1) if clf else "",
        })

    # 3. Update aspects.json.
    aspects = json.loads(ASPECTS_PATH.read_text())
    keep = [a for a in aspects if a["aspect_id"] not in
            ("a124", "a133", "a160", "a173", "a176")]
    existing_ids = {a["aspect_id"] for a in keep}
    for row in new_aspect_rows:
        if row["aspect_id"] in existing_ids:
            print(f"  WARN: aspect_id {row['aspect_id']} already exists, skipping")
            continue
        keep.append(row)
        existing_ids.add(row["aspect_id"])
    ASPECTS_PATH.write_text(json.dumps(keep, indent=1, ensure_ascii=False))
    print(f"aspects.json: {len(aspects)} -> {len(keep)} entries")

    return new_aspect_rows


if __name__ == "__main__":
    rows = main()
    print(f"DONE: {len(rows)} new aspects")
