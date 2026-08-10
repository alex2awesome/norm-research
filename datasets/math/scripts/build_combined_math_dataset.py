#!/usr/bin/env python
"""Build the unified ~100K-row math V/A/T dataset on sk3.

Sources combined under a single schema:
  - Math.SE  (Posts.xml, multi-answer questions filtered to proof-tags)
  - ProofBench (HF: wenjiema02/ProofBench, expert 0-7 ratings)
  - IMO-GradingBench (HF: Hwilner/imo-gradingbench, expert Points/Reward)
  - Open Proof Corpus (NOT FOUND on HF as of run; documented and skipped)

Schema (per row):
  source, group_id, row_id, problem_text, answer_body, proof_code, prose,
  taste_label, expert_grade, group_n_answers, group_score_spread,
  len_chars, n_latex_blocks, n_steps, created_age_days, raw_score

Outputs:
  /lfs/skampere3/0/alexspan/norm-research/datasets/combined_math/combined.parquet
  /lfs/skampere3/0/alexspan/norm-research/datasets/combined_math/manifest.json

Run on sk3 (HOME=/lfs/skampere3/0/alexspan).
"""

from __future__ import annotations

import gzip
import html
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd

# ---------- paths ----------
ROOT = Path("/lfs/skampere3/0/alexspan/norm-research")
OUT_DIR = ROOT / "datasets" / "combined_math"
OUT_DIR.mkdir(parents=True, exist_ok=True)
POSTS_XML = ROOT / "datasets" / "math-stackexchange" / "raw_dump" / "Posts.xml"
PREF_PAIRS = ROOT / "datasets" / "math-stackexchange" / "preference_pairs.jsonl.gz"

PROOF_TAGS = {
    "proof-verification",
    "proof-writing",
    "proof-explanation",
    "solution-verification",
    "alternative-proof",
    "proof-strategy",
}

# ---------- helpers ----------
HTML_TAG = re.compile(r"<[^>]+>")
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')
DOLLAR_INLINE = re.compile(r"(?<!\\)\$(?!\$)([^$\n]{1,400}?)(?<!\\)\$")
DOLLAR_BLOCK = re.compile(r"(?<!\\)\$\$([\s\S]{1,2000}?)(?<!\\)\$\$")
BRACKET_BLOCK = re.compile(r"\\\[([\s\S]{1,2000}?)\\\]")
BEGIN_BLOCK = re.compile(r"\\begin\{[a-zA-Z*]+\}[\s\S]*?\\end\{[a-zA-Z*]+\}")
STEP_RE = re.compile(r"(?im)^\s*(?:step\s*\d+|case\s*\d+|claim\s*\d+|lemma\s*\d+)[:.\s]")


def parse_row_attrs(line: str) -> dict:
    """Parse a Stack Exchange <row .../> line. Returns dict of unescaped attrs."""
    out = {}
    for m in ATTR_RE.finditer(line):
        out[m.group(1)] = html.unescape(m.group(2))
    return out


def html_to_text(body: str) -> str:
    """Strip Stack Exchange HTML to plaintext, preserving LaTeX."""
    if not body:
        return ""
    txt = html.unescape(body)
    # Replace <br>, </p> with newlines
    txt = re.sub(r"</p\s*>", "\n\n", txt, flags=re.I)
    txt = re.sub(r"<br\s*/?>", "\n", txt, flags=re.I)
    txt = HTML_TAG.sub("", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
    return txt


def count_latex_blocks(text: str) -> int:
    """Rough count of LaTeX delimiters / environments."""
    n = 0
    n += len(DOLLAR_INLINE.findall(text))
    n += len(DOLLAR_BLOCK.findall(text))
    n += len(BRACKET_BLOCK.findall(text))
    n += len(BEGIN_BLOCK.findall(text))
    return n


def extract_proof_code(text: str) -> tuple[str, str]:
    """Return (proof_code, prose). proof_code = LaTeX/math fragments concatenated;
    prose = text with those fragments removed (the commentary)."""
    code_parts = []

    def grab(pat, txt):
        nonlocal code_parts
        out = []
        for m in pat.finditer(txt):
            code_parts.append(m.group(0))
            out.append("[MATH]")
        return out  # unused

    prose = text
    # Order: longest/most-specific first
    for pat in (BEGIN_BLOCK, BRACKET_BLOCK, DOLLAR_BLOCK, DOLLAR_INLINE):
        prose = pat.sub(" [MATH] ", prose)
    # Re-run on original to collect actual code (since we replaced in prose)
    code_parts = []
    for pat in (BEGIN_BLOCK, BRACKET_BLOCK, DOLLAR_BLOCK, DOLLAR_INLINE):
        for m in pat.finditer(text):
            code_parts.append(m.group(0))
    proof_code = "\n".join(code_parts)
    prose = re.sub(r"\s+\[MATH\]\s+", " [MATH] ", prose)
    prose = re.sub(r"\s{2,}", " ", prose).strip()
    return proof_code, prose


def count_steps(text: str) -> int:
    explicit = len(STEP_RE.findall(text))
    # also fall back to newline-paragraph count
    if explicit > 0:
        return explicit
    paras = [p for p in text.split("\n\n") if p.strip()]
    return len(paras)


def days_since(iso: str, now_ts: float) -> float:
    try:
        # 2010-07-20T19:09:27.200
        dt = datetime.strptime(iso[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        return (now_ts - dt.timestamp()) / 86400.0
    except Exception:
        return float("nan")


# ---------- Math.SE: stream Posts.xml ----------
def parse_math_se(posts_xml: Path, log) -> pd.DataFrame:
    """Two-pass over Posts.xml.

    Pass 1: index questions -> (tags, title, body, AnswerCount, CreationDate).
    Pass 2: collect answers, filter to those whose parent question has any proof tag
            and has AnswerCount >= 2; keep answers with len(body) >= 50.
    Build per-group (question) labels = score > group median.
    """
    log(f"[math_se] streaming {posts_xml}")
    questions = {}  # qid -> dict
    keep_qids = set()
    t0 = time.time()
    n_lines = 0
    n_q = 0
    n_a = 0

    with open(posts_xml, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            n_lines += 1
            if "<row" not in line:
                continue
            # cheap check first
            if 'PostTypeId="1"' in line:
                attrs = parse_row_attrs(line)
                tags = attrs.get("Tags", "")
                # Tags formatted like "|tag1|tag2|"
                tag_list = [t for t in tags.split("|") if t]
                has_proof = any(t in PROOF_TAGS for t in tag_list)
                try:
                    n_ans = int(attrs.get("AnswerCount", "0"))
                except ValueError:
                    n_ans = 0
                if has_proof and n_ans >= 2:
                    qid = attrs.get("Id", "")
                    questions[qid] = {
                        "tags": tag_list,
                        "title": attrs.get("Title", ""),
                        "body": html_to_text(attrs.get("Body", "")),
                        "answer_count_meta": n_ans,
                        "created": attrs.get("CreationDate", ""),
                    }
                    keep_qids.add(qid)
                n_q += 1
            if n_lines % 500_000 == 0:
                log(
                    f"[math_se] pass1 line={n_lines:,} q_total={n_q:,} q_keep={len(keep_qids):,}"
                    f" elapsed={time.time()-t0:.0f}s"
                )

    log(f"[math_se] pass1 done. q_total={n_q:,} q_keep={len(keep_qids):,} elapsed={time.time()-t0:.0f}s")

    # Pass 2: collect answers for kept questions
    answers_by_q: dict[str, list[dict]] = defaultdict(list)
    t1 = time.time()
    n_lines = 0
    with open(posts_xml, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            n_lines += 1
            if 'PostTypeId="2"' not in line:
                continue
            attrs = parse_row_attrs(line)
            pid = attrs.get("ParentId", "")
            if pid not in keep_qids:
                continue
            body_txt = html_to_text(attrs.get("Body", ""))
            if len(body_txt) < 50:
                continue
            try:
                score = int(attrs.get("Score", "0"))
            except ValueError:
                score = 0
            answers_by_q[pid].append(
                {
                    "aid": attrs.get("Id", ""),
                    "body": body_txt,
                    "score": score,
                    "created": attrs.get("CreationDate", ""),
                }
            )
            n_a += 1
            if n_lines % 500_000 == 0:
                log(
                    f"[math_se] pass2 line={n_lines:,} answers_kept={n_a:,}"
                    f" elapsed={time.time()-t1:.0f}s"
                )

    log(f"[math_se] pass2 done. answers_kept={n_a:,} elapsed={time.time()-t1:.0f}s")

    # Build rows with within-group taste labels
    now_ts = time.time()
    rows = []
    for qid, answers in answers_by_q.items():
        if len(answers) < 2:
            continue
        q = questions[qid]
        scores = [a["score"] for a in answers]
        med = sorted(scores)[len(scores) // 2]
        spread = max(scores) - min(scores)
        if spread == 0:
            # No signal — skip this group
            continue
        problem_text = (q["title"] + "\n\n" + q["body"]).strip()
        for a in answers:
            body = a["body"]
            proof_code, prose = extract_proof_code(body)
            rows.append(
                {
                    "source": "math_se",
                    "group_id": f"mse_{qid}",
                    "row_id": f"mse_a{a['aid']}",
                    "problem_text": problem_text,
                    "answer_body": body,
                    "proof_code": proof_code,
                    "prose": prose,
                    "taste_label": int(a["score"] > med),
                    "expert_grade": None,
                    "group_n_answers": len(answers),
                    "group_score_spread": spread,
                    "len_chars": len(body),
                    "n_latex_blocks": count_latex_blocks(body),
                    "n_steps": count_steps(body),
                    "created_age_days": days_since(a["created"], now_ts),
                    "raw_score": a["score"],
                    "tags": "|".join(q["tags"]),
                }
            )

    df = pd.DataFrame(rows)
    log(f"[math_se] built {len(df):,} rows across {df.group_id.nunique() if len(df) else 0:,} groups")
    return df


# ---------- Math.SE preference-pairs supplement ----------
# Math content tags we'll keep as the broader "math content" filter when proof-tagged set is exhausted.
MATH_CONTENT_TAGS = {
    "real-analysis",
    "complex-analysis",
    "abstract-algebra",
    "linear-algebra",
    "general-topology",
    "functional-analysis",
    "measure-theory",
    "number-theory",
    "elementary-number-theory",
    "group-theory",
    "ring-theory",
    "field-theory",
    "galois-theory",
    "category-theory",
    "differential-geometry",
    "algebraic-geometry",
    "algebraic-topology",
    "logic",
    "set-theory",
    "elementary-set-theory",
    "model-theory",
    "combinatorics",
    "graph-theory",
    "probability-theory",
    "probability",
    "statistics",
    "inequality",
    "induction",
    "sequences-and-series",
    "limits",
    "integration",
    "ordinary-differential-equations",
    "partial-differential-equations",
}


def parse_math_se_pairs(pref_pairs: Path, exclude_qids: set, log) -> pd.DataFrame:
    """Use preference_pairs.jsonl.gz to add more Math.SE rows beyond the proof-tag set.

    Filter: question has at least one tag in MATH_CONTENT_TAGS or PROOF_TAGS,
    chosen_text and rejected_text both >= 50 chars, score_diff >= 1.
    Skips qids already in exclude_qids (the proof-tag set we already built).
    Each pair contributes 2 rows (chosen + rejected) with binary taste_label.
    """
    log(f"[math_se_pairs] streaming {pref_pairs}")
    rows = []
    seen_q = 0
    kept_q = 0
    t0 = time.time()
    with gzip.open(pref_pairs, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            try:
                d = json.loads(line)
            except Exception:
                continue
            seen_q += 1
            qid = str(d.get("question_id", ""))
            if qid in exclude_qids:
                continue
            tags_str = str(d.get("question_tags") or "")
            tag_list = [t for t in tags_str.split("|") if t] if "|" in tags_str else [t for t in tags_str.split() if t]
            tag_set = set(tag_list)
            has_proof = bool(tag_set & PROOF_TAGS)
            has_math = bool(tag_set & MATH_CONTENT_TAGS)
            if not (has_proof or has_math):
                continue
            chosen = d.get("chosen_text") or ""
            rejected = d.get("rejected_text") or ""
            if len(chosen) < 50 or len(rejected) < 50:
                continue
            if int(d.get("score_diff", 0)) < 1:
                continue
            kept_q += 1
            problem = d.get("question_text") or ""
            n_ans = int(d.get("n_answers", 2))
            spread = int(d.get("score_diff", 0))
            for which, body, lab, raw, aid in [
                ("chosen", chosen, 1, int(d.get("chosen_score", 0)), str(d.get("chosen_id", ""))),
                ("rejected", rejected, 0, int(d.get("rejected_score", 0)), str(d.get("rejected_id", ""))),
            ]:
                proof_code, prose = extract_proof_code(body)
                rows.append(
                    {
                        "source": "math_se",
                        "group_id": f"mse_{qid}",
                        "row_id": f"mse_a{aid}",
                        "problem_text": problem,
                        "answer_body": body,
                        "proof_code": proof_code,
                        "prose": prose,
                        "taste_label": lab,
                        "expert_grade": None,
                        "group_n_answers": n_ans,
                        "group_score_spread": spread,
                        "len_chars": len(body),
                        "n_latex_blocks": count_latex_blocks(body),
                        "n_steps": count_steps(body),
                        "created_age_days": None,
                        "raw_score": float(raw),
                        "tags": "|".join(tag_list),
                    }
                )
            if i % 50_000 == 0:
                log(f"[math_se_pairs] line={i:,} seen_q={seen_q:,} kept_q={kept_q:,} elapsed={time.time()-t0:.0f}s")
    df = pd.DataFrame(rows)
    log(f"[math_se_pairs] built {len(df):,} rows across {df.group_id.nunique() if len(df) else 0:,} groups")
    return df


# ---------- ProofBench ----------
def parse_proofbench(log) -> pd.DataFrame:
    log("[proofbench] loading from HF")
    from datasets import load_dataset

    ds = load_dataset("wenjiema02/ProofBench")
    parts = []
    for split in ds:
        d = ds[split]
        for ex in d:
            try:
                grade = float(ex["expert_rating"])
            except (TypeError, ValueError):
                continue
            sol = ex.get("model_solution") or ""
            if len(sol) < 50:
                continue
            proof_code, prose = extract_proof_code(sol)
            parts.append(
                {
                    "source": "proofbench",
                    "group_id": f"pb_{ex['problem_id']}",
                    "row_id": f"pb_{ex['problem_id']}_{ex.get('generator','?')}_{ex.get('response_number','0')}_{split}",
                    "problem_text": ex.get("problem", ""),
                    "answer_body": sol,
                    "proof_code": proof_code,
                    "prose": prose,
                    "taste_label": None,  # filled after group median
                    "expert_grade": grade,
                    "group_n_answers": None,
                    "group_score_spread": None,
                    "len_chars": len(sol),
                    "n_latex_blocks": count_latex_blocks(sol),
                    "n_steps": count_steps(sol),
                    "created_age_days": None,
                    "raw_score": grade,
                    "tags": "",
                }
            )
    df = pd.DataFrame(parts)
    if len(df) == 0:
        return df
    # within-problem labels: > median grade; if degenerate, fall back to > 4
    out = []
    for gid, g in df.groupby("group_id"):
        if g.expert_grade.nunique() > 1:
            med = g.expert_grade.median()
            g = g.assign(taste_label=(g.expert_grade > med).astype(int))
        else:
            g = g.assign(taste_label=(g.expert_grade > 4).astype(int))
        g = g.assign(
            group_n_answers=len(g),
            group_score_spread=float(g.expert_grade.max() - g.expert_grade.min()),
        )
        out.append(g)
    df = pd.concat(out, ignore_index=True)
    log(f"[proofbench] built {len(df):,} rows across {df.group_id.nunique():,} groups")
    return df


# ---------- IMO-GradingBench ----------
def parse_imo_grading(log) -> pd.DataFrame:
    log("[imo_grading] loading from HF")
    from datasets import load_dataset

    ds = load_dataset("Hwilner/imo-gradingbench")
    parts = []
    for split in ds:
        d = ds[split]
        for ex in d:
            try:
                pts = float(ex["Points"])
            except (TypeError, ValueError):
                pts = None
            resp = ex.get("Response") or ""
            if len(resp) < 50:
                continue
            proof_code, prose = extract_proof_code(resp)
            parts.append(
                {
                    "source": "imo_grading",
                    "group_id": f"imo_{ex['Problem ID']}",
                    "row_id": f"imo_{ex['Grading ID']}",
                    "problem_text": ex.get("Problem", ""),
                    "answer_body": resp,
                    "proof_code": proof_code,
                    "prose": prose,
                    "taste_label": None,
                    "expert_grade": pts,
                    "group_n_answers": None,
                    "group_score_spread": None,
                    "len_chars": len(resp),
                    "n_latex_blocks": count_latex_blocks(resp),
                    "n_steps": count_steps(resp),
                    "created_age_days": None,
                    "raw_score": pts,
                    "tags": str(ex.get("Problem Source") or ""),
                }
            )
    df = pd.DataFrame(parts)
    if len(df) == 0:
        return df
    out = []
    for gid, g in df.groupby("group_id"):
        if g.expert_grade.nunique() > 1:
            med = g.expert_grade.median()
            g = g.assign(taste_label=(g.expert_grade > med).astype(int))
        else:
            # Fall back to using Reward == "Correct" if available
            g = g.assign(taste_label=(g.expert_grade > 0).astype(int) if g.expert_grade.notna().any() else 0)
        g = g.assign(
            group_n_answers=len(g),
            group_score_spread=(
                float(g.expert_grade.max() - g.expert_grade.min())
                if g.expert_grade.notna().any()
                else None
            ),
        )
        out.append(g)
    df = pd.concat(out, ignore_index=True)
    log(f"[imo_grading] built {len(df):,} rows across {df.group_id.nunique():,} groups")
    return df


# ---------- main ----------
def main():
    log_path = OUT_DIR / "build.log"
    log_fp = open(log_path, "w")

    def log(msg):
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)
        log_fp.write(f"[{ts}] {msg}\n")
        log_fp.flush()

    log(f"OUT_DIR={OUT_DIR}")

    manifest = {
        "sources": {},
        "filters": {
            "math_se": {
                "proof_tags": sorted(PROOF_TAGS),
                "rule": "PostTypeId=1 has any PROOF_TAGS, AnswerCount>=2; PostTypeId=2 children len>=50",
                "label": "score > within-group median; drop groups with zero score spread",
            },
            "proofbench": {
                "label": "expert_rating > within-problem median, fallback > 4",
            },
            "imo_grading": {
                "label": "Points > within-problem median, fallback Points > 0",
            },
        },
        "schema": [
            "source",
            "group_id",
            "row_id",
            "problem_text",
            "answer_body",
            "proof_code",
            "prose",
            "taste_label",
            "expert_grade",
            "group_n_answers",
            "group_score_spread",
            "len_chars",
            "n_latex_blocks",
            "n_steps",
            "created_age_days",
            "raw_score",
            "tags",
        ],
        "notes": {
            "open_proof_corpus": (
                "Searched HuggingFace for 'open proof corpus', 'open-proof-corpus', "
                "'step proof annotation', 'proof step annotated' — no clear public match found. "
                "Documented and skipped per user fallback."
            ),
            "mathnet": "Skipped — problems only, no graded solutions.",
            "author_field": "Math.SE OwnerUserId NOT included in schema (per user spec, avoid author leakage).",
        },
    }

    frames = []

    # ProofBench (fast)
    try:
        df_pb = parse_proofbench(log)
        frames.append(df_pb)
        manifest["sources"]["proofbench"] = {
            "rows": int(len(df_pb)),
            "groups": int(df_pb.group_id.nunique()) if len(df_pb) else 0,
            "label_y1_rate": float(df_pb.taste_label.mean()) if len(df_pb) else None,
        }
    except Exception as e:
        log(f"[proofbench] FAIL {e}")
        manifest["sources"]["proofbench"] = {"error": str(e)}

    # IMO-GradingBench (fast)
    try:
        df_imo = parse_imo_grading(log)
        frames.append(df_imo)
        manifest["sources"]["imo_grading"] = {
            "rows": int(len(df_imo)),
            "groups": int(df_imo.group_id.nunique()) if len(df_imo) else 0,
            "label_y1_rate": float(df_imo.taste_label.mean()) if len(df_imo) else None,
        }
    except Exception as e:
        log(f"[imo_grading] FAIL {e}")
        manifest["sources"]["imo_grading"] = {"error": str(e)}

    # Math.SE (slow) — proof-tag strict pass
    df_mse_proof = pd.DataFrame()
    if POSTS_XML.exists():
        try:
            df_mse_proof = parse_math_se(POSTS_XML, log)
        except Exception as e:
            log(f"[math_se] FAIL {e}")
            import traceback

            traceback.print_exc()
            manifest["sources"]["math_se"] = {"error": str(e)}
    else:
        log(f"[math_se] Posts.xml not found at {POSTS_XML}")
        manifest["sources"]["math_se"] = {"error": "Posts.xml missing"}

    # Math.SE — supplement with preference pairs to push toward ~100K total
    df_mse_pairs = pd.DataFrame()
    target_total = 100_000
    current = sum(len(f) for f in frames) + len(df_mse_proof)
    budget = max(0, target_total - current)
    log(f"[math_se_pairs] current total after proof-tag pass = {current:,}; budget = {budget:,}")
    if budget > 0 and PREF_PAIRS.exists():
        try:
            exclude = set()
            if len(df_mse_proof):
                exclude = {gid.replace("mse_", "") for gid in df_mse_proof.group_id.unique()}
            df_mse_pairs = parse_math_se_pairs(PREF_PAIRS, exclude, log)
            # Subsample to budget if oversized (uniform random by group_id for determinism)
            if len(df_mse_pairs) > budget:
                # Keep whole groups (2 rows each) — shuffle group_ids deterministically
                gids = df_mse_pairs.group_id.unique()
                import numpy as np

                rng = np.random.default_rng(42)
                rng.shuffle(gids)
                # Each group is 2 rows
                keep_n_groups = budget // 2
                keep_gids = set(gids[:keep_n_groups])
                df_mse_pairs = df_mse_pairs[df_mse_pairs.group_id.isin(keep_gids)].reset_index(drop=True)
                log(f"[math_se_pairs] subsampled to {len(df_mse_pairs):,} rows ({len(keep_gids):,} groups)")
        except Exception as e:
            log(f"[math_se_pairs] FAIL {e}")
            import traceback

            traceback.print_exc()

    df_mse = pd.concat([df_mse_proof, df_mse_pairs], ignore_index=True) if (len(df_mse_proof) or len(df_mse_pairs)) else pd.DataFrame()

    if len(df_mse):
        # Drop dupes by row_id (chosen/rejected ids should be globally unique within Math.SE)
        before = len(df_mse)
        df_mse = df_mse.drop_duplicates(subset=["row_id"]).reset_index(drop=True)
        log(f"[math_se] combined proof-tag + pairs: {before:,} -> {len(df_mse):,} after row_id dedupe")
        frames.append(df_mse)
        tag_counter = Counter()
        for t in df_mse.tags.dropna():
            for tag in str(t).split("|"):
                if tag:
                    tag_counter[tag] += 1
        manifest["sources"]["math_se"] = {
            "rows": int(len(df_mse)),
            "groups": int(df_mse.group_id.nunique()),
            "label_y1_rate": float(df_mse.taste_label.mean()),
            "proof_tag_pass_rows": int(len(df_mse_proof)),
            "preference_pairs_pass_rows": int(len(df_mse_pairs)),
            "top_30_tags": tag_counter.most_common(30),
            "proof_tag_counts": {t: tag_counter.get(t, 0) for t in PROOF_TAGS},
        }

    if not frames:
        log("FATAL: no frames assembled")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    # Force consistent dtypes
    combined["taste_label"] = combined["taste_label"].astype("Int64")
    combined["expert_grade"] = combined["expert_grade"].astype("Float64")
    combined["group_n_answers"] = combined["group_n_answers"].astype("Int64")
    combined["group_score_spread"] = combined["group_score_spread"].astype("Float64")
    combined["len_chars"] = combined["len_chars"].astype("Int64")
    combined["n_latex_blocks"] = combined["n_latex_blocks"].astype("Int64")
    combined["n_steps"] = combined["n_steps"].astype("Int64")
    combined["created_age_days"] = combined["created_age_days"].astype("Float64")
    combined["raw_score"] = combined["raw_score"].astype("Float64")

    parquet_path = OUT_DIR / "combined.parquet"
    combined.to_parquet(parquet_path, index=False)
    log(f"wrote {parquet_path} ({len(combined):,} rows)")

    # Manifest stats
    manifest["combined"] = {
        "rows": int(len(combined)),
        "by_source": {
            s: {
                "rows": int((combined.source == s).sum()),
                "y1_rate": float(combined[combined.source == s].taste_label.dropna().mean())
                if (combined.source == s).any()
                else None,
            }
            for s in combined.source.unique()
        },
        "label_y1_rate_overall": float(combined.taste_label.dropna().mean()),
        "len_chars": {
            "p10": float(combined.len_chars.quantile(0.10)),
            "p50": float(combined.len_chars.quantile(0.50)),
            "p90": float(combined.len_chars.quantile(0.90)),
            "p99": float(combined.len_chars.quantile(0.99)),
            "max": int(combined.len_chars.max()),
        },
        "n_latex_blocks": {
            "p10": float(combined.n_latex_blocks.quantile(0.10)),
            "p50": float(combined.n_latex_blocks.quantile(0.50)),
            "p90": float(combined.n_latex_blocks.quantile(0.90)),
        },
        "n_steps": {
            "p10": float(combined.n_steps.quantile(0.10)),
            "p50": float(combined.n_steps.quantile(0.50)),
            "p90": float(combined.n_steps.quantile(0.90)),
        },
        "group_n_answers": {
            "p10": float(combined.group_n_answers.dropna().quantile(0.10))
            if combined.group_n_answers.notna().any()
            else None,
            "p50": float(combined.group_n_answers.dropna().quantile(0.50))
            if combined.group_n_answers.notna().any()
            else None,
            "p90": float(combined.group_n_answers.dropna().quantile(0.90))
            if combined.group_n_answers.notna().any()
            else None,
        },
    }
    manifest_path = OUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    log(f"wrote {manifest_path}")

    log_fp.close()


if __name__ == "__main__":
    main()
