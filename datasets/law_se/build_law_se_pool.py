#!/usr/bin/env python3
"""Law.SE position-matched pool build (Stage 1).

Direct 1:1 port of CR.SE v2 (build_crse_pool_v2.py), which itself converged on
the Math.SE v3.3 recipe (build_v3_position_matched.py + propensity_balance_v3_3.py).
The label is the PRACTITIONER-CROWD revealed preference: Law Stack Exchange
community votes + asker acceptance on legal-exposition answers.

Recipe (identical to Math.SE / CR.SE):
  - Pos (label=1): accepted AND Score >= --pos-min-score (default 3)
  - Neg (label=0): NOT accepted AND Score <= --neg-max-score (default 0)
      strict recipe default NEG_MAX_SCORE = 0; CR.SE widened to 1 for its
      right-shifted score distribution. Auto-fallback below.
  - Question-disjoint filter (drop questions that have BOTH classes).
  - Min answer length 50 chars after HTML strip (stub filter).
  - **Position matching** within (primary_tag x question_len_bin x
    answer_len_bin x position{1,2,3+} x 3-year bin). THE dominant confound on
    SE is answer POSITION (earlier answers accrue more votes); this kills it.
  - Carry per-row metadata: question_id, answer_id, answer_position,
    n_answers_on_question, answer_age_gap_days, answer_year, score, accepted,
    primary_tag, question_tags, text.
  - Hash-based 80/10/10 group split via splits.py (shared module) — all answers
    to a question land on ONE side.
  - Time-order audit (P(positive posted earlier)) baked into the manifest.

Auto-fallback for NEG_MAX_SCORE (same as CR.SE v2):
  Try --neg-max-score=0 first (recipe-strict). If the matched POOL is
  < --min-pool-rows (default 4000 for this SMALL site), re-run with
  --neg-max-score=1 and document the fallback in the manifest.

Presentation normalization: the canonical `text` column uses the SAME HTML
stripper as Math.SE/CR.SE (so floors are comparable). A parallel
`answer_text_norm` column applies NFKC, curly->straight quotes, dash/ellipsis
normalization, whitespace collapse (per task spec) and is carried for
downstream consumers; raw stripped text is retained in `text`.

Floor tests are NOT run here — propensity_balance_law_se.py owns the floor
diagnostics and the fail-loud gate.

Outputs:
  built/law_se_pool.csv.gz
  built/law_se_pool.manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from splits import split_of  # noqa: E402

POSITION_BIN_LABELS = ("1", "2", "3+")


# -- text utilities ----------------------------------------------------------

def strip_html(html_text: str) -> str:
    """HTML stripper IDENTICAL to Math.SE v3 / CR.SE v2 — strips tags only and
    decodes the five core entities. Floor measurement uses question-only text
    so this must match the other SE sites byte-for-byte."""
    if not isinstance(html_text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", html_text)
    text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    text = text.replace("&quot;", '"').replace("&#39;", "'")
    text = " ".join(text.split())
    return text.strip()


def presentation_normalize(text: str) -> str:
    """NFKC + curly->straight quotes, dashes, ellipsis, whitespace collapse.
    Carried as answer_text_norm (task spec); does NOT replace the canonical
    `text` field used for floor comparability."""
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = (t.replace("‘", "'").replace("’", "'")
           .replace("“", '"').replace("”", '"')
           .replace("–", "-").replace("—", "-")
           .replace("…", "..."))
    t = " ".join(t.split())
    return t.strip()


def parse_tags(tag_str: str) -> str:
    if not isinstance(tag_str, str) or not tag_str:
        return ""
    if tag_str.startswith("|") or tag_str.endswith("|"):
        return "|".join(t for t in tag_str.split("|") if t)
    tags = re.findall(r"<([^>]+)>", tag_str)
    return "|".join(tags)


def position_bin(position: int) -> int:
    return min(int(position), 3)


def position_distribution(items: pd.DataFrame) -> dict:
    counts = {label: 0 for label in POSITION_BIN_LABELS}
    for p in items["answer_position"]:
        counts[POSITION_BIN_LABELS[position_bin(p) - 1]] += 1
    total = len(items) or 1
    return {label: round(c / total, 4) for label, c in counts.items()}


# -- XML load ----------------------------------------------------------------

def load_posts_xml(posts_xml: str) -> pd.DataFrame:
    """Stream Posts.xml into a DataFrame with the columns we need. Law.SE's
    Posts.xml is ~137 MB, so a single iterparse pass is fine."""
    print(f"[{datetime.now():%H:%M:%S}] parsing {posts_xml}", flush=True)
    rows = []
    n = 0
    for _event, elem in ET.iterparse(str(posts_xml), events=("end",)):
        if elem.tag != "row":
            continue
        n += 1
        ptid = elem.get("PostTypeId")
        if ptid not in ("1", "2"):
            elem.clear()
            continue
        rows.append({
            "Id": int(elem.get("Id")),
            "PostTypeId": int(ptid),
            "ParentId": int(elem.get("ParentId")) if elem.get("ParentId") else -1,
            "AcceptedAnswerId": (int(elem.get("AcceptedAnswerId"))
                                 if elem.get("AcceptedAnswerId") else -1),
            "Score": int(elem.get("Score", 0)),
            "Title": elem.get("Title", "") or "",
            "Body": elem.get("Body", "") or "",
            "Tags": elem.get("Tags", "") or "",
            "OwnerUserId": (int(elem.get("OwnerUserId"))
                            if elem.get("OwnerUserId") else -1),
            "CreationDate": elem.get("CreationDate", "") or "",
        })
        if n % 200000 == 0:
            print(f"  rows scanned: {n}", flush=True)
        elem.clear()
    df = pd.DataFrame(rows)
    print(f"[{datetime.now():%H:%M:%S}] parsed {len(df)} Q+A rows "
          f"({(df.PostTypeId == 1).sum()} questions, "
          f"{(df.PostTypeId == 2).sum()} answers)", flush=True)
    return df


# -- core build --------------------------------------------------------------

def build_labeled_pool(df: pd.DataFrame,
                       pos_min_score: int,
                       neg_max_score: int,
                       min_chars: int) -> tuple[pd.DataFrame, dict, dict]:
    """Label, attach position metadata, question-disjoint filter.
    Identical to CR.SE v2 build_labeled_pool()."""
    stage: dict = {}

    qs = df[df.PostTypeId == 1].copy()
    ans = df[df.PostTypeId == 2].copy()
    stage["questions_total"] = int(len(qs))
    stage["answers_total"] = int(len(ans))

    print(f"[{datetime.now():%H:%M:%S}] indexing questions", flush=True)
    qs["title"] = qs.Title.fillna("")
    qs["body_stripped"] = qs.Body.fillna("").map(strip_html)
    qs["tags_parsed"] = qs.Tags.fillna("").map(parse_tags)
    qs["question_len"] = qs.title.str.len() + 2 + qs.body_stripped.str.len()
    qs_map_full = qs.set_index("Id")[
        ["title", "body_stripped", "tags_parsed", "question_len"]
    ].to_dict(orient="index")

    accepted_ids = set(qs[qs.AcceptedAnswerId > 0]
                       .AcceptedAnswerId.astype(int).tolist())
    stage["n_accepted_answer_ids"] = len(accepted_ids)

    # Position metadata over ALL answers (BEFORE any filter).
    print(f"[{datetime.now():%H:%M:%S}] computing position metadata over all answers",
          flush=True)
    ans = ans.sort_values(["ParentId", "CreationDate", "Id"]).reset_index(drop=True)
    ans["answer_position"] = ans.groupby("ParentId").cumcount() + 1
    ans["n_answers_on_question"] = ans.groupby("ParentId")["Id"].transform("count")
    first_dt = ans.groupby("ParentId")["CreationDate"].transform("first")
    ans["_first_dt"] = pd.to_datetime(first_dt, errors="coerce")
    ans["_self_dt"] = pd.to_datetime(ans.CreationDate, errors="coerce")
    ans["answer_age_gap_days"] = (
        (ans["_self_dt"] - ans["_first_dt"]).dt.total_seconds() / 86400.0
    ).round(3).fillna(0.0)
    ans["answer_year"] = ans.CreationDate.str[:4].astype(int)

    print(f"[{datetime.now():%H:%M:%S}] stripping answer HTML", flush=True)
    ans["answer_text"] = ans.Body.fillna("").map(strip_html)
    ans["answer_text_norm"] = ans.answer_text.map(presentation_normalize)
    ans["answer_len"] = ans.answer_text.str.len()

    # Label
    ans["accepted"] = ans.Id.isin(accepted_ids)
    is_pos = ans.accepted & (ans.Score >= pos_min_score)
    is_neg = (~ans.accepted) & (ans.Score <= neg_max_score)
    stage["labeled_positive_pre_qdisjoint"] = int(is_pos.sum())
    stage["labeled_negative_pre_qdisjoint"] = int(is_neg.sum())

    keep_mask = ((is_pos | is_neg)
                 & (ans.answer_len >= min_chars)
                 & (ans.ParentId.isin(qs_map_full)))
    cand = ans[keep_mask].copy()
    cand["judgement"] = is_pos[keep_mask].astype(int).values
    stage["after_length_and_parent_filter"] = int(len(cand))

    # Question-disjoint filter
    pos_qids = set(cand.loc[cand.judgement == 1, "ParentId"].tolist())
    neg_qids = set(cand.loc[cand.judgement == 0, "ParentId"].tolist())
    overlap = pos_qids & neg_qids
    cand = cand[~cand.ParentId.isin(overlap)].copy()
    stage["overlap_questions_dropped"] = int(len(overlap))
    stage["after_question_disjoint_pos"] = int((cand.judgement == 1).sum())
    stage["after_question_disjoint_neg"] = int((cand.judgement == 0).sum())

    def primary_tag(qid):
        rec = qs_map_full.get(qid)
        if not rec:
            return "_none_"
        tags = rec["tags_parsed"].split("|")
        return tags[0] if tags and tags[0] else "_none_"

    cand["question_tags"] = cand.ParentId.map(
        lambda q: qs_map_full.get(q, {}).get("tags_parsed", ""))
    cand["primary_tag"] = cand.ParentId.map(primary_tag)
    cand["question_id"] = cand.ParentId.astype(str)
    cand["answer_id"] = cand.Id.astype(str)
    cand["question_len"] = cand.ParentId.map(
        lambda q: qs_map_full.get(q, {}).get("question_len", 0))

    return cand, qs_map_full, stage


def time_order_audit_pre_disjoint(df: pd.DataFrame, pos_min_score: int,
                                  neg_max_score: int, min_chars: int) -> dict:
    """P(positive posted earlier) over within-question pos/neg pairs, computed
    on the PRE-disjoint candidate pool (the disjoint filter drops exactly these
    mixed-class questions). Identical logic to CR.SE v2."""
    qs = df[df.PostTypeId == 1]
    accepted_ids = set(qs[qs.AcceptedAnswerId > 0]
                       .AcceptedAnswerId.astype(int).tolist())
    ans = df[df.PostTypeId == 2].copy()
    ans = ans.sort_values(["ParentId", "CreationDate", "Id"]).reset_index(drop=True)
    ans["answer_position"] = ans.groupby("ParentId").cumcount() + 1
    ans["accepted"] = ans.Id.isin(accepted_ids)
    ans["answer_text"] = ans.Body.fillna("").map(strip_html)
    ans["answer_len"] = ans.answer_text.str.len()

    is_pos = ans.accepted & (ans.Score >= pos_min_score)
    is_neg = (~ans.accepted) & (ans.Score <= neg_max_score)
    keep = (is_pos | is_neg) & (ans.answer_len >= min_chars)
    sub = ans[keep].copy()
    sub["judgement"] = is_pos[keep].astype(int).values

    pos_pairs = 0
    n_pairs = 0
    n_q_both = 0
    for _qid, g in sub.groupby("ParentId"):
        p = g[g.judgement == 1]
        n = g[g.judgement == 0]
        if p.empty or n.empty:
            continue
        n_q_both += 1
        pp = p["answer_position"].values
        nn = n["answer_position"].values
        pos_pairs += int((pp[:, None] < nn[None, :]).sum())
        n_pairs += int(len(pp) * len(nn))
    return {
        "n_questions_with_both_classes": int(n_q_both),
        "n_pos_neg_pairs": int(n_pairs),
        "p_positive_posted_earlier": (pos_pairs / n_pairs) if n_pairs else float("nan"),
        "note": ("Computed on pre-disjoint candidate pairs (within-question "
                 "pos vs neg) using CreationDate-ordered position rank. The "
                 "disjoint filter removes exactly these mixed-class questions, "
                 "so this MUST be measured before it."),
    }


def position_matched_downsample(cand: pd.DataFrame, n_len_bins: int,
                                seed: int) -> tuple[pd.DataFrame, dict]:
    """Per-tag + (q_len_bin x a_len_bin x position{1,2,3+} x year3) matched
    downsampling. Identical algorithm to Math.SE build_v3_position_matched.py
    balanced_downsample() / CR.SE v2 position_matched_downsample()."""
    rng = random.Random(seed)

    pos = cand[cand.judgement == 1]
    neg = cand[cand.judgement == 0]

    pos_by_tag = {t: g for t, g in pos.groupby("primary_tag")}
    neg_by_tag = {t: g for t, g in neg.groupby("primary_tag")}
    all_tags = set(pos_by_tag) | set(neg_by_tag)

    kept_pos_idx: list[int] = []
    kept_neg_idx: list[int] = []
    n_one_class_tags = 0
    cell_counts: list[dict] = []
    cells_with_overlap = 0
    cells_dropped_no_overlap = 0

    for tag in sorted(all_tags):
        tp = pos_by_tag.get(tag)
        tn = neg_by_tag.get(tag)
        if tp is None or tn is None or len(tp) == 0 or len(tn) == 0:
            n_one_class_tags += 1
            continue

        small_is_pos = len(tp) <= len(tn)
        smaller = tp if small_is_pos else tn
        larger = tn if small_is_pos else tp

        q_edges = np.percentile(smaller["question_len"].values,
                                np.linspace(0, 100, n_len_bins + 1))
        a_edges = np.percentile(smaller["answer_len"].values,
                                np.linspace(0, 100, n_len_bins + 1))
        q_edges[-1] += 1
        a_edges[-1] += 1

        def joint_bin_vec(d: pd.DataFrame):
            qb = np.searchsorted(q_edges[1:], d["question_len"].values)
            ab = np.searchsorted(a_edges[1:], d["answer_len"].values)
            pb = np.minimum(d["answer_position"].values, 3)
            yb = d["answer_year"].values // 3
            return list(zip(qb.tolist(), ab.tolist(), pb.tolist(), yb.tolist()))

        smaller = smaller.copy()
        larger = larger.copy()
        smaller["_cell"] = joint_bin_vec(smaller)
        larger["_cell"] = joint_bin_vec(larger)

        small_by_cell = {cell: g.index.tolist()
                         for cell, g in smaller.groupby("_cell")}
        large_by_cell = {cell: g.index.tolist()
                         for cell, g in larger.groupby("_cell")}

        for cell, s_idx in small_by_cell.items():
            l_idx = large_by_cell.get(cell)
            if not l_idx:
                cells_dropped_no_overlap += 1
                continue
            cells_with_overlap += 1
            n_take = min(len(s_idx), len(l_idx))
            sampled_s = rng.sample(s_idx, n_take)
            sampled_l = rng.sample(l_idx, n_take)
            if small_is_pos:
                kept_pos_idx.extend(sampled_s)
                kept_neg_idx.extend(sampled_l)
            else:
                kept_neg_idx.extend(sampled_s)
                kept_pos_idx.extend(sampled_l)
            cell_counts.append({
                "tag": tag, "cell": str(cell),
                "small_avail": int(len(s_idx)),
                "large_avail": int(len(l_idx)),
                "kept_per_class": int(n_take),
            })

    matched = pd.concat([cand.loc[kept_pos_idx], cand.loc[kept_neg_idx]])
    matched = matched.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    cell_sizes = [c["kept_per_class"] for c in cell_counts]
    summary = {
        "tags_processed": int(len(all_tags) - n_one_class_tags),
        "tags_skipped_one_class": int(n_one_class_tags),
        "cells_with_overlap": int(cells_with_overlap),
        "cells_dropped_no_overlap": int(cells_dropped_no_overlap),
        "kept_per_class_quantiles": {
            "min": int(min(cell_sizes)) if cell_sizes else 0,
            "p25": float(np.percentile(cell_sizes, 25)) if cell_sizes else 0.0,
            "p50": float(np.percentile(cell_sizes, 50)) if cell_sizes else 0.0,
            "p75": float(np.percentile(cell_sizes, 75)) if cell_sizes else 0.0,
            "p99": float(np.percentile(cell_sizes, 99)) if cell_sizes else 0.0,
            "max": int(max(cell_sizes)) if cell_sizes else 0,
        },
        "total_rows_matched": int(len(matched)),
        "pos_rate": float(matched.judgement.mean()) if len(matched) else float("nan"),
        "position_dist": {
            "positive": position_distribution(matched[matched.judgement == 1]),
            "negative": position_distribution(matched[matched.judgement == 0]),
        },
    }
    return matched, summary


def write_pool(matched: pd.DataFrame, qs_map: dict, out_path: Path,
               seed: int) -> dict:
    matched = matched.copy()

    def _make_text(row):
        q = qs_map.get(int(row.question_id), {})
        title = q.get("title", "")
        body = q.get("body_stripped", "")
        question = (title + "\n\n" + body).strip()
        return f"Question: {question}\n\nAnswer: {row.answer_text}"

    matched["text"] = matched.apply(_make_text, axis=1)
    matched["split"] = matched["question_id"].map(split_of)

    out_cols = [
        "text", "judgement", "split", "question_id", "answer_id",
        "answer_position", "n_answers_on_question", "answer_age_gap_days",
        "answer_year", "score", "accepted", "primary_tag", "question_tags",
        "answer_text_norm",
    ]
    out_df = matched.rename(columns={"Score": "score"})[out_cols]
    out_df = out_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, compression="gzip")

    splits = {
        s: {
            "rows": int((out_df.split == s).sum()),
            "positive": int(((out_df.split == s) & (out_df.judgement == 1)).sum()),
            "negative": int(((out_df.split == s) & (out_df.judgement == 0)).sum()),
        } for s in ("train", "eval", "test")
    }
    return {"rows": int(len(out_df)), "pos_rate": float(out_df.judgement.mean()),
            "splits": splits}


# -- driver ------------------------------------------------------------------

def _hash_splits() -> str:
    p = Path(__file__).parent / "splits.py"
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _neg_max_attempts(args) -> list[int]:
    if args.no_fallback:
        return [args.neg_max_score]
    if args.neg_max_score == 0:
        return [0, 1]
    return [args.neg_max_score]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    base = Path("/lfs/skampere3/0/alexspan/norm-research/datasets/law_se")
    ap.add_argument("--posts-xml", default=str(base / "raw_dump" / "Posts.xml"))
    ap.add_argument("--pos-min-score", type=int, default=3)
    ap.add_argument("--neg-max-score", type=int, default=0,
                    help="Strict recipe default (Math.SE used 0). Auto-falls "
                         "back to 1 if matched pool < --min-pool-rows.")
    ap.add_argument("--min-pool-rows", type=int, default=4000,
                    help="If matched pool < this with neg_max_score=0, fall "
                         "back to neg_max_score=1. SMALL site -> lower than "
                         "CR.SE's 15K.")
    ap.add_argument("--min-chars", type=int, default=50)
    ap.add_argument("--n-len-bins", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(base / "built" / "law_se_pool.csv.gz"))
    ap.add_argument("--manifest",
                    default=str(base / "built" / "law_se_pool.manifest.json"))
    ap.add_argument("--no-fallback", action="store_true")
    args = ap.parse_args()

    df_all = load_posts_xml(args.posts_xml)

    manifest: dict = {
        "args": vars(args),
        "build_date": datetime.now().isoformat(timespec="seconds"),
        "source": args.posts_xml,
        "splits_module_hash": _hash_splits(),
        "n_xml_rows_q_plus_a": int(len(df_all)),
    }

    attempts: list[dict] = []
    chosen = None
    for attempt_neg_max in _neg_max_attempts(args):
        print(f"\n=== ATTEMPT: neg_max_score = {attempt_neg_max} ===", flush=True)
        cand, qs_map, stage = build_labeled_pool(
            df_all, args.pos_min_score, attempt_neg_max, args.min_chars)
        audit = time_order_audit_pre_disjoint(
            df_all, args.pos_min_score, attempt_neg_max, args.min_chars)
        matched, matching_summary = position_matched_downsample(
            cand, n_len_bins=args.n_len_bins, seed=args.seed)
        attempt_log = {
            "neg_max_score": attempt_neg_max,
            "labeling": stage,
            "time_order_audit": audit,
            "matching": matching_summary,
            "matched_rows": int(len(matched)),
        }
        attempts.append(attempt_log)
        if len(matched) >= args.min_pool_rows:
            chosen = (cand, qs_map, matched, attempt_log)
            break
        print(f"[fallback] matched={len(matched)} < min_pool_rows="
              f"{args.min_pool_rows}", flush=True)
    if chosen is None:
        chosen = (cand, qs_map, matched, attempt_log)
        attempt_log["forced_use_below_min"] = True

    cand, qs_map, matched, attempt_log = chosen
    manifest["attempts"] = attempts
    manifest["chosen"] = {
        "neg_max_score": attempt_log["neg_max_score"],
        "matched_rows": attempt_log["matched_rows"],
        "fallback_triggered": (attempt_log["neg_max_score"] != args.neg_max_score),
    }

    write_stats = write_pool(matched, qs_map, Path(args.out), seed=args.seed)
    manifest["pool"] = write_stats

    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\n[{datetime.now():%H:%M:%S}] wrote pool {args.out} "
          f"({write_stats['rows']} rows; neg_max_score="
          f"{manifest['chosen']['neg_max_score']})", flush=True)
    print(f"  manifest -> {args.manifest}", flush=True)


if __name__ == "__main__":
    main()
