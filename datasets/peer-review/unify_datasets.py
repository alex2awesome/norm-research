#!/usr/bin/env python3
"""
Unify all peer-review datasets into a common schema.

Produces two output files:
  - unified_papers.csv.gz   — one row per paper
  - unified_reviews.csv.gz  — one row per review (linked to papers by paper_id)

Datasets processed:
  1. PeerRead OpenReview (ICLR, NeurIPS, ICML, COLM, TMLR)
  2. PeerRead legacy (ACL 2017, CoNLL 2016, ICLR 2017)
  3. eLife
  4. F1000Research

Skipped (no review text): ORB metadata-only, NeurIPS 2013-2017 PDFs, arXiv,
  Berenslab (overlaps with OpenReview ICLR), Seafoodair (analysis artifacts).

Usage:
    python unify_datasets.py [--output-dir OUTPUT_DIR]
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Domain classification
# ---------------------------------------------------------------------------
VENUE_DOMAIN = {
    "iclr": "AI/ML",
    "neurips": "AI/ML",
    "icml": "AI/ML",
    "colm": "AI/ML",
    "tmlr": "AI/ML",
    "acl_2017": "NLP",
    "conll_2016": "NLP",
    "iclr_2017": "AI/ML",
    "elife": "Life Sciences",
    "f1000research": "Multidisciplinary",
}

# ---------------------------------------------------------------------------
# Score normalization — map venue-specific scales to [0, 1]
# ---------------------------------------------------------------------------
# ICLR: typically 1-10 (but text-prefixed like "5: marginally below...")
# NeurIPS: 1-10
# ACL: 1-5
# TMLR: binary Yes/No or no score
# COLM: appears 1-10
SCORE_RANGES = {
    "iclr": (1, 10),
    "neurips": (1, 10),
    "icml": (1, 10),
    "colm": (1, 10),
    "tmlr": (1, 10),
    "acl_2017": (1, 5),
    "conll_2016": (1, 5),
    "iclr_2017": (1, 10),
}


def normalize_score(raw_value, venue_key):
    """Extract numeric value and normalize to [0, 1]."""
    if raw_value is None:
        return None, None
    # Handle string scores like "5: marginally below the acceptance threshold"
    if isinstance(raw_value, str):
        m = re.match(r"(\d+(?:\.\d+)?)", raw_value.strip())
        if m:
            num = float(m.group(1))
        elif raw_value.strip().lower() in ("yes", "true"):
            return 1.0, 1.0
        elif raw_value.strip().lower() in ("no", "false"):
            return 0.0, 0.0
        else:
            return None, None
    elif isinstance(raw_value, (int, float)):
        num = float(raw_value)
    else:
        return None, None

    lo, hi = SCORE_RANGES.get(venue_key, (1, 10))
    if hi == lo:
        return num, 0.5
    normalized = (num - lo) / (hi - lo)
    return num, round(max(0.0, min(1.0, normalized)), 4)


# ---------------------------------------------------------------------------
# Decision unification
# ---------------------------------------------------------------------------
def unify_decision(raw_decision, accepted_flag=None, source="openreview"):
    """Map dataset-specific decisions to a unified label."""
    if accepted_flag is True:
        return "accept"
    if accepted_flag is False:
        return "reject"

    if raw_decision is None:
        return "no_decision"

    d = str(raw_decision).strip().lower()

    # Accept variants
    if d in ("accept", "accepted", "accept (oral)", "accept (spotlight)",
             "accept (poster)", "accept (notable-top-5%)",
             "accept (notable-top-25%)"):
        return "accept"
    if "accept" in d and "not" not in d and "reserv" not in d:
        return "accept"

    # Reject variants
    if d in ("reject", "rejected", "withdrawn", "desk reject",
             "not approved", "not_approved"):
        return "reject"
    if "reject" in d:
        return "reject"

    # Conditional / reservations
    if "approved with reservations" in d or "conditional" in d:
        return "conditional"
    if d in ("approved", "approve"):
        return "accept"

    # eLife assessment-only
    if source == "elife":
        return "assessment_only"

    return "no_decision"


# ---------------------------------------------------------------------------
# Loaders for each dataset
# ---------------------------------------------------------------------------

def load_openreview_venue(venue_key, venue_dir):
    """Load a PeerRead OpenReview venue (conference with year dirs or TMLR flat)."""
    papers = []
    reviews = []

    # Determine if year-based or flat (TMLR)
    review_dirs = []
    reviews_direct = venue_dir / "reviews"
    if reviews_direct.exists():
        # Flat layout (TMLR)
        review_dirs.append((None, reviews_direct))
    else:
        # Year-based layout
        for year_dir in sorted(venue_dir.iterdir()):
            if year_dir.is_dir() and year_dir.name.isdigit():
                rd = year_dir / "reviews"
                if rd.exists():
                    review_dirs.append((int(year_dir.name), rd))

    for year, rd in review_dirs:
        for fpath in rd.glob("*.json"):
            try:
                data = json.loads(fpath.read_text())
            except Exception:
                continue

            paper_id = f"{venue_key}_{data.get('id', fpath.stem)}"
            raw_decision = data.get("decision", "")
            accepted = data.get("accepted")

            # Extract year from conference string if not from directory
            paper_year = year
            if paper_year is None:
                conf = data.get("conference", "")
                m = re.search(r"20\d{2}", conf)
                if m:
                    paper_year = int(m.group())

            venue_display = venue_key.upper()
            if paper_year:
                venue_display = f"{venue_key.upper()} {paper_year}"

            papers.append({
                "paper_id": paper_id,
                "source": "peerread_openreview",
                "venue": venue_display,
                "venue_key": venue_key,
                "year": paper_year,
                "domain": VENUE_DOMAIN.get(venue_key, "AI/ML"),
                "title": data.get("title", ""),
                "abstract": data.get("abstract", ""),
                "authors": "; ".join(data.get("authors", [])) if isinstance(data.get("authors"), list) else str(data.get("authors", "")),
                "keywords": "; ".join(data.get("keywords", [])) if isinstance(data.get("keywords"), list) else "",
                "decision_raw": str(raw_decision),
                "decision_unified": unify_decision(raw_decision, accepted),
            })

            for i, rev in enumerate(data.get("reviews", [])):
                is_meta = bool(rev.get("IS_META_REVIEW", False))
                rec_raw = rev.get("RECOMMENDATION")
                rec_num, rec_norm = normalize_score(rec_raw, venue_key)
                conf_raw = rev.get("REVIEWER_CONFIDENCE")
                conf_num, conf_norm = normalize_score(conf_raw, venue_key)

                reviews.append({
                    "review_id": f"{paper_id}_r{i}",
                    "paper_id": paper_id,
                    "review_text": rev.get("comments", ""),
                    "is_meta_review": is_meta,
                    "recommendation_raw": str(rec_raw) if rec_raw is not None else "",
                    "recommendation_numeric": rec_num,
                    "recommendation_normalized": rec_norm,
                    "confidence_raw": str(conf_raw) if conf_raw is not None else "",
                    "confidence_numeric": conf_num,
                    "confidence_normalized": conf_norm,
                    "reviewer_name": "",
                    "has_author_response": False,
                    "author_response_text": "",
                })

    return papers, reviews


def load_legacy_venue(venue_key, venue_dir):
    """Load a PeerRead legacy venue (ACL, CoNLL, ICLR 2017)."""
    papers = []
    reviews = []

    # Legacy venues have train/dev/test splits
    for split_dir in venue_dir.iterdir():
        if not split_dir.is_dir():
            continue
        reviews_dir = split_dir / "reviews"
        if not reviews_dir.exists():
            continue

        for fpath in reviews_dir.glob("*.json"):
            try:
                data = json.loads(fpath.read_text())
            except Exception:
                continue

            paper_id = f"{venue_key}_{data.get('id', fpath.stem)}"

            # Legacy ACL/CoNLL don't have explicit accept/reject in the JSON;
            # acceptance is inferred from the train/dev/test split structure.
            # ICLR 2017 has an "accepted" field.
            accepted = data.get("accepted")
            raw_decision = data.get("decision", "")

            # Extract year from venue key
            year_match = re.search(r"(\d{4})", venue_key)
            paper_year = int(year_match.group(1)) if year_match else None

            venue_display = venue_key.upper().replace("_", " ")

            papers.append({
                "paper_id": paper_id,
                "source": "peerread_legacy",
                "venue": venue_display,
                "venue_key": venue_key,
                "year": paper_year,
                "domain": VENUE_DOMAIN.get(venue_key, "AI/ML"),
                "title": data.get("title", ""),
                "abstract": data.get("abstract", ""),
                "authors": "; ".join(data.get("authors", [])) if isinstance(data.get("authors"), list) else str(data.get("authors", "")),
                "keywords": "",
                "decision_raw": str(raw_decision) if raw_decision else ("accepted" if accepted else "rejected" if accepted is False else ""),
                "decision_unified": unify_decision(raw_decision, accepted),
            })

            for i, rev in enumerate(data.get("reviews", [])):
                is_meta = bool(rev.get("IS_META_REVIEW") or rev.get("is_meta_review"))
                rec_raw = rev.get("RECOMMENDATION")
                rec_num, rec_norm = normalize_score(rec_raw, venue_key)
                conf_raw = rev.get("REVIEWER_CONFIDENCE")
                conf_num, conf_norm = normalize_score(conf_raw, venue_key)

                reviews.append({
                    "review_id": f"{paper_id}_r{i}",
                    "paper_id": paper_id,
                    "review_text": rev.get("comments", ""),
                    "is_meta_review": is_meta,
                    "recommendation_raw": str(rec_raw) if rec_raw is not None else "",
                    "recommendation_numeric": rec_num,
                    "recommendation_normalized": rec_norm,
                    "confidence_raw": str(conf_raw) if conf_raw is not None else "",
                    "confidence_numeric": conf_num,
                    "confidence_normalized": conf_norm,
                    "reviewer_name": "",
                    "has_author_response": False,
                    "author_response_text": "",
                })

    return papers, reviews


def load_elife():
    """Load eLife reviewed preprints."""
    data_dir = BASE / "elife" / "data"
    if not data_dir.exists():
        return [], []

    papers = []
    reviews = []

    for fpath in sorted(data_dir.glob("*.json")):
        try:
            data = json.loads(fpath.read_text())
        except Exception:
            continue

        paper_id = f"elife_{data.get('id', fpath.stem)}"

        # Extract year from published or reviewed_date
        pub_date = data.get("published", data.get("reviewed_date", ""))
        year_match = re.search(r"(20\d{2})", pub_date)
        paper_year = int(year_match.group(1)) if year_match else None

        # eLife assessment as decision proxy
        assessment = data.get("elife_assessment", {})
        significance = assessment.get("significance", [])
        strength = assessment.get("strength", [])
        decision_raw = ""
        if significance or strength:
            sig_str = ", ".join(significance) if significance else ""
            str_str = ", ".join(strength) if strength else ""
            decision_raw = f"significance: {sig_str}; strength: {str_str}"

        subjects = data.get("subjects", [])

        papers.append({
            "paper_id": paper_id,
            "source": "elife",
            "venue": "eLife",
            "venue_key": "elife",
            "year": paper_year,
            "domain": "Life Sciences",
            "title": data.get("title", ""),
            "abstract": "",  # eLife data doesn't include abstracts
            "authors": data.get("author_line", ""),
            "keywords": "; ".join(subjects),
            "decision_raw": decision_raw,
            "decision_unified": "assessment_only",
        })

        # Regular reviews
        has_response = data.get("has_author_response", False)
        response_text = data.get("author_response", "")

        for i, rev in enumerate(data.get("reviews", [])):
            reviews.append({
                "review_id": f"{paper_id}_r{i}",
                "paper_id": paper_id,
                "review_text": rev.get("text", ""),
                "is_meta_review": False,
                "recommendation_raw": "",
                "recommendation_numeric": None,
                "recommendation_normalized": None,
                "confidence_raw": "",
                "confidence_numeric": None,
                "confidence_normalized": None,
                "reviewer_name": rev.get("reviewer", ""),
                "has_author_response": has_response,
                "author_response_text": response_text if i == 0 else "",  # attach once
            })

        # eLife assessment as a meta-review
        if assessment.get("text"):
            reviews.append({
                "review_id": f"{paper_id}_assessment",
                "paper_id": paper_id,
                "review_text": assessment["text"],
                "is_meta_review": True,
                "recommendation_raw": decision_raw,
                "recommendation_numeric": None,
                "recommendation_normalized": None,
                "confidence_raw": "",
                "confidence_numeric": None,
                "confidence_normalized": None,
                "reviewer_name": "eLife Assessment",
                "has_author_response": False,
                "author_response_text": "",
            })

    return papers, reviews


def load_f1000research():
    """Load F1000Research articles with open peer review."""
    data_dir = BASE / "f1000research" / "data"
    if not data_dir.exists():
        return [], []

    papers = []
    reviews = []

    for fpath in sorted(data_dir.glob("*.json")):
        try:
            data = json.loads(fpath.read_text())
        except Exception:
            continue

        paper_id = f"f1000_{data.get('article_id', fpath.stem)}_v{data.get('version', 1)}"

        # Extract year from pub_date
        pub_date = data.get("pub_date", "")
        year_match = re.search(r"(20\d{2})", pub_date)
        paper_year = int(year_match.group(1)) if year_match else None

        # Authors
        authors_list = data.get("authors", [])
        if isinstance(authors_list, list):
            author_names = []
            for a in authors_list:
                if isinstance(a, dict):
                    name = a.get("name", "")
                    if not name:
                        gn = a.get("given_names", "")
                        sn = a.get("surname", "")
                        name = f"{gn} {sn}".strip()
                    author_names.append(name)
                else:
                    author_names.append(str(a))
            authors_str = "; ".join(author_names)
        else:
            authors_str = str(authors_list)

        # Aggregate decisions from reviewer reports
        report_decisions = []
        reports = data.get("reviewer_reports", [])
        for r in reports:
            rec = r.get("recommendation", r.get("recommendation_raw", ""))
            if rec:
                report_decisions.append(rec)

        if report_decisions:
            decision_raw = "; ".join(report_decisions)
            # Overall decision: worst review wins
            if any("not" in d.lower() for d in report_decisions):
                decision_unified = "reject"
            elif all("approved" == d.lower() or "approve" == d.lower() for d in report_decisions):
                decision_unified = "accept"
            elif any("reserv" in d.lower() for d in report_decisions):
                decision_unified = "conditional"
            else:
                decision_unified = unify_decision(report_decisions[0])
        else:
            decision_raw = ""
            decision_unified = "no_decision"

        papers.append({
            "paper_id": paper_id,
            "source": "f1000research",
            "venue": "F1000Research",
            "venue_key": "f1000research",
            "year": paper_year,
            "domain": "Multidisciplinary",
            "title": data.get("title", ""),
            "abstract": data.get("abstract", ""),
            "authors": authors_str,
            "keywords": "; ".join(data.get("keywords", [])) if isinstance(data.get("keywords"), list) else "",
            "decision_raw": decision_raw,
            "decision_unified": decision_unified,
        })

        for i, rep in enumerate(reports):
            has_response = bool(rep.get("author_response"))
            response_text = ""
            if has_response and isinstance(rep["author_response"], dict):
                response_text = rep["author_response"].get("text", "")

            rec = rep.get("recommendation", rep.get("recommendation_raw", ""))

            reviews.append({
                "review_id": f"{paper_id}_r{i}",
                "paper_id": paper_id,
                "review_text": rep.get("text", ""),
                "is_meta_review": False,
                "recommendation_raw": str(rec),
                "recommendation_numeric": None,
                "recommendation_normalized": None,
                "confidence_raw": "",
                "confidence_numeric": None,
                "confidence_normalized": None,
                "reviewer_name": rep.get("reviewer_name", ""),
                "has_author_response": has_response,
                "author_response_text": response_text,
            })

    return papers, reviews


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_stats(papers_df, reviews_df):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("  UNIFIED DATASET STATISTICS")
    print("=" * 70)

    print(f"\nTotal papers:  {len(papers_df):,}")
    print(f"Total reviews: {len(reviews_df):,}")

    # By source
    print("\n--- By Source ---")
    src = papers_df.groupby("source").agg(
        papers=("paper_id", "count"),
    ).reset_index()
    src_rev = reviews_df.merge(papers_df[["paper_id", "source"]], on="paper_id")
    src_rev_counts = src_rev.groupby("source").size().reset_index(name="reviews")
    src = src.merge(src_rev_counts, on="source", how="left").fillna(0)
    print(src.to_string(index=False))

    # By venue and domain
    print("\n--- By Venue & Domain ---")
    venue = papers_df.groupby(["domain", "venue"]).agg(
        papers=("paper_id", "count"),
    ).reset_index().sort_values(["domain", "papers"], ascending=[True, False])
    print(venue.to_string(index=False))

    # By decision
    print("\n--- By Decision (unified) ---")
    dec = papers_df.groupby("decision_unified").size().reset_index(name="count")
    dec["pct"] = (dec["count"] / len(papers_df) * 100).round(1)
    print(dec.sort_values("count", ascending=False).to_string(index=False))

    # Accept/reject by venue
    print("\n--- Accept/Reject by Venue ---")
    ar = papers_df[papers_df["decision_unified"].isin(["accept", "reject"])]
    if len(ar) > 0:
        ar_venue = ar.groupby(["venue", "decision_unified"]).size().unstack(fill_value=0)
        if "accept" in ar_venue.columns and "reject" in ar_venue.columns:
            ar_venue["total"] = ar_venue.sum(axis=1)
            ar_venue["accept_rate"] = (ar_venue["accept"] / ar_venue["total"] * 100).round(1)
            print(ar_venue.sort_values("total", ascending=False).to_string())

    # By year
    print("\n--- By Year ---")
    yr = papers_df.dropna(subset=["year"])
    yr_counts = yr.groupby("year").agg(
        papers=("paper_id", "count"),
    ).reset_index()
    yr_counts["year"] = yr_counts["year"].astype(int)
    print(yr_counts.sort_values("year").to_string(index=False))

    # Reviews with text
    has_text = reviews_df["review_text"].str.len() > 0
    print(f"\nReviews with text: {has_text.sum():,} / {len(reviews_df):,} ({has_text.mean()*100:.1f}%)")
    meta = reviews_df["is_meta_review"].sum()
    print(f"Meta-reviews:      {meta:,}")
    has_score = reviews_df["recommendation_numeric"].notna().sum()
    print(f"With numeric score: {has_score:,}")
    has_response = reviews_df["has_author_response"].sum()
    print(f"With author response: {has_response:,}")


def main():
    parser = argparse.ArgumentParser(description="Unify peer-review datasets")
    parser.add_argument("--output-dir", type=str, default=str(BASE),
                        help="Directory to write output files")
    parser.add_argument("--stats-only", action="store_true",
                        help="Print statistics without writing files")
    args = parser.parse_args()

    all_papers = []
    all_reviews = []

    # 1. OpenReview venues
    openreview_dir = BASE / "PeerRead" / "data" / "openreview"
    if openreview_dir.exists():
        for venue_name in ["iclr", "neurips", "icml", "colm", "tmlr"]:
            venue_dir = openreview_dir / venue_name
            if venue_dir.exists():
                print(f"Loading OpenReview {venue_name}...")
                p, r = load_openreview_venue(venue_name, venue_dir)
                print(f"  -> {len(p):,} papers, {len(r):,} reviews")
                all_papers.extend(p)
                all_reviews.extend(r)

    # 2. Legacy PeerRead venues
    legacy_venues = {
        "acl_2017": BASE / "PeerRead" / "data" / "acl_2017",
        "conll_2016": BASE / "PeerRead" / "data" / "conll_2016",
        "iclr_2017": BASE / "PeerRead" / "data" / "iclr_2017",
    }
    for venue_key, venue_dir in legacy_venues.items():
        if venue_dir.exists():
            print(f"Loading legacy {venue_key}...")
            p, r = load_legacy_venue(venue_key, venue_dir)
            print(f"  -> {len(p):,} papers, {len(r):,} reviews")
            all_papers.extend(p)
            all_reviews.extend(r)

    # 3. eLife
    print("Loading eLife...")
    p, r = load_elife()
    print(f"  -> {len(p):,} papers, {len(r):,} reviews")
    all_papers.extend(p)
    all_reviews.extend(r)

    # 4. F1000Research
    print("Loading F1000Research...")
    p, r = load_f1000research()
    print(f"  -> {len(p):,} papers, {len(r):,} reviews")
    all_papers.extend(p)
    all_reviews.extend(r)

    # Build DataFrames
    papers_df = pd.DataFrame(all_papers)
    reviews_df = pd.DataFrame(all_reviews)

    # Add num_reviews to papers
    rev_counts = reviews_df.groupby("paper_id").size().reset_index(name="num_reviews")
    papers_df = papers_df.merge(rev_counts, on="paper_id", how="left")
    papers_df["num_reviews"] = papers_df["num_reviews"].fillna(0).astype(int)

    # Print stats
    print_stats(papers_df, reviews_df)

    # Write output
    if not args.stats_only:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        papers_out = out_dir / "unified_papers.csv.gz"
        reviews_out = out_dir / "unified_reviews.csv.gz"

        print(f"\nWriting {papers_out}...")
        papers_df.to_csv(papers_out, index=False, compression="gzip")
        print(f"Writing {reviews_out}...")
        reviews_df.to_csv(reviews_out, index=False, compression="gzip")

        print(f"\nDone! Output files:")
        print(f"  {papers_out} ({papers_out.stat().st_size / 1e6:.1f} MB)")
        print(f"  {reviews_out} ({reviews_out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
