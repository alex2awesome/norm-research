#!/usr/bin/env python3
"""
Analyze all peer review datasets we have, showing coverage and gaps.

Scans:
  - PeerRead/data/openreview/iclr/ (crawled ICLR data)
  - orb-dataset/ (ORB from Zenodo)
  - f1000research/ (URL list)
  - elife/ (API metadata)

Usage:
    python analyze_all_data.py
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent


def analyze_iclr_crawl():
    """Analyze our existing ICLR crawled data."""
    iclr_dir = BASE / "PeerRead" / "data" / "openreview" / "iclr"
    if not iclr_dir.exists():
        return None

    results = {}
    for year_dir in sorted(iclr_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        year = year_dir.name
        reviews_dir = year_dir / "reviews"
        if not reviews_dir.exists():
            continue

        stats = {
            "total": 0,
            "accepted": 0,
            "rejected": 0,
            "no_decision": 0,
            "with_reviews": 0,
            "total_reviews": 0,
            "with_meta_review": 0,
        }

        for f in reviews_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
            except Exception:
                continue

            stats["total"] += 1
            accepted = data.get("accepted")
            if accepted is True:
                stats["accepted"] += 1
            elif accepted is False:
                stats["rejected"] += 1
            else:
                stats["no_decision"] += 1

            reviews = data.get("reviews", [])
            non_meta = [r for r in reviews if not r.get("IS_META_REVIEW")]
            meta = [r for r in reviews if r.get("IS_META_REVIEW")]

            if non_meta:
                stats["with_reviews"] += 1
            stats["total_reviews"] += len(non_meta)
            if meta:
                stats["with_meta_review"] += 1

        results[year] = stats

    return results


def analyze_orb():
    """Analyze the ORB dataset JSON."""
    orb_file = BASE / "orb-dataset" / "OrbDataset_mainfile.json"
    if not orb_file.exists():
        return None

    print("  Loading ORB JSON (may take a moment)...")
    try:
        data = json.loads(orb_file.read_text())
    except Exception as e:
        print(f"  Error loading ORB: {e}")
        return None

    # ORB structure: list of submissions or dict with submissions
    submissions = []
    if isinstance(data, list):
        submissions = data
    elif isinstance(data, dict):
        # Try common keys
        for key in ["submissions", "data", "items", "papers"]:
            if key in data:
                submissions = data[key]
                break
        if not submissions:
            # Maybe the whole dict is keyed by ID
            print(f"  ORB top-level keys: {list(data.keys())[:10]}")
            return {"_raw_keys": list(data.keys())[:20], "_total_keys": len(data)}

    venue_stats = defaultdict(lambda: {
        "total": 0, "accepted": 0, "rejected": 0,
        "no_decision": 0, "total_reviews": 0
    })

    for sub in submissions:
        # Try to extract venue
        venue = None
        if isinstance(sub, dict):
            venue = sub.get("venue", sub.get("conference", sub.get("venue_id", "unknown")))
            if isinstance(venue, dict):
                venue = venue.get("name", venue.get("id", "unknown"))

            decision = sub.get("decision", sub.get("decision_flag"))
            reviews = sub.get("reviews", [])

            v = venue_stats[venue]
            v["total"] += 1
            v["total_reviews"] += len(reviews) if isinstance(reviews, list) else 0

            if isinstance(decision, bool):
                if decision:
                    v["accepted"] += 1
                else:
                    v["rejected"] += 1
            elif isinstance(decision, str):
                dl = decision.lower()
                if "accept" in dl:
                    v["accepted"] += 1
                elif "reject" in dl:
                    v["rejected"] += 1
                else:
                    v["no_decision"] += 1
            else:
                v["no_decision"] += 1

    return dict(venue_stats)


def analyze_f1000():
    """Analyze F1000Research URL list."""
    url_file = BASE / "f1000research" / "xml_urls_unique.txt"
    if not url_file.exists():
        return None

    urls = url_file.read_text().strip().split("\n")

    # Parse article IDs and versions
    articles = defaultdict(list)
    for url in urls:
        # URL format: https://f1000research.com/articles/8-53/v3/xml
        parts = url.split("/")
        try:
            article_id = parts[4]  # e.g., "8-53"
            version = parts[5]     # e.g., "v3"
            articles[article_id].append(version)
        except IndexError:
            continue

    # Count by volume (first part of article ID)
    volume_counts = Counter()
    for article_id in articles:
        vol = article_id.split("-")[0]
        volume_counts[vol] += 1

    return {
        "total_unique_urls": len(urls),
        "unique_articles": len(articles),
        "articles_with_multiple_versions": sum(1 for vs in articles.values() if len(vs) > 1),
        "by_volume": dict(sorted(volume_counts.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0)),
    }


def analyze_elife():
    """Check eLife API availability."""
    # Just report what the API offers
    return {
        "api_base": "https://api.elifesciences.org",
        "estimated_articles": 18961,
        "note": "API accessible, not yet downloaded. Has reviewed preprints with public reviews (post-2023 model).",
    }


def print_section(title, char="="):
    print(f"\n{char * 60}")
    print(f"  {title}")
    print(f"{char * 60}")


def main():
    print_section("PEER REVIEW DATA INVENTORY")

    # 1. ICLR crawled data
    print_section("1. ICLR (OpenReview Crawl)", "-")
    iclr = analyze_iclr_crawl()
    if iclr:
        total_papers = sum(s["total"] for s in iclr.values())
        total_reviews = sum(s["total_reviews"] for s in iclr.values())
        total_accepted = sum(s["accepted"] for s in iclr.values())
        total_rejected = sum(s["rejected"] for s in iclr.values())

        print(f"{'Year':<8} {'Papers':>8} {'Reviews':>8} {'Accept':>8} {'Reject':>8} {'No Dec':>8} {'Acc%':>8}")
        print("-" * 60)
        for year in sorted(iclr.keys()):
            s = iclr[year]
            acc_pct = f"{100*s['accepted']/s['total']:.1f}%" if s['total'] else "N/A"
            print(f"{year:<8} {s['total']:>8} {s['total_reviews']:>8} {s['accepted']:>8} {s['rejected']:>8} {s['no_decision']:>8} {acc_pct:>8}")
        print("-" * 60)
        acc_pct = f"{100*total_accepted/total_papers:.1f}%" if total_papers else "N/A"
        print(f"{'TOTAL':<8} {total_papers:>8} {total_reviews:>8} {total_accepted:>8} {total_rejected:>8} {'':>8} {acc_pct:>8}")
    else:
        print("  No ICLR data found.")

    # 2. ORB dataset
    print_section("2. ORB Dataset (Zenodo)", "-")
    orb = analyze_orb()
    if orb:
        if "_raw_keys" in orb:
            print(f"  ORB loaded but structure unclear. Top keys: {orb['_raw_keys']}")
            print(f"  Total top-level keys: {orb['_total_keys']}")
        else:
            total_papers = sum(v["total"] for v in orb.values())
            total_reviews = sum(v["total_reviews"] for v in orb.values())
            print(f"  Total submissions: {total_papers}")
            print(f"  Total reviews: {total_reviews}")
            print(f"  Venues: {len(orb)}")
            print(f"\n{'Venue':<45} {'Papers':>8} {'Reviews':>8} {'Accept':>8} {'Reject':>8}")
            print("-" * 80)
            for venue, stats in sorted(orb.items(), key=lambda x: -x[1]["total"]):
                name = str(venue)[:44]
                print(f"{name:<45} {stats['total']:>8} {stats['total_reviews']:>8} {stats['accepted']:>8} {stats['rejected']:>8}")
    else:
        print("  ORB not yet downloaded or could not be loaded.")

    # 3. F1000Research
    print_section("3. F1000Research (XML Corpus)", "-")
    f1000 = analyze_f1000()
    if f1000:
        print(f"  Unique articles: {f1000['unique_articles']}")
        print(f"  Unique article-version URLs: {f1000['total_unique_urls']}")
        print(f"  Articles with multiple versions: {f1000['articles_with_multiple_versions']}")
        print(f"\n  By volume (year proxy):")
        for vol, count in f1000["by_volume"].items():
            print(f"    Vol {vol}: {count} articles")
    else:
        print("  F1000 data not available.")

    # 4. eLife
    print_section("4. eLife (API)", "-")
    elife = analyze_elife()
    print(f"  API: {elife['api_base']}")
    print(f"  Estimated articles: {elife['estimated_articles']}")
    print(f"  Note: {elife['note']}")

    # 5. Summary
    print_section("SUMMARY & GAPS")

    iclr_total = sum(s["total"] for s in iclr.values()) if iclr else 0
    iclr_reviews = sum(s["total_reviews"] for s in iclr.values()) if iclr else 0

    print(f"""
  Source                    Papers    Reviews   Accept+Reject?  Domain
  ────────────────────────  ────────  ────────  ──────────────  ──────────────
  ICLR crawl (2018-2025)   {iclr_total:>8}  {iclr_reviews:>8}  Yes (both)      AI/ML
  ORB (OpenReview+SciPost)  (loading) (loading) Yes (both)      AI/ML + Physics
  F1000Research             {f1000['unique_articles'] if f1000 else '?':>8}  (in XML)  3-level rating  Multidisciplinary
  eLife                     {elife['estimated_articles']:>8}  (API)     Assessment only  Life sciences

  NOT YET COVERED:
  - NeurIPS (2021+): Run crawler with --years 2021-2025
  - ICML (2023+): Run crawler with --years 2023-2025
  - TMLR: Run crawler with TMLR flag
  - COLM (2024-2025): Accepted papers only
  - Nature Communications: Reviews in supplementary PDFs (hard to bulk-download)
  - PeerJ: Covered by ORB (updated version), or scrape directly
""")


if __name__ == "__main__":
    main()
