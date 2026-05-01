#!/usr/bin/env python3
"""
Build a newsworthiness preference dataset from Internet Archive news homepage snapshots.

Pipeline:
1. Download hyperlinks.json + accessibility.json from archive.org
2. Classify article vs furniture links using trained LogisticRegression
3. Filter to visible articles in top 30% of page
4. Extract richer text from accessibility tree (headline + summary)
5. Label: 1 = top half of 30% zone, 0 = bottom half
6. Output: CSV with text and judgement columns

Usage:
    # Download data for all outlets onto sk3:
    python build_homepage_dataset.py download --outlets nytimes,wsj,latimes,bbc --output-dir raw_data

    # Build dataset:
    python build_homepage_dataset.py build --input-dir raw_data --output homepage_newsworthiness.csv.gz

    # All-in-one:
    python build_homepage_dataset.py all --outlets nytimes,wsj,latimes,bbc
"""
import argparse
import csv
import gzip
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd
import requests

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DEFAULT_OUTLETS = [
    "nytimes", "wsj", "latimes", "bbc",
    "washingtonpost", "cnn", "guardian", "reuters",
]

CLASSIFIER_PATH = "article_classifier.joblib"
TOP_ZONE_FRACTION = 0.30


# ─────────────────────────────────────────────
# STEP 1: DOWNLOAD
# ─────────────────────────────────────────────

def get_item_ids(outlet):
    """Get all Internet Archive item IDs for an outlet."""
    url = (
        "https://archive.org/advancedsearch.php"
        "?q=collection:news-homepages AND identifier:%s*"
        "&output=json&rows=100&fl=identifier"
    ) % outlet
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return sorted(doc["identifier"] for doc in resp.json()["response"]["docs"])


def get_file_list(item_id, suffix):
    """List files with a given suffix in an IA item."""
    url = "https://archive.org/metadata/%s" % item_id
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return [f["name"] for f in resp.json().get("files", []) if f["name"].endswith(suffix)]


def download_file(item_id, filename, output_dir):
    """Download a single file from IA. Returns True if downloaded, False if skipped/failed."""
    outpath = output_dir / filename
    if outpath.exists():
        return False

    dl_url = "https://archive.org/download/%s/%s" % (item_id, requests.utils.quote(filename))
    try:
        resp = requests.get(dl_url, timeout=60)
        resp.raise_for_status()
        outpath.write_bytes(resp.content)
        return True
    except Exception as e:
        print("  Error: %s" % e)
        return False


def download_outlet(outlet, output_dir, max_snapshots=10000):
    """Download hyperlinks.json and accessibility.json for an outlet."""
    out_path = Path(output_dir) / outlet
    out_path.mkdir(parents=True, exist_ok=True)

    existing = set(f.name for f in out_path.glob("*.json"))
    print("[%s] %d files already on disk" % (outlet, len(existing)))

    item_ids = get_item_ids(outlet)
    print("[%s] Found %d archive items: %s" % (outlet, len(item_ids), ", ".join(item_ids)))

    downloaded = 0
    for item_id in item_ids:
        hl_files = get_file_list(item_id, ".hyperlinks.json")
        print("[%s/%s] %d hyperlink snapshots" % (outlet, item_id, len(hl_files)))

        for hl_fname in hl_files:
            if downloaded >= max_snapshots:
                break

            # Derive corresponding accessibility filename
            acc_fname = hl_fname.replace(".hyperlinks.json", ".accessibility.json")

            # Download both
            got_hl = download_file(item_id, hl_fname, out_path)
            got_acc = download_file(item_id, acc_fname, out_path)

            if got_hl or got_acc:
                downloaded += 1
                if downloaded % 50 == 0:
                    print("[%s] Downloaded %d snapshots" % (outlet, downloaded))
                time.sleep(0.1)  # polite

        if downloaded >= max_snapshots:
            break

    print("[%s] Done: %d snapshots downloaded" % (outlet, downloaded))


# ─────────────────────────────────────────────
# STEP 2: PARSE ACCESSIBILITY TREE
# ─────────────────────────────────────────────

def extract_accessibility_text(acc_path):
    """
    Extract article text from accessibility.json.
    Returns dict: url -> {headline, summary} with richer text than hyperlinks.json.
    """
    if not acc_path.exists():
        return {}

    try:
        with open(acc_path) as f:
            tree = json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {}

    # Walk the tree and collect text near links
    url_text = {}

    def walk(node):
        role = node.get("role", "")
        name = node.get("name", "")
        children = node.get("children", [])

        if role == "link" and name and len(name) > 15:
            # The link's name in the accessibility tree often contains
            # headline + summary concatenated
            # Try to find a URL — check children for nested links
            url_text[name] = name

        for child in children:
            walk(child)

    walk(tree)
    return url_text


def match_accessibility_text(hyperlinks, acc_texts):
    """
    Match hyperlink entries with richer accessibility tree text.
    Returns enriched hyperlinks with 'rich_text' field.
    """
    # Build index: short text -> full accessibility text
    # The accessibility tree often has "Headline Summary Text 5 min read"
    # while hyperlinks just has "Headline"
    for link in hyperlinks:
        hl_text = link.get("text", "").strip()
        if not hl_text or len(hl_text) < 10:
            link["rich_text"] = hl_text
            continue

        # Find the best matching accessibility text
        best_match = hl_text
        best_len = len(hl_text)
        for acc_text in acc_texts.values():
            if hl_text in acc_text and len(acc_text) > best_len:
                best_match = acc_text
                best_len = len(acc_text)

        link["rich_text"] = best_match

    return hyperlinks


# ─────────────────────────────────────────────
# STEP 3: CLASSIFY AND LABEL
# ─────────────────────────────────────────────

def parse_timestamp(filename):
    """Extract date string from filename."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    return m.group(1) if m else "unknown"


def process_snapshot(hl_path, acc_path, outlet, classifier):
    """
    Process one snapshot into labeled examples.

    Returns list of {text, judgement} dicts.
    """
    try:
        with open(hl_path) as f:
            links = json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return []

    # Filter to visible links
    visible = [
        l for l in links
        if l.get("top", 0) > 0
        and l.get("bottom", 0) > l.get("top", 0)
        and l.get("right", 0) > l.get("left", 0)
    ]

    if not visible:
        return []

    page_height = max(l["bottom"] for l in visible)
    if page_height <= 0:
        return []

    # Classify article vs furniture using bulk sklearn predict
    rows = []
    for l in visible:
        url = l.get("url", "")
        try:
            parsed = urlparse(url if url.startswith("http") else "https://example.com" + url)
            url_path = parsed.path or "/"
        except Exception:
            url_path = url
        rows.append({
            "url_path": url_path,
            "text": l.get("text", ""),
            "top": l["top"],
            "bottom": l["bottom"],
            "url": url,
            "raw_text": l.get("text", ""),
        })

    df = pd.DataFrame(rows)
    preds = classifier.predict(df[["url_path", "text"]])
    df["is_story"] = preds

    # Filter to articles only
    articles = df[df["is_story"] == 1].copy()
    # Deduplicate by URL path, keeping highest (lowest y) occurrence
    articles = articles.sort_values("top").drop_duplicates(subset=["url_path"])

    # Enrich with accessibility text
    acc_texts = extract_accessibility_text(acc_path)
    if acc_texts:
        for idx, row in articles.iterrows():
            raw = row["raw_text"]
            best = raw
            best_len = len(raw)
            for acc_text in acc_texts.values():
                if raw and raw in acc_text and len(acc_text) > best_len:
                    best = acc_text
                    best_len = len(acc_text)
            articles.at[idx, "rich_text"] = best
    else:
        articles["rich_text"] = articles["raw_text"]

    # Filter to top 30% zone
    cutoff = page_height * TOP_ZONE_FRACTION
    in_zone = articles[articles["top"] < cutoff]

    if len(in_zone) < 4:
        return []

    # Split at midpoint
    midpoint = cutoff / 2.0
    date = parse_timestamp(hl_path.name)

    # Collect all cleaned headlines for context building
    import random as _rng
    all_headlines_raw = in_zone["raw_text"].tolist()
    all_headlines_clean = []
    for h in all_headlines_raw:
        h = h.strip()[:80] if h else ""
        if h:
            all_headlines_clean.append(h)

    examples = []
    for _, row in in_zone.iterrows():
        headline = row.get("rich_text", row["raw_text"]).strip()
        if not headline or len(headline) < 15:
            continue

        # Clean up: remove "X min read" suffixes, image tags, HTML fragments
        headline = re.sub(r"\d+\s*min read", "", headline).strip()
        headline = re.sub(r"<img[^>]*>", "", headline).strip()
        headline = re.sub(r"<[^>]+>", "", headline).strip()
        if not headline or len(headline) < 15:
            continue

        label = 1 if row["top"] < midpoint else 0

        # Build context: other headlines EXCLUDING this one, SHUFFLED
        # This prevents position-in-context from leaking the label
        headline_prefix = headline[:50]
        other_headlines = [h for h in all_headlines_clean if headline_prefix not in h]
        _rng.shuffle(other_headlines)
        context = "; ".join(other_headlines)

        text = "HEADLINE: %s\n\nCONTEXT: %s" % (
            headline,
            context,
        )
        examples.append({"text": text, "judgement": label, "outlet": outlet})

    return examples


# ─────────────────────────────────────────────
# STEP 4: BUILD DATASET
# ─────────────────────────────────────────────

def build_dataset(input_dir, output_path, classifier):
    """Process all downloaded snapshots and build the dataset."""
    input_dir = Path(input_dir)
    all_examples = []

    for outlet_dir in sorted(input_dir.iterdir()):
        if not outlet_dir.is_dir():
            continue
        outlet = outlet_dir.name

        hl_files = sorted(outlet_dir.glob("*.hyperlinks.json"))
        print("[%s] Processing %d snapshots..." % (outlet, len(hl_files)))

        outlet_examples = []
        for hl_path in hl_files:
            acc_path = hl_path.with_name(
                hl_path.name.replace(".hyperlinks.json", ".accessibility.json")
            )
            examples = process_snapshot(hl_path, acc_path, outlet, classifier)
            outlet_examples.extend(examples)

        label_1 = sum(1 for e in outlet_examples if e["judgement"] == 1)
        label_0 = sum(1 for e in outlet_examples if e["judgement"] == 0)
        print("[%s] %d examples (top=%d, bottom=%d, ratio=%.0f/%.0f)" % (
            outlet, len(outlet_examples), label_1, label_0,
            label_1 / max(1, len(outlet_examples)) * 100,
            label_0 / max(1, len(outlet_examples)) * 100,
        ))
        all_examples.extend(outlet_examples)

    # Balance per-outlet: downsample to 50/50 within each outlet
    # This prevents the model from using outlet identity as a shortcut
    import random
    rng = random.Random(42)
    balanced = []
    outlet_groups = defaultdict(lambda: {0: [], 1: []})
    for e in all_examples:
        outlet_groups[e["outlet"]][e["judgement"]].append(e)

    print("\nBalancing per-outlet:")
    for outlet in sorted(outlet_groups):
        pos = outlet_groups[outlet][1]
        neg = outlet_groups[outlet][0]
        n = min(len(pos), len(neg))
        rng.shuffle(pos)
        rng.shuffle(neg)
        balanced.extend(pos[:n])
        balanced.extend(neg[:n])
        print("  [%s] %d per class (was %d/%d)" % (outlet, n, len(pos), len(neg)))

    rng.shuffle(balanced)

    # Write (strip outlet column from output)
    print("\nWriting %d total examples to %s" % (len(balanced), output_path))
    with gzip.open(output_path, "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "judgement"])
        writer.writeheader()
        for e in balanced:
            writer.writerow({"text": e["text"], "judgement": e["judgement"]})

    dist = Counter(e["judgement"] for e in balanced)
    total = len(balanced)
    print("Label 1 (top): %d (%.0f%%)" % (dist[1], dist[1] / total * 100))
    print("Label 0 (bottom): %d (%.0f%%)" % (dist[0], dist[0] / total * 100))
    print("Done!")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build homepage newsworthiness dataset")
    subparsers = parser.add_subparsers(dest="command")

    # Download
    dl = subparsers.add_parser("download")
    dl.add_argument("--outlets", default=",".join(DEFAULT_OUTLETS))
    dl.add_argument("--max-per-outlet", type=int, default=10000)
    dl.add_argument("--output-dir", default="raw_data")

    # Build
    bd = subparsers.add_parser("build")
    bd.add_argument("--input-dir", default="raw_data")
    bd.add_argument("--output", default="homepage_newsworthiness.csv.gz")
    bd.add_argument("--classifier", default=CLASSIFIER_PATH)

    # All-in-one
    al = subparsers.add_parser("all")
    al.add_argument("--outlets", default=",".join(DEFAULT_OUTLETS))
    al.add_argument("--max-per-outlet", type=int, default=10000)
    al.add_argument("--output-dir", default="raw_data")
    al.add_argument("--output", default="homepage_newsworthiness.csv.gz")
    al.add_argument("--classifier", default=CLASSIFIER_PATH)

    args = parser.parse_args()

    if args.command == "download":
        outlets = [o.strip() for o in args.outlets.split(",")]
        for outlet in outlets:
            download_outlet(outlet, args.output_dir, args.max_per_outlet)

    elif args.command == "build":
        clf = joblib.load(args.classifier)
        build_dataset(args.input_dir, args.output, clf)

    elif args.command == "all":
        outlets = [o.strip() for o in args.outlets.split(",")]
        for outlet in outlets:
            download_outlet(outlet, args.output_dir, args.max_per_outlet)
        clf = joblib.load(args.classifier)
        build_dataset(args.output_dir, args.output, clf)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
