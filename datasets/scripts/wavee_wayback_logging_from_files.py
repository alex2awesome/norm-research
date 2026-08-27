#!/usr/bin/env python3
"""
wavee_wayback_logging_from_files.py — Walk each task's online-rubrics/raw/
for wavee_*.html files containing Wayback Machine markers, extract the
wayback URL from the HTML content, and append CSV log rows.

This avoids any network calls to web.archive.org.
"""
import os, csv, re, datetime, glob


DATASETS_ROOT = "/Users/spangher/Projects/stanford-research/norm-research/datasets"
TASKS = [
    "code-review", "creative-writing", "grant-funding", "humor",
    "legal-outcome-prediction", "math/stackexchange", "news-homepages",
    "notice-and-comment", "patents", "peer-review", "press-releases",
]

# Patterns that uniquely identify Wayback Machine archived content
WAYBACK_MARKERS = [
    "archive.org/includes/athena.js",
    "BEGIN WAYBACK TOOLBAR",
    "/web/static/",
    'src="//archive.org/',
]

# Extract the canonical web/<14digit-ts>/<orig> from the HTML
RE_WB = re.compile(r'https?://web\.archive\.org/web/(\d{14})/(https?://[^\s"\'<>]+)')


def is_wayback_html(content: bytes) -> bool:
    head = content[:20000].decode("utf-8", errors="replace")
    return any(m in head for m in WAYBACK_MARKERS)


def extract_wayback_url(content: bytes) -> str:
    """Extract the canonical https://web.archive.org/web/<ts>/<orig> URL.
    Returns the most-frequent timestamp seen (the snapshot's ts)."""
    text = content[:200000].decode("utf-8", errors="replace")
    matches = RE_WB.findall(text)
    if not matches:
        return ""
    # Find most common ts
    from collections import Counter
    ts_counts = Counter(ts for ts, _ in matches)
    best_ts = ts_counts.most_common(1)[0][0]
    # Get matching original URL for that ts (first one)
    for ts, orig in matches:
        if ts == best_ts:
            return f"https://web.archive.org/web/{ts}/{orig}"
    return ""


def load_logged_urls(csv_path: str) -> set:
    s = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if row:
                    s.add(row[0])
    return s


def append_rows(csv_path: str, rows: list):
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["url", "query", "filename", "http_status", "bytes", "fetched_at"])
        for row in rows:
            w.writerow(row)


def main():
    QUERY_LABEL = "wave_e"
    summary = {}
    for task in TASKS:
        raw = os.path.join(DATASETS_ROOT, task, "online-rubrics", "raw")
        csv_path = os.path.join(DATASETS_ROOT, task, "online-rubrics", "urls-visited.csv")
        if not os.path.isdir(raw):
            continue
        logged = load_logged_urls(csv_path)
        new_rows = []
        files = glob.glob(os.path.join(raw, "wavee_*.html"))
        n_wayback = 0
        n_logged = 0
        for f in files:
            try:
                with open(f, "rb") as fh:
                    content = fh.read(200000)
                if not is_wayback_html(content):
                    continue
                n_wayback += 1
                wb_url = extract_wayback_url(content)
                if not wb_url:
                    continue
                if wb_url in logged:
                    continue
                fname = os.path.basename(f)
                sz = os.path.getsize(f)
                now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                new_rows.append((wb_url, QUERY_LABEL, fname, 200, sz, now))
                logged.add(wb_url)
                n_logged += 1
            except Exception as e:
                pass
        if new_rows:
            append_rows(csv_path, new_rows)
        summary[task] = (n_wayback, n_logged)
        print(f"[{task}] wayback files found: {n_wayback}, newly logged: {n_logged}", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    total_w = sum(v[0] for v in summary.values())
    total_l = sum(v[1] for v in summary.values())
    print(f"  Total wayback files: {total_w}", flush=True)
    print(f"  Newly logged rows: {total_l}", flush=True)


if __name__ == "__main__":
    main()
