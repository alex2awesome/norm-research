#!/usr/bin/env python3
"""
wavee_wayback_logging_recover.py — Match unlogged wavee_*.html files in raw/
that contain Wayback Machine markers back to their CDX URLs by re-running CDX
and computing the same sha1 hash filename, then appending CSV log rows.

Usage: python wavee_wayback_logging_recover.py
"""
import os, sys, csv, json, hashlib, glob, urllib.request, urllib.parse, time, datetime, ssl
import importlib.util
spec = importlib.util.spec_from_file_location(
    "wfetch",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "wavee_wayback_fetch.py"))
wfetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wfetch)


def main():
    from concurrent.futures import ThreadPoolExecutor, as_completed
    QUERY_LABEL = "wave_e"
    DATASETS_ROOT = wfetch.DATASETS_ROOT

    # Build expected (task, hash, wb_url) for all queued URLs - parallel CDX
    print("[recover] Re-running CDX in parallel to map files...", flush=True)
    expected = {}  # hash -> (task, wb_url)
    jobs = [(task, orig) for task, origs in wfetch.TARGETS.items() for orig in origs]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(wfetch.cdx_lookup, orig): (task, orig) for task, orig in jobs}
        for fut in as_completed(futs):
            task, orig = futs[fut]
            try: snaps = fut.result()
            except Exception: snaps = []
            if not snaps:
                continue
            picks = wfetch.yearly_strides(snaps, max_n=6)
            for ts, original_in_cdx in picks:
                wb = wfetch.build_wayback_url(ts, original_in_cdx)
                h = wfetch.hash_url(wb)
                expected[h] = (task, wb)

    print(f"[recover] {len(expected)} expected hashes from CDX", flush=True)

    # Walk each task's raw/ dir and check for files with names matching expected hashes
    new_rows_per_task = {}
    for h, (task, wb) in expected.items():
        raw = os.path.join(DATASETS_ROOT, task, "online-rubrics", "raw")
        # check both .html and .pdf
        for ext in (".html", ".pdf", ".txt"):
            path = os.path.join(raw, f"wavee_{h}{ext}")
            if os.path.exists(path):
                # check it's not already logged
                csv_path = os.path.join(DATASETS_ROOT, task, "online-rubrics", "urls-visited.csv")
                already_logged = False
                if os.path.exists(csv_path):
                    with open(csv_path, newline="") as f:
                        for row in csv.reader(f):
                            if row and row[0] == wb:
                                already_logged = True
                                break
                if not already_logged:
                    sz = os.path.getsize(path)
                    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    new_rows_per_task.setdefault(task, []).append(
                        (wb, QUERY_LABEL, f"wavee_{h}{ext}", 200, sz, now))
                break

    total = 0
    for task, rows in new_rows_per_task.items():
        csv_path = os.path.join(DATASETS_ROOT, task, "online-rubrics", "urls-visited.csv")
        wfetch.append_rows(csv_path, rows)
        print(f"[{task}] recovered {len(rows)} log rows", flush=True)
        total += len(rows)
    print(f"[recover] TOTAL recovered: {total}", flush=True)


if __name__ == "__main__":
    main()
