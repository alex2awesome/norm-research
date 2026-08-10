#!/usr/bin/env python3
"""
wavee_wayback_fetch_pass2.py — Second pass: only fetch URLs that haven't been
successfully archived yet. Fully sequential with longer delays to be polite.
"""
import os, sys, csv, hashlib, datetime, time, ssl, urllib.request, urllib.error
import importlib.util

spec = importlib.util.spec_from_file_location(
    "wfetch",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "wavee_wayback_fetch.py"))
wfetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wfetch)


SLEEP_BETWEEN = 1.0  # seconds; HTTP is faster
USE_HTTP = True  # Wayback HTTPS is blocked from our IP; HTTP works


def fetch_one(url, retries=3):
    if USE_HTTP and url.startswith("https://web.archive.org/"):
        url = "http://" + url[len("https://"):]
    """Sequential fetch with backoff."""
    last_err = (0, "", b"")
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": wfetch.UA,
                "Accept": "text/html,application/pdf,*/*;q=0.8",
            })
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
                ctype = r.headers.get("Content-Type", "")
                data = r.read(wfetch.MAX_BYTES + 1)
                if len(data) > wfetch.MAX_BYTES:
                    data = data[:wfetch.MAX_BYTES]
                return r.status, ctype, data
        except urllib.error.HTTPError as e:
            try: data = e.read()
            except Exception: data = b""
            ctype = (e.headers.get("Content-Type", "") if e.headers else "")
            if e.code in (429, 503) and attempt < retries:
                time.sleep(8 + 5 * attempt)
                continue
            return e.code, ctype, data
        except Exception as e:
            last_err = (0, str(e)[:120], b"")
            if attempt < retries:
                time.sleep(5 + 3 * attempt)
                continue
    return last_err


def main():
    from concurrent.futures import ThreadPoolExecutor, as_completed
    only = set(sys.argv[1:]) if len(sys.argv) > 1 else None

    print("[pass2] Phase 1: Parallel CDX lookup for missing URLs...", flush=True)
    # Phase 1: Parallel CDX
    all_targets_per_task = {}  # task -> [(wb, orig)]
    cdx_jobs = [(task, orig) for task, origs in wfetch.TARGETS.items() for orig in origs
                if not only or task in only]
    cdx_results = {}  # (task, orig) -> [(ts, original)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(wfetch.cdx_lookup, orig): (task, orig) for task, orig in cdx_jobs}
        for fut in as_completed(futs):
            key = futs[fut]
            try: snaps = fut.result()
            except Exception: snaps = []
            cdx_results[key] = snaps
    print(f"[pass2] CDX done for {len(cdx_results)} originals", flush=True)

    # Build task -> [wb_url] (only missing ones)
    n_total_missing = 0
    for task, originals in wfetch.TARGETS.items():
        if only and task not in only:
            continue
        raw, csv_path = wfetch.task_paths(task)
        if not raw:
            continue
        attempted = set()
        if os.path.exists(csv_path):
            with open(csv_path, newline="") as f:
                r = csv.reader(f)
                next(r, None)
                for row in r:
                    if row and row[1] == "wave_e":
                        attempted.add(row[0])
        targets = []
        for orig in originals:
            snaps = cdx_results.get((task, orig), [])
            if not snaps:
                continue
            picks = wfetch.yearly_strides(snaps, max_n=6)
            for ts, original_in_cdx in picks:
                wb = wfetch.build_wayback_url(ts, original_in_cdx)
                if wb not in attempted:
                    targets.append(wb)
        all_targets_per_task[task] = targets
        n_total_missing += len(targets)
        print(f"[{task}] {len(targets)} missing", flush=True)
    print(f"[pass2] Phase 2: Sequential HTTP fetch of {n_total_missing} missing URLs", flush=True)

    # Phase 2: Sequential fetch (HTTP works when HTTPS doesn't)
    n_fetched = 0
    for task, targets in all_targets_per_task.items():
        if not targets:
            continue
        raw, csv_path = wfetch.task_paths(task)
        rows = []
        for i, wb in enumerate(targets):
            status, ctype, data = fetch_one(wb)
            now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if status == 200 and len(data) >= 400:
                ext = wfetch.ext_for(wb, ctype)
                fname = f"wavee_{wfetch.hash_url(wb)}{ext}"
                fpath = os.path.join(raw, fname)
                with open(fpath, "wb") as out:
                    out.write(data)
                rows.append((wb, "wave_e", fname, status, len(data), now))
                n_fetched += 1
            else:
                rows.append((wb, "wave_e", "", status, len(data), now))
            if (i + 1) % 5 == 0:
                # incremental flush
                wfetch.append_rows(csv_path, rows)
                rows = []
                print(f"[{task}] {i+1}/{len(targets)} ok={n_fetched}", flush=True)
            time.sleep(SLEEP_BETWEEN)
        if rows:
            wfetch.append_rows(csv_path, rows)
        print(f"[{task}] DONE; total fetched so far: {n_fetched}", flush=True)

    print(f"[pass2] FINAL: {n_fetched} new files fetched", flush=True)


if __name__ == "__main__":
    main()
