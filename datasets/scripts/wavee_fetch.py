#!/usr/bin/env python3
"""
wavee_fetch.py — Fetch a list of URLs into a per-task online-rubrics/raw/ dir.

Usage:
    python wavee_fetch.py <task> <urls_file> <query_label>

  - <task>: dataset task name (e.g., peer-review, creative-writing)
  - <urls_file>: text file with one URL per line
  - <query_label>: short label written to urls-visited.csv (e.g., wavee:slush_pile)

Output:
  - online-rubrics/raw/wavee_<hash>.html|.pdf|.txt
  - appends rows to online-rubrics/urls-visited.csv

Behavior:
  - Skips URLs already present in urls-visited.csv (per-task seen-set).
  - Threaded fetch with sane timeout / UA / size cap.
"""
import sys, os, csv, hashlib, time, datetime, mimetypes
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request, urllib.error, urllib.parse, ssl

DATASETS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
TIMEOUT = 25
MAX_BYTES = 6 * 1024 * 1024  # 6 MB cap
NUM_WORKERS = 8

def hash_url(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]

def ext_for(url: str, ctype: str) -> str:
    u = url.lower().split("?")[0].split("#")[0]
    for ex in (".pdf", ".html", ".htm", ".txt", ".json", ".xml"):
        if u.endswith(ex):
            return ".pdf" if ex == ".pdf" else (".html" if ex in (".htm",) else ex)
    if ctype:
        c = ctype.lower()
        if "pdf" in c: return ".pdf"
        if "json" in c: return ".json"
        if "xml" in c: return ".xml"
        if "text/plain" in c: return ".txt"
    return ".html"

def fetch(url: str):
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": UA,
            "Accept": "text/html,application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=ctx) as r:
            ctype = r.headers.get("Content-Type", "")
            data = r.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                data = data[:MAX_BYTES]
            return r.status, ctype, data
    except urllib.error.HTTPError as e:
        try:
            data = e.read()
        except Exception:
            data = b""
        return e.code, e.headers.get("Content-Type", "") if e.headers else "", data
    except Exception as e:
        return 0, str(e)[:120], b""

def main():
    if len(sys.argv) != 4:
        print("usage: wavee_fetch.py <task> <urls_file> <query_label>", file=sys.stderr)
        sys.exit(2)
    task, urls_file, query = sys.argv[1], sys.argv[2], sys.argv[3]
    online = os.path.join(DATASETS_ROOT, task, "online-rubrics")
    raw = os.path.join(online, "raw")
    csv_path = os.path.join(online, "urls-visited.csv")
    if not os.path.isdir(raw):
        print(f"raw dir not found: {raw}", file=sys.stderr); sys.exit(2)

    # seen set
    seen = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            r = csv.reader(f)
            try: next(r)  # header
            except StopIteration: pass
            for row in r:
                if row: seen.add(row[0])

    with open(urls_file) as f:
        urls = [u.strip() for u in f if u.strip() and u.strip().startswith("http")]

    todo = [u for u in dict.fromkeys(urls) if u not in seen]
    print(f"[{task}] {len(urls)} input, {len(todo)} new (skipping {len(urls)-len(todo)} dupes)")

    new_rows = []
    n_ok = 0
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futs = {pool.submit(fetch, u): u for u in todo}
        for fut in as_completed(futs):
            url = futs[fut]
            try:
                status, ctype, data = fut.result()
            except Exception as e:
                status, ctype, data = 0, str(e)[:120], b""
            if status != 200 or len(data) < 400:
                # log failure too, with empty bytes/no file
                new_rows.append((url, query, "", status, len(data),
                                 datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")))
                continue
            ext = ext_for(url, ctype)
            fname = f"wavee_{hash_url(url)}{ext}"
            fpath = os.path.join(raw, fname)
            try:
                with open(fpath, "wb") as out:
                    out.write(data)
                n_ok += 1
                new_rows.append((url, query, fname, status, len(data),
                                 datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")))
            except Exception as e:
                new_rows.append((url, query, "", status, len(data),
                                 datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")))

    # append CSV (create header if needed)
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["url","query","filename","http_status","bytes","fetched_at"])
        for row in new_rows:
            w.writerow(row)

    print(f"[{task}] wrote {n_ok} files, logged {len(new_rows)} rows")

if __name__ == "__main__":
    main()
