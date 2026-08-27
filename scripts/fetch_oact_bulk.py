#!/usr/bin/env python3
"""Download ALL USPTO OACT weekly archives (full-text office actions, 2020+).

OACT = "Office Actions Weekly Archives" on the Open Data Portal: weekly zips,
each a JSONL of CTNF/CTFR records with oaText (full text), structured
rejections (type 102/103 + position), and per-section rejection text.
981 files / ~66GB total, covering OAs MAILED 2020-01-06 -> present.
(Does NOT cover the OARD-era 2012-2017 OA events of the 1.1M cohort —
the per-app API downloader remains the route for those.)

Compliance: STRICTLY SEQUENTIAL (ODP advises one API call per key at a time).
The file endpoint returns a signed CloudFront URL with a quota of 20
URL-generations per file per YEAR — so: skip files already on disk that pass
zip integrity, max 2 attempts per file, never loop on failures.
"""
import argparse
import json
import os
import subprocess
import time
import urllib.request
import zipfile

OUT = os.path.expanduser("~/norm-research/datasets/patents/raw/oact")
KEY_FILE = "/lfs/skampere3/0/alexspan/.uspto-open-data-api-key.txt"
LIST_URL = ("https://api.uspto.gov/api/v1/datasets/products/OACT"
            "?fileDataFromDate={}&fileDataToDate={}")


def api_json(url, key):
    req = urllib.request.Request(url, headers={"X-API-KEY": key,
                                               "Accept": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def zip_ok(path):
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-year", type=int, default=2007)
    ap.add_argument("--to-year", type=int, default=2026)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    key = open(KEY_FILE).read().strip()

    # ---- enumerate all weekly files, year by year ----
    files = {}
    for y in range(args.from_year, args.to_year + 1):
        url = LIST_URL.format(f"{y}-01-01", f"{y}-12-31")
        try:
            r = api_json(url, key)
        except Exception as e:
            print(f"list {y}: {e}", flush=True)
            continue
        bag = r.get("bulkDataProductBag", [])
        fb = (bag[0].get("productFileBag") or bag[0].get("fileDataBag")
              if bag else None) or r.get("fileDataBag") or []
        if isinstance(fb, dict):
            fb = fb.get("fileDataBag", [])
        for f in fb:
            files[f["fileName"]] = f
        print(f"list {y}: +{len(fb)} files", flush=True)
        time.sleep(1)
    names = sorted(files)
    print(f"total files: {len(names)}", flush=True)

    n_done = n_new = n_fail = 0
    for i, name in enumerate(names, 1):
        dst = os.path.join(OUT, name)
        if os.path.exists(dst) and zip_ok(dst):
            n_done += 1
            continue
        ok = False
        for attempt in range(2):  # NEVER more: 20 signed URLs/file/YEAR quota
            try:
                req = urllib.request.Request(
                    files[name]["fileDownloadURI"],
                    headers={"X-API-KEY": key})
                body = urllib.request.urlopen(req, timeout=120).read()
                if body[:4] == b"PK\x03\x04":  # direct zip binary
                    with open(dst + ".tmp", "wb") as f:
                        f.write(body)
                else:  # JSON/text message carrying a signed redirect URL
                    txt = body.decode(errors="replace")
                    try:
                        txt = json.loads(txt)
                    except Exception:
                        pass
                    if not isinstance(txt, str) or "redirect URL" not in txt:
                        raise RuntimeError(f"unexpected response: {txt[:160]}")
                    signed = txt.split("Use redirect URL to download: ")[1]\
                        .split(". IMPORTANT")[0]
                    subprocess.run(["curl", "-s", "-o", dst + ".tmp", signed],
                                   check=True, timeout=900)
                if zip_ok(dst + ".tmp"):
                    os.replace(dst + ".tmp", dst)
                    ok = True
                    break
                os.remove(dst + ".tmp")
            except Exception as e:
                print(f"  {name} attempt {attempt + 1}: {e}", flush=True)
                time.sleep(5)
        if ok:
            n_new += 1
        else:
            n_fail += 1
            print(f"  FAILED (will retry on next run): {name}", flush=True)
        if (n_new + n_fail) % 10 == 0 or i == len(names):
            done_gb = sum(os.path.getsize(os.path.join(OUT, n))
                          for n in os.listdir(OUT)
                          if n.endswith(".zip")) / 1e9
            print(f"  {i}/{len(names)}  cached={n_done} new={n_new} "
                  f"fail={n_fail}  disk={done_gb:.1f}GB", flush=True)
    print(f"OACT-BULK-DONE cached={n_done} new={n_new} fail={n_fail}",
          flush=True)


if __name__ == "__main__":
    main()
