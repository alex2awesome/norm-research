#!/usr/bin/env python3
"""Build a compact CFR part-existence index from the eCFR structure API.

Fetches https://www.ecfr.gov/api/versioner/v1/structure/2024-01-01/title-{N}.json
for N in 1..50, walks the JSON tree collecting every node with type=='part'
(identifier + its title number), and writes a compact gzip json set to
cfr_parts_index.json.gz as {"<title>": ["<part>", "<part>", ...], ...}.

This is used by authority_lookup_h2.py to VERIFY that a cited "N CFR Part M"
authority actually exists (not just that it is syntactically well-formed).

Usage: python build_cfr_index.py [--out cfr_parts_index.json.gz] [--sleep 1.0]
"""
import argparse, gzip, json, pathlib, sys, time, urllib.error, urllib.request

HERE = pathlib.Path(__file__).resolve().parent
URL_TMPL = "https://www.ecfr.gov/api/versioner/v1/structure/2024-01-01/title-{n}.json"


def _collect_parts(node, out):
    if node.get("type") == "part":
        ident = node.get("identifier")
        if ident:
            out.append(str(ident))
    for c in node.get("children") or []:
        _collect_parts(c, out)


def fetch_title(n, timeout=60):
    url = URL_TMPL.format(n=n)
    req = urllib.request.Request(url, headers={"User-Agent": "norm-research-cfr-index/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "cfr_parts_index.json.gz"))
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--min-title", type=int, default=1)
    ap.add_argument("--max-title", type=int, default=50)
    a = ap.parse_args()

    index = {}
    n_ok, n_fail = 0, 0
    for n in range(a.min_title, a.max_title + 1):
        try:
            data = fetch_title(n)
        except Exception as e:  # noqa: BLE001 - skip failures gracefully
            print(f"[cfr-index] title {n}: FAILED ({e!r}) — skipping", flush=True)
            n_fail += 1
            time.sleep(a.sleep)
            continue
        parts = []
        _collect_parts(data, parts)
        parts = sorted(set(parts), key=lambda s: (len(s), s))
        index[str(n)] = parts
        n_ok += 1
        print(f"[cfr-index] title {n}: {len(parts)} parts", flush=True)
        time.sleep(a.sleep)

    out_path = pathlib.Path(a.out)
    with gzip.open(out_path, "wt", encoding="utf-8") as fh:
        json.dump(index, fh)

    total_parts = sum(len(v) for v in index.values())
    print(f"[cfr-index] done: {n_ok} titles ok, {n_fail} failed, "
          f"{total_parts} total parts -> {out_path}", flush=True)
    print("CFR_INDEX_DONE", flush=True)


if __name__ == "__main__":
    main()
