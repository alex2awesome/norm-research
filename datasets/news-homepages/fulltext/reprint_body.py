#!/usr/bin/env python3
"""Stage-2: LLM body-reprint. Takes fulltext.jsonl rows (BS4 get_text(' ') soup — headline, nav,
promos, article body, footer all mixed) and has Gemma reprint ONLY the article body verbatim.
Strict instructions + verbatim spot-check gate (sampled 12-grams must appear in source text).
Run on sk3 (Gemma server on :8006): python3 reprint_body.py [--limit N]
Output: fulltext_body.jsonl rows {url, outlet, body(<=20k), n_verbatim_checked, verbatim_ok, ts}
"""
import json, os, re, sys, time, argparse, random, urllib.request
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "fulltext.jsonl")
OUT = os.path.join(BASE, "fulltext_body.jsonl")
GEMMA = os.environ.get("GEMMA_URL", "http://127.0.0.1:8006/v1")
MODEL = "gemma"

PROMPT = """Below is the raw text extracted from a news-article web page. It contains navigation \
menus, cookie banners, promos, related-links, captions, and the ARTICLE BODY mixed together.

Reprint ONLY the article body, VERBATIM. Rules:
- Copy the body text EXACTLY as written - do not paraphrase, summarize, fix typos, or add anything.
- Start from the first sentence of the article (after the headline/byline/dateline).
- Include every body paragraph in order. Stop at the end of the article.
- EXCLUDE: navigation, menus, cookie/subscription banners, bylines, timestamps, image captions/credits,
  "Read more"/"Related" links, newsletters, comments, footers, ads, share buttons.
- If there is no article body in the text, output exactly: NO_BODY

RAW PAGE TEXT:
{page}

ARTICLE BODY (verbatim):"""

def gemma(prompt, max_tokens=6000, timeout=180):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0}
    req = urllib.request.Request(GEMMA.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]

def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", (s or "").lower()))

def verbatim_check(body, source, k=5, n=12, seed=0):
    """Sample k random n-gram windows from the reprinted body; each must appear in the source."""
    bw = norm(body).split(); sw = norm(source)
    if len(bw) < n + 2: return 0, False
    rng = random.Random(seed)
    ok = 0
    for _ in range(k):
        i = rng.randrange(0, len(bw) - n)
        if " ".join(bw[i:i + n]) in sw: ok += 1
    return k, ok >= k - 1  # allow 1 miss (boundary artifacts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--min_len", type=int, default=800)
    args = ap.parse_args()
    done = set()
    if os.path.exists(OUT):
        for ln in open(OUT):
            try: done.add(json.loads(ln)["url"])
            except Exception: pass
    todo = []
    for ln in open(SRC):
        try: r = json.loads(ln)
        except Exception: continue
        if r.get("route") == "FAIL" or r.get("text_len", 0) < args.min_len: continue
        if r["url"] in done: continue
        todo.append(r)
    if args.limit: todo = todo[:args.limit]
    print(f"[reprint] todo={len(todo)} (done={len(done)})", flush=True)
    lock = Lock(); fout = open(OUT, "a"); n = [0, 0]
    def work(r):
        try:
            # guardian_api is already clean body -> pass through
            if r.get("route") == "guardian_api":
                body, checked, vok = r["text"][:20000], 0, True
            else:
                out = gemma(PROMPT.format(page=r["text"][:16000]))
                body = (out or "").strip()
                if body.upper().startswith("NO_BODY") or len(body) < 300:
                    body, checked, vok = "", 0, False
                else:
                    checked, vok = verbatim_check(body, r["text"])
                    body = body[:20000]
            row = {"url": r["url"], "outlet": r["outlet"], "route": r.get("route"),
                   "body": body if vok else "", "raw_reprint_len": len(body),
                   "n_verbatim_checked": checked, "verbatim_ok": vok, "ts": int(time.time())}
        except Exception as e:
            row = {"url": r["url"], "outlet": r["outlet"], "err": str(e)[:80], "verbatim_ok": False}
        with lock:
            fout.write(json.dumps(row) + "\n"); fout.flush()
            n[0] += 1; n[1] += bool(row.get("verbatim_ok"))
            if n[0] % 25 == 0:
                print(f"[reprint] {n[0]}/{len(todo)} verbatim_ok={n[1]} ({n[1]/max(n[0],1):.0%})", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    print(f"[reprint] DONE {n[0]}, verbatim_ok={n[1]}", flush=True)

if __name__ == "__main__":
    main()
