#!/usr/bin/env python3
"""Stage-2 v2: COPY-PROOF body extraction. Gemma NEVER generates body text — it only outputs
BOUNDARY MARKERS (the exact first and last ~8 words of the article body). Code then SLICES the
source text between the markers. The output is a literal substring of the BS4 text — verbatim
by construction, zero paraphrase risk.

Fallback ladder per doc:
  1) marker-slice (copy-proof)             -> quality flag "slice"
  2) retry once with temperature 0.3       -> "slice_retry"
  3) reprint+strict verbatim gate (legacy) -> "reprint_gated" (only if 12-gram checks pass)
  4) give up -> body="" (never emit unverified text)
Run on sk3: GEMMA_URL=http://127.0.0.1:8006/v1 python3 reprint_body2.py [--limit N]
"""
import json, os, re, sys, time, argparse, random, urllib.request
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "fulltext.jsonl")
OUT = os.path.join(BASE, "fulltext_body.jsonl")
GEMMA = os.environ.get("GEMMA_URL", "http://127.0.0.1:8006/v1")
MODEL = "gemma"

MARKER_PROMPT = """Below is raw text extracted from a news-article web page (navigation, banners, \
promos, and the ARTICLE BODY are mixed together).

Identify the article body's BOUNDARIES. Output JSON with EXACTLY two fields:
{{"start_words": "<the EXACT first 8-12 consecutive words of the article body, copied character-for-character from the text>",
"end_words": "<the EXACT last 8-12 consecutive words of the article body, copied character-for-character from the text>"}}

Rules:
- Copy the words EXACTLY as they appear (same punctuation, same capitalization). Do not fix or normalize anything.
- The body starts at the first sentence of the news article itself (after headline/byline/dateline) and ends at its last sentence (before footers/related-links/comments).
- If there is no article body, output {{"start_words": "", "end_words": ""}}

RAW PAGE TEXT:
{page}

JSON:"""

REPRINT_PROMPT = """Below is raw text from a news-article web page. Reprint ONLY the article body \
VERBATIM - copy exactly, never paraphrase or summarize. Exclude navigation/banners/captions/\
related-links/footers. If no body: output NO_BODY.

RAW PAGE TEXT:
{page}

ARTICLE BODY (verbatim):"""

def gemma(prompt, max_tokens=400, temperature=0.0, timeout=150):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature}
    req = urllib.request.Request(GEMMA.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]

def _fuzzy_find(source, snippet, from_end=False):
    """Locate snippet in source; exact first, then whitespace-normalized, then drop
    first/last word (boundary slop). Returns (start_idx, end_idx) of the match or None."""
    if not snippet or len(snippet) < 10: return None
    cands = [snippet, snippet.strip(' "\'')]
    words = snippet.split()
    if len(words) > 5:
        cands += [" ".join(words[1:]), " ".join(words[:-1]), " ".join(words[1:-1])]
    for c in cands:
        i = source.rfind(c) if from_end else source.find(c)
        if i >= 0: return (i, i + len(c))
        # whitespace-insensitive regex match
        pat = r"\s+".join(re.escape(w) for w in c.split())
        m = None
        for m_ in re.finditer(pat, source):
            m = m_
            if not from_end: break
        if m: return (m.start(), m.end())
    return None

def marker_slice(source, temperature=0.0):
    raw = gemma(MARKER_PROMPT.format(page=source[:16000]), temperature=temperature)
    m = re.search(r"\{[\s\S]*\}", raw or "")
    if not m: return None
    try: obj = json.loads(m.group(0))
    except Exception: return None
    sw, ew = (obj.get("start_words") or "").strip(), (obj.get("end_words") or "").strip()
    if not sw or not ew: return None
    a = _fuzzy_find(source, sw, from_end=False)
    b = _fuzzy_find(source, ew, from_end=True)
    if not a or not b: return None
    start, end = a[0], b[1]
    if end <= start or (end - start) < 400: return None
    return source[start:end]

def norm(s): return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", (s or "").lower()))

def verbatim_gate(body, source, k=5, n=12, seed=0):
    bw = norm(body).split(); sw = norm(source)
    if len(bw) < n + 2: return False
    rng = random.Random(seed); ok = 0
    for _ in range(k):
        i = rng.randrange(0, len(bw) - n)
        ok += (" ".join(bw[i:i + n]) in sw)
    return ok >= k - 1

def extract_body(source):
    body = marker_slice(source, 0.0)
    if body: return body, "slice"
    body = marker_slice(source, 0.3)
    if body: return body, "slice_retry"
    out = (gemma(REPRINT_PROMPT.format(page=source[:16000]), max_tokens=6000) or "").strip()
    if out and not out.upper().startswith("NO_BODY") and len(out) > 300 and verbatim_gate(out, source):
        return out, "reprint_gated"
    return "", "fail"

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
    print(f"[body2] todo={len(todo)} (done={len(done)})", flush=True)
    lock = Lock(); fout = open(OUT, "a"); n = [0, 0]
    import collections; methods = collections.Counter()
    def work(r):
        try:
            if r.get("route") == "guardian_api":
                body, method = r["text"][:40000], "guardian_clean"
            else:
                body, method = extract_body(r["text"])
            row = {"url": r["url"], "outlet": r["outlet"], "route": r.get("route"),
                   "method": method, "body": body[:40000], "body_len": len(body),
                   "ts": int(time.time())}
        except Exception as e:
            row = {"url": r["url"], "outlet": r["outlet"], "method": "err",
                   "err": str(e)[:80], "body": "", "body_len": 0}
        with lock:
            fout.write(json.dumps(row) + "\n"); fout.flush()
            n[0] += 1; n[1] += bool(row["body"]); methods[row["method"]] += 1
            if n[0] % 25 == 0:
                print(f"[body2] {n[0]}/{len(todo)} ok={n[1]} methods={dict(methods)}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    print(f"[body2] DONE {n[0]} ok={n[1]} methods={dict(methods)}", flush=True)

if __name__ == "__main__":
    main()
