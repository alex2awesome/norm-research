#!/usr/bin/env python3
"""Bibliography-disambiguated gold arm v2 (fixes from funnel diagnosis):
resolve reviewer-named prior work -> citation string (review's own [n] defs, else paper bib)
-> gold paper via, in order:
   (a) arXiv ID in the citation string -> OpenAlex direct lookup (exact)
   (b) title-segment extraction (the '. '-bounded segment containing the name term, plus the
       following segment for author-name matches) -> OpenAlex title.search, overlap-verified
   (c) raw-string search fallback, overlap-verified
Abstract: OpenAlex inverted index, else Semantic Scholar fallback by title.
Then SS102 + SS103 verdicts on matched pairs vs MISMATCHED control pairs.
GATE G2': title-verified resolution >= .40; fired >= .35 on matched AND >= 2x mismatched.
Run on sk3: python -m methods.claim_verification.run_gold_bib"""
import json, os, re, sqlite3, sys, time, urllib.parse, urllib.request, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "methods")
import numpy as np
from claim_verification.core import Cache
from claim_verification.run_check_v2 import pa_check
from claim_verification.run_delta_check import delta_check
from claim_verification.run_gold_openalex import GENERIC

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
PDF_DB = f"{ROOT}/datasets/peer-review/peer_review_pdfs.db"
OUTD = f"{ROOT}/outputs/checks_v2"
MAILTO = "alex2awesome@gmail.com"

def http_json(url, cache, tries=7, sleep=1.0):
    # S2 shared pool 429s hard when the citation fetcher runs concurrently — long backoff
    hit = cache.get("url:" + url)
    if hit is not None: return hit
    headers = {"User-Agent": f"research ({MAILTO})"}
    kp = os.path.expanduser("~/.s2_api_key.txt")
    if "semanticscholar" in url and os.path.exists(kp):
        headers["x-api-key"] = open(kp).read().strip()
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as r:
                out = json.loads(r.read())
            cache.put("url:" + url, out); time.sleep(sleep)
            return out
        except Exception:
            time.sleep(min(8 * (t + 1), 45))
    cache.put("url:" + url, None)
    return None

def http_text(url, cache, tries=4, sleep=1.0):
    hit = cache.get("txt:" + url)
    if hit is not None: return hit
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": f"research ({MAILTO})"})
            with urllib.request.urlopen(req, timeout=30) as r:
                out = r.read().decode("utf-8", "ignore")
            cache.put("txt:" + url, out); time.sleep(sleep)
            return out
        except Exception:
            time.sleep(3 * (t + 1))
    return ""

def arxiv_lookup(aid, cache):
    """arXiv Atom API — free, no quota. Returns {title, year, abstract} or None."""
    xml = http_text(f"http://export.arxiv.org/api/query?id_list={aid}", cache, sleep=0.5)
    m = re.search(r"<entry>.*?<title>(.*?)</title>.*?<summary>(.*?)</summary>", xml, re.S)
    y = re.search(r"<published>(\d{4})", xml)
    if not m: return None
    title = re.sub(r"\s+", " ", m.group(1)).strip()
    ab = re.sub(r"\s+", " ", m.group(2)).strip()[:1200]
    if len(ab) < 150: return None
    return {"title": title, "year": int(y.group(1)) if y else None, "abstract": ab}

def s2_match(cand, cache):
    """Semantic Scholar title match (free shared pool; 429-tolerant via http_json retries)."""
    q = urllib.parse.quote(re.sub(r"[^A-Za-z0-9 \-]", " ", cand)[:130].strip())
    if len(q) < 15: return None
    r = http_json(f"https://api.semanticscholar.org/graph/v1/paper/search/match?query={q}"
                  f"&fields=title,abstract,year", cache, sleep=1.2)
    for d in (r or {}).get("data", [])[:1]:
        if d.get("abstract") and title_overlap(d.get("title", ""), cand) >= 0.6:
            return {"title": d.get("title", ""), "year": d.get("year"),
                    "abstract": (d.get("abstract") or "")[:1200]}
    return None

def title_overlap(title, context):
    words = [w for w in re.findall(r"[a-z]{4,}", (title or "").lower()) if w not in GENERIC]
    if not words: return 0.0
    low = (context or "").lower()
    return sum(1 for w in words if w in low) / len(words)

def review_ref_defs(review_text):
    defs = {}
    for m in re.finditer(r"^\s*\[(\d{1,2})\][.:]?\s+(.{25,400}?)(?=\n|$)", review_text, re.M):
        defs[m.group(1)] = m.group(2).strip()
    return defs

def segments_around(text, term):
    """'. '-bounded segment containing term + the following segment (title candidates)."""
    low = text.lower()
    out = []
    start = 0
    for _ in range(3):
        i = low.find(term.lower(), start)
        if i < 0: break
        a = text.rfind(". ", max(0, i - 300), i)
        a = a + 2 if a >= 0 else max(0, i - 150)
        b = text.find(". ", i)
        b = b if b >= 0 else min(len(text), i + 250)
        seg = text[a:b].strip().replace("\n", " ")
        nxt = text[b + 2:b + 2 + 200].split(". ")[0].strip().replace("\n", " ")
        out += [seg, nxt]
        start = i + 1
    return [s for s in out if 15 < len(s) < 250]

def resolve_gold(citation, prior, cache):
    """citation string + reviewer's prior text -> {title, year, abstract} or None.
    arXiv API first (free, exact), Semantic Scholar title-match otherwise
    (OpenAlex free budget exhausts daily — do not depend on it)."""
    m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", citation, re.I) or \
        re.search(r"arXiv[:\s]+(\d{4}\.\d{4,5})", citation)
    if m:
        g = arxiv_lookup(m.group(1), cache)
        if g: return {**g, "how": "arxiv"}
    terms = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", prior)
             if t.lower() not in GENERIC][:4]
    cands = []
    for t in terms: cands += segments_around(citation, t)
    cands.append(citation[:180])
    seen = set()
    for cand in cands:
        key = cand[:60]
        if key in seen: continue
        seen.add(key)
        g = s2_match(cand, cache)
        if g: return {**g, "how": "title_seg"}
    return None

def main():
    nov = [json.loads(l) for l in open(f"{ROOT}/outputs/reviewer_flags/novelty_flags.jsonl")]
    named = [r for r in nov if r["flag"] and len(r.get("prior", "")) > 5
             and len(r.get("claim", "")) > 20]
    con = sqlite3.connect(PDF_DB)
    cache = Cache(f"{OUTD}/oa_cache2.jsonl")
    stats = {"named": len(named), "cit_from_review": 0, "cit_from_bib": 0,
             "resolved_arxiv": 0, "resolved_title": 0}
    resolved = []
    for r in named:
        forum = r["paper"].replace("iclr_", "")
        citation = None
        marks = re.findall(r"\[(\d{1,2})\]", r["prior"]) or \
                re.findall(r"\[(\d{1,2})\]", r["sent"])
        if marks:
            row = con.execute("SELECT review_text FROM reviews WHERE paper_id=? AND "
                              "review_text LIKE ?", (forum, f"%{r['sent'][:80]}%")).fetchone()
            if row and row[0]:
                defs = review_ref_defs(row[0])
                for n_ in marks:
                    if n_ in defs:
                        citation = defs[n_]; stats["cit_from_review"] += 1; break
        if not citation:
            terms = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", r["prior"])
                     if t.lower() not in GENERIC][:5]
            if terms:
                srow = con.execute("SELECT sections FROM pdf_versions WHERE paper_id=? "
                                   "AND version=0", (forum,)).fetchone()
                if srow and srow[0]:
                    try: s = json.loads(srow[0])
                    except Exception: s = {}
                    refs = s.get("references") or ""
                    if len(refs) > 200:
                        low = refs.lower()
                        for t in sorted(terms, key=len, reverse=True):
                            i = low.find(t.lower())
                            if i >= 0:
                                citation = refs[max(0, i - 300):i + 450]
                                stats["cit_from_bib"] += 1; break
        if not citation: continue
        gold = resolve_gold(citation, r["prior"], cache)
        if not gold: continue
        stats[f"resolved_{'arxiv' if gold['how']=='arxiv' else 'title'}"] += 1
        resolved.append({"paper": r["paper"], "claim": r["claim"], "prior": r["prior"],
                         **{k: gold[k] for k in ("title", "year", "abstract", "how")}})
    n_res = stats["resolved_arxiv"] + stats["resolved_title"]
    res_rate = n_res / max(stats["named"], 1)
    print(f"[bib2] {stats}", flush=True)
    print(f"[bib2] resolution rate: {res_rate:.3f} ({n_res}/{stats['named']})", flush=True)
    rows = []
    for shift, arm in ((0, "matched"), (max(1, len(resolved) // 2), "mismatched")):
        for i, x in enumerate(resolved):
            g = resolved[(i + shift) % len(resolved)]
            c102 = [(str(g["year"]), f"{g['title']}. {g['abstract']}")]
            c103 = [f"{g['title']} ({g['year']}): {g['abstract']}"]
            try:
                v102 = pa_check(x["claim"], 2024, c102, cache)["claim_verdict"]
                v103 = delta_check(x["claim"], c103, cache)["verdict"]
            except Exception:
                continue
            rows.append({"arm": arm, "paper": x["paper"], "claim": x["claim"],
                         "prior": x["prior"], "gold_title": g["title"],
                         "v102": v102, "v103": v103})
    for arm in ("matched", "mismatched"):
        vs = [x for x in rows if x["arm"] == arm]
        n = max(len(vs), 1)
        f102 = sum(1 for x in vs if x["v102"] == "ANTICIPATED") / n
        f103 = sum(1 for x in vs if x["v103"] == "TRIVIAL_DELTA") / n
        fired = sum(1 for x in vs if x["v102"] == "ANTICIPATED"
                    or x["v103"] == "TRIVIAL_DELTA") / n
        print(f"  {arm:10} n={n:3d} ANTICIPATED={f102:.3f} TRIVIAL_DELTA={f103:.3f} "
              f"fired={fired:.3f}", flush=True)
    m = [x for x in rows if x["arm"] == "matched"]
    mm = [x for x in rows if x["arm"] == "mismatched"]
    fm = np.mean([x["v102"] == "ANTICIPATED" or x["v103"] == "TRIVIAL_DELTA" for x in m]) if m else 0
    fmm = np.mean([x["v102"] == "ANTICIPATED" or x["v103"] == "TRIVIAL_DELTA" for x in mm]) if mm else 1
    gate = res_rate >= 0.40 and fm >= 0.35 and fm >= 2 * fmm
    print(f"  GATE G2 (res>=.40 & fired>=.35 & >=2x mismatched): "
          f"{'PASS' if gate else 'FAIL'} (res={res_rate:.3f}, fired={fm:.3f}, "
          f"mism={fmm:.3f})", flush=True)
    with open(f"{OUTD}/gold_bib_results.jsonl", "w") as f:
        for x in rows: f.write(json.dumps(x) + "\n")
    print("\n  examples (matched + fired):", flush=True)
    k = 0
    for x in m:
        if (x["v102"] == "ANTICIPATED" or x["v103"] == "TRIVIAL_DELTA") and k < 6:
            print(f"    {x['paper']}: disputes '{x['claim'][:75]}'\n      named '{x['prior'][:45]}'"
                  f" -> {str(x['gold_title'])[:75]} [{x['v102']}/{x['v103']}]", flush=True)
            k += 1
    print("GOLD_BIB_DONE", flush=True)

if __name__ == "__main__":
    main()
