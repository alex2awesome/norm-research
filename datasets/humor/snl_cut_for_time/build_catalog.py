#!/usr/bin/env python3
"""Build snl_catalog.jsonl from raw scrapes.

- cut_for_time rows: from Fandom Cut_For_Time wikitext (lineup3 templates,
  "Host-S## E##-Title").
- aired rows: from snltranscripts.jt.org season category listings (seasons
  2013+; categories 13..18 = seasons starting 2013..2018, 2019..2023 =
  seasons starting those years). These are transcripts of broadcast sketches.
"""
import html as ihtml
import json, os, re, glob

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "raw")
OUT = os.path.join(BASE, "snl_catalog.jsonl")

# SNL season number -> season start year (S39 = 2013-14)
def season_year(snum):
    return 1975 + (snum - 1)

def cut_rows():
    d = json.load(open(os.path.join(RAW, "cut_for_time_wikitext.json")))
    wt = d["parse"]["wikitext"]["*"]
    rows = []
    for m in re.finditer(r"\{\{lineup3\|[^|]*\|([^|]+)\|[^|]*\|(.*?)\}\}", wt, re.S):
        entry, summary = m.group(1).strip(), m.group(2).strip()
        em = re.match(r"(.+?)-S(\d+)\s*E(\d+)-(.+)", entry)
        if not em:
            continue
        host, s, e, title = em.group(1), int(em.group(2)), int(em.group(3)), em.group(4)
        rows.append({
            "title": title.strip(), "host": host.strip(),
            "season": s, "episode": f"S{s}E{e}", "year": season_year(s),
            "verdict": "cut_for_time", "summary": summary,
            "source": "snl.fandom.com/wiki/Cut_For_Time (MediaWiki API)",
            "transcript_path": None, "url": None,
        })
    return rows

CAT_SEASON = {  # category slug -> season start year
    "13": 2013, "14": 2014, "15": 2015, "16": 2016, "17": 2017, "18": 2018,
    "2019": 2019, "2020": 2020, "2021": 2021, "2022": 2022, "2023": 2023,
}
def year_to_season(y):
    return y - 1975 + 1  # 2013 -> S39

def aired_rows():
    rows, seen = [], set()
    art_re = re.compile(
        r'<h2 class="entry-title[^"]*">\s*<a href="(https://snltranscripts\.jt\.org/[^"]+)"[^>]*>(.*?)</a>',
        re.S)
    for fp in sorted(glob.glob(os.path.join(RAW, "category_*_p*.html"))):
        m = re.match(r"category_(\w+)_p(\d+)\.html", os.path.basename(fp))
        cat = m.group(1)
        y = CAT_SEASON[cat]
        html = open(fp, errors="ignore").read()
        found = art_re.findall(html)
        if not found:  # fallback: any transcript link with title text
            found = re.findall(
                r'<a href="(https://snltranscripts\.jt\.org/(?:%s|20\d\d)/[^"]+\.phtml)"[^>]*(?:rel="bookmark")?[^>]*>([^<]{4,})</a>' % cat,
                html)
        for url, title in found:
            title = ihtml.unescape(re.sub(r"<[^>]+>", "", title)).strip()
            if not title or url in seen:
                continue
            seen.add(url)
            # snltranscripts also transcribes some Cut For Time sketches;
            # label those correctly (they were cut after dress, not aired).
            verdict = "cut_for_time" if "cut for time" in title.lower() else "aired"
            rows.append({
                "title": title, "host": None,
                "season": year_to_season(y), "episode": None, "year": y,
                "verdict": verdict, "summary": None,
                "source": f"snltranscripts.jt.org category/{cat}",
                "transcript_path": None, "url": url,
            })
    return rows

def merge_pilot(rows):
    pf = os.path.join(BASE, "yt_pilot_results.jsonl")
    if not os.path.exists(pf):
        return
    pilot = {(r["title"], r["episode"]): r for r in map(json.loads, open(pf))}
    for r in rows:
        p = pilot.get((r["title"], r["episode"]))
        if p and r["verdict"] == "cut_for_time":
            r["url"] = r["url"] or p.get("url")
            r["transcript_path"] = r["transcript_path"] or p.get("transcript_path")

def merge_transcript_samples(rows):
    tdir = os.path.join(RAW, "transcript_samples")
    if not os.path.isdir(tdir):
        return
    have = {f[:-5] for f in os.listdir(tdir) if f.endswith(".html")}
    for r in rows:
        if r.get("url"):
            slug = re.sub(r"[^A-Za-z0-9]+", "_",
                          r["url"].split("/")[-1].replace(".phtml", ""))[:80]
            if slug in have:
                r["transcript_path"] = f"raw/transcript_samples/{slug}.html"

def main():
    rows = cut_rows() + aired_rows()
    merge_transcript_samples(rows)
    merge_pilot(rows)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(len(rows), Counter(r["verdict"] for r in rows))

if __name__ == "__main__":
    main()
