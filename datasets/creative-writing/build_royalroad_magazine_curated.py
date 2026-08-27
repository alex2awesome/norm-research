#!/usr/bin/env python3
"""U1 CURATED cell — RoyalRoad Community Magazine contest picks.

Design: each magazine edition compiles the FIRST CHAPTER of every contest
entry; invited judges pick the best ~3 entries (prizes announced in the RR
blog). X = the entry chapter text (single renderer: strip tags -> unescape ->
collapse whitespace). y = 1 if the entry's SOURCE fiction is in the blog
winner pool. group = edition (judged-together stratum).

Entry->source join: the "A note from RRM ... found at /fiction/<id>" author
note on each chapter page; fallback = any non-magazine /fiction/ link on the
page. Organizer chapters (intro/afterword/voting; no source link) excluded.

Editions whose winners never got a parsable blog announcement are kept in the
file but flagged edition_labeled=False (and excluded from the default
population view). Output: royalroad_magazine_cell/.
"""
import glob
import gzip
import hashlib
import html
import json
import os
import re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "royalroad_expansion")
OUT_DIR = os.path.join(HERE, "royalroad_magazine_cell")
MAG_IDS = {50199, 55346, 63158, 69303, 79296, 88331, 103100, 120257, 147817, 173505}

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def render(s):
    return WS_RE.sub(" ", TAG_RE.sub(" ", html.unescape(s or ""))).strip()


def norm_title(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def winner_pool():
    winners, winner_titles = set(), set()
    per_blog = {}
    for f in glob.glob(os.path.join(EXP, "blog_raw", "*.html.gz")):
        t = gzip.open(f, "rt", errors="replace").read()
        ttl = re.search(r"<title>(.*?)</title>", t, re.S)
        ttl = (ttl.group(1) if ttl else "").lower()
        if "magazine" not in ttl and "we have a winner" not in ttl:
            continue
        pairs = re.findall(r'/fiction/(\d+)/[\w\-%]*"[^>]*>(.*?)</a>', t, re.S)
        ids = []
        for fid, anchor in pairs:
            fid = int(fid)
            if fid in MAG_IDS:
                continue
            ids.append(fid)
            winner_titles.add(norm_title(render(anchor)))
        ids = list(dict.fromkeys(ids))
        if ids:
            per_blog[os.path.basename(f)] = ids
            winners.update(ids)
    winner_titles.discard("")
    return winners, winner_titles, per_blog


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "rr_magazine_population.jsonl.gz")
    if os.path.exists(out_path):
        i = 2
        while os.path.exists(out_path.replace(".jsonl.gz", f"_v{i}.jsonl.gz")):
            i += 1
        out_path = out_path.replace(".jsonl.gz", f"_v{i}.jsonl.gz")

    winners, winner_titles, per_blog = winner_pool()
    print(f"winner pool {len(winners)} ids / {len(winner_titles)} titles from {len(per_blog)} blogs")

    rows, excl = [], Counter()
    for f in sorted(glob.glob(os.path.join(EXP, "magazine_raw", "edition_*_ch*.html.gz"))):
        m = re.search(r"edition_(\d{4}-\d{2})_ch(\d+)", f)
        ed, cid = m.group(1), m.group(2)
        t = gzip.open(f, "rt", errors="replace").read()
        mt = re.search(r"<h1[^>]*>(.*?)</h1>", t, re.S)
        title = render(mt.group(1)) if mt else ""
        mc = re.search(r'<div class="chapter-inner[^"]*"[^>]*>(.*?)</div>', t, re.S)
        body = render(mc.group(1)) if mc else ""
        # source fiction: author-note first, then any non-magazine fiction link
        notes = re.findall(r'<div class="portlet-body author-note">(.*?)</div>', t, re.S)
        src = None
        for blob in notes + [t]:
            ids = [int(x) for x in re.findall(r"/fiction/(\d+)", blob) if int(x) not in MAG_IDS]
            if ids:
                src = Counter(ids).most_common(1)[0][0]
                break
        low = title.lower()
        if any(k in low for k in ("introduction", "afterword", "voting", "foreword", "table of contents", "rules", "winner", "results", "announcement", "prizes", "judges", "sign up", "closing")):
            excl["organizer"] += 1
            continue
        if src is None:
            excl["no_source_kept_title_matched"] += 1
        if len(body) < 500:
            excl["short_body"] += 1
            continue
        rows.append(dict(
            row_id=f"rrm_{ed}_{cid}", chapter_id=int(cid), edition=ed, group=ed,
            source_fiction_id=src, title=title, text=body,
            judgement=int((src in winners) if src is not None else (norm_title(title) in winner_titles)),
        ))
    # dedupe: one entry per source fiction per edition (organizer chapters that
    # link a past winner as an example are already title-filtered; if a source
    # still appears twice, keep the longest body = the actual entry chapter)
    best = {}
    for r in rows:
        k = (r["edition"], r["source_fiction_id"] if r["source_fiction_id"] is not None else f"cid{r['chapter_id']}")
        if k not in best or len(r["text"]) > len(best[k]["text"]):
            best[k] = r
    dropped_dupes = len(rows) - len(best)
    rows = sorted(best.values(), key=lambda r: r["row_id"])
    excl["dupe_source_in_edition"] = dropped_dupes
    print(f"entries kept {len(rows)}; excluded {dict(excl)}")

    per_ed = Counter(r["edition"] for r in rows)
    per_ed_pos = Counter(r["edition"] for r in rows if r["judgement"])
    labeled_eds = {ed for ed in per_ed if per_ed_pos.get(ed, 0) > 0}
    for ed in sorted(per_ed):
        print(f"  {ed}: {per_ed[ed]} entries, {per_ed_pos.get(ed, 0)} winners"
              + ("" if ed in labeled_eds else "  [NO WINNER ANNOUNCEMENT -> edition_labeled=False]"))
    for r in rows:
        r["edition_labeled"] = r["edition"] in labeled_eds
        h = int(hashlib.sha256(f"rr_magazine::{r['row_id']}".encode()).hexdigest(), 16) % 1000
        r["split"] = "train" if h < 800 else ("eval" if h < 900 else "test")

    with gzip.open(out_path, "wt") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
    lab = [r for r in rows if r["edition_labeled"]]
    manifest = dict(
        cell="cw_royalroad_magazine_curated",
        built_utc=__import__("datetime").datetime.utcnow().isoformat(),
        n_all=len(rows), n_labeled=len(lab), pos=sum(r["judgement"] for r in lab),
        editions={ed: [per_ed[ed], per_ed_pos.get(ed, 0)] for ed in sorted(per_ed)},
        winner_pool=sorted(winners), winner_blogs=per_blog,
        y="entry's source fiction in blog winner pool (judge picks, ~3/edition)",
        x="entry chapter text (magazine compiles each entry's FIRST chapter); single renderer",
        group="edition (judged-together stratum)",
        split_rule="sha256('rr_magazine::'+row_id)%1000 <800/<900/rest",
        power_caveat="POSITIVES ARE FEW (~3/edition); pre-kill checklist minority-count "
                     "rule applies to every readout; PILOT flag mandatory",
        excluded=dict(excl),
    )
    mpath = out_path.replace(".jsonl.gz", "_manifest.json")
    json.dump(manifest, open(mpath, "w"), indent=1)
    print(f"WROTE {out_path}\n      {mpath}")


if __name__ == "__main__":
    main()
