#!/usr/bin/env python3
"""Parse raw HTML (BCG forum threads, newsbiscuit forum, garryabbott blog)
into newsjack_rejects.jsonl.

Unit = one forum post (or one blog section). A post may contain several
one-liners; joke_text preserves the post body with paragraph breaks.

Verdict rules (documented in README.md):
- In dedicated rejects threads: default "rejected"; flipped to "aired-claimed"
  when the poster claims material got on the show.
- In general series-discussion threads: only posts with an explicit reject cue
  ("rejected"/"my rejects"/"didn't make it" ...) or an aired claim are kept.
- garryabbott.com posts are titled "Not good enough for the BBC" -> rejected
  (part 5, "Good enough / not good enough", labels its aired items in-text).
"""
import html as ihtml
import json, os, re, glob

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "raw")
OUT = os.path.join(BASE, "newsjack_rejects.jsonl")

THREAD_META = {  # tid -> (label, series, title)
    34847: ("rejects", 20, "Newsjack rejects series 20"),
    35265: ("rejects", 21, "Newsjack Series 21 rejects"),
    35458: ("rejects", 22, "Newsjack Series 22 rejects"),
    35790: ("rejects", 23, "Newsjack Series 23 rejects"),
    33939: ("rejects", None, "Newsjack Rejects (Autumn 2017)"),
    23867: ("rejects", 6, "NewsJack Rejections. Series 6"),
    13736: ("rejects", None, "Newsjack salon de refuse: voxpops"),
    36041: ("series", 24, "Newsjack Series 24"), 35764: ("series", 23, "Newsjack Series 23"),
    35454: ("series", 22, "Newsjack 2020"), 35232: ("series", 21, "Newsjack Series 21"),
    34731: ("series", 20, "Newsjack Series 20"), 34590: ("series", 19, "Newsjack Series 19"),
    34201: ("series", 18, "Newsjack Series 18"), 33509: ("series", None, "Newsjack 2017"),
    32679: ("series", None, "Newsjack 2016"), 32186: ("series", 13, "Newsjack Series 13"),
    31434: ("series", 12, "Newsjack Series 12"), 30665: ("series", 11, "Newsjack Series 11"),
    29520: ("series", 10, "Newsjack Series 10"), 28560: ("series", 9, "Newsjack Series 9"),
    26584: ("series", 8, "Newsjack Series 8"), 26577: ("series", 8, "Newsjack 2013"),
    25345: ("series", 7, "Newsjack Series 7"), 23389: ("series", 6, "Newsjack Series 6"),
    21761: ("series", 5, "Newsjack Series 5"), 19817: ("series", 4, "Newsjack Series 4"),
    17746: ("series", 3, "Newsjack Series 3"), 16165: ("series", 2, "Newsjack Series 2"),
}

AIRED_RE = re.compile(
    r"\b(got (?:one|two|1|2|a couple|a one[- ]?liner|a joke|a line|a sketch|something|it|my \w+ )?(?:in|on|read out|used|through)\b"
    r"|made it (?:in|on(?:to)? (?:the (?:show|air))?)"
    r"|made the (?:show|final cut|broadcast|edit)"
    r"|was (?:used|read out|broadcast|aired|picked)"
    r"|were (?:used|read out|broadcast|aired|picked)"
    r"|got (?:on|onto) (?:the )?(?:show|air|radio)"
    r"|one that got in|first (?:hit|credit)|got a credit|my first credit)\b", re.I)
NEG_AIRED_RE = re.compile(
    r"\b(haven'?t|hasn'?t|didn'?t|never|not|nothing|no luck|nowt|none)\W+(?:\w+\W+){0,6}?"
    r"(got|get|made|make|used|in|on|through)\b", re.I)
OTHERS_RE = re.compile(r"\b(well done|congrats?|congratulations|glad someone|good on you|"
                       r"anyone (?:else )?(?:get|got)|who got)\b", re.I)
REJECT_RE = re.compile(
    r"\b(reject|rejected|rejects|didn'?t (?:make|get)|not used|unused|no luck|"
    r"dud|failed to make|binned|spiked|didn'?t get (?:in|on|used)|cast[- ]?offs?|"
    r"none of (?:mine|my)|swings? and misses)\b", re.I)

def strip_tags(s):
    s = re.sub(r"<br\s*/?>", "\n", s)
    s = re.sub(r"</p>\s*<p>", "\n", s)
    s = re.sub(r"<blockquote[^>]*>", "\n[quote] ", s)
    s = re.sub(r"</blockquote>", " [/quote]\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = ihtml.unescape(s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()

def parse_bcg():
    recs = []
    post_re = re.compile(
        r'<div id="P(\d+)" class="post">.*?<div class="author-details">\s*'
        r'<p><a href="[^"]*">([^<]*)</a></p>\s*<p>([^<]*)</p>'
        r'.*?<div class="post-body">(.*?)</div>\s*<ul class="post-options"',
        re.S)
    for fp in sorted(glob.glob(os.path.join(RAW, "thread_*_p*.html"))):
        m = re.match(r"thread_(\d+)_p(\d+)\.html", os.path.basename(fp))
        tid, page = int(m.group(1)), int(m.group(2))
        label, series, title = THREAD_META.get(tid, ("series", None, f"thread {tid}"))
        url = f"https://www.comedy.co.uk/forums/thread/{tid}/" + (f"{page}/" if page > 1 else "")
        html = open(fp, errors="ignore").read()
        for pm in post_re.finditer(html):
            pid, author, date, body = pm.groups()
            full = strip_tags(body)
            # own words only: drop quoted blocks (other posters' text) before
            # cue detection and output, so quoting someone's reject doesn't
            # relabel the quoting post
            text = re.sub(r"\[quote\].*?\[/quote\]", "", full, flags=re.S).strip()
            if len(text) < 25:
                continue
            aired = bool(AIRED_RE.search(text)) and not NEG_AIRED_RE.search(text) \
                    and not OTHERS_RE.search(text)
            rejectcue = bool(REJECT_RE.search(text)) or bool(NEG_AIRED_RE.search(text))
            if label == "rejects":
                verdict = "aired-claimed" if aired and not rejectcue else "rejected"
            else:
                if aired and not rejectcue:
                    verdict = "aired-claimed"
                elif rejectcue:
                    verdict = "rejected"
                else:
                    continue  # chatter in discussion threads
            recs.append({
                "source": "comedy.co.uk-forum", "source_url": url,
                "thread_id": tid, "thread_title": title, "thread_kind": label,
                "post_id": pid, "author": author.strip(), "date": date.strip(),
                "series": series, "joke_text": text, "verdict": verdict,
                "aired_cue": aired, "reject_cue": rejectcue,
            })
    return recs

def parse_newsbiscuit():
    recs = []
    for fp in sorted(glob.glob(os.path.join(RAW, "newsbiscuit_*_p*.html"))):
        m = re.match(r"newsbiscuit_(\d+)_p(\d+)\.html", os.path.basename(fp))
        tid, page = m.group(1), int(m.group(2))
        url = f"http://newsbiscuit.com/forum/topic.php?id={tid}" + (f"&page={page}" if page > 1 else "")
        html = open(fp, errors="ignore").read()
        title = "Newsjack Rejects" if tid == "120350" else "Newsjack Series 19"
        # bbPress-style: <div class="threadpost"...> or <li id="post-..">
        posts = re.findall(r'<li id="post-(\d+)"[^>]*>(.*?)</li>', html, re.S)
        if not posts:
            posts = re.findall(r'<div class="threadpost[^"]*"[^>]*>(.*?)</div>\s*</div>', html, re.S)
            posts = [(str(i), p) for i, p in enumerate(posts)]
        for pid, block in posts:
            am = re.search(r'<strong>([^<]+)</strong>', block)
            author = am.group(1).strip() if am else ""
            bm = re.search(r'<div class="post">(.*)', block, re.S)
            text = strip_tags(bm.group(1) if bm else block)
            if len(text) < 25:
                continue
            aired = bool(AIRED_RE.search(text))
            rejectcue = bool(REJECT_RE.search(text))
            verdict = "aired-claimed" if aired and not rejectcue else "rejected"
            recs.append({
                "source": "newsbiscuit-forum(wayback)", "source_url": url,
                "thread_id": tid, "thread_title": title, "thread_kind": "rejects",
                "post_id": pid, "author": author, "date": None, "series": None,
                "joke_text": text, "verdict": verdict,
                "aired_cue": aired, "reject_cue": rejectcue,
            })
    return recs

GA_URLS = {
    1: ("https://garryabbott.com/2014/04/16/newsjack-series-ten-critique-with-bonus-jokes/", 10),
    2: ("https://garryabbott.com/2014/10/01/not-good-enough-for-the-bbc-newsjack-series-11/", 11),
    3: ("https://garryabbott.com/2014/10/08/not-good-enough-for-the-bbc-newsjack-series-11-part-2/", 11),
    4: ("https://garryabbott.com/2014/10/15/not-good-enough-for-the-bbc-newsjack-series-11-part-3/", 11),
    5: ("https://garryabbott.com/2014/10/22/not-good-enough-for-the-bbc-newsjack-series-11-part-4/", 11),
    6: ("https://garryabbott.com/2014/10/29/good-enough-not-good-enough-for-the-bbc-newsjack-series-11-part-5/", 11),
    7: ("https://garryabbott.com/2014/11/05/not-good-enough-for-the-bbc-newsjack-series-11-part-6/", 11),
}

def parse_garryabbott():
    recs = []
    for i, (url, series) in GA_URLS.items():
        fp = os.path.join(RAW, f"garryabbott_wayback_{i}.html")
        if not os.path.exists(fp):
            continue
        html = open(fp, errors="ignore").read()
        em = re.search(r'<div class="entry-content[^"]*">(.*?)<footer', html, re.S) or \
             re.search(r'<div class="entry-content[^"]*">(.*?)</div>', html, re.S)
        if not em:
            continue
        text = strip_tags(em.group(1))
        # split on blank lines; each chunk with content = candidate joke/sketch bit
        chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if len(c.strip()) > 40]
        for j, c in enumerate(chunks):
            low = c.lower()
            aired = bool(re.search(r"\b(this one (?:got|was) (?:in|used|read)|made it (?:in|on)|was used|was broadcast|got in)\b", low))
            recs.append({
                "source": "garryabbott.com(wayback)", "source_url": url,
                "thread_id": None, "thread_title": None, "thread_kind": "blog-rejects",
                "post_id": f"chunk{j}", "author": "Garry Abbott", "date": None,
                "series": series, "joke_text": c,
                "verdict": "aired-claimed" if aired else "rejected",
                "aired_cue": aired, "reject_cue": True,
            })
    return recs

def main():
    recs = parse_bcg() + parse_newsbiscuit() + parse_garryabbott()
    with open(OUT, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(len(recs), "records")
    print(Counter((r["source"], r["verdict"]) for r in recs))

if __name__ == "__main__":
    main()
