#!/usr/bin/env python3
"""Pilot v2: transcripts for up to 40 Cut-For-Time sketches.

- yt-dlp search (flat, no download) on the official Saturday Night Live
  channel; STRICT title match (normalized sketch title must appear in the
  video title) to avoid wrong-video hits.
- Captions via youtube_transcript_api (manual "en" preferred, else generated).
  No video downloads at any point.
- 4-5s sleep between videos. Results -> yt_pilot_results.jsonl,
  transcripts -> raw/yt_auto_subs/*.txt
"""
import json, os, re, subprocess, time, unicodedata

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "raw")
SUBS = os.path.join(RAW, "yt_auto_subs")
os.makedirs(SUBS, exist_ok=True)
RES = os.path.join(BASE, "yt_pilot_results.jsonl")

def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"weekend update-", "weekend update: ", s.lower())
    return re.sub(r"[^a-z0-9 ]+", "", s).strip()

def cut_titles():
    d = json.load(open(os.path.join(RAW, "cut_for_time_wikitext.json")))
    wt = d["parse"]["wikitext"]["*"]
    out = []
    for m in re.finditer(r"\{\{lineup3\|[^|]*\|([^|]+)\|", wt):
        em = re.match(r"(.+?)-S(\d+)\s*E(\d+)-(.+)", m.group(1).strip())
        if em:
            out.append((em.group(4).strip(), int(em.group(2)), int(em.group(3)), em.group(1).strip()))
    return out

def get_transcript(vid):
    from youtube_transcript_api import YouTubeTranscriptApi
    api = YouTubeTranscriptApi()
    ts = api.list(vid)
    pick = None
    for t in ts:
        if t.language_code.startswith("en") and not t.is_generated:
            pick = t; break
    if pick is None:
        for t in ts:
            if t.language_code.startswith("en"):
                pick = t; break
    if pick is None:
        return None, None
    segs = pick.fetch()
    text = "\n".join(s.text for s in segs)
    return text, ("manual" if not pick.is_generated else "auto")

def main():
    f = open(RES, "w")
    n = 0
    for title, s, e, host in cut_titles():
        if n >= 40:
            break
        n += 1
        clean = re.sub(r"^Weekend Update-", "Weekend Update: ", title)
        query = f"ytsearch5:SNL Cut for Time {clean}"
        r = subprocess.run(["yt-dlp", query, "--skip-download", "--flat-playlist",
                            "--print", "%(id)s\t%(title)s\t%(channel)s"],
                           capture_output=True, text=True, timeout=120)
        best = None
        tn = norm(clean)
        for ln in r.stdout.strip().splitlines():
            parts = ln.split("\t")
            if len(parts) != 3:
                continue
            vid, vtitle, chan = parts
            if "saturday night live" in (chan or "").lower() and tn in norm(vtitle):
                best = (vid, vtitle, chan)
                break
        rec = {"title": title, "season": s, "episode": f"S{s}E{e}", "host": host}
        if best:
            vid, vtitle, chan = best
            rec.update(url=f"https://www.youtube.com/watch?v={vid}",
                       yt_title=vtitle, channel=chan)
            try:
                text, kind = get_transcript(vid)
            except Exception as ex:
                text, kind = None, type(ex).__name__
            if text:
                slug = re.sub(r"[^A-Za-z0-9]+", "_", title)[:60]
                fp = os.path.join(SUBS, f"S{s}E{e}__{slug}.txt")
                open(fp, "w").write(text)
                rec.update(transcript_path=f"raw/yt_auto_subs/S{s}E{e}__{slug}.txt",
                           transcript_kind=kind, status="ok")
            else:
                rec.update(transcript_path=None, status=f"no_transcript:{kind}")
        else:
            rec.update(url=None, transcript_path=None, status="no_confident_match")
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        print(n, title, rec["status"], flush=True)
        time.sleep(4.5)

if __name__ == "__main__":
    main()
