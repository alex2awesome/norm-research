#!/usr/bin/env python3
"""
wavee_wayback_fetch.py — Fetch historical snapshots of canonical rubric URLs
from the Internet Archive Wayback Machine into per-task online-rubrics/raw/.

Strategy per original URL:
  1. Hit the CDX API to list all snapshots.
  2. Down-select to one snapshot per year (earliest of that year), keeping
     up to ~6 strides spanning the URL's history.
  3. Fetch each selected snapshot via the standard Wayback URL pattern
     https://web.archive.org/web/<ts>/<original>.
  4. Save to <task>/online-rubrics/raw/wavee_<hash>.html with longer timeout
     (Wayback is slow). Append rows to urls-visited.csv with query=wave_e.
"""
import os, sys, csv, json, hashlib, datetime, ssl, time, urllib.request, urllib.error, urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

DATASETS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
TIMEOUT = 90
MAX_BYTES = 6 * 1024 * 1024
NUM_WORKERS = 4  # Wayback rate-limits aggressive parallelism
INTER_REQUEST_SLEEP = 0.4
RETRIES = 2
QUERY_LABEL = "wave_e"

# Map task -> list of canonical original URLs to seek historical snapshots for
TARGETS = {
    "creative-writing": [
        "https://www.writersdigest.com/literary-agents",
        "https://pen.org/grants/",
        "https://pen.org/literary-awards/",
        "https://pushcartprize.com/about/",
        "http://strangehorizons.com/submit/fiction-submission-guidelines/",
        "http://www.clarkesworldmagazine.com/submissions/",
        "https://nanowrimo.org/pep-talks",
        "https://www.poetryfoundation.org/poetrymagazine/submit",
        "https://tinhouse.com/submission-guidelines/",
        "https://www.theparisreview.org/submissions",
        "https://www.poetsandwriters.com/writing_contests",
    ],
    "peer-review": [
        "https://iclr.cc/Conferences/2018/ReviewerGuidelines",
        "https://iclr.cc/Conferences/2020/ReviewerGuide",
        "https://iclr.cc/Conferences/2022/ReviewerGuide",
        "https://iclr.cc/Conferences/2024/ReviewerGuide",
        "https://www.nature.com/info/peer-review",
        "https://elifesciences.org/about/peer-review",
        "https://publicationethics.org/guidance/Guidelines",
        "https://www.icmje.org/recommendations/",
        "https://nips.cc/Conferences/2020/PaperInformation/ReviewerGuidelines",
        "https://aclweb.org/adminwiki/index.php/ACL_Policies_for_Review_and_Citation",
        "https://www.cambridge.org/core/services/authors/peer-review",
    ],
    "math/stackexchange": [
        "https://www.ams.org/notices/author-info",
        "https://www.maa.org/press/periodicals/convergence",
        "https://arxiv.org/help/submit",
        "https://mathoverflow.net/help",
        "https://math.stackexchange.com/help",
        "https://proofwiki.org/wiki/ProofWiki:Style",
        "https://www.ams.org/journals/notices/",
        "https://aimsciences.org/journal/dcds-a/about",
    ],
    "news-homepages": [
        "https://www.ap.org/about/news-values-and-principles/",
        "http://www.bbc.co.uk/editorialguidelines/",
        "https://www.nytimes.com/editorial-standards/ethical-journalism.html",
        "https://www.reuters.com/about/trust-principles",
        "https://www.npr.org/about-npr/688418842/special-english-edition-of-npr-s-ethics-handbook",
        "https://www.spj.org/ethicscode.asp",
        "https://www.washingtonpost.com/policies-and-standards/",
        "https://www.theguardian.com/info/guardian-editorial-code",
    ],
    "press-releases": [
        "https://www.prsa.org/about/ethics/prsa-code-of-ethics",
        "https://www.cipr.co.uk/CIPR/Our_work/Policy/Code_of_Conduct/CIPR/Policy/Code_of_Conduct.aspx",
        "https://www.prnewswire.com/knowledge-center/",
        "https://www.eurekalert.org/about",
        "https://www.businesswire.com/portal/site/home/about/",
        "https://www.prsa.org/professional-development/accreditation",
    ],
    "code-review": [
        "https://google.github.io/styleguide/",
        "https://about.gitlab.com/handbook/",
        "https://www.python.org/dev/peps/pep-0008/",
        "https://www.kernel.org/doc/html/latest/process/coding-style.html",
        "https://www.kernel.org/doc/html/v4.10/process/coding-style.html",
        "https://google.github.io/eng-practices/review/",
        "https://google.github.io/eng-practices/review/reviewer/standard.html",
        "https://github.com/thoughtbot/guides",
    ],
    "grant-funding": [
        "https://grants.nih.gov/grants/peer-review.htm",
        "https://www.nsf.gov/bfa/dias/policy/papp.jsp",
        "https://erc.europa.eu/funding",
        "https://wellcome.org/grant-funding",
        "https://grants.nih.gov/grants/policy/nihgps/index.htm",
        "https://www.nsf.gov/pubs/policydocs/pappg/index.jsp",
        "https://www.nih.gov/about-nih/what-we-do/mission-goals",
    ],
    "humor": [
        "https://ucbcomedy.com/training",
        "http://www.secondcity.com/training/",
        "https://www.newyorker.com/cartoons",
        "https://www.mcsweeneys.net/pages/submission-guidelines",
        "https://www.theonion.com/about",
        "https://www.uprightcitizensbrigade.com/",
    ],
    "legal-outcome-prediction": [
        "https://www.americanbar.org/groups/professional_responsibility/publications/model_rules_of_professional_conduct/",
        "https://www.supremecourt.gov/ctrules/2017RulesoftheCourt.pdf",
        "https://www.supremecourt.gov/ctrules/rulesofthecourt.aspx",
        "https://www.uscourts.gov/rules-policies",
        "https://www.uscourts.gov/rules-policies/current-rules-practice-procedure",
        "https://www.law.cornell.edu/rules/frcp",
        "https://www.law.cornell.edu/rules/fre",
    ],
    "notice-and-comment": [
        "https://www.regulations.gov/about",
        "https://www.federalregister.gov/reader-aids",
        "https://obamawhitehouse.archives.gov/omb/circulars_a004_a-4/",
        "https://www.acus.gov/recommendations",
        "https://www.reginfo.gov/public/jsp/Utilities/faq.myjsp",
        "https://www.whitehouse.gov/omb/information-regulatory-affairs/regulatory-matters/",
    ],
    "patents": [
        "https://www.uspto.gov/web/offices/pac/mpep/index.html",
        "https://www.uspto.gov/patents/laws/manual-patent-examining-procedure",
        "https://www.epo.org/law-practice/legal-texts/guidelines.html",
        "https://www.wipo.int/pct/en/texts/articles/atoc.html",
        "https://www.jpo.go.jp/e/system/laws/rule/guideline/patent/tukujitu_kijun/index.html",
        "https://www.uspto.gov/patents/basics",
        "https://www.uspto.gov/learning-and-resources/patent-and-trademark-practitioners",
    ],
}


def hash_url(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def cdx_lookup(url: str, retries: int = 1) -> list:
    """Return list of (timestamp, original) tuples from CDX API."""
    # Build URL manually to allow multiple `filter` params
    base = "http://web.archive.org/cdx/search/cdx"
    params = [
        ("url", url),
        ("output", "json"),
        ("limit", "200"),
        ("filter", "statuscode:200"),
        ("filter", "mimetype:text/html"),
        ("collapse", "timestamp:6"),  # one snapshot per month
    ]
    qs = "&".join(f"{k}={urllib.parse.quote(str(v), safe='')}" for k, v in params)
    api = f"{base}?{qs}"
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(api, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=45) as r:
                data = json.loads(r.read().decode("utf-8", errors="replace"))
            if not data or len(data) < 2:
                return []
            header = data[0]
            ti = header.index("timestamp")
            oi = header.index("original")
            return [(row[ti], row[oi]) for row in data[1:]]
        except Exception:
            if attempt < retries:
                time.sleep(2 + attempt * 2)
                continue
            return []
    return []


def yearly_strides(snaps: list, max_n: int = 6) -> list:
    """Pick at most one snapshot per year (earliest), then sample evenly up to max_n."""
    by_year = {}
    for ts, orig in snaps:
        y = ts[:4]
        if y not in by_year:
            by_year[y] = (ts, orig)
    picks = sorted(by_year.values(), key=lambda x: x[0])
    if len(picks) <= max_n:
        return picks
    # Sample evenly across the range
    step = len(picks) / max_n
    out = []
    for i in range(max_n):
        out.append(picks[int(i * step)])
    # Always include the most recent
    if picks[-1] not in out:
        out[-1] = picks[-1]
    return out


def build_wayback_url(ts: str, original: str) -> str:
    return f"https://web.archive.org/web/{ts}/{original}"


def fetch(url: str):
    last_err = (0, "", b"")
    for attempt in range(RETRIES + 1):
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
                if INTER_REQUEST_SLEEP:
                    time.sleep(INTER_REQUEST_SLEEP)
                return r.status, ctype, data
        except urllib.error.HTTPError as e:
            try:
                data = e.read()
            except Exception:
                data = b""
            ctype = (e.headers.get("Content-Type", "") if e.headers else "")
            # 429 / 503 -> backoff & retry
            if e.code in (429, 503) and attempt < RETRIES:
                time.sleep(5 + 5 * attempt)
                continue
            return e.code, ctype, data
        except Exception as e:
            last_err = (0, str(e)[:120], b"")
            if attempt < RETRIES:
                time.sleep(3 + 2 * attempt)
                continue
    return last_err


def ext_for(url: str, ctype: str) -> str:
    u = url.lower().split("?")[0].split("#")[0]
    for ex in (".pdf", ".html", ".htm", ".txt", ".json", ".xml"):
        if u.endswith(ex):
            return ".pdf" if ex == ".pdf" else (".html" if ex == ".htm" else ex)
    if ctype:
        c = ctype.lower()
        if "pdf" in c: return ".pdf"
        if "json" in c: return ".json"
        if "xml" in c: return ".xml"
        if "text/plain" in c: return ".txt"
    return ".html"


def task_paths(task: str):
    online = os.path.join(DATASETS_ROOT, task, "online-rubrics")
    raw = os.path.join(online, "raw")
    csv_path = os.path.join(online, "urls-visited.csv")
    if not os.path.isdir(raw):
        return None, None
    return raw, csv_path


def load_seen(csv_path: str, only_success: bool = True) -> set:
    """Load URLs already attempted. If only_success=True, return only the
    successfully-fetched URLs so failures will be retried."""
    seen = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            r = csv.reader(f)
            try: next(r)
            except StopIteration: pass
            for row in r:
                if not row: continue
                if only_success:
                    # row: url, query, filename, http_status, bytes, fetched_at
                    try:
                        if row[2] and int(row[3]) == 200:
                            seen.add(row[0])
                    except (IndexError, ValueError):
                        pass
                else:
                    seen.add(row[0])
    return seen


def append_rows(csv_path: str, rows: list):
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["url", "query", "filename", "http_status", "bytes", "fetched_at"])
        for row in rows:
            w.writerow(row)


def cdx_for_task(task: str, originals: list, max_per_url: int = 6):
    """Run CDX lookups for a task in parallel. Returns (targets, notable)."""
    raw, csv_path = task_paths(task)
    if raw is None:
        print(f"[{task}] raw dir missing, skipping", file=sys.stderr, flush=True)
        return [], []
    seen = load_seen(csv_path)
    targets = []
    notable = []

    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(cdx_lookup, o): o for o in originals}
        for fut in as_completed(futs):
            orig = futs[fut]
            try:
                snaps = fut.result()
            except Exception as e:
                snaps = []
            if not snaps:
                print(f"[{task}] no snapshots for {orig}", flush=True)
                continue
            picks = yearly_strides(snaps, max_n=max_per_url)
            years = []
            for ts, original_in_cdx in picks:
                wb = build_wayback_url(ts, original_in_cdx)
                if wb not in seen:
                    targets.append((task, wb, ts[:4], orig))
                    years.append(ts[:4])
            notable.append((orig, years))
            print(f"[{task}] {orig} -> picked {len(picks)} snaps, {len(years)} new ({','.join(years)})", flush=True)
    return targets, notable


def fetch_targets(targets):
    """Fetch (task, wb_url, year, orig) tuples; group rows per task; return per-task counts."""
    per_task_rows = {}
    per_task_ok = {}
    if not targets:
        return per_task_rows, per_task_ok

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futs = {pool.submit(fetch, t[1]): t for t in targets}
        for fut in as_completed(futs):
            task, wb, year, orig = futs[fut]
            raw, _ = task_paths(task)
            try:
                status, ctype, data = fut.result()
            except Exception as e:
                status, ctype, data = 0, str(e)[:120], b""
            now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if status != 200 or len(data) < 400:
                per_task_rows.setdefault(task, []).append((wb, QUERY_LABEL, "", status, len(data), now))
                continue
            ext = ext_for(wb, ctype)
            fname = f"wavee_{hash_url(wb)}{ext}"
            fpath = os.path.join(raw, fname)
            try:
                with open(fpath, "wb") as out:
                    out.write(data)
                per_task_ok[task] = per_task_ok.get(task, 0) + 1
                per_task_rows.setdefault(task, []).append((wb, QUERY_LABEL, fname, status, len(data), now))
            except Exception:
                per_task_rows.setdefault(task, []).append((wb, QUERY_LABEL, "", status, len(data), now))
    return per_task_rows, per_task_ok


def main():
    only = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    notable_all = {}
    all_targets = []
    for task, originals in TARGETS.items():
        if only and task not in only:
            continue
        targets, notable = cdx_for_task(task, originals)
        notable_all[task] = notable
        all_targets.extend(targets)

    print(f"\n[CDX done] {len(all_targets)} fetches queued across tasks", flush=True)

    per_task_rows, per_task_ok = fetch_targets(all_targets)

    # Append CSV rows per-task
    for task, rows in per_task_rows.items():
        _, csv_path = task_paths(task)
        if csv_path:
            append_rows(csv_path, rows)

    print("\n=== SUMMARY ===")
    total = 0
    # Compute per-task attempted (rows length) and ok
    for task in sorted(set(list(per_task_rows.keys()) + list(per_task_ok.keys()))):
        ok = per_task_ok.get(task, 0)
        att = len(per_task_rows.get(task, []))
        print(f"  {task}: {ok} new files (of {att} attempted)")
        total += ok
    print(f"  TOTAL: {total} new files")

    print("\n=== NOTABLE HISTORICAL CAPTURES ===")
    for task, items in notable_all.items():
        for orig, years in items:
            if years:
                print(f"  [{task}] {orig} :: {','.join(years)}")


if __name__ == "__main__":
    main()
