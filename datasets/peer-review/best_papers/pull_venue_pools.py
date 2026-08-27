#!/usr/bin/env python3
"""Pull y=0 candidate pools: same-venue same-year works from OpenAlex.

OpenAlex conference->source mapping is messy: most venues are split across a
MAG-era generic source ("International Conference on Software Engineering"),
per-year IEEE/ACM proceedings sources ("2015 IEEE/ACM 37th IEEE International
Conference on Software Engineering"), and sometimes a journal that carries the
proceedings (VLDB -> "Proceedings of the VLDB Endowment", SIGMETRICS ->
POMACS, SIGMOD 2023+ -> "Proceedings of the ACM on Management of Data").

Resolution strategy, per venue:
  1. empirical: primary_location.source of award papers matched in
     best_papers_joined.csv (excluding repositories/arXiv) -- ground truth
     for at least the award years;
  2. name search: sources?filter=display_name.search:<query>, paginated,
     filtered by per-venue ACCEPT/REJECT regexes.
  Union of both, every accepted source logged to sources_resolved.json.

Then cursor-paginates works (locations.source.id OR-filter, chunked by 40,
type:article, publication_year min..max award year per venue) into
pools/{venue}.parquet. Reports per-(venue, award-year) pool sizes; pairs with
pool < MIN_POOL works are flagged unresolved in pool_coverage.csv.
"""
import json
import os
import re
import time
import unicodedata

import pandas as pd
import requests

MAILTO = "alex2awesome@gmail.com"
WORKS_API = "https://api.openalex.org/works"
SOURCES_API = "https://api.openalex.org/sources"
MIN_POOL = 20

# (display_name.search queries, accept regex, reject regex)
# NOTE on multiplexed ACM journals: PACMPL ("Proceedings of the ACM on
# Programming Languages") carries PLDI but also POPL/OOPSLA/ICFP -- it must
# NOT be accepted for PLDI (pool would mix sibling conferences). POMACS is
# SIGMETRICS-only and PACMMOD is SIGMOD-only; those are accepted.
VENUE_CFG = {
    "AAAI": ("AAAI Conference on Artificial Intelligence",
             r"AAAI Conference on Artificial Intelligence$", r"interactive"),
    "ACL": ("Annual Meeting of the Association for Computational Linguistics",
            r"Annual Meeting of the Association for Computational Linguistics",
            r"workshop|student|demonstr|tutorial|findings"),
    "CHI": ("Conference on Human Factors in Computing Systems",
            r"(SIGCHI|CHI) Conference on Human Factors in Computing Systems",
            r"extended abstracts|companion|workshop|adjunct"),
    "CIKM": ("Conference on Information and Knowledge Management",
             r"Conference on Information (and|&) Knowledge Management",
             r"education"),
    "CVPR": ("Computer Vision and Pattern Recognition",
             r"(Conference on )?Computer Vision and Pattern Recognition",
             r"workshop|advances in|image|asian|chinese"),
    "FOCS": ("Symposium on Foundations of Computer Science",
             r"Symposium on Foundations of Computer Science", None),
    "FSE": ("Foundations of Software Engineering",
            r"Foundations of Software Engineering", r"companion|workshop"),
    "ICCV": ("International Conference on Computer Vision",
             r"(IEEE|CVF|^)[^,]*International Conference on Computer Vision"
             r"( \(ICCV\))?$",
             r"workshop|theory|image|pattern|graphics"),
    "ICML": ("International Conference on Machine Learning",
             r"International Conference on Machine Learning$",
             r"applications|big data|communications|cybernetics|workshop"),
    "ICSE": ("International Conference on Software Engineering",
             r"International Conference on Software Engineering",
             r"companion|workshop|practice|education|new ideas|proceedings/"),
    "IJCAI": ("International Joint Conference on Artificial Intelligence",
              r"International Joint Conference on Artificial Intelligence",
              None),
    "INFOCOM": ("IEEE INFOCOM",
                r"INFOCOM", r"workshop"),
    "ISCA": ("International Symposium on Computer Architecture",
             r"International Symposium on Computer Architecture", None),
    "KDD": ("Knowledge Discovery and Data Mining",
            r"(SIGKDD|Knowledge Discovery (and|&) Data Mining)",
            r"^data mining|wiley|chapman|lecture|pacific|workshop"),
    "MOBICOM": ("Mobile Computing and Networking",
                r"Mobile Computing and Networking", r"workshop"),
    "NSDI": ("Networked Systems Design and Implementation",
             r"Networked Systems Design and Implementation", None),
    "NeurIPS": ("Neural Information Processing Systems",
                r"Neural Information Processing Systems$", r"advances"),
    "OSDI": ("Operating Systems Design and Implementation",
             r"Operating Systems Design and Implementation", None),
    "PLDI": ("Programming Language Design and Implementation",
             r"Programming Language Design and Implementation", None),
    "PODS": ("Principles of Database Systems",
             r"Principles of Database Systems", None),
    "S&P": ("IEEE Symposium on Security and Privacy",
            r"(IEEE )?Symposium on Security and Privacy", r"workshop"),
    "SIGCOMM": (["SIGCOMM", "Data Communication"],
                r"(SIGCOMM|Special Interest Group on Data Communication)",
                r"workshop|posters|review|education|latin|asia"),
    "SIGGRAPH": ("SIGGRAPH",
                 r"^(Proceedings of |ACM )?SIGGRAPH (\d{4}|Conference)",
                 r"asia|computer graphics$|courses|talks|posters|emerging"),
    "SIGIR": ("SIGIR Conference on Research and Development in Information "
              "Retrieval",
              r"Research (and|&) Development in Information Retrieval",
              r"workshop"),
    "SIGMETRICS": ("SIGMETRICS",
                   r"(SIGMETRICS|Measurement and Modeling of Computer Systems"
                   r"|Measurement and Analysis of Computing Systems)",
                   r"performance evaluation review"),
    "SIGMOD": ("Management of Data",
               r"(International Conference on Management of Data"
               r"|ACM on Management of Data)",
               r"big data economy"),
    "SODA": ("Symposium on Discrete Algorithms",
             r"Symposium on Discrete Algorithms", None),
    "SOSP": ("Symposium on Operating Systems Principles",
             r"Symposium on Operating Systems Principles", None),
    "STOC": ("Symposium on Theory of Computing",
             r"Symposium on (the )?Theory of Computing", None),
    "UIST": ("User Interface Software and Technology",
             r"User Interface Software and Technology", r"adjunct"),
    "VLDB": ("Very Large Data Bases",
             r"(Very Large Data Bases|VLDB Endowment)", r"journal"),
    "WWW": (["World Wide Web Conference", "The Web Conference",
             "World Wide Web"],
            r"(World Wide Web Conference|The (ACM )?Web Conference"
            r"|International Conference on World Wide Web)",
            r"companion|workshop|internet"),
}


def norm_text(s):
    """NFKC + curly->straight quotes; identical to the Task B normalization
    (formatting leaks are a standing hazard)."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = (s.replace("‘", "'").replace("’", "'")
          .replace("“", '"').replace("”", '"')
          .replace("–", "-").replace("—", "-"))
    return s


def decode_abstract(inv):
    if not inv:
        return ""
    pos = [(i, w) for w, idxs in inv.items() for i in idxs]
    pos.sort()
    return " ".join(w for _, w in pos)


def api_get(session, url, params):
    for attempt in range(6):
        try:
            r = session.get(url, params=params, timeout=120)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError(f"API gave up: {url} {params}")


def discover_sources(session, venue, candidates=None):
    """Candidate sources by name, filtered by accept/reject regexes.

    If `candidates` (pre-fetched {venue: [{id,name,type,works}]} json) is
    given, filter those instead of hitting the API: display_name.search is a
    search-type request (10 credits each under the 2026 OpenAlex credit
    system) and the candidate lists were already fetched once.
    """
    queries, acc, rej = VENUE_CFG[venue]
    if isinstance(queries, str):
        queries = [queries]
    out = {}

    def consider(sid, name, typ, works):
        if not re.search(acc, name, re.I):
            return
        if rej and re.search(rej, name, re.I):
            return
        if typ not in ("conference", "journal"):
            return
        if (works or 0) < 10:
            return
        out[sid] = {"name": name, "type": typ, "works": works,
                    "via": "name_search"}

    if candidates is not None:
        for c in candidates.get(venue, []):
            consider(c["id"], c["name"], c["type"], c["works"])
        return out

    for query in queries:
        page = 1
        while True:
            j = api_get(session, SOURCES_API, {
                "filter": f"display_name.search:{query}",
                "select": "id,display_name,type,works_count",
                "per-page": "200", "page": str(page), "mailto": MAILTO})
            for s in j["results"]:
                consider(s["id"].split("/")[-1], s["display_name"],
                         s["type"], s["works_count"])
            if len(j["results"]) < 200:
                break
            page += 1
            time.sleep(0.15)
        time.sleep(0.15)
    return out


def award_sources(joined):
    """Sources where matched award papers actually live (non-repository)."""
    out = {}
    for r in joined.itertuples():
        if not r.matched or not isinstance(r.oa_source, str):
            continue
        out.setdefault(r.venue, set()).add(r.oa_source)
    return out


def main():
    candidates = None
    if os.path.exists("venue_source_candidates.json"):
        with open("venue_source_candidates.json") as f:
            candidates = json.load(f)
        print("using pre-fetched venue_source_candidates.json "
              "(no search-credit spend)")
    joined = pd.read_csv("best_papers_joined.csv")
    award_years = (joined.groupby("venue").year.agg(["min", "max"])
                   .to_dict("index"))
    award_vy = set(zip(joined.venue, joined.year))

    session = requests.Session()
    session.headers["User-Agent"] = f"norm-research (mailto:{MAILTO})"

    # empirical sources from award joins: need source IDs, so re-query the
    # matched works' primary_location.source.id in bulk via the works API
    ids = [i.split("/")[-1] for i in
           joined.loc[joined.matched, "openalex_id"].dropna()]
    emp = {}  # venue -> {source_id: name}
    id2venue = dict(zip(joined.openalex_id.fillna(""), joined.venue))
    for chunk_start in range(0, len(ids), 50):
        chunk = ids[chunk_start:chunk_start + 50]
        j = api_get(session, WORKS_API, {
            "filter": "openalex_id:" + "|".join(chunk),
            "select": "id,primary_location", "per-page": "50",
            "mailto": MAILTO})
        for w in j["results"]:
            src = (w.get("primary_location") or {}).get("source") or {}
            if not src.get("id") or src.get("type") == "repository":
                continue
            venue = id2venue.get(w["id"])
            if venue:
                emp.setdefault(venue, {})[src["id"].split("/")[-1]] = \
                    src.get("display_name")
        time.sleep(0.15)

    os.makedirs("pools", exist_ok=True)
    resolved = {}
    coverage_rows = []
    for venue in sorted(VENUE_CFG):
        srcs = discover_sources(session, venue, candidates)
        # award-derived sources must pass the same name regexes: matched
        # award papers sometimes live in journal versions (CVPR -> TPAMI,
        # NeurIPS -> JACM) or are outright mismatches; without this filter
        # those sources inject tens of thousands of foreign works
        acc, rej = VENUE_CFG[venue][1], VENUE_CFG[venue][2]
        for sid, name in emp.get(venue, {}).items():
            if sid in srcs:
                srcs[sid]["via"] = srcs[sid]["via"] + "+award"
                continue
            if not name or not re.search(acc, name, re.I):
                continue
            if rej and re.search(rej, name, re.I):
                continue
            srcs[sid] = {"name": name, "type": "?", "works": None,
                         "via": "award_paper"}
        resolved[venue] = srcs
        if not srcs:
            print(f"{venue}: NO SOURCES RESOLVED", flush=True)
            continue

        pq_path = f"pools/{venue.replace('&', 'and')}.parquet"
        if os.path.exists(pq_path):  # resume after budget exhaustion
            df = pd.read_parquet(pq_path)
            print(f"{venue}: parquet exists ({len(df)} works), skipping pull",
                  flush=True)
            per_year = df.groupby("publication_year").size()
            y0, y1 = award_years[venue]["min"], award_years[venue]["max"]
            for y in range(y0, y1 + 1):
                if (venue, y) in award_vy:
                    n = int(per_year.get(y, 0))
                    coverage_rows.append({"venue": venue, "year": y,
                                          "pool_n": n,
                                          "resolved": n >= MIN_POOL})
            continue
        y0, y1 = award_years[venue]["min"], award_years[venue]["max"]
        sids = list(srcs)
        rows = []
        for cs in range(0, len(sids), 40):
            filt = ("locations.source.id:" + "|".join(sids[cs:cs + 40]) +
                    f",publication_year:{y0}-{y1},type:article")
            cursor = "*"
            while cursor:
                j = api_get(session, WORKS_API, {
                    "filter": filt,
                    "select": ("id,doi,title,publication_year,type,"
                               "cited_by_count,abstract_inverted_index,"
                               "primary_location"),
                    "per-page": "200", "cursor": cursor, "mailto": MAILTO})
                for w in j["results"]:
                    src = (w.get("primary_location") or {}).get("source") or {}
                    rows.append({
                        "venue": venue,
                        "openalex_id": w["id"],
                        "doi": w.get("doi"),
                        "title": w.get("title"),
                        "publication_year": w.get("publication_year"),
                        "cited_by_count": w.get("cited_by_count"),
                        "abstract": decode_abstract(
                            w.get("abstract_inverted_index")),
                        "primary_source": src.get("display_name"),
                    })
                cursor = j["meta"].get("next_cursor")
                if not j["results"]:
                    break
                time.sleep(0.12)
        if not rows:
            print(f"{venue}: {len(srcs)} sources but ZERO pool works "
                  f"({y0}-{y1})", flush=True)
            for y in range(y0, y1 + 1):
                if (venue, y) in award_vy:
                    coverage_rows.append({"venue": venue, "year": y,
                                          "pool_n": 0, "resolved": False})
            continue
        df = pd.DataFrame(rows).drop_duplicates("openalex_id")
        df["title"] = df.title.map(norm_text)
        df["abstract"] = df.abstract.map(norm_text)
        df["text"] = (df.title + "\n\n" + df.abstract).str.strip()
        df.to_parquet(f"pools/{venue.replace('&', 'and')}.parquet",
                      index=False)
        per_year = df.groupby("publication_year").size()
        for y in range(y0, y1 + 1):
            if (venue, y) not in award_vy:
                continue
            n = int(per_year.get(y, 0))
            coverage_rows.append({"venue": venue, "year": y, "pool_n": n,
                                  "resolved": n >= MIN_POOL})
        print(f"{venue}: {len(srcs)} sources, {len(df)} pool works "
              f"({y0}-{y1})", flush=True)

    with open("sources_resolved.json", "w") as f:
        json.dump(resolved, f, indent=1)
    cov = pd.DataFrame(coverage_rows)
    cov.to_csv("pool_coverage.csv", index=False)
    print("\nresolution summary (award venue-years with pool >= "
          f"{MIN_POOL} works):")
    print(cov.groupby("venue").resolved.agg(["mean", "size"]).to_string())


if __name__ == "__main__":
    main()
