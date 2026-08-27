#!/usr/bin/env python3
"""Parse the AoPS Wiki scrape (full.jsonl.gz) into clean tables.

Inputs:  datasets/math/aops/wiki/full.jsonl.gz
Outputs (datasets/math/aops/wiki/):
  problems.parquet     — one row per (variant, problem)
  solutions.parquet    — one row per non-video solution section
  answer_keys.parquet  — one row per (variant, problem_n)
  wiki_parse_manifest.json — validation stats (also printed)

CPU-only, local. Run: python3 datasets/math/aops/scripts/parse_wiki.py
"""
import gzip
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # datasets/math/aops
WIKI = ROOT / "wiki"
INPUT = WIKI / "full.jsonl.gz"

HEADING_RE = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$", re.M)
ANSWER_LINE_RE = re.compile(r"^#\s*(.*?)\s*$", re.M)
REDIRECT_RE = re.compile(r"(?i)^\s*#redirect\s*\[\[(.*?)\]\]")
CHINA_RE = re.compile(r"(?i)china|chinese\s+test")


# ---------------------------------------------------------------- sectioning

def split_sections(wikitext):
    """Return (preamble, [ {level, title, body} ]) from MediaWiki source."""
    matches = list(HEADING_RE.finditer(wikitext))
    preamble = wikitext[: matches[0].start()].strip() if matches else wikitext.strip()
    sections = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(wikitext)
        sections.append(
            {"level": len(m.group(1)), "title": m.group(2), "body": wikitext[start:end].strip()}
        )
    return preamble, sections


def classify(title):
    t = title.strip().lower()
    if "video" in t:
        return "video"
    if re.match(r"^problems?\b", t) or re.match(r"^problem\s*\d", t):
        return "problem"
    if re.match(r"^solutions?\b", t):
        return "solution"
    if re.match(r"^see\s+also\b", t):
        return "nav"
    return "other"  # Note, Remark, Diagram, ...


def parse_problem_page(row):
    """Return (problem_statement, [(heading, text)] solutions, n_video)."""
    preamble, sections = split_sections(row["wikitext"])

    # problem statement: first 'Problem*' section, else preamble before headings
    statement = None
    for s in sections:
        if classify(s["title"]) == "problem":
            statement = s["body"]
            break
    if statement is None:
        statement = preamble

    n_video = sum(1 for s in sections if classify(s["title"]) == "video")

    # solutions: every 'Solution*' (non-video) section with a non-empty body.
    # Deeper-nested 'other' sections (e.g. ===Note=== under ==Solution 2==)
    # are appended to the owning solution's text.
    solutions = []  # list of [heading, text, level]
    for s in sections:
        kind = classify(s["title"])
        if kind == "solution":
            # 'Solutions' wrapper with no direct body -> skip (subsections are own rows)
            if s["body"]:
                solutions.append([s["title"], s["body"], s["level"]])
        elif kind == "other" and solutions and s["level"] > solutions[-1][2]:
            solutions[-1][1] += "\n\n== {} ==\n{}".format(s["title"], s["body"])
        elif kind in ("video", "nav", "problem"):
            continue
    return statement, [(h, t) for h, t, _ in solutions], n_video


# ---------------------------------------------------------------- answer keys

VALID_TOKEN_RE = re.compile(r"^(?:\d{1,3}|[A-E])$")
TOKEN_RE = re.compile(r"\b(\d{1,3}|[A-E])\b")


def parse_answer_key(row, violations):
    """Return list of dicts (one per problem_n) from a numbered-list key page."""
    contest = row["contest"]
    out = []
    lines = ANSWER_LINE_RE.findall(row["wikitext"])
    for i, raw in enumerate(lines, start=1):
        raw = raw.strip()
        if VALID_TOKEN_RE.match(raw):
            answer, answer_all, valid = raw, raw, True
        else:
            toks = TOKEN_RE.findall(raw)
            # keep only tokens of the right type for the contest
            want_digits = contest == "AIME"
            toks = [t for t in toks if t.isdigit() == want_digits]
            if toks:
                answer, answer_all, valid = toks[0], "|".join(dict.fromkeys(toks)), False
            else:
                answer, answer_all, valid = raw, "", False
            violations.append(
                {"variant": row["variant"], "problem_n": i, "raw": raw, "parsed": answer_all}
            )
        # format validation
        if valid:
            pat = r"^\d{1,3}$" if contest == "AIME" else r"^[A-E]$"
            if not re.match(pat, answer):
                valid = False
                violations.append(
                    {"variant": row["variant"], "problem_n": i, "raw": raw,
                     "parsed": answer, "note": f"wrong format for {contest}"}
                )
        out.append(
            {"contest": contest, "year": row["year"], "variant": row["variant"],
             "problem_n": i, "answer": answer, "answer_all": answer_all,
             "answer_raw": raw, "is_valid": valid}
        )
    return out


# ---------------------------------------------------------------- boxed check

def extract_boxed(text):
    """Return list of \\boxed{...} contents with balanced braces."""
    out = []
    for m in re.finditer(r"\\boxed\s*\{", text):
        depth, i = 1, m.end()
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            out.append(text[m.end(): i - 1])
    return out


LETTER_RE = re.compile(r"\(\s*([A-E])\s*\)")


def boxed_to_answer(content, contest):
    """Normalize a boxed payload to a comparable answer, or None."""
    if contest == "AIME":
        # fractions can collapse to fake integers after brace-stripping
        if re.search(r"\\[dt]?c?frac|/", content):
            return None
        # strip latex commands/braces/whitespace, keep digits
        cleaned = re.sub(r"\\[a-zA-Z]+", " ", content)
        cleaned = re.sub(r"[{}\s~,\\$]", "", cleaned)
        # allow 'n=108' style final boxes
        m = re.fullmatch(r"(?:[A-Za-z]{1,4}=)?(\d{1,3})(?:\.0*)?", cleaned)
        if m:
            return str(int(m.group(1)))
        return None
    else:  # AMC: find the choice letter
        m = LETTER_RE.search(content)
        if m:
            return m.group(1)
        cleaned = re.sub(r"\\[a-zA-Z]+|[{}\s().]", "", content)
        if re.fullmatch(r"[A-E]", cleaned):
            return cleaned
        return None


def key_to_comparable(answer_all, contest):
    """Accepted-answer set in the same normalized form as boxed_to_answer."""
    vals = set()
    for tok in answer_all.split("|"):
        tok = tok.strip()
        if not tok:
            continue
        if contest == "AIME" and tok.isdigit():
            vals.add(str(int(tok)))
        elif contest != "AIME" and re.fullmatch(r"[A-E]", tok):
            vals.add(tok)
    return vals


# ---------------------------------------------------------------- main

def main():
    random.seed(2026)
    rows = [json.loads(l) for l in gzip.open(INPUT, "rt")]
    ok = [r for r in rows if r["status"] == "ok"]
    prob_rows = [r for r in ok if r["kind"] == "problem"]
    key_rows = [r for r in ok if r["kind"] == "answer_key"]

    # ---- problems + solutions (resolving cross-contest #redirect pages)
    by_title = {r["title"]: r for r in prob_rows}
    problems, solutions = [], []
    empty_statements, unresolved_redirects = [], []
    n_redirects = 0
    for r in prob_rows:
        redirect_to = ""
        src = r
        m = REDIRECT_RE.match(r["wikitext"])
        if m:
            n_redirects += 1
            target = m.group(1).split("|")[0].strip().replace(" ", "_")
            tgt = by_title.get(target)
            if tgt is not None and not REDIRECT_RE.match(tgt["wikitext"]):
                redirect_to = target
                src = tgt
            else:
                unresolved_redirects.append({"title": r["title"], "target": target})
                src = None
        if src is None:
            statement, sols, n_video = "", [], 0
        else:
            statement, sols, n_video = parse_problem_page(src)
        if not statement:
            empty_statements.append(r["title"])
        problems.append(
            {"contest": r["contest"], "year": r["year"], "variant": r["variant"],
             "problem_n": r["problem"], "problem_statement": statement,
             "n_solutions": len(sols), "n_video_solutions": n_video,
             "is_redirect": bool(m), "redirect_to": redirect_to}
        )
        for idx, (heading, text) in enumerate(sols, start=1):
            solutions.append(
                {"contest": r["contest"], "year": r["year"], "variant": r["variant"],
                 "problem_n": r["problem"], "solution_index": idx,
                 "solution_heading": heading, "solution_text": text,
                 "is_redirect": bool(m), "redirect_to": redirect_to}
            )

    # ---- answer keys
    key_violations = []
    answer_keys = []
    skipped_keys = []
    for r in key_rows:
        entries = parse_answer_key(r, key_violations)
        if not entries:  # e.g. delete-template stub pages
            skipped_keys.append(r["variant"])
            continue
        answer_keys.extend(entries)

    df_p = pd.DataFrame(problems).sort_values(["contest", "year", "variant", "problem_n"])
    df_s = pd.DataFrame(solutions).sort_values(["contest", "year", "variant", "problem_n", "solution_index"])
    df_k = pd.DataFrame(answer_keys).sort_values(["contest", "year", "variant", "problem_n"])

    df_p.to_parquet(WIKI / "problems.parquet", index=False)
    df_s.to_parquet(WIKI / "solutions.parquet", index=False)
    df_k.to_parquet(WIKI / "answer_keys.parquet", index=False)

    # ============================================================ validation
    manifest = {}
    manifest["input_rows"] = len(rows)
    manifest["input_ok"] = len(ok)
    manifest["n_problems"] = len(df_p)
    manifest["n_solutions"] = len(df_s)
    manifest["n_answer_key_entries"] = len(df_k)
    manifest["skipped_key_pages"] = skipped_keys
    manifest["n_redirect_pages_resolved"] = n_redirects - len(unresolved_redirects)
    manifest["unresolved_redirects"] = unresolved_redirects
    manifest["n_empty_problem_statements"] = len(empty_statements)
    manifest["empty_problem_statement_titles"] = empty_statements[:20]

    manifest["problems_per_contest"] = df_p.groupby("contest").size().to_dict()
    manifest["solutions_per_contest"] = df_s.groupby("contest").size().to_dict()
    manifest["key_entries_per_contest"] = df_k.groupby("contest").size().to_dict()

    pct_with_sol = float((df_p["n_solutions"] >= 1).mean() * 100)
    manifest["pct_problems_with_nonvideo_solution"] = round(pct_with_sol, 2)
    manifest["pct_problems_with_nonvideo_solution_per_contest"] = {
        c: round(float(g.mean() * 100), 2)
        for c, g in (df_p["n_solutions"] >= 1).groupby(df_p["contest"])
    }

    # answer-key coverage: problems whose variant has a key AND problem_n present
    key_index = {(v, n) for v, n in zip(df_k["variant"], df_k["problem_n"])}
    answerable = df_p[df_p["contest"].isin(["AIME", "AMC8", "AMC10", "AMC12"])]
    covered = sum((v, n) in key_index for v, n in zip(answerable["variant"], answerable["problem_n"]))
    manifest["answer_key_coverage"] = {
        "answerable_problems (AIME/AMC)": len(answerable),
        "covered": covered,
        "pct": round(100 * covered / len(answerable), 2),
    }
    cov_by = {}
    for c, g in answerable.groupby("contest"):
        cv = sum((v, n) in key_index for v, n in zip(g["variant"], g["problem_n"]))
        cov_by[c] = {"n": len(g), "covered": cv, "pct": round(100 * cv / len(g), 2)}
    manifest["answer_key_coverage_per_contest"] = cov_by

    manifest["answer_key_violations"] = key_violations

    # ---- boxed-vs-key check (the V-check dry run)
    key_lookup = {
        (v, n): (a, c)
        for v, n, a, c in zip(df_k["variant"], df_k["problem_n"], df_k["answer_all"], df_k["contest"])
    }
    checked = matched = last_matched = 0
    mismatches = []
    skipped_china = []
    per_contest = defaultdict(lambda: [0, 0])
    for _, s in df_s.iterrows():
        kv = key_lookup.get((s["variant"], s["problem_n"]))
        if kv is None:
            continue
        answer_all, contest = kv
        accepted = key_to_comparable(answer_all, contest)
        if not accepted:
            continue
        # China-testpaper alternate versions answer a different exam; skip them
        if CHINA_RE.search(s["solution_heading"]):
            skipped_china.append(f"{s['variant']} P{s['problem_n']} {s['solution_heading']}")
            continue
        text = s["solution_text"]
        # some solutions append a China-variant answer at the end; truncate there
        cm = re.search(r"(?i)\[only for certain chinese testpapers\]", text)
        if cm:
            text = text[: cm.start()]
        boxed_vals = [boxed_to_answer(b, contest) for b in extract_boxed(text)]
        boxed_vals = [b for b in boxed_vals if b is not None]
        if not boxed_vals:
            continue
        checked += 1
        per_contest[contest][1] += 1
        if boxed_vals[-1] in accepted:
            last_matched += 1
        # a solution counts as matching if ANY parseable boxed value equals the key
        # (some solutions box intermediate quantities as well as the final answer)
        if any(b in accepted for b in boxed_vals):
            matched += 1
            per_contest[contest][0] += 1
        else:
            mismatches.append(
                {"variant": s["variant"], "problem_n": int(s["problem_n"]),
                 "solution_heading": s["solution_heading"],
                 "key": answer_all,
                 "all_boxed_in_solution": boxed_vals[-5:]}
            )
    manifest["boxed_vs_key"] = {
        "solutions_checked": checked,
        "matched_any_boxed": matched,
        "match_rate_pct": round(100 * matched / checked, 2) if checked else None,
        "matched_last_boxed": last_matched,
        "match_rate_last_boxed_pct": round(100 * last_matched / checked, 2) if checked else None,
        "per_contest": {c: {"matched": m, "checked": n, "pct": round(100 * m / n, 2)}
                        for c, (m, n) in per_contest.items()},
        "skipped_china_variant_solutions": skipped_china,
        "n_mismatches": len(mismatches),
        "mismatch_examples": mismatches[:30],
    }

    # ---- 5-random-AIME spot check
    aime_sols = df_s[(df_s["contest"] == "AIME") & df_s["solution_text"].str.contains(r"\\boxed\{", regex=True)]
    aime_probs = sorted({(v, n) for v, n in zip(aime_sols["variant"], aime_sols["problem_n"])
                         if (v, n) in key_lookup})
    spot = []
    for v, n in random.sample(aime_probs, min(5, len(aime_probs))):
        sub = aime_sols[(aime_sols["variant"] == v) & (aime_sols["problem_n"] == n)]
        answer_all, contest = key_lookup[(v, n)]
        accepted = key_to_comparable(answer_all, contest)
        for _, s in sub.iterrows():
            bv = [boxed_to_answer(b, contest) for b in extract_boxed(s["solution_text"])]
            bv = [b for b in bv if b is not None]
            spot.append({"variant": v, "problem_n": int(n), "heading": s["solution_heading"],
                         "boxed": bv[-1] if bv else None, "key": answer_all,
                         "match": bool(bv and bv[-1] in accepted)})
    manifest["aime_spot_check_5_problems"] = spot

    out = WIKI / "wiki_parse_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))

    # ---------------------------------------------------------------- print
    print(json.dumps({k: v for k, v in manifest.items()
                      if k not in ("boxed_vs_key", "aime_spot_check_5_problems",
                                   "answer_key_violations", "empty_problem_statement_titles")},
                     indent=2))
    print("\nanswer_key_violations:")
    for v in key_violations:
        print("  ", v)
    print("\nAIME spot check (5 random problems):")
    for s in spot:
        print("  ", s)
    bk = manifest["boxed_vs_key"]
    print(f"\nBOXED vs KEY: {bk['matched_any_boxed']}/{bk['solutions_checked']} "
          f"= {bk['match_rate_pct']}% match (any boxed); "
          f"last-boxed only: {bk['match_rate_last_boxed_pct']}%")
    print("skipped China-testpaper variant solutions:", len(bk["skipped_china_variant_solutions"]))
    print("per contest:", bk["per_contest"])
    print(f"mismatches: {bk['n_mismatches']} (first 15 below)")
    for m in bk["mismatch_examples"][:15]:
        print("  ", m)
    print(f"\nwrote {WIKI/'problems.parquet'} ({len(df_p)} rows)")
    print(f"wrote {WIKI/'solutions.parquet'} ({len(df_s)} rows)")
    print(f"wrote {WIKI/'answer_keys.parquet'} ({len(df_k)} rows)")
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
