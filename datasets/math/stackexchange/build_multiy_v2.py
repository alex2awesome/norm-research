#!/usr/bin/env python3
"""math.SE V2 REBUILD — un-binarized multi-y populations, straight from the raw
Stack Exchange dump (task V2 / #44).

Why this exists: every math.SE population in the program descends from
`build_binary_dataset.py`, whose y is "accepted AND score >= 3" vs "score <= 0"
— it DROPS every answer with score 1 or 2 and every high-scoring unaccepted
answer. That "signal gap" makes the cell ineligible for the taste decomposition,
because the two distinct preference signals the site actually records (the
asker's ACCEPT verdict and the crowd's VOTE score) are fused and then censored.
This script rebuilds both signals separately, with no score censoring.

Populations produced (both question-grouped, both label-blind at feature time):

  accepted_verdict : among questions that have >= 2 answers AND a recorded
      accepted answer, y = 1 for the accepted answer, 0 for the others.
      This is the asker's VERDICT. No vote information enters y.

  vote_score       : among questions that have >= 2 answers, y = 1 if the
      answer's raw vote score is strictly above the median score of the answers
      on its own question, 0 if strictly below (ties at the median are dropped,
      exactly the caption crowd-C construction). This is the CROWD signal.
      No accept information enters y.

Both are within-question contrasts, so question-level popularity, age, topic and
difficulty are differenced out by construction, and the grouping unit
(`question_id`) is the container.

Input : Posts.xml from the math.stackexchange dump (streamed with iterparse;
        the file is ~5.8 GB and is never held in memory).
Output: <outdir>/mathse_v2_accepted_verdict.csv.gz
        <outdir>/mathse_v2_vote_score.csv.gz
        <outdir>/mathse_v2_manifest.json
        columns: row_id, question_id, answer_id, group, text, judgement,
                 score, accepted, answer_position, n_answers, answer_year,
                 primary_tag

Usage (CPU only, ~15-30 min on sk3):
  python3 build_multiy_v2.py \
      --posts /lfs/skampere3/0/alexspan/norm-research/datasets/math-stackexchange/raw_dump/Posts.xml \
      --outdir /lfs/skampere3/0/alexspan/norm-research/datasets/math-stackexchange/v2_multiy
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

TAG_RE = re.compile(r"<([^<>]+)>")
CODE_RE = re.compile(r"<pre><code>.*?</code></pre>", re.S)
HTML_TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"[ \t]+")
MIN_CHARS = 50
MAX_CHARS = 12000


def clean_body(raw: str) -> str:
    """HTML -> plain text, preserving LaTeX. Deterministic, label-blind."""
    s = raw or ""
    s = CODE_RE.sub(lambda m: "\n" + html.unescape(HTML_TAG_RE.sub("", m.group(0))) + "\n", s)
    s = re.sub(r"</p>|<br\s*/?>", "\n", s, flags=re.I)
    s = HTML_TAG_RE.sub("", s)
    s = html.unescape(s)
    s = WS_RE.sub(" ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def parse_tags(t: str):
    return TAG_RE.findall(t or "")


def stream_posts(path):
    """Yield (post_type, attrib) for every post row, memory-flat."""
    ctx = ET.iterparse(path, events=("start",))
    for _, elem in ctx:
        if elem.tag == "row":
            yield elem.attrib
        elem.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-rows", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    questions = {}          # qid -> dict(title, tags, accepted_id)
    answers_by_q = defaultdict(list)
    n_seen = 0
    for at in stream_posts(a.posts):
        pt = at.get("PostTypeId")
        n_seen += 1
        if a.max_rows and n_seen > a.max_rows:
            break
        if n_seen % 2_000_000 == 0:
            print(f"  ... {n_seen:,} rows, {len(questions):,} questions, "
                  f"{sum(len(v) for v in answers_by_q.values()):,} answers", flush=True)
        if pt == "1":
            questions[at["Id"]] = {
                "title": html.unescape(at.get("Title", "")),
                "tags": parse_tags(at.get("Tags", "")),
                "accepted": at.get("AcceptedAnswerId"),
            }
        elif pt == "2":
            qid = at.get("ParentId")
            if not qid:
                continue
            answers_by_q[qid].append({
                "answer_id": at["Id"],
                "score": int(at.get("Score", 0)),
                "date": at.get("CreationDate", ""),
                "body": at.get("Body", ""),
            })
    print(f"parsed: {n_seen:,} rows | {len(questions):,} questions | "
          f"{sum(len(v) for v in answers_by_q.values()):,} answers", flush=True)

    cols = ["row_id", "question_id", "answer_id", "group", "text", "judgement",
            "score", "accepted", "answer_position", "n_answers", "answer_year",
            "primary_tag"]
    acc_rows, vote_rows = [], []
    n_multi = 0
    for qid, ans in answers_by_q.items():
        if len(ans) < 2:
            continue
        q = questions.get(qid)
        if q is None:
            continue
        n_multi += 1
        ans = sorted(ans, key=lambda r: r["date"])
        prepped = []
        for pos, r in enumerate(ans):
            body = clean_body(r["body"])
            if not (MIN_CHARS <= len(body) <= MAX_CHARS):
                prepped.append(None)
                continue
            text = (f"QUESTION: {q['title']}\n\nANSWER:\n{body}")
            prepped.append({
                "row_id": hashlib.sha1(f"{qid}|{r['answer_id']}".encode()).hexdigest()[:20],
                "question_id": qid, "answer_id": r["answer_id"], "group": qid,
                "text": text, "score": r["score"],
                "accepted": int(q["accepted"] == r["answer_id"]),
                "answer_position": pos, "n_answers": len(ans),
                "answer_year": r["date"][:4],
                "primary_tag": q["tags"][0] if q["tags"] else "",
            })
        keep = [p for p in prepped if p is not None]
        if len(keep) < 2:
            continue

        # --- y (a) accepted verdict -----------------------------------------
        if q["accepted"] and any(p["accepted"] for p in keep):
            for p in keep:
                row = dict(p); row["judgement"] = p["accepted"]
                acc_rows.append(row)

        # --- y (b) vote score, within-question median split -------------------
        med = statistics.median([p["score"] for p in keep])
        for p in keep:
            if p["score"] == med:
                continue
            row = dict(p); row["judgement"] = int(p["score"] > med)
            vote_rows.append(row)

    def write(rows, name):
        path = out / name
        with gzip.open(path, "wt", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r[c] for c in cols})
        pos = sum(r["judgement"] for r in rows)
        g = len(set(r["group"] for r in rows))
        print(f"wrote {path}: n={len(rows):,} pos_rate={pos/max(len(rows),1):.4f} "
              f"groups={g:,}", flush=True)
        return {"path": str(path), "n": len(rows), "pos_rate": pos / max(len(rows), 1),
                "n_groups": g}

    manifest = {
        "source": a.posts,
        "n_posts_seen": n_seen,
        "n_questions": len(questions),
        "n_questions_multi_answer": n_multi,
        "body_filter": f"{MIN_CHARS} <= plain-text chars <= {MAX_CHARS}",
        "accepted_verdict": write(acc_rows, "mathse_v2_accepted_verdict.csv.gz"),
        "vote_score": write(vote_rows, "mathse_v2_vote_score.csv.gz"),
        "y_definitions": {
            "accepted_verdict": "1 = this answer is the question's accepted answer, "
                                "0 = a non-accepted answer on the same question "
                                "(questions with >=2 answers and a recorded accept)",
            "vote_score": "1 = raw vote score strictly ABOVE the median score of the "
                          "answers on its own question, 0 = strictly below; ties at "
                          "the median dropped. NO score censoring, NO accept "
                          "information.",
        },
        "group_column": "question_id",
    }
    (out / "mathse_v2_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print("MATHSE_V2_BUILD_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
