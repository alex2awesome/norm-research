#!/usr/bin/env python3
"""Phase-2 mathlib fetch: review-thread comment TEXT for PRs with threads.

Phase 1 (fetch_pr_reviews_graphql.py) got thread COUNTS; this gets the
comments themselves — the articulated-norm corpus (math analog of CR.SE
reviewer prose) feeding label 3b (revision pairs) and norm extraction.

Input: friction_full_v2.csv.gz (merged PRs; we fetch those with
n_review_threads > 0). Batches 10 PRs per GraphQL query via aliases.
Resume-safe: skips PR numbers already present in the output file.

Usage (sk3):
  python3 fetch_thread_comments_graphql.py \
      --friction-full friction_full_v2.csv.gz \
      --token-file /lfs/skampere3/0/alexspan/secrets_github_token \
      --out pr_thread_comments.jsonl
"""
import argparse
import csv
import gzip
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

API = "https://api.github.com/graphql"

FRAGMENT = """
fragment T on PullRequest {
  number
  reviewThreads(first: 50) {
    totalCount
    nodes {
      isResolved
      isOutdated
      path
      line
      comments(first: 20) {
        totalCount
        nodes { body author { login } createdAt }
      }
    }
  }
}
"""


def build_query(numbers):
    aliases = "\n".join(
        f'p{i}: pullRequest(number: {n}) {{ ...T }}'
        for i, n in enumerate(numbers))
    return ("query { rateLimit { remaining resetAt } "
            'repository(owner: "leanprover-community", name: "mathlib4") { '
            + aliases + " } }" + FRAGMENT)


def gql(token, query, retries=6):
    body = json.dumps({"query": query}).encode()
    for attempt in range(retries):
        req = urllib.request.Request(
            API, data=body,
            headers={"Authorization": f"bearer {token}",
                     "Content-Type": "application/json",
                     "User-Agent": "norm-research-fetcher"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                out = json.loads(resp.read())
            if out.get("data"):
                return out
            raise RuntimeError(f"GraphQL errors: {str(out.get('errors'))[:200]}")
        except Exception as e:
            wait = min(2 ** attempt * 5, 300)
            print(f"[{datetime.now():%H:%M:%S}] request failed ({e}); "
                  f"retry {attempt+1}/{retries} in {wait}s", flush=True)
            time.sleep(wait)
    raise SystemExit("giving up")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--friction-full", required=True)
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=10)
    args = ap.parse_args()
    token = open(args.token_file).read().strip()

    targets = []
    with gzip.open(args.friction_full, "rt") as f:
        for row in csv.DictReader(f):
            if int(float(row["n_review_threads"] or 0)) > 0:
                targets.append(int(row["number"]))

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["number"])
                except Exception:
                    pass
    todo = [n for n in targets if n not in done]
    print(f"[{datetime.now():%H:%M:%S}] {len(targets)} PRs with threads; "
          f"{len(done)} done; {len(todo)} to fetch", flush=True)

    with open(args.out, "a") as fout:
        for i in range(0, len(todo), args.batch):
            batch = todo[i:i + args.batch]
            out = gql(token, build_query(batch))
            repo = out["data"]["repository"]
            for j, n in enumerate(batch):
                node = repo.get(f"p{j}")
                if node is None:
                    node = {"number": n, "error": "null_node"}
                fout.write(json.dumps(node) + "\n")
            fout.flush()
            rl = out["data"]["rateLimit"]
            if (i // args.batch) % 50 == 0:
                print(f"[{datetime.now():%H:%M:%S}] {i + len(batch)}/{len(todo)} "
                      f"rate remaining {rl['remaining']}", flush=True)
            if rl["remaining"] < 200:
                reset = datetime.fromisoformat(
                    rl["resetAt"].replace("Z", "+00:00"))
                wait = max((reset - datetime.now(timezone.utc)).total_seconds(),
                           0) + 30
                print(f"sleeping {wait:.0f}s for rate limit", flush=True)
                time.sleep(wait)
            time.sleep(0.4)
    print(f"[{datetime.now():%H:%M:%S}] DONE", flush=True)


if __name__ == "__main__":
    main()
