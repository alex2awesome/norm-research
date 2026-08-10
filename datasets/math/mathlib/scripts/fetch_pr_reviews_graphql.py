#!/usr/bin/env python3
"""Fetch all closed/merged PRs of a repo with review-level detail via GitHub GraphQL.

Built for leanprover-community/mathlib4 (Bors-merged: merge fact lives in the
'[Merged by Bors]' title prefix / 'ready-to-merge' label, NOT in mergedAt).

Per PR we fetch what the review-friction label and first-push V metrics need:
  - structural: number, title, state, timestamps, authorAssociation, author,
    additions/deletions/changedFiles, labels
  - review signal: reviews (state APPROVED/CHANGES_REQUESTED/COMMENTED, body,
    author, submittedAt), comment count, review-thread count
  - first-push state: first commit oid + committedDate + CI statusCheckRollup,
    head/merge-commit oids, force-push timeline events (before/after oids) so
    the true first-pushed tree is recoverable from refs/pull/N/head history

Resume-safe: cursor checkpointed to <out>.state.json after every page; restart
just re-runs the same command. Rate-limit aware: reads rateLimit{remaining,
resetAt} from every response and sleeps through the reset when low.

Usage (on sk3, HOME pinned to /lfs per feedback_sk3_afs_tokens):
  GITHUB_TOKEN=$(cat /lfs/.../github_token) python3 fetch_pr_reviews_graphql.py \
      --owner leanprover-community --repo mathlib4 \
      --out /lfs/skampere3/0/alexspan/norm-research/datasets/math/mathlib/pr_reviews_mathlib4.jsonl
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

API = "https://api.github.com/graphql"

QUERY = """
query($owner: String!, $repo: String!, $cursor: String) {
  rateLimit { cost remaining resetAt }
  repository(owner: $owner, name: $repo) {
    pullRequests(states: [MERGED, CLOSED], first: 25, after: $cursor,
                 orderBy: {field: CREATED_AT, direction: ASC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number title state createdAt closedAt mergedAt authorAssociation
        author { login }
        additions deletions changedFiles
        baseRefName headRefOid
        mergeCommit { oid }
        labels(first: 20) { nodes { name } }
        reviews(first: 30) {
          totalCount
          nodes { state submittedAt body author { login } }
        }
        comments(first: 1) { totalCount }
        reviewThreads(first: 1) { totalCount }
        commits(first: 1) {
          totalCount
          nodes { commit { oid committedDate statusCheckRollup { state } } }
        }
        timelineItems(first: 20, itemTypes: [HEAD_REF_FORCE_PUSHED_EVENT]) {
          totalCount
          nodes {
            ... on HeadRefForcePushedEvent {
              createdAt
              beforeCommit { oid }
              afterCommit { oid }
            }
          }
        }
      }
    }
  }
}
"""


def gql(token: str, variables: dict, retries: int = 6) -> dict:
    body = json.dumps({"query": QUERY, "variables": variables}).encode()
    for attempt in range(retries):
        req = urllib.request.Request(
            API, data=body,
            headers={"Authorization": f"bearer {token}",
                     "Content-Type": "application/json",
                     "User-Agent": "norm-research-fetcher"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                out = json.loads(resp.read())
            if "errors" in out and not out.get("data"):
                raise RuntimeError(f"GraphQL errors: {out['errors'][:2]}")
            return out
        except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError,
                TimeoutError, json.JSONDecodeError) as e:
            wait = min(2 ** attempt * 5, 300)
            log(f"request failed ({e}); retry {attempt+1}/{retries} in {wait}s")
            time.sleep(wait)
    raise SystemExit("giving up after retries")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def maybe_sleep_for_ratelimit(rl: dict) -> None:
    if rl["remaining"] < 200:
        reset = datetime.fromisoformat(rl["resetAt"].replace("Z", "+00:00"))
        wait = max((reset - datetime.now(timezone.utc)).total_seconds(), 0) + 30
        log(f"rate limit low ({rl['remaining']}); sleeping {wait:.0f}s")
        time.sleep(wait)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", default="leanprover-community")
    ap.add_argument("--repo", default="mathlib4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--token-file", default=None,
                    help="path to a file holding the token (else $GITHUB_TOKEN)")
    args = ap.parse_args()

    token = (open(args.token_file).read().strip() if args.token_file
             else os.environ.get("GITHUB_TOKEN", ""))
    if not token:
        sys.exit("no token: set GITHUB_TOKEN or --token-file")

    state_path = args.out + ".state.json"
    cursor, n_done = None, 0
    if os.path.exists(state_path):
        st = json.load(open(state_path))
        cursor, n_done = st["cursor"], st["n_done"]
        log(f"resuming from cursor={cursor!r} n_done={n_done}")

    mode = "a" if cursor else "w"
    with open(args.out, mode) as fout:
        while True:
            out = gql(token, {"owner": args.owner, "repo": args.repo,
                              "cursor": cursor})
            data = out["data"]
            prs = data["repository"]["pullRequests"]
            for node in prs["nodes"]:
                if node is None:
                    continue
                fout.write(json.dumps(node) + "\n")
                n_done += 1
            fout.flush()
            page = prs["pageInfo"]
            cursor = page["endCursor"]
            json.dump({"cursor": cursor, "n_done": n_done,
                       "ts": datetime.now(timezone.utc).isoformat()},
                      open(state_path, "w"))
            rl = data["rateLimit"]
            if n_done % 500 < 25:
                log(f"{n_done} PRs fetched; rate remaining {rl['remaining']}")
            maybe_sleep_for_ratelimit(rl)
            if not page["hasNextPage"]:
                break
            time.sleep(0.3)
    log(f"DONE: {n_done} PRs -> {args.out}")


if __name__ == "__main__":
    main()
