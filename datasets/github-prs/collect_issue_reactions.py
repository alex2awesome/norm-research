#!/usr/bin/env python3
"""PR -> linked-issue reactions (community demand signal) for gold repos.

For each closed PR: merged?, its own reactions/comments, and for each issue it
closes: that issue's reactions + comments. GraphQL, 50 PRs/page, resumable
per-repo via cursor files. gh CLI supplies auth.
"""
import functools
import gzip
import json
import os
import subprocess
import time

print = functools.partial(print, flush=True)
HERE = os.path.dirname(os.path.abspath(__file__))
import sys
REPOS = sys.argv[1:] or [
    "hashicorp/consul", "traefik/traefik", "hashicorp/nomad",
    "hashicorp/packer", "beego/beego", "opentofu/opentofu",
    "XTLS/Xray-core", "helm/helm", "etcd-io/etcd", "containerd/containerd",
    "spf13/cobra", "gin-gonic/gin", "tektoncd/pipeline"]

Q = """query($owner:String!,$name:String!,$cursor:String){
repository(owner:$owner,name:$name){
 pullRequests(first:50,states:[MERGED,CLOSED],orderBy:{field:CREATED_AT,direction:DESC},after:$cursor){
  pageInfo{hasNextPage endCursor}
  nodes{number title createdAt merged mergedAt additions deletions changedFiles
   author{login}
   reactions{totalCount} comments{totalCount}
   reviews(first:1){totalCount}
   closingIssuesReferences(first:5){nodes{number createdAt
    reactions{totalCount} comments{totalCount}
    reactionGroups{content reactors{totalCount}}}}}}}}"""


def gql(owner, name, cursor):
    cmd = ["gh", "api", "graphql", "-f", f"query={Q}",
           "-f", f"owner={owner}", "-f", f"name={name}"]
    if cursor:
        cmd += ["-f", f"cursor={cursor}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[:200])
    return json.loads(r.stdout)


def main():
    for repo in REPOS:
        owner, name = repo.split("/")
        slug = repo.replace("/", "__")
        outp = os.path.join(HERE, f"issue_reactions__{slug}.jsonl.gz")
        curp = os.path.join(HERE, f".cursor__{slug}")
        if os.path.exists(curp) and open(curp).read().strip() == "DONE":
            continue
        cursor = open(curp).read().strip() or None if os.path.exists(curp) else None
        out = gzip.open(outp, "ab" if cursor else "wb")
        n = 0
        while True:
            try:
                d = gql(owner, name, cursor)
            except Exception as e:
                print(f"{repo}: {e}; sleep 120")
                time.sleep(120)
                continue
            pr = d["data"]["repository"]["pullRequests"]
            for node in pr["nodes"]:
                node["repo"] = repo
                out.write((json.dumps(node) + "\n").encode())
            n += len(pr["nodes"])
            cursor = pr["pageInfo"]["endCursor"]
            open(curp, "w").write(cursor or "")
            if not pr["pageInfo"]["hasNextPage"]:
                break
            if n % 1000 == 0:
                out.flush()
                print(f"{repo}: {n} PRs")
            time.sleep(2)
        out.close()
        open(curp, "w").write("DONE")
        print(f"{repo}: DONE, {n} PRs")
    print("ALL DONE")


if __name__ == "__main__":
    main()
