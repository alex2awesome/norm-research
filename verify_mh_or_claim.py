#!/usr/bin/env python3
"""
Verify the claim: MH OR drops from 2.33 to 1.43 when removing helm/helm (38.4% decline)
Focus on P2F (regression) -> rejection analysis.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
import math

def load_repo_cache(cache_path):
    """Load repo cache to get language mapping."""
    repo_info = {}
    with open(cache_path) as f:
        for line in f:
            try:
                repo = json.loads(line)
                key = f"{repo['owner']}/{repo['repo']}"
                repo_info[key] = repo
            except:
                pass
    return repo_info

def normalize_judgement(judgement):
    """Normalize judgement values."""
    if judgement in ['accepted', 'merged']:
        return 'accept'
    elif judgement == 'rejected':
        return 'reject'
    elif judgement in ['true', '1', '0', 'false', 'unknown']:
        return None  # Exclude infra placeholders
    return judgement

def load_verdict_data(verdict_dir):
    """Load and aggregate all verdict data."""
    verdict_path = Path(verdict_dir)
    all_records = []

    for host_dir in verdict_path.iterdir():
        if not host_dir.is_dir():
            continue
        host = host_dir.name

        for repo_dir in host_dir.iterdir():
            if not repo_dir.is_dir():
                continue

            verdict_file = repo_dir / "verdicts.jsonl"
            if not verdict_file.exists():
                continue

            with open(verdict_file) as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        # Use the repo field from the JSON, not directory name
                        record['_host'] = host
                        all_records.append(record)
                    except:
                        pass

    return all_records

def dedup_and_filter(records):
    """Dedup by (repo, paper_id) using majority verdict."""
    from collections import Counter

    # Group by (repo, paper_id)
    groups = defaultdict(list)
    for r in records:
        norm_judgement = normalize_judgement(r.get('judgement'))
        if norm_judgement is None:
            continue
        key = (r.get('repo'), r.get('paper_id'))
        groups[key].append({
            'verdict': r.get('verdict'),
            'judgement': norm_judgement,
            'repo': r.get('repo'),
            'host': r.get('_host'),
            'transitions': r.get('transitions', {})
        })

    # Take majority verdict for each group
    deduped = []
    for key, group_records in groups.items():
        # Count verdicts
        verdict_counts = Counter(r['verdict'] for r in group_records)
        majority_verdict = verdict_counts.most_common(1)[0][0]

        # Get a representative record with majority verdict
        for r in group_records:
            if r['verdict'] == majority_verdict:
                deduped.append(r)
                break

    return deduped

def calculate_p2f_mh_or(records, exclude_repos=None):
    """
    Calculate Mantel-Haenszel OR for P2F (regression) -> rejection.
    P2F = fail_to_pass transitions (regressions)
    """
    if exclude_repos is None:
        exclude_repos = set()

    # Group by repo
    repo_data = defaultdict(lambda: {'a': 0, 'b': 0, 'c': 0, 'd': 0})
    # 2x2 table per repo:
    #              Reject    Accept
    # P2F (reg)      a         b
    # Other          c         d

    for r in records:
        repo = r.get('repo', '')
        if repo in exclude_repos:
            continue

        judgement = r.get('judgement')
        transitions = r.get('transitions', {})

        # P2F = regression = fail_to_pass (tests that passed before but fail after)
        p2f_count = transitions.get('fail_to_pass', {}).get('count', 0)

        # We're only interested in PRs that have at least one P2F transition
        is_p2f = (p2f_count > 0)

        if is_p2f:
            if judgement == 'reject':
                repo_data[repo]['a'] += 1
            elif judgement == 'accept':
                repo_data[repo]['b'] += 1
        else:
            if judgement == 'reject':
                repo_data[repo]['c'] += 1
            elif judgement == 'accept':
                repo_data[repo]['d'] += 1

    # Calculate MH OR
    numerator = 0.0
    denominator = 0.0

    total_repos = 0
    for repo, counts in repo_data.items():
        a, b, c, d = counts['a'], counts['b'], counts['c'], counts['d']

        # Skip repos with no data
        if a + b + c + d == 0:
            continue

        total_repos += 1

        # MH contribution from this repo
        # a*d / (b*c) but with MH formula: (a * d) / n
        n = a + b + c + d

        # MH weighted sum
        if a * d > 0:
            numerator += (a * d) / n
        if b * c > 0:
            denominator += (b * c) / n

    if denominator == 0:
        mh_or = float('inf') if numerator > 0 else 1.0
    else:
        mh_or = numerator / denominator

    return mh_or, total_repos, repo_data

def main():
    verdict_dir = "/tmp/1k_audit"
    repo_cache = "/Users/spangher/Projects/stanford-research/norm-research/datasets/code-review/pr_test_execution/factory/.triage_repo_cache.json"

    print("Loading verdict data...")
    records = load_verdict_data(verdict_dir)
    print(f"Loaded {len(records)} raw verdict records")

    print("Deduplicating and filtering...")
    records = dedup_and_filter(records)
    print(f"After dedup: {len(records)} records")

    # Calculate baseline MH OR
    print("\n=== BASELINE MH OR (all repos) ===")
    mh_or, n_repos, repo_data = calculate_p2f_mh_or(records)
    print(f"MH OR: {mh_or:.2f}")
    print(f"Repos: {n_repos}")

    # Calculate MH OR excluding helm/helm
    print("\n=== EXCLUDING helm/helm ===")
    mh_or_exclude, n_repos_exclude, _ = calculate_p2f_mh_or(records, exclude_repos={'helm/helm'})
    print(f"MH OR: {mh_or_exclude:.2f}")
    print(f"Repos: {n_repos_exclude}")

    # Calculate the decline
    decline_pct = ((mh_or - mh_or_exclude) / mh_or) * 100 if mh_or > 0 else 0
    print(f"\nDecline: {decline_pct:.1f}%")

    # Check for helm/helm in the data
    print("\n=== CHECKING helm/helm ===")
    helm_records = [r for r in records if r.get('repo') == 'helm/helm']
    print(f"helm/helm records: {len(helm_records)}")
    if helm_records:
        helm_p2f = sum(1 for r in helm_records if r.get('transitions', {}).get('fail_to_pass', {}).get('count', 0) > 0)
        helm_p2f_reject = sum(1 for r in helm_records if r.get('judgement') == 'reject' and r.get('transitions', {}).get('fail_to_pass', {}).get('count', 0) > 0)
        helm_p2f_accept = sum(1 for r in helm_records if r.get('judgement') == 'accept' and r.get('transitions', {}).get('fail_to_pass', {}).get('count', 0) > 0)
        print(f"helm/helm P2F: {helm_p2f}")
        print(f"helm/helm P2F->reject: {helm_p2f_reject}")
        print(f"helm/helm P2F->accept: {helm_p2f_accept}")

    # Verify the claim
    print("\n=== CLAIM VERIFICATION ===")
    print(f"Claim: MH OR drops from 2.33 to 1.43 when removing helm/helm")
    print(f"Actual baseline: {mh_or:.2f}")
    print(f"Actual excluded: {mh_or_exclude:.2f}")
    print(f"Claim decline: 38.4%")
    print(f"Actual decline: {decline_pct:.1f}%")

    if abs(mh_or - 2.33) < 0.01 and abs(mh_or_exclude - 1.43) < 0.01:
        print("\n✓ CLAIM CONFIRMED")
    else:
        print("\n✗ CLAIM REFUTED")
        print(f"  Baseline OR: expected 2.33, got {mh_or:.2f}")
        print(f"  Excluded OR: expected 1.43, got {mh_or_exclude:.2f}")

if __name__ == "__main__":
    main()
