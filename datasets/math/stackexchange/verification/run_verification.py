#!/usr/bin/env python3
"""CPU verification step: run the harness over an extracted claims JSONL.

Thin wrapper around harness.verify_claim with:
  * shard-parallel workers (each worker is an independent forked process
    that verifies its slice sequentially; each claim additionally runs in
    its own hard-timeout child, so one pathological claim can never stall
    a worker for more than timeout + grace)
  * resume from existing results (by claim_id)
  * per-answer aggregation into the V-feature CSV described in README.md:
      n_claims, n_verified, n_refuted, frac_checkable,
      has_refuted_load_bearing (+ diagnostic counts)

Usage:
  python3 run_verification.py --claims claims_eval.jsonl \
      [--results results_eval.jsonl] [--features features_eval.csv] \
      [--timeout 10] [--workers 8]
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness  # noqa: E402

SKIP_TYPES = {"EXTRACTION_FAILED"}  # not claims; carried through to features


def load_claims(path: Path):
    claims = []
    with path.open() as fh:
        for ln, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                claims.append(json.loads(line))
            except Exception:
                print(f"warn: skipping unparseable line {ln}", file=sys.stderr)
    return claims


def load_done_claim_ids(path: Path):
    done = set()
    if not path.exists():
        return done
    with path.open() as fh:
        for line in fh:
            try:
                d = json.loads(line)
                if "claim_id" in d:
                    done.add(d["claim_id"])
            except Exception:
                continue
    return done


def _shard_worker(shard, timeout_s, out_file):
    """Runs in its own (non-daemonic) process; writes one shard file."""
    with open(out_file, "w") as fout:
        for c in shard:
            if c.get("claim_type") in SKIP_TYPES:
                res = {**{k: c.get(k) for k in harness.PASSTHROUGH_FIELDS
                          if k in c},
                       "claim_type": c["claim_type"],
                       "verdict": "EXTRACTION_FAILED", "method": "none",
                       "detail": {"errors": c.get("errors", [])[:5]}}
            else:
                res = harness.verify_claim(c, timeout_s=timeout_s)
            fout.write(json.dumps(res, default=str) + "\n")
            fout.flush()
    os._exit(0)


def run_verification(claims, timeout_s, workers, results_path: Path):
    done = load_done_claim_ids(results_path)
    pending = [c for c in claims if c.get("claim_id") not in done or
               c.get("claim_id") is None]
    print(f"{len(claims):,} claim records, {len(done):,} already verified, "
          f"{len(pending):,} pending")
    if not pending:
        return

    workers = max(1, min(workers, len(pending)))
    shards = [pending[i::workers] for i in range(workers)]
    ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods()
                         else "spawn")
    procs = []
    shard_files = []
    t0 = time.time()
    for i, shard in enumerate(shards):
        sf = str(results_path) + f".shard{i}"
        shard_files.append(sf)
        p = ctx.Process(target=_shard_worker, args=(shard, timeout_s, sf),
                        daemon=False)
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    # merge shards into the append-only results file
    with results_path.open("a") as fout:
        for sf in shard_files:
            if os.path.exists(sf):
                with open(sf) as fin:
                    for line in fin:
                        fout.write(line)
                os.remove(sf)
    print(f"verified {len(pending):,} claims in {time.time() - t0:.0f}s "
          f"({workers} workers)")


def aggregate_features(results_path: Path, features_path: Path):
    """Per-answer V features (see README.md aggregation plan)."""
    by_row = defaultdict(list)
    with results_path.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "row_id" in r:
                by_row[r["row_id"]].append(r)

    rows = []
    for row_id, results in sorted(by_row.items()):
        meta = results[0]
        c = Counter(r["verdict"] for r in results)
        n_claims = sum(v for k, v in c.items()
                       if k not in ("NO_CLAIM", "EXTRACTION_FAILED"))
        n_verified = c["VERIFIED_SYMBOLIC"] + c["VERIFIED_NUMERIC"]
        n_refuted = c["REFUTED"]
        has_refuted_lb = any(
            r["verdict"] == "REFUTED" and bool(r.get("load_bearing"))
            for r in results)
        rows.append({
            "row_id": row_id,
            "question_id": meta.get("question_id"),
            "split": meta.get("split"),
            "judgement": meta.get("judgement"),
            "n_claims": n_claims,
            "n_verified": n_verified,
            "n_verified_symbolic": c["VERIFIED_SYMBOLIC"],
            "n_verified_numeric": c["VERIFIED_NUMERIC"],
            "n_refuted": n_refuted,
            "n_inconclusive": c["INCONCLUSIVE"],
            "n_parse_fail": c["PARSE_FAIL"],
            "frac_checkable": round((n_verified + n_refuted) / n_claims, 4)
            if n_claims else 0.0,
            "has_refuted_load_bearing": int(has_refuted_lb),
            "no_claims": int(n_claims == 0),
            "extraction_failed": int(c["EXTRACTION_FAILED"] > 0),
        })

    with features_path.open("w", newline="") as fh:
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"features for {len(rows):,} answers -> {features_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--claims", required=True)
    ap.add_argument("--results", default=None,
                    help="default: results_<claims-suffix>.jsonl alongside claims")
    ap.add_argument("--features", default=None,
                    help="default: features_<claims-suffix>.csv alongside claims")
    ap.add_argument("--timeout", type=float, default=harness.DEFAULT_TIMEOUT_S)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    claims_path = Path(args.claims)
    stem = claims_path.stem.replace("claims_", "") or claims_path.stem
    results_path = Path(args.results) if args.results else \
        claims_path.parent / f"results_{stem}.jsonl"
    features_path = Path(args.features) if args.features else \
        claims_path.parent / f"features_{stem}.csv"

    claims = load_claims(claims_path)
    run_verification(claims, args.timeout, args.workers, results_path)

    tally = Counter()
    with results_path.open() as fh:
        for line in fh:
            try:
                tally[json.loads(line)["verdict"]] += 1
            except Exception:
                continue
    print("verdict tally:", dict(tally))
    aggregate_features(results_path, features_path)


if __name__ == "__main__":
    main()
