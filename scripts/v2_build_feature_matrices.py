"""Build per-task feature matrices from v2 judge responses.

For each task, parses judge_responses_{claude,llama_bf16}/ and the qwen pool.
Produces:
  outputs/v2_analysis/{task}__{judge}__p{0,1,2}.parquet
    columns: dp_id, aspect_id, applicable, score, reason
  outputs/v2_analysis/{task}__coverage.parquet
    per (dp, aspect, paraphrase) coverage indicator across 3 judges

Plus prints headline coverage stats.
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

TASKS_DEFAULT = ["peer_review", "math", "notice_and_comment", "press_releases",
                 "humor", "news_homepages", "patents", "code_review", "creative_writing"]

JUDGE_DIRS = {
    "claude": "judge_responses_claude",
    "llama": "judge_responses_llama_bf16",
}


import re as _re

def _fix_bad_escapes(raw):
    return _re.sub(r'\\(?![\"\\/bfnrtu])', r'\\\\', raw)

def _try_parse_one(candidate):
    for variant in [candidate, _fix_bad_escapes(candidate)]:
        try:
            obj = json.loads(variant)
            if isinstance(obj, dict) and "results" in obj:
                return obj
        except: pass
    return None

def _smart_parse(raw):
    """Same logic as sk3_v2_judge_runner._smart_parse."""
    raw = raw.strip()
    # Direct
    obj = _try_parse_one(raw)
    if obj: return obj
    # Fenced code block
    m = _re.search(r"```(?:json)?\s*\n(.*?)```", raw, _re.S)
    if m:
        obj = _try_parse_one(m.group(1).strip())
        if obj: return obj
    # </think> suffix
    if "</think>" in raw:
        suffix = raw.rsplit("</think>", 1)[1].strip()
        starts = [i for i, c in enumerate(suffix) if c == "{"]
        for start in starts[:3]:
            end = suffix.rfind("}")
            while end > start:
                obj = _try_parse_one(suffix[start:end+1])
                if obj: return obj
                end = suffix.rfind("}", start, end)
    # General fallback
    starts = [i for i, c in enumerate(raw) if c == "{"]
    if not starts: return None
    candidates = list(starts[-5:]) + list(starts[:5])
    seen = set()
    for start in candidates:
        if start in seen: continue
        seen.add(start)
        end = raw.rfind("}")
        while end > start:
            obj = _try_parse_one(raw[start:end+1])
            if obj: return obj
            end = raw.rfind("}", start, end)
    return None


def parse_response(p: Path):
    """Returns list of (dp_id, aspect_id, applicable, score, reason) tuples."""
    raw = p.read_text()
    d = _smart_parse(raw)
    if d is None:
        return None, "smart_parse_failed"
    if not isinstance(d, dict) or "results" not in d:
        return None, "no_results"
    rows = []
    for r in d.get("results", []):
        if not isinstance(r, dict): continue
        dp_id = r.get("text_id") or r.get("dp_id")
        if dp_id is None:
            continue
        for s in r.get("scores", []):
            if not isinstance(s, dict): continue
            aspect_id = s.get("aspect_id")
            if aspect_id is None:
                continue
            rows.append((dp_id, aspect_id,
                         bool(s.get("applicable", False)),
                         s.get("score"),
                         (s.get("reason") or "")[:200]))
    return rows, None


def build_for_task(task: str, repo_root: Path, out_dir: Path,
                   qwen_resp_dir: Path | None = None):
    task_dir = repo_root / "runs" / "validity_full" / "v2" / task
    manifest = json.loads((task_dir / "judge_manifest.json").read_text())
    key_to_meta = {m["key"]: m for m in manifest}

    per_judge_dfs = {}
    for judge, sub in JUDGE_DIRS.items():
        rdir = task_dir / sub
        if not rdir.exists():
            continue
        rows = []
        n_manifest = 0
        n_targeted = 0
        n_parse_err = 0
        n_skipped_unknown = 0
        for fp in rdir.glob("*.json"):
            key = fp.stem
            res, err = parse_response(fp)
            if err:
                n_parse_err += 1
                continue
            if key in key_to_meta:
                meta = key_to_meta[key]
                p_idx, b_id, c_idx = meta["paraphrase_idx"], meta["bundle_id"], meta["chunk_idx"]
                n_manifest += 1
                source = "manifest"
            elif key.startswith("t") and "__b" in key:
                # Targeted prompt: t{X}__b{Y} — always p0; no chunk/bundle in manifest sense
                p_idx, b_id, c_idx = 0, key, -1
                n_targeted += 1
                source = "targeted"
            else:
                n_skipped_unknown += 1
                continue
            for dp_id, aspect_id, app, score, reason in res:
                rows.append({
                    "dp_id": dp_id,
                    "aspect_id": aspect_id,
                    "paraphrase_idx": p_idx,
                    "bundle_id": b_id,
                    "chunk_idx": c_idx,
                    "applicable": app,
                    "score": score,
                    "reason": reason,
                    "source": source,
                })
        if rows:
            df = pd.DataFrame(rows)
            per_judge_dfs[judge] = df
            print(f"  {task}/{judge}: manifest={n_manifest} targeted={n_targeted} "
                  f"unknown={n_skipped_unknown} parse_err={n_parse_err}, total cells={len(df)}",
                  file=sys.stderr)

    # Qwen: prefix-stripped from shared pool
    if qwen_resp_dir is not None and qwen_resp_dir.exists():
        rows = []
        prefix = f"{task}__"
        n_files = 0
        n_parse_err = 0
        for fp in qwen_resp_dir.glob(f"{prefix}*.json"):
            key = fp.stem[len(prefix):]
            if key not in key_to_meta:
                continue
            n_files += 1
            res, err = parse_response(fp)
            if err:
                n_parse_err += 1
                continue
            meta = key_to_meta[key]
            for dp_id, aspect_id, app, score, reason in res:
                rows.append({
                    "dp_id": dp_id,
                    "aspect_id": aspect_id,
                    "paraphrase_idx": meta["paraphrase_idx"],
                    "bundle_id": meta["bundle_id"],
                    "chunk_idx": meta["chunk_idx"],
                    "applicable": app,
                    "score": score,
                    "reason": reason,
                })
        if rows:
            per_judge_dfs["qwen"] = pd.DataFrame(rows)
            print(f"  {task}/qwen: {n_files} files ({n_parse_err} parse errors), {len(per_judge_dfs['qwen'])} cells",
                  file=sys.stderr)

    # Save per-judge per-paraphrase parquet
    for judge, df in per_judge_dfs.items():
        for p_idx in sorted(df["paraphrase_idx"].unique()):
            sub = df[df["paraphrase_idx"] == p_idx]
            (out_dir / f"{task}__{judge}__p{p_idx}.parquet").parent.mkdir(parents=True, exist_ok=True)
            sub.to_parquet(out_dir / f"{task}__{judge}__p{p_idx}.parquet", index=False)

    # Coverage summary: per (dp_id, aspect_id, paraphrase_idx, judge) — 1 if has any score
    cov_rows = []
    for judge, df in per_judge_dfs.items():
        g = df.groupby(["dp_id", "aspect_id", "paraphrase_idx"]).size().reset_index(name="n")
        g["judge"] = judge
        cov_rows.append(g)
    if cov_rows:
        cov = pd.concat(cov_rows, ignore_index=True)
        cov.to_parquet(out_dir / f"{task}__coverage.parquet", index=False)
        # Quick stats
        print(f"  {task}: ===== headline coverage =====", file=sys.stderr)
        for judge in per_judge_dfs:
            sub = cov[cov["judge"] == judge]
            for p_idx in sorted(sub["paraphrase_idx"].unique()):
                ss = sub[sub["paraphrase_idx"] == p_idx]
                n_dps = ss["dp_id"].nunique()
                n_asp = ss["aspect_id"].nunique()
                n_cells = len(ss)
                # dps with >=80% of seen aspects covered
                if n_asp > 0:
                    dp_aspect_counts = ss.groupby("dp_id")["aspect_id"].nunique()
                    dps_full = (dp_aspect_counts >= 0.95 * n_asp).sum()
                else:
                    dps_full = 0
                print(f"    {judge} p{p_idx}: dps={n_dps}, aspects={n_asp}, cells={n_cells}, "
                      f"dps_with_≥95%_aspects={dps_full}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="outputs/v2_analysis")
    ap.add_argument("--tasks", nargs="+", default=TASKS_DEFAULT)
    ap.add_argument("--qwen-gpu3", default="runs/validity_full/v2/_qwen_pool/gpu3_responses")
    ap.add_argument("--qwen-gpu5", default="runs/validity_full/v2/_qwen_pool/gpu5_responses")
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    out = repo / args.out
    out.mkdir(parents=True, exist_ok=True)
    qwen3 = repo / args.qwen_gpu3
    qwen5 = repo / args.qwen_gpu5
    # Qwen is split across gpu3 and gpu5 pools by task — we'll pass both, the function
    # will pick whichever holds prefixed files for the task
    for t in args.tasks:
        # Try gpu5 first (covers 5 tasks); fall back to gpu3 (covers other 4)
        qwen_dir = None
        if qwen5.exists() and any(qwen5.glob(f"{t}__*.json")):
            qwen_dir = qwen5
        elif qwen3.exists() and any(qwen3.glob(f"{t}__*.json")):
            qwen_dir = qwen3
        try:
            build_for_task(t, repo, out, qwen_dir)
        except FileNotFoundError as e:
            print(f"  {t}: SKIP ({e})", file=sys.stderr)
        print("", file=sys.stderr)


if __name__ == "__main__":
    main()
