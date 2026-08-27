"""Build the unified judge-cells database.

One row per (task, judge_model, aspect_id, datapoint_id, paraphrase_idx, level).

Columns:
  task            peer_review | math | notice_and_comment | press_releases | humor |
                    news_homepages | patents | code_review | creative_writing
  judge_model     claude | llama_bf16 | qwen_thinking_fp8
  aspect_id       e.g., 'a4'
  datapoint_id    e.g., 'd00042'
  paraphrase_idx  0 | 1 | 2
  level           R2 | R2_post           (which aspect-inventory the score is for)
  score           0.0 | 0.5 | 1.0 | NaN
  applicable      bool
  reason          str (max 200 chars, for inspection)
  date_scored     ISO date from response file mtime (UTC)
  response_file   filename of the source response (.json)
  response_key    e.g., 'b0__p0__c0' or 't1__b36' or for qwen 'humor__b0__p0__c0'
  parse_status    ok | smart_parse_failed | no_json_chars | empty

Layout: writes to outputs/v2_db/cells_v1/task=<task>/judge=<judge>/data.parquet
(Hive-partitioned for fast filter pushdown.)

This script can run locally (claude) or on sk3 (llama/qwen). It only reads source
dirs that exist locally. Run on both machines, then concatenate parquets.

Usage:
  python3 scripts/build_cells_db.py --judges claude
  python3 scripts/build_cells_db.py --judges llama_bf16 qwen
"""
import argparse, datetime, json, re, sys
from pathlib import Path
from collections import Counter

import pandas as pd

TASKS = ["peer_review", "math", "notice_and_comment", "press_releases",
         "humor", "news_homepages", "patents", "code_review", "creative_writing"]


# ---- smart parser (mirrors sk3_v2_judge_runner._smart_parse) ----
def _fix_bad_escapes(raw):
    return re.sub(r'\\(?![\"\\/bfnrtu])', r'\\\\', raw)

def _try_parse_one(candidate):
    for variant in [candidate, _fix_bad_escapes(candidate)]:
        try:
            obj = json.loads(variant)
            if isinstance(obj, dict) and "results" in obj:
                return obj
        except: pass
    return None

def _smart_parse(raw):
    raw = raw.strip()
    if not raw: return None, "empty"
    obj = _try_parse_one(raw)
    if obj: return obj, "ok"
    m = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.S)
    if m:
        obj = _try_parse_one(m.group(1).strip())
        if obj: return obj, "ok"
    if "</think>" in raw:
        suffix = raw.rsplit("</think>", 1)[1].strip()
        starts = [i for i, c in enumerate(suffix) if c == "{"]
        for start in starts[:3]:
            end = suffix.rfind("}")
            while end > start:
                obj = _try_parse_one(suffix[start:end+1])
                if obj: return obj, "ok"
                end = suffix.rfind("}", start, end)
    starts = [i for i, c in enumerate(raw) if c == "{"]
    if not starts: return None, "no_json_chars"
    candidates = list(starts[-5:]) + list(starts[:5])
    seen = set()
    for start in candidates:
        if start in seen: continue
        seen.add(start)
        end = raw.rfind("}")
        while end > start:
            obj = _try_parse_one(raw[start:end+1])
            if obj: return obj, "ok"
            end = raw.rfind("}", start, end)
    return None, "smart_parse_failed"


# ---- per-file parser ----
def parse_response_file(fp: Path):
    """Return list of cell dicts + parse_status."""
    raw = fp.read_text()
    obj, status = _smart_parse(raw)
    if obj is None:
        return [], status
    rows = []
    if obj is None:
        return [], "parse_null"
    if not isinstance(obj, dict):
        return [], "parse_not_dict"
    for r in (obj.get("results") or []):
        if not isinstance(r, dict): continue
        dp_id = r.get("text_id") or r.get("dp_id")
        if dp_id is None: continue
        for s in (r.get("scores") or []):
            if not isinstance(s, dict): continue
            aspect_id = s.get("aspect_id")
            if aspect_id is None: continue
            sc = s.get("score")
            try: sc = float(sc) if sc is not None else None
            except: sc = None
            rows.append({
                "aspect_id": aspect_id,
                "datapoint_id": dp_id,
                "applicable": bool(s.get("applicable", False)),
                "score": sc,
                "reason": (s.get("reason") or "")[:200],
            })
    return rows, status


# ---- source-dir walkers ----
def claude_files(task_dir: Path):
    """Yields (response_file, response_key, paraphrase_idx, level)."""
    d = task_dir / "judge_responses_claude"
    if not d.exists(): return
    # Read manifest for p_idx of standard keys
    manifest = json.loads((task_dir / "judge_manifest.json").read_text())
    key_to_p = {m["key"]: m["paraphrase_idx"] for m in manifest}
    for fp in d.iterdir():
        if not fp.name.endswith(".json"): continue
        key = fp.stem
        if key in key_to_p:
            yield fp, key, key_to_p[key], "R2"
        elif key.startswith("t") and "__b" in key:
            # Targeted prompts: always p0
            yield fp, key, 0, "R2"
        # else skip unknown keys


def llama_files(task_dir: Path, variant: str = "bf16"):
    """Llama variants: judge_responses_llama_{bf16,fp8,fp8_smoketest}."""
    d = task_dir / f"judge_responses_llama_{variant}"
    if not d.exists(): return
    manifest = json.loads((task_dir / "judge_manifest.json").read_text())
    key_to_p = {m["key"]: m["paraphrase_idx"] for m in manifest}
    for fp in d.iterdir():
        if not fp.name.endswith(".json"): continue
        key = fp.stem
        if key in key_to_p:
            yield fp, key, key_to_p[key], "R2"


def qwen_per_task_files(task_dir: Path, variant: str):
    """Old per-task qwen response dirs (judge_responses_qwen_thinking[_vN], _nothink, _fp8).

    These use standard manifest keys (b__p__c). The per-task dirs predate the
    shared _qwen_pool structure.
    """
    d = task_dir / f"judge_responses_qwen_{variant}"
    if not d.exists(): return
    manifest = json.loads((task_dir / "judge_manifest.json").read_text())
    key_to_p = {m["key"]: m["paraphrase_idx"] for m in manifest}
    for fp in d.iterdir():
        if not fp.name.endswith(".json"): continue
        key = fp.stem
        if key in key_to_p:
            yield fp, key, key_to_p[key], "R2"


def qwen_files(repo_root: Path, task: str):
    """Qwen lives in shared pool dirs with task-prefixed names."""
    manifest = json.loads((repo_root / "runs/validity_full/v2" / task / "judge_manifest.json").read_text())
    key_to_p = {m["key"]: m["paraphrase_idx"] for m in manifest}
    pool_root = repo_root / "runs/validity_full/v2/_qwen_pool"
    for pool in ["gpu3_responses", "gpu5_responses"]:
        d = pool_root / pool
        if not d.exists(): continue
        prefix = f"{task}__"
        for fp in d.iterdir():
            if not fp.name.startswith(prefix): continue
            key = fp.stem[len(prefix):]
            if key in key_to_p:
                yield fp, key, key_to_p[key], "R2"


def qwen_20x1_files(task_dir: Path):
    """Smoke-test 20×1 responses (single dp per file; paraphrase=0)."""
    d = task_dir / "judge_responses_qwen_20x1"
    if not d.exists(): return
    for fp in d.iterdir():
        if not fp.name.endswith(".json"): continue
        yield fp, fp.stem, 0, "R2"




def qwen_20x1_r2post_files(repo_root, task: str):
    """r2_post pool + p1/p2 pools; task-prefixed; paraphrase optional in key."""
    import re
    pool_roots = [
        repo_root / "runs/validity_full/v2/_qwen_20x1_r2p_pool",
        repo_root / "runs/validity_full/v2/_qwen_s2_p1_pool",
        repo_root / "runs/validity_full/v2/_qwen_s3_p2_pool",
        repo_root / "runs/validity_full/v2/_qwen_s5_p0_pool",
    ]
    prefix = f"{task}__"
    p_re = re.compile(r"__p(\d+)__")
    for pool_root in pool_roots:
        if not pool_root.exists(): continue
        for sd in pool_root.glob("shard_*_responses"):
            if not sd.exists(): continue
            for fp in sd.iterdir():
                if not fp.name.endswith(".json"): continue
                if not fp.name.startswith(prefix): continue
                key = fp.stem[len(prefix):]  # strip "<task>__"
                m = p_re.search(key)
                p_idx = int(m.group(1)) if m else 0
                yield fp, key, p_idx, "R2_post"


# ---- main builder per (judge, task) ----
def build(judge: str, repo_root: Path, out_root: Path):
    print(f"\n=== judge: {judge} ===")
    walkers = {
        "claude": lambda task_dir, repo, task: claude_files(task_dir),
        "llama_bf16": lambda task_dir, repo, task: llama_files(task_dir, "bf16"),
        "llama_fp8": lambda task_dir, repo, task: llama_files(task_dir, "fp8"),
        "llama_fp8_smoketest": lambda task_dir, repo, task: llama_files(task_dir, "fp8_smoketest"),
        "qwen_thinking_fp8": lambda task_dir, repo, task: qwen_files(repo, task),
        "qwen_thinking_fp8_20x1": lambda task_dir, repo, task: qwen_20x1_files(task_dir),
        "qwen_thinking_fp8_20x1_r2post": lambda task_dir, repo, task: qwen_20x1_r2post_files(repo, task),
        # Older per-task variants (peer_review and math have these)
        "qwen_thinking_v1": lambda task_dir, repo, task: qwen_per_task_files(task_dir, "thinking"),
        "qwen_thinking_v2": lambda task_dir, repo, task: qwen_per_task_files(task_dir, "thinking_v2"),
        "qwen_thinking_v3": lambda task_dir, repo, task: qwen_per_task_files(task_dir, "thinking_v3"),
        "qwen_nothink": lambda task_dir, repo, task: qwen_per_task_files(task_dir, "nothink"),
        "qwen_fp8_early": lambda task_dir, repo, task: qwen_per_task_files(task_dir, "fp8"),
    }
    if judge not in walkers:
        print(f"  unknown judge: {judge}; skip"); return

    for task in TASKS:
        task_dir = repo_root / "runs/validity_full/v2" / task
        if not task_dir.exists(): continue
        rows = []
        statuses = Counter()
        n_files = 0
        for fp, key, p_idx, level in walkers[judge](task_dir, repo_root, task):
            n_files += 1
            cells, status = parse_response_file(fp)
            statuses[status] += 1
            date_str = datetime.datetime.fromtimestamp(fp.stat().st_mtime, tz=datetime.timezone.utc).isoformat()[:19]
            for c in cells:
                rows.append({
                    "task": task,
                    "judge_model": judge,
                    "aspect_id": c["aspect_id"],
                    "datapoint_id": c["datapoint_id"],
                    "paraphrase_idx": p_idx,
                    "level": level,
                    "score": c["score"],
                    "applicable": c["applicable"],
                    "reason": c["reason"],
                    "date_scored": date_str,
                    "response_file": fp.name,
                    "response_key": key,
                    "parse_status": status,
                })
        if n_files == 0:
            print(f"  {task:<22} no files"); continue
        out_dir = out_root / f"task={task}" / f"judge={judge}"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Try parquet first; fall back to plain csv (sk3 has broken pandas)
        wrote = False
        try:
            out_path = out_dir / "data.parquet"
            pd.DataFrame(rows).to_parquet(out_path, index=False, compression="zstd")
            wrote = True
        except Exception:
            pass
        if not wrote:
            # Plain Python csv.gz (works without pyarrow/pandas csv module)
            import csv, gzip
            out_path = out_dir / "data.csv.gz"
            if rows:
                cols = ["task","judge_model","aspect_id","datapoint_id","paraphrase_idx",
                        "level","score","applicable","reason","date_scored",
                        "response_file","response_key","parse_status"]
                with gzip.open(out_path, "wt", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                    w.writeheader()
                    for r in rows: w.writerow(r)
        parse_rate = (statuses["ok"] / n_files * 100) if n_files else 0
        print(f"  {task:<22} files={n_files:>6} cells={len(rows):>8} parse_rate={parse_rate:>5.1f}% → {out_path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="outputs/v2_db/cells_v1")
    ap.add_argument("--judges", nargs="+", default=["claude"],
                    help="Which judges to ingest. Available: "
                         "claude, llama_bf16, qwen_thinking_fp8, qwen_thinking_fp8_20x1")
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    out_root = repo / args.out
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Repo: {repo}\nOutput root: {out_root}\nJudges: {args.judges}")
    for j in args.judges:
        build(j, repo, out_root)


if __name__ == "__main__":
    main()
