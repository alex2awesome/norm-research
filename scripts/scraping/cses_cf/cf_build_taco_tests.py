"""Build CF stdin/stdout test bank from TACO raw (urls -> canonical_pid).

Input:  datasets/taco_raw/ALL/*.parquet  (columns: source, url, input_output, ...)
Output: datasets/codeforces_delta/cf_tests_taco.parquet
        [canonical_pid, n_tests, tests_json]   tests_json = JSON list of
        {"input": str, "output": str} dicts (same shape consumed by
        scripts/competition_exec/run_candidate_tests.py stdio mode).

APPEND-ONLY: writes its own new output file only.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import pandas as pd

TACO = Path("/lfs/skampere3/0/alexspan/norm-research/datasets/taco_raw/ALL")
OUT = Path("/lfs/skampere3/0/alexspan/norm-research/datasets/codeforces_delta/cf_tests_taco.parquet")

URL_RE = re.compile(
    r"codeforces\.com/(?:problemset/problem/(\d+)/([A-Z]\d?)|contest/(\d+)/problem/([A-Z]\d?))",
    re.I,
)

rows = []
for f in sorted(TACO.glob("*.parquet")):
    df = pd.read_parquet(f, columns=["source", "url", "input_output"])
    df = df[df["source"].astype(str).str.lower().str.contains("codeforces", na=False)]
    for url, io_raw in zip(df["url"], df["input_output"]):
        if not url or not io_raw:
            continue
        m = URL_RE.search(str(url))
        if not m:
            continue
        cid = m.group(1) or m.group(3)
        letter = (m.group(2) or m.group(4)).lower()
        pid = f"cf:{cid}_{letter}"
        try:
            io = json.loads(io_raw)
            ins, outs = io.get("inputs", []), io.get("outputs", [])
        except Exception:
            continue
        tests = []
        for i, o in zip(ins, outs):
            if isinstance(i, list):
                i = "\n".join(map(str, i))
            if isinstance(o, list):
                o = "\n".join(map(str, o))
            if isinstance(i, str) and isinstance(o, str) and i.strip() and o.strip():
                tests.append({"input": i, "output": o})
        if tests:
            rows.append({"canonical_pid": pid, "n_tests": len(tests),
                         "tests_json": json.dumps(tests[:25])})
    print(f"{f.name}: cum rows={len(rows)}", flush=True)

out = pd.DataFrame(rows).drop_duplicates("canonical_pid")
out.to_parquet(OUT, index=False)
print(f"wrote {OUT}: {len(out)} CF problems with tests, "
      f"median n_tests={out['n_tests'].median()}")
