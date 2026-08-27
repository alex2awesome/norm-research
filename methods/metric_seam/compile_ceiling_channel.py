"""Compile the ceiling arm additively against an already-filtered prompt-jobs bundle.

The v3 bundle was produced by filtering v2 down to the 18 cells that survived the
independent cross-audit.  Recompiling from scratch would have to re-derive that
filter and risks drift, so this tool instead reads v3 and emits the ceiling channel
for exactly the cells, items, and passes it already contains.  v3 is never modified.

The ceiling arm discloses the complete program source, digest-bound to the frozen
audit.  It exists because v3 had no upper anchor: every channel was an impoverished
articulation, so a low rho could not be separated from disclosure loss.  It calls no
model and reads no outcome label.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from methods.metric_seam.hierarchy_prompt_batch import (
    PASS_SEED_SALT,
    SYSTEM_PROMPT,
    RESPONSE_SCHEMA,
    PromptBatchError,
    _digest_bytes,
    _full_contract_prompt,
    _sampling_seed,
)

CEILING_CHANNEL = "full_executable_contract"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-jobs", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--jobs-out", type=Path, required=True)
    args = parser.parse_args(argv)

    audit = json.loads(args.audit.read_text())
    metric_name = {row["cell_id"]: row["metric_name"] for row in audit["rows"]}

    cells: dict[str, dict] = {}
    items: dict[str, str] = {}
    passes: set[int] = set()
    for line in gzip.open(args.prompt_jobs, "rt"):
        meta = json.loads(line)["audit_metadata"]
        cells.setdefault(meta["cell_id"], meta)
        items.setdefault(meta["item_key"], meta["ctext"])
        passes.add(meta["pass_id"])

    sources: dict[str, str] = {}
    for cell_id, meta in cells.items():
        raw = Path(meta["source_path"]).read_bytes()
        if _digest_bytes(raw) != meta["source_sha256"]:
            raise PromptBatchError(
                f"{cell_id}: program source digest does not match the frozen audit"
            )
        sources[cell_id] = raw.decode("utf-8")

    jobs = []
    for cell_id, meta in sorted(cells.items()):
        row = {"metric_name": metric_name[cell_id]}
        for item_key, ctext in sorted(items.items()):
            prompt = _full_contract_prompt(row, sources[cell_id], ctext)
            for pass_id in sorted(passes):
                jobs.append({
                    "request_id": f"{cell_id}::{CEILING_CHANNEL}::p{pass_id}::{item_key}",
                    "request": {"system": SYSTEM_PROMPT, "user": prompt},
                    "executor_metadata": {
                        "sampling_seed": _sampling_seed(
                            cell_id, CEILING_CHANNEL, item_key, pass_id
                        ),
                        "temperature": 0.2,
                        "top_p": 1.0,
                        "stateless_separate_call": True,
                        "cache_and_context_reuse_forbidden": True,
                        "response_schema": RESPONSE_SCHEMA,
                    },
                    "audit_metadata": {
                        "cell_id": cell_id,
                        "aspect_id": meta["aspect_id"],
                        "source_path": meta["source_path"],
                        "source_sha256": meta["source_sha256"],
                        "level": meta["level"],
                        "channel": CEILING_CHANNEL,
                        "pass_id": pass_id,
                        "item_key": item_key,
                        "ctext": ctext,
                        "ctext_sha256": _digest_bytes(ctext.encode()),
                        "source_only_subrelation": None,
                    },
                })

    jobs.sort(key=lambda job: _digest_bytes(job["request_id"].encode()))
    if len({job["request_id"] for job in jobs}) != len(jobs):
        raise PromptBatchError("duplicate prompt request ids")

    args.jobs_out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.jobs_out, "wt", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(json.dumps(job, sort_keys=True) + "\n")

    print(f"cells   : {len(cells)}")
    print(f"items   : {len(items)}")
    print(f"passes  : {sorted(passes)}")
    print(f"programs: {len({m['source_sha256'] for m in cells.values()})} unique, digest-verified")
    print(f"jobs    : {len(jobs)}")
    print(f"wrote   : {args.jobs_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
