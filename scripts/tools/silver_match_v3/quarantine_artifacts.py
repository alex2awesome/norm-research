#!/usr/bin/env python3
"""Hash and quarantine immutable artifacts that must not enter selection."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from .common import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--prohibition", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    artifacts = [Path(path).resolve() for path in args.artifact]
    payload = {
        "schema_version": "silver-match-v3-artifact-quarantine-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reason": args.reason,
        "prohibitions": args.prohibition,
        "artifacts": {
            str(path): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in artifacts
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
