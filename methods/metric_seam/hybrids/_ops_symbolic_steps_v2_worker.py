#!/usr/bin/env python
"""Private process worker for :mod:`ops_symbolic_steps_v2`."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from methods.metric_seam.hybrids.ops_symbolic_steps_v2 import (
    _canonical_bytes,
    execute_process_request,
)


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} REQUEST OUTPUT")
    request_path, output_path = map(Path, sys.argv[1:])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    result = execute_process_request(request)
    output_path.write_bytes(_canonical_bytes(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
