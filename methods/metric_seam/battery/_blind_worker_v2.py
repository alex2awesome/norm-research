"""Private subprocess worker for blind_reconstruction_v2.py.

The parent process writes a request containing only the compiler bundle.  This worker
never imports battery_common, evaluation results, or a split map.
"""

from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import signal
import sys


HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from split_ops_v2 import SplitScopedOps  # noqa: E402


class _ItemTimeout(Exception):
    pass


def _alarm(_signum, _frame):
    raise _ItemTimeout("candidate score timed out")


def _load_candidate(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("blind_candidate", path)
    if spec is None or spec.loader is None:
        raise ImportError("could not create candidate module spec")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "score", None)):
        raise TypeError("candidate must define callable score(text, extracted, ops)")
    return mod


def execute(request: dict) -> dict:
    bundle = request["bundle"]
    items = bundle["train_items"]
    corpus = {item["item_key"]: item["ctext"] for item in items}
    ops = SplitScopedOps(corpus, bundle["allowed"]["capabilities"])
    mod = _load_candidate(pathlib.Path(request["candidate_path"]))

    declared = getattr(mod, "LLM_FIELDS", {}) or {}
    if (not isinstance(declared, dict)
            or any(not isinstance(k, str) or not isinstance(v, str)
                   for k, v in declared.items())):
        raise TypeError("LLM_FIELDS must be a mapping from field name to prompt")
    field_specs = bundle["allowed"]["fields"]
    allowed_fields = set(field_specs)
    undeclared = sorted(set(declared) - allowed_fields)
    if undeclared:
        raise ValueError(f"candidate requests fields outside allowlist: {undeclared}")
    prompt_mismatch = sorted(
        name for name, prompt in declared.items()
        if prompt != field_specs[name]["prompt"]
    )
    if prompt_mismatch:
        raise ValueError(
            "candidate prompt does not match frozen cached-field provenance for: "
            f"{prompt_mismatch}"
        )

    timeout_s = float(request["timeout_per_item"])
    if timeout_s <= 0:
        raise ValueError("timeout_per_item must be positive")
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _alarm)

    rows, errors = [], []
    for item in items:
        key = item["item_key"]
        try:
            if hasattr(signal, "setitimer"):
                signal.setitimer(signal.ITIMER_REAL, timeout_s)
            value = float(mod.score(item["ctext"], dict(item.get("fields", {})),
                                    ops.for_item(key)))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("score must be finite and in [0,1]")
            rows.append({"item_key": key, "score": value})
        except Exception as exc:  # error type/message are safe label-free feedback
            rows.append({"item_key": key, "score": None})
            errors.append({"item_key": key, "type": type(exc).__name__,
                           "message": str(exc)[:300]})
        finally:
            if hasattr(signal, "setitimer"):
                signal.setitimer(signal.ITIMER_REAL, 0)

    scoreable = sum(row["score"] is not None for row in rows)
    values = [row["score"] for row in rows if row["score"] is not None]
    return {
        "schema": "metric-seam.blind-reconstruction.execution-result.v2",
        "candidate_declared_fields": sorted(declared),
        "n_items": len(rows),
        "n_scoreable": scoreable,
        "coverage": scoreable / len(rows) if rows else 0.0,
        "output_min": min(values) if values else None,
        "output_max": max(values) if values else None,
        "outputs": rows,
        "errors": errors,
    }


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: _blind_worker_v2.py REQUEST CANDIDATE OUTPUT")
    request_path, candidate_path, output_path = map(pathlib.Path, sys.argv[1:])
    request = json.loads(request_path.read_text())
    request["candidate_path"] = str(candidate_path)
    result = execute(request)
    output_path.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
