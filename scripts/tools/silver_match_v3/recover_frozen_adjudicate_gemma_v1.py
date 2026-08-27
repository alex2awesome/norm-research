#!/usr/bin/env python3
"""Recover the exact pre-remediation adjudicate_gemma source by recorded diffs.

The Humor fresh-select queue executed with a pinned source artifact.  A later,
unrelated parser remediation updated the shared checkout before selection.
This tool reverses only the three recorded post-freeze edits and refuses to
emit an archive unless the resulting bytes equal the queue's pinned identity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import sha256_file


CURRENT_SHA256 = "25c9d7909804ad3a920a71ed741fa46ffd8f7328d68d4d9e01f7c5eb4f021cbb"
FROZEN_SHA256 = "66e5bd7f2785a2597fe550be3c10d40eaac51f1e8e3e15a0b16612306fe17208"
FROZEN_BYTES = 21668


def recover(source: str) -> str:
    replacements = [
        (
            "import os\nimport re\nimport time\n",
            "import os\nimport time\n",
            "remove_post_freeze_literal_backslash_repair_import",
        ),
        (
            "from .common import read_jsonl, sha256_file\n",
            "from .common import metric_card, read_jsonl, sha256_file\n",
            "restore_frozen_import_identity",
        ),
        (
            '''        suffix = (raw or "")[start:]
        try:
            value, _ = decoder.raw_decode(suffix)
        except (json.JSONDecodeError, TypeError):
            # Models occasionally emit a literal LaTeX/code backslash in an
            # otherwise valid JSON string (for example ``"`\\|`"``). JSON
            # permits only the escapes in ["\\\\/bfnrtu]. Escape any other
            # backslash and retry without changing the semantic answer.
            repaired = re.sub(r'\\\\(?!["\\\\/bfnrtu])', r"\\\\\\\\", suffix)
            if repaired == suffix:
                continue
            try:
                value, _ = decoder.raw_decode(repaired)
            except (json.JSONDecodeError, TypeError):
                continue
''',
            '''        try:
            value, _ = decoder.raw_decode((raw or "")[start:])
        except (json.JSONDecodeError, TypeError):
            continue
''',
            "reverse_post_freeze_literal_backslash_parser_repair",
        ),
        (
            '''    raw_confidence = parsed.get("confidence")
    confidence_raw = None
    if isinstance(raw_confidence, (int, float)) and not isinstance(raw_confidence, bool):
        confidence_raw = float(raw_confidence)
        if not 0.0 <= confidence_raw <= 1.0:
            return None, "unknown_confidence"
        confidence = (
            "high"
            if confidence_raw >= 0.8
            else "medium"
            if confidence_raw >= 0.5
            else "low"
        )
    else:
        confidence = str(raw_confidence or "").strip().lower()
''',
            '    confidence = str(parsed.get("confidence") or "").strip().lower()\n',
            "reverse_local_only_numeric_confidence_extension",
        ),
        (
            '''    result = {
        "decision": decision,
        "metric_id": metric_id,
        "confidence": confidence,
        "reason": reason,
    }
    # A frozen prompt may request calibrated [0,1] confidence while the
    # downstream row contract remains categorical.  Preserve the raw value in
    # the parsed object; inference runs that need byte-level provenance also
    # retain ``raw_response`` with ``--keep-raw``.
    if confidence_raw is not None:
        result["confidence_raw"] = confidence_raw
    return result, None
''',
            '''    return {
        "decision": decision,
        "metric_id": metric_id,
        "confidence": confidence,
        "reason": reason,
    }, None
''',
            "reverse_local_only_numeric_confidence_return_extension",
        ),
    ]
    for old, new, name in replacements:
        if source.count(old) != 1:
            raise ValueError(f"recorded diff anchor is not unique: {name}")
        source = source.replace(old, new, 1)
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    report = Path(args.report).resolve()
    if output.exists() or report.exists():
        raise FileExistsError("recovered source and report are append-only")
    if source.stat().st_size != 23106 or sha256_file(source) != CURRENT_SHA256:
        raise ValueError("source is not the recorded post-freeze remediation artifact")
    recovered = recover(source.read_text(encoding="utf-8")).encode("utf-8")
    import hashlib

    recovered_sha = hashlib.sha256(recovered).hexdigest()
    if len(recovered) != FROZEN_BYTES or recovered_sha != FROZEN_SHA256:
        raise ValueError("recorded diffs do not reconstruct the frozen artifact")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(recovered)
    payload = {
        "schema_version": "silver-match-v3-frozen-source-recovery-v1",
        "status": "EXACT_FROZEN_BYTES_RECOVERED_FROM_RECORDED_POST_FREEZE_DIFFS",
        "source": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        },
        "recovered": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
        "post_freeze_edits_reversed": [
            "local_only_numeric_confidence_extension",
            "literal_backslash_parser_repair",
            "unused_metric_card_import_cleanup",
        ],
        "model_outputs_changed": False,
        "queue_or_thresholds_changed": False,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({**payload, "report_sha256": sha256_file(report)}, sort_keys=True))


if __name__ == "__main__":
    main()
