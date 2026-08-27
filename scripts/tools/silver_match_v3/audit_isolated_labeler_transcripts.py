#!/usr/bin/env python3
"""Audit Codex labeler transcripts for truth/proposal isolation.

The structured-label validator proves schema and UID coverage.  This companion
audit proves that each Codex subprocess only opened the immutable guide, its
order-permuted bank, and its assigned item chunk.  It fails closed on shell
discovery commands or any repository/data path outside that per-chunk allowlist.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path
from typing import Any

from .common import sha256_file


COMMAND_BOUNDARY = r"(?:^|[\s'\";&|(])"
DISCOVERY_COMMAND = re.compile(
    COMMAND_BOUNDARY + r"(?:/[^\s]+/)?(?:find|fd|fdfind|locate|ls|git)\b"
)
TARGETABLE_RG = re.compile(COMMAND_BOUNDARY + r"(?:/[^\s]+/)?(?:rg|ripgrep)\b")
RECURSIVE_GREP = re.compile(r"\bgrep\s+[^;&|]*(?:-[A-Za-z]*[Rr]|--recursive)\b")
NETWORK_OR_INTERPRETER = re.compile(
    COMMAND_BOUNDARY
    + r"(?:/[^\s]+/)?(?:curl|wget|ssh|scp|rsync|python\d*|perl|ruby|node)\b"
)
REPOSITORY_PATH = re.compile(
    r"(?:^|[\s'\"=:(])"
    r"((?:\.\./|~/|/Users/|/lfs/|/home/|/tmp/|scripts/|outputs/|datasets/|data/)"
    r"[^\s'\";&|)]*)"
)


def _command_lines(text: str) -> list[str]:
    lines = text.splitlines()
    commands: list[str] = []
    for index, line in enumerate(lines[:-1]):
        if line.strip() != "exec":
            continue
        value = lines[index + 1].strip()
        # Codex appends `` in <cwd>`` outside the quoted shell command.
        if " in " in value:
            value = value.rsplit(" in ", 1)[0]
        commands.append(value)
    return commands


def _display_forms(path: Path, repo: Path, cwd: Path | None = None) -> set[str]:
    forms = {str(path.resolve())}
    for root in (repo, cwd):
        if root is None:
            continue
        try:
            forms.add(str(path.resolve().relative_to(root.resolve())))
        except ValueError:
            pass
    return forms


def _unapproved_paths(command: str, allowed: set[str]) -> list[str]:
    scrubbed = command
    # Longest first prevents replacing a parent prefix before its child path.
    for value in sorted(allowed, key=len, reverse=True):
        scrubbed = scrubbed.replace(value, "__ALLOWED_FILE__")
    return sorted({match.group(1) for match in REPOSITORY_PATH.finditer(scrubbed)})


def _has_untargeted_rg(command: str, allowed: set[str]) -> bool:
    """Return true when an rg stage can read beyond a frozen file or its stdin.

    A common safe inspection idiom is ``jq ... frozen-bank.json | rg PATTERN``.
    The old clause-by-clause check rejected that because the ``rg`` stage has no
    file operand, even though its stdin is produced from the allowlisted bank.
    Preserve fail-closed behavior by accepting stdin-only ``rg`` only when an
    earlier stage of the same shell pipeline names an allowlisted file and the
    rg stage has at most one non-option argument (its pattern).
    """

    try:
        outer = shlex.split(command, posix=True)
    except ValueError:
        return bool(TARGETABLE_RG.search(command))
    script = command
    if outer and Path(outer[0]).name in {"bash", "dash", "sh", "zsh"}:
        script = outer[-1]
    lexer = shlex.shlex(script, posix=True, punctuation_chars=";&|()")
    lexer.whitespace_split = True
    try:
        words = list(lexer)
    except ValueError:
        return bool(TARGETABLE_RG.search(command))
    stage: list[str] = []
    pipeline: list[list[str]] = []
    pipelines: list[list[list[str]]] = []
    for word in words:
        if word and set(word) <= set(";&|()"):
            if stage:
                pipeline.append(stage)
                stage = []
            if word != "|" and pipeline:
                pipelines.append(pipeline)
                pipeline = []
        else:
            stage.append(word)
    if stage:
        pipeline.append(stage)
    if pipeline:
        pipelines.append(pipeline)
    for pipeline in pipelines:
        upstream_is_frozen = False
        for stage in pipeline:
            stage_is_frozen = any(value in stage for value in allowed)
            rg_positions = [
                index
                for index, value in enumerate(stage)
                if Path(value).name in {"rg", "ripgrep"}
            ]
            for index in rg_positions:
                if stage_is_frozen:
                    continue
                positional = [
                    value for value in stage[index + 1 :] if not value.startswith("-")
                ]
                if not upstream_is_frozen or len(positional) > 1:
                    return True
            upstream_is_frozen = upstream_is_frozen or stage_is_frozen
    return False


def audit(pack_root: Path, guides: list[Path], repo: Path) -> dict[str, Any]:
    pack_root = pack_root.resolve()
    repo = repo.resolve()
    guides = [path.resolve() for path in guides]
    bank = pack_root / "bank.json"
    items = pack_root / "items.jsonl"
    pack_validation = pack_root / "validation.json"
    chunks = sorted((pack_root / "chunks").glob("part-*.jsonl"))
    if not bank.is_file() or not chunks:
        raise FileNotFoundError("pack must contain bank.json and part-*.jsonl chunks")
    if any(not path.is_file() for path in guides):
        raise FileNotFoundError("one or more labeling guides are missing")

    rows: list[dict[str, Any]] = []
    violations: list[dict[str, str]] = []
    common_allowed: set[str] = set()
    for path in [bank, *guides]:
        common_allowed.update(_display_forms(path, repo, pack_root.parent))

    for chunk in chunks:
        log = pack_root / "logs" / f"{chunk.stem}.log"
        raw = pack_root / "raw_labels" / f"{chunk.stem}.json"
        if not log.is_file() or not raw.is_file():
            violations.append(
                {
                    "chunk": chunk.stem,
                    "kind": "MISSING_TRANSCRIPT_OR_RAW_LABEL",
                    "detail": f"log={log.is_file()} raw={raw.is_file()}",
                }
            )
            continue
        text = log.read_text(encoding="utf-8", errors="replace")
        commands = _command_lines(text)
        allowed = set(common_allowed)
        allowed.update(_display_forms(chunk, repo, pack_root.parent))
        observed_allowed = {value for value in allowed if value in text}
        required = [bank, chunk, *guides]
        missing_reads = [
            str(path)
            for path in required
            if not (_display_forms(path, repo, pack_root.parent) & observed_allowed)
        ]
        if missing_reads:
            violations.append(
                {
                    "chunk": chunk.stem,
                    "kind": "REQUIRED_INPUT_NOT_OBSERVED",
                    "detail": json.dumps(missing_reads, sort_keys=True),
                }
            )
        if "sandbox: read-only" not in text or "approval: never" not in text:
            violations.append(
                {
                    "chunk": chunk.stem,
                    "kind": "RUNTIME_ISOLATION_NOT_OBSERVED",
                    "detail": "expected read-only sandbox and approval-never transcript markers",
                }
            )
        if not commands:
            violations.append(
                {
                    "chunk": chunk.stem,
                    "kind": "NO_EXEC_COMMANDS_OBSERVED",
                    "detail": "transcript does not expose file reads",
                }
            )
        for ordinal, command in enumerate(commands, 1):
            reasons: list[str] = []
            if DISCOVERY_COMMAND.search(command):
                reasons.append("repository discovery command")
            # ``rg`` against the one immutable bank/chunk/guide file is a
            # targeted read, not repository discovery.  A bare or otherwise
            # untargeted rg remains fail-closed.
            if _has_untargeted_rg(command, allowed):
                reasons.append("repository discovery command")
            if RECURSIVE_GREP.search(command):
                reasons.append("recursive grep")
            if NETWORK_OR_INTERPRETER.search(command):
                reasons.append("network or general-purpose interpreter command")
            unexpected = _unapproved_paths(command, allowed)
            if unexpected:
                reasons.append(f"unapproved paths={unexpected}")
            if reasons:
                violations.append(
                    {
                        "chunk": chunk.stem,
                        "kind": "UNAPPROVED_EXECUTION",
                        "detail": f"command {ordinal}: {'; '.join(reasons)} :: {command}",
                    }
                )
        rows.append(
            {
                "chunk": chunk.stem,
                "chunk_sha256": sha256_file(chunk),
                "log_sha256": sha256_file(log),
                "raw_label_sha256": sha256_file(raw),
                "command_count": len(commands),
            }
        )

    return {
        "schema_version": "silver-match-v3-isolated-labeler-transcript-audit-v1",
        "status": "PASS" if not violations else "FAIL",
        "complete": not violations and len(rows) == len(chunks),
        "pack_root": str(pack_root),
        "bank": {"path": str(bank), "sha256": sha256_file(bank)},
        "items": (
            {"path": str(items), "sha256": sha256_file(items)}
            if items.is_file()
            else None
        ),
        "pack_validation": (
            {"path": str(pack_validation), "sha256": sha256_file(pack_validation)}
            if pack_validation.is_file()
            else None
        ),
        "full_pack_artifact_binding": items.is_file() and pack_validation.is_file(),
        "guides": [{"path": str(path), "sha256": sha256_file(path)} for path in guides],
        "expected_chunks": len(chunks),
        "audited_chunks": len(rows),
        "command_count": sum(row["command_count"] for row in rows),
        "chunks": rows,
        "violations": violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--guide", action="append", required=True)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite transcript audit: {output}")
    result = audit(
        Path(args.pack_root), [Path(value) for value in args.guide], Path(args.repo)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {**result, "output": str(output), "output_sha256": sha256_file(output)},
            sort_keys=True,
        )
    )
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
