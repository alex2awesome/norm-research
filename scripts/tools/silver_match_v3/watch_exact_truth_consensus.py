#!/usr/bin/env python3
"""Persist exact multi-pass truth collection through disagreement-only rounds.

Two externally started full-pack passes are required.  This watcher waits for
both, validates their exact chunk coverage, freezes a consensus stage, and
labels only the still-unresolved UIDs in independently shuffled resolver
packs.  No disagreement labels are exposed to a later annotator.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .common import sha256_file


SCHEMA = "silver-match-v3-exact-truth-consensus-watch-v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_named_roots(values: Iterable[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"pass must be NAME=PACK_ROOT: {value}")
        name, raw_root = value.split("=", 1)
        if not name or not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"invalid pass name: {name!r}")
        if name in seen:
            raise ValueError(f"duplicate pass name: {name}")
        parsed.append((name, Path(raw_root).resolve()))
        seen.add(name)
    if len(parsed) != 2:
        raise ValueError("exactly two independent initial passes are required")
    return parsed


def raw_pack_progress(pack_root: Path) -> dict[str, int | bool]:
    chunks = sorted((pack_root / "chunks").glob("part-*.jsonl"))
    if not chunks:
        raise ValueError(f"pack has no chunks: {pack_root}")
    raw_root = pack_root / "raw_labels"
    present = sum((raw_root / f"{path.stem}.json").is_file() for path in chunks)
    return {"expected": len(chunks), "present": present, "complete": present == len(chunks)}


def _all_or_none(paths: Iterable[Path], label: str) -> bool:
    values = [path.exists() for path in paths]
    if any(values) and not all(values):
        raise RuntimeError(f"partial append-only {label} artifacts exist")
    return all(values)


class Watcher:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.source = Path(args.source_pack).resolve()
        self.output = Path(args.output_root).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output / "WATCH_EVENTS.jsonl"
        self.passes = _parse_named_roots(args.initial_pass)

    def event(self, event: str, **payload: Any) -> None:
        row = {"schema_version": SCHEMA, "created_at": _now(), "event": event, **payload}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()

    def run_command(self, command: list[str], *, label: str, log_path: Path | None = None) -> None:
        self.event("COMMAND_STARTED", label=label, command=command)
        if log_path is None:
            completed = subprocess.run(command, check=False)
        else:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                completed = subprocess.run(
                    command,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                    text=True,
                )
        if completed.returncode:
            self.event("COMMAND_FAILED", label=label, returncode=completed.returncode)
            raise RuntimeError(f"{label} failed with return code {completed.returncode}")
        self.event("COMMAND_COMPLETED", label=label)

    def wait_for_raw(self, name: str, pack: Path) -> None:
        previous: tuple[int, int] | None = None
        while True:
            progress = raw_pack_progress(pack)
            state = int(progress["present"]), int(progress["expected"])
            if state != previous:
                self.event(
                    "RAW_PASS_PROGRESS",
                    pass_name=name,
                    pack_root=str(pack),
                    present=state[0],
                    expected=state[1],
                )
                previous = state
            if progress["complete"]:
                return
            time.sleep(self.args.poll_seconds)

    def validate_pass(self, name: str, pack: Path) -> Path:
        pass_root = self.output / "validated_passes"
        labels = pass_root / f"{name}.labels.jsonl"
        report = pass_root / f"{name}.report.json"
        if _all_or_none((labels, report), f"validated pass {name}"):
            return labels
        command = [
            sys.executable,
            "-m",
            "scripts.tools.silver_match_v3.validate_independent_teacher_labels",
            "--pack-root",
            str(pack),
            "--raw-label-dir",
            str(pack / "raw_labels"),
            "--annotator",
            self.args.annotator,
            "--label-source",
            f"{self.args.label_source_prefix}:{name}",
            "--output",
            str(labels),
            "--report",
            str(report),
        ]
        self.run_command(command, label=f"validate-{name}")
        self.event(
            "PASS_VALIDATED",
            pass_name=name,
            labels={"path": str(labels), "sha256": sha256_file(labels)},
            report={"path": str(report), "sha256": sha256_file(report)},
        )
        return labels

    def finalize_stage(
        self,
        pass_specs: list[tuple[str, Path, Path]],
    ) -> tuple[dict[str, Any], Path]:
        ordinal = len(pass_specs)
        stage = self.output / f"consensus_after_{ordinal:02d}_passes"
        resolved = stage / "resolved.jsonl"
        unresolved = stage / "unresolved.jsonl"
        disagreements = stage / "disagreements.jsonl"
        report_path = stage / "report.json"
        if _all_or_none(
            (resolved, unresolved, disagreements, report_path),
            f"consensus stage {ordinal}",
        ):
            return json.loads(report_path.read_text(encoding="utf-8")), unresolved
        command = [
            sys.executable,
            "-m",
            "scripts.tools.silver_match_v3.finalize_exact_multi_pass_truth",
            "--pack-root",
            str(self.source),
            "--output",
            str(resolved),
            "--unresolved-output",
            str(unresolved),
            "--disagreements-output",
            str(disagreements),
            "--report",
            str(report_path),
            "--gepa-role",
            "evaluation",
        ]
        for name, pack, labels in pass_specs:
            command.extend(("--label-pass", f"{name}={labels}", "--pass-pack", f"{name}={pack}"))
        self.run_command(command, label=f"finalize-after-{ordinal}-passes")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.event(
            "CONSENSUS_FROZEN",
            pass_count=ordinal,
            resolved_count=report["resolved_count"],
            unresolved_count=report["unresolved_count"],
            report={"path": str(report_path), "sha256": sha256_file(report_path)},
        )
        return report, unresolved

    def prepare_resolver(self, ordinal: int, unresolved: Path) -> Path:
        resolver = self.output / f"resolver_pass_{ordinal:02d}"
        pack = resolver / "pack"
        validation = pack / "validation.json"
        if pack.exists() and not validation.exists():
            raise RuntimeError(f"partial resolver pack exists: {pack}")
        if not validation.exists():
            self.run_command(
                [
                    sys.executable,
                    "-m",
                    "scripts.tools.silver_match_v3.prepare_exact_unresolved_resolver_pack",
                    "--pack-root",
                    str(self.source),
                    "--unresolved",
                    str(unresolved),
                    "--output-root",
                    str(pack),
                    "--seed",
                    str(self.args.resolver_seed_base + ordinal),
                    "--chunk-size",
                    str(self.args.chunk_size),
                ],
                label=f"prepare-resolver-{ordinal}",
            )
        return pack

    def label_resolver(self, ordinal: int, pack: Path) -> None:
        if raw_pack_progress(pack)["complete"]:
            return
        command = [
            sys.executable,
            "-u",
            "-m",
            "scripts.tools.silver_match_v3.run_codex_pack_labels",
            "--pack-root",
            str(pack),
            "--task",
            self.args.task,
            "--pass-name",
            f"{self.args.task}-exact-resolver-{ordinal}",
            "--model",
            self.args.model,
            "--reasoning-effort",
            self.args.reasoning_effort,
            "--concurrency",
            str(self.args.concurrency),
            "--timeout-seconds",
            str(self.args.timeout_seconds),
            "--chunk-attempts",
            str(self.args.chunk_attempts),
        ]
        for guide in self.args.boundary_guide:
            command.extend(("--boundary-guide", str(Path(guide).resolve())))
        self.run_command(
            command,
            label=f"label-resolver-{ordinal}",
            log_path=pack.parent / "runner.log",
        )
        if not raw_pack_progress(pack)["complete"]:
            raise RuntimeError(f"resolver {ordinal} returned without complete raw coverage")

    def run(self) -> None:
        self.event(
            "WATCH_STARTED",
            source_pack=str(self.source),
            source_validation_sha256=sha256_file(self.source / "validation.json"),
            max_passes=self.args.max_passes,
        )
        pass_specs: list[tuple[str, Path, Path]] = []
        for name, pack in self.passes:
            self.wait_for_raw(name, pack)
            pass_specs.append((name, pack, self.validate_pass(name, pack)))
        report, unresolved = self.finalize_stage(pass_specs)
        for ordinal in range(3, self.args.max_passes + 1):
            if report["complete"]:
                break
            pack = self.prepare_resolver(ordinal, unresolved)
            self.label_resolver(ordinal, pack)
            name = f"resolver_{ordinal:02d}"
            labels = self.validate_pass(name, pack)
            pass_specs.append((name, pack, labels))
            report, unresolved = self.finalize_stage(pass_specs)
        if report["complete"] and self.args.training_truth_output:
            truth_output = Path(self.args.training_truth_output).resolve()
            manifest_path = truth_output / "MANIFEST.json"
            if truth_output.exists() and not manifest_path.exists():
                raise RuntimeError(f"partial training-truth output exists: {truth_output}")
            if not manifest_path.exists():
                consensus_report = (
                    self.output
                    / f"consensus_after_{len(pass_specs):02d}_passes"
                    / "report.json"
                )
                self.run_command(
                    [
                        sys.executable,
                        "-m",
                        "scripts.tools.silver_match_v3.materialize_consensus_training_truth",
                        "--pack-root",
                        str(self.source),
                        "--consensus-report",
                        str(consensus_report),
                        "--output-root",
                        str(truth_output),
                    ],
                    label="materialize-training-truth",
                )
            self.event(
                "TRAINING_TRUTH_MATERIALIZED",
                manifest={"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            )
            if self.args.ce_truth_output_root:
                ce_root = Path(self.args.ce_truth_output_root).resolve()
                ce_paths = (
                    ce_root / "truth.ce-eligible.jsonl",
                    ce_root / "truth.typed-only.jsonl",
                    ce_root / "REPORT.json",
                )
                if ce_root.exists() and not _all_or_none(ce_paths, "CE truth partition"):
                    raise RuntimeError(f"partial CE truth partition exists: {ce_root}")
                if not ce_paths[2].exists():
                    self.run_command(
                        [
                            sys.executable,
                            "-m",
                            "scripts.tools.silver_match_v3.prepare_ce_eligible_truth",
                            "--truth",
                            str(truth_output / "truth.all.jsonl"),
                            "--output",
                            str(ce_paths[0]),
                            "--excluded",
                            str(ce_paths[1]),
                            "--report",
                            str(ce_paths[2]),
                        ],
                        label="partition-ce-eligible-truth",
                    )
                self.event(
                    "CE_TRUTH_PARTITIONED",
                    report={"path": str(ce_paths[2]), "sha256": sha256_file(ce_paths[2])},
                )
        terminal = "CONSENSUS_COMPLETE" if report["complete"] else "MAX_PASSES_REACHED"
        self.event(
            terminal,
            pass_count=len(pass_specs),
            resolved_count=report["resolved_count"],
            unresolved_count=report["unresolved_count"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pack", required=True)
    parser.add_argument("--initial-pass", action="append", required=True, help="NAME=PACK_ROOT")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--annotator", default="codex-gpt-5.6-sol-high")
    parser.add_argument("--label-source-prefix", default="independent_exact_full_bank")
    parser.add_argument("--boundary-guide", action="append", default=[])
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--chunk-attempts", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-passes", type=int, default=6)
    parser.add_argument("--resolver-seed-base", type=int, default=2026071400)
    parser.add_argument(
        "--training-truth-output",
        help="Optional append-only output root for frozen-split canonical training truth.",
    )
    parser.add_argument(
        "--ce-truth-output-root",
        help="Optional append-only pairwise-CE truth partition output root.",
    )
    args = parser.parse_args()
    if args.max_passes < 3 or args.concurrency < 1 or args.poll_seconds < 1:
        parser.error("invalid max-passes, concurrency, or poll-seconds")
    Watcher(args).run()


if __name__ == "__main__":
    main()
