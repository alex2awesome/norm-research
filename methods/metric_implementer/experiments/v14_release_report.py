"""Structural completion audit and compact release report for CR-3 v14."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .v14_panel_design import canonical_sha256
from .v14_value_bound import validate_state_tables
from .v14_scoring_lanes import assert_release_rows_are_cert


RELEASE_SCHEMA = "cr3-v14-release-report-v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_release(root: str | Path, *, expected_metrics: int = 35) -> dict:
    source = Path(root).resolve()
    campaign = json.loads((source / "campaign_manifest.json").read_text())
    frame = pd.read_parquet(source / "results.parquet")
    expected_rows = int(expected_metrics) * 2 * 3
    failures = []
    try:
        assert_release_rows_are_cert(frame)
    except ValueError as exc:
        failures.append(str(exc))
    if len(frame) != expected_rows:
        failures.append(f"results has {len(frame)} rows; expected {expected_rows}")
    identity = ["metric_key", "instrument", "channel", "arm"]
    if frame.duplicated(identity).any():
        failures.append("results contains duplicate metric/instrument/channel/arm rows")
    if frame["metric_key"].nunique() != int(expected_metrics):
        failures.append("results does not contain the declared metric population")
    for required in ("template_freeze.json", "preregistration.json", "sentinel_report.json"):
        if not (source / required).is_file():
            failures.append(f"missing campaign freeze artifact {required}")
    mhat_manifest = source / "mhat_archive" / "manifest.json"
    if not mhat_manifest.is_file():
        failures.append("missing release-blocking full M-hat archive")
    else:
        archive = json.loads(mhat_manifest.read_text())
        if archive.get("schema") != "cr3-v14-mhat-archive-v1" or int(
            archive.get("n_identity_rows", 0)
        ) <= 0:
            failures.append("invalid or empty M-hat archive")
    sentinel_path = source / "sentinel_report.json"
    if sentinel_path.is_file() and not bool(json.loads(sentinel_path.read_text()).get("passed")):
        failures.append("control-based sentinel liveness report did not pass")
    if not bool(campaign.get("control_liveness_gate_applied", False)):
        failures.append("control-based sentinel liveness gate was not applied")
    required_status = {"RESOLVED", "PLATEAUED", "RISING", "UNRESOLVED", "DEAD_INSTRUMENT", "ZERO_CAP"}
    if not set(frame["status"]).issubset(required_status):
        failures.append("results contains an unsupported status")
    artifacts = []
    for _, row in frame.iterrows():
        certificate_path = Path(row["certificate_path"])
        if not certificate_path.is_file():
            failures.append(f"missing certificate {certificate_path}")
            continue
        certificate = json.loads(certificate_path.read_text())
        core = dict(certificate)
        observed = str(core.pop("certificate_sha256", ""))
        if observed != canonical_sha256(core):
            failures.append(f"certificate checksum mismatch {certificate_path}")
        table_path = certificate_path.parent / "state_tables.npz"
        prompt_path = certificate_path.parent / "prompt_values.parquet"
        design_path = certificate_path.parent / "design_manifest.json"
        novelty_path = certificate_path.parent / "novelty_curves.parquet"
        for path in (table_path, prompt_path, design_path, novelty_path):
            if not path.is_file():
                failures.append(f"missing artifact {path}")
        if table_path.is_file():
            with np.load(table_path, allow_pickle=False) as state:
                try:
                    shape = np.asarray(state["raw_lift"]).shape
                    if len(shape) != 2 or shape[0] != 50 or shape[1] not in {64, 256}:
                        failures.append(f"state table is not exhaustive 50x64/256: {table_path}")
                    else:
                        validate_state_tables(
                            state["raw_lift"], state["clipped_value"],
                            panel_size=6 if shape[1] == 64 else 8,
                        )
                except Exception as exc:
                    failures.append(f"invalid state table {table_path}: {exc}")
        if prompt_path.is_file():
            prompt_frame = pd.read_parquet(prompt_path)
            if not {"prompt_id", "raw_lift", "value", "fidelity_bits"}.issubset(
                prompt_frame.columns
            ):
                failures.append(f"prompt table lacks fidelity/legibility columns {prompt_path}")
        cap = float(certificate["free_recombination_cap"])
        achieved = float(certificate["achieved_value"])
        if not np.isfinite([cap, achieved]).all() or cap + 1e-12 < achieved:
            failures.append(f"invalid cap/achieved relation {certificate_path}")
        bounds = certificate.get("process_relative_bounds")
        if not isinstance(bounds, dict) or set(bounds) != {"100", "300"}:
            failures.append(f"missing process-relative horizon bounds {certificate_path}")
        if not isinstance(certificate.get("ceiling_table"), dict):
            failures.append(f"missing layered ceiling table {certificate_path}")
        artifacts.extend(path for path in (
            certificate_path, table_path, prompt_path, design_path,
            certificate_path.parent / "novelty_curves.parquet",
        ) if path.is_file())
    audit_complete = bool(campaign.get("fresh_audit_complete", False))
    if not audit_complete:
        failures.append("fresh 400-draw audit stage is incomplete")
    for metric_key in sorted(set(map(str, frame["metric_key"]))):
        audit_path = source / "audit" / "signatures" / f"{metric_key}.npz"
        if not audit_path.is_file():
            failures.append(f"missing fresh audit signatures {audit_path}")
            continue
        with np.load(audit_path, allow_pickle=True) as audit:
            if np.asarray(audit["sigs"]).shape[0] != 400 or str(
                audit["evidence_role"]
            ) != "never_absorbed_pure_audit":
                failures.append(f"invalid fresh 400-draw audit {audit_path}")
    checksum_rows = [
        {"path": str(path), "sha256": _file_sha256(path), "bytes": path.stat().st_size}
        for path in sorted(set(artifacts))
    ]
    scientific_rows = []
    for keys, group in frame.groupby(["instrument", "channel", "arm"], dropna=False):
        instrument, channel, arm = keys
        scientific_rows.append({
            "instrument": str(instrument), "channel": str(channel),
            "arm": None if pd.isna(arm) else str(arm), "n": len(group),
            "mean_achieved_value": float(np.mean(group["achieved_value"])),
            "mean_structural_cap": float(np.mean(group["structural_cap"])),
            "mean_structural_gap": float(np.mean(group["structural_gap"])),
        })
    correlations = np.asarray(
        frame["mcq_behavioral_spearman"].dropna(), dtype=float,
    )
    exemplar_gaps = np.asarray(
        frame["exemplar_vs_rule_achieved_gap_bits"].dropna(), dtype=float,
    )
    report = {
        "schema": RELEASE_SCHEMA,
        "complete": not bool(failures),
        "failures": failures,
        "n_metrics": int(frame["metric_key"].nunique()),
        "n_results": len(frame),
        "expected_results": expected_rows,
        "status_counts": {
            str(key): int(value) for key, value in frame["status"].value_counts().items()
        },
        "fresh_audit_complete": audit_complete,
        "scientific_summary": scientific_rows,
        "median_mcq_behavioral_spearman": (
            float(np.median(correlations)) if len(correlations) else None
        ),
        "mean_exemplar_vs_rule_achieved_gap_bits": (
            float(np.mean(exemplar_gaps)) if len(exemplar_gaps) else None
        ),
        "n_rows_allowing_optimal_prompt_ranking": int(np.sum(frame["ranking_allowed"])),
        "artifact_checksums": checksum_rows,
    }
    report["release_report_sha256"] = canonical_sha256(report)
    return report


def write_release_outputs(root: str | Path, report: Mapping[str, object]) -> None:
    source = Path(root).resolve()
    status = {
        "schema": "cr3-v14-run-status-v1",
        "state": "complete" if report["complete"] else "structural_failure",
        "release_report_sha256": report["release_report_sha256"],
        "failures": report["failures"],
    }
    (source / "run_status.json").write_text(json.dumps(status, indent=2) + "\n")
    checksums = {
        "schema": "cr3-v14-artifact-checksums-v1",
        "artifacts": report["artifact_checksums"],
    }
    checksums["manifest_sha256"] = canonical_sha256(checksums)
    (source / "artifact_checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
    (source / "release_report.json").write_text(json.dumps(dict(report), indent=2) + "\n")
    lines = [
        "# CR-3 v14 campaign report",
        "",
        f"Structural completion: **{'PASS' if report['complete'] else 'FAIL'}**",
        f"Metrics: {report['n_metrics']}; result rows: {report['n_results']}.",
        f"Fresh audit complete: {report['fresh_audit_complete']}.",
        "",
        "## Status counts",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in report["status_counts"].items())
    lines.extend([
        "", "## Scientific summary", "",
        "| Instrument | Channel | Arm | Mean achieved | Mean cap | Mean gap |",
        "|---|---|---|---:|---:|---:|",
    ])
    lines.extend(
        "| {instrument} | {channel} | {arm} | {mean_achieved_value:.6f} | "
        "{mean_structural_cap:.6f} | {mean_structural_gap:.6f} |".format(
            **{**row, "arm": row["arm"] or "—"}
        )
        for row in report["scientific_summary"]
    )
    lines.extend([
        "",
        f"Median MCQ/behavioral Spearman: {report['median_mcq_behavioral_spearman']}.",
        "Mean unconstrained-minus-no-verbatim achieved gap: "
        f"{report['mean_exemplar_vs_rule_achieved_gap_bits']} bits.",
        "Optimal-prompt ranking allowed for "
        f"{report['n_rows_allowing_optimal_prompt_ranking']} result rows.",
    ])
    if report["failures"]:
        lines.extend(["", "## Structural failures", ""])
        lines.extend(f"- {value}" for value in report["failures"])
    (source / "report.md").write_text("\n".join(lines) + "\n")
