import json
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.aggregate_nemotron_ce_seed_consensus import (
    CONSENSUS_REPORT_SCHEMA,
    CONSENSUS_SCHEMA,
)
from scripts.tools.silver_match_v3.audit_progressive_nemotron_ce_dev_policy import (
    freeze_policy,
)
from scripts.tools.silver_match_v3.common import sha256_file
from scripts.tools.silver_match_v3.materialize_nemotron_ce_production_pairs import (
    META_SCHEMA,
    PAIR_SCHEMA,
    UNIVERSE_SCHEMA,
)
from scripts.tools.silver_match_v3.materialize_progressive_nemotron_ce_pairs import (
    MANIFEST_SCHEMA,
    materialize,
)
from scripts.tools.silver_match_v3.run_frozen_progressive_nemotron_ce_production import (
    _artifact,
    _materialize_active_pairs,
    _merge_final,
    _partition_trial,
)


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _pair(uid: str, metric: str, rank: int, depth: int) -> dict:
    return {
        "schema_version": PAIR_SCHEMA,
        "task": "humor",
        "corpus": "jokes",
        "norm_uid": uid,
        "source_group": f"group-{uid}",
        "split": "production",
        "query": f"norm {uid}",
        "metric_id": metric,
        "metric_card": f"metric {metric}",
        "candidate_rank": rank,
        "current_bank_source_sha256": "b" * 64,
        "source_depth": depth,
    }


def _pair_report(
    root: Path,
    *,
    name: str,
    depth: int,
    metrics: list[str],
    bank: Path,
    universe: Path,
) -> Path:
    pairs = root / f"{name}.pairs.jsonl"
    _jsonl(
        pairs,
        [
            _pair(uid, metric, rank, depth)
            for uid in ("n1", "n2")
            for rank, metric in enumerate(metrics, 1)
        ],
    )
    report = root / f"{name}.report.json"
    _json(
        report,
        {
            "schema_version": META_SCHEMA,
            "status": "FROZEN_COMPLETE_UNLABELED_PRODUCTION_PAIR_UNIVERSE",
            "task": "humor",
            "bank": {
                "path": str(bank),
                "sha256": sha256_file(bank),
                "source_sha256": "b" * 64,
                "metric_count": 4,
            },
            "corpus_order": ["jokes"],
            "norm_count": 2,
            "candidate_depth": depth,
            "pair_count": 2 * depth,
            "pairs": {"path": str(pairs), "sha256": sha256_file(pairs)},
            "norm_universe": {
                "path": str(universe),
                "sha256": sha256_file(universe),
                "count": 2,
            },
            "labels_present": False,
            "release_ready": False,
        },
    )
    return report


def test_materializer_partitions_full_bank_exactly_once(tmp_path: Path) -> None:
    bank = tmp_path / "bank.json"
    _json(
        bank,
        {
            "source_sha256": "b" * 64,
            "metrics": [{"metric_id": f"m{i}"} for i in range(1, 5)],
        },
    )
    universe = tmp_path / "universe.jsonl"
    _jsonl(
        universe,
        [
            {
                "schema_version": UNIVERSE_SCHEMA,
                "task": "humor",
                "corpus": "jokes",
                "norm_uid": uid,
                "source_group": f"group-{uid}",
                "split": "production",
            }
            for uid in ("n1", "n2")
        ],
    )
    primary = _pair_report(
        tmp_path,
        name="primary",
        depth=3,
        metrics=["m1", "m2", "m3"],
        bank=bank,
        universe=universe,
    )
    full = _pair_report(
        tmp_path,
        name="full",
        depth=4,
        metrics=["m3", "m4", "m1", "m2"],
        bank=bank,
        universe=universe,
    )
    candidates = tmp_path / "primary.candidates.jsonl"
    candidate_rows = []
    for uid in ("n1", "n2"):
        candidate_rows.append(
            {
                "task": "humor",
                "corpus": "jokes",
                "norm_uid": uid,
                "candidates": [
                    {"metric_id": "m1", "rank": 1, "lane_ranks": {"a": 1, "b": 3}},
                    {"metric_id": "m2", "rank": 2, "lane_ranks": {"a": 4, "b": 2}},
                    {"metric_id": "m3", "rank": 3, "lane_ranks": {}},
                ],
            }
        )
    _jsonl(candidates, candidate_rows)
    _json(
        candidates.with_suffix(candidates.suffix + ".meta.json"),
        {
            "output_sha256": sha256_file(candidates),
            "task": "humor",
            "bank_source_sha256": "b" * 64,
            "input_count": 2,
            "output_k": 3,
            "union": {
                "lanes": [
                    {"name": "a", "kind": "complete-bank"},
                    {"name": "b", "kind": "complete-bank"},
                ]
            },
        },
    )
    result = materialize(
        task="humor",
        primary_report_path=primary,
        fullbank_report_path=full,
        primary_candidates_path=candidates,
        output_root=tmp_path / "progressive",
        component_depths=[1, 2],
    )
    assert result["schema_version"] == MANIFEST_SCHEMA
    assert result["coverage_contract"]["total_pair_count"] == 8
    assert [row["pairs"]["count"] for row in result["trials"]] == [2, 2, 2, 2]
    per_uid: dict[str, set[str]] = {"n1": set(), "n2": set()}
    for trial in result["trials"]:
        for row in map(json.loads, Path(trial["pairs"]["path"]).read_text().splitlines()):
            metric = row["metric_id"]
            assert metric not in per_uid[row["norm_uid"]]
            per_uid[row["norm_uid"]].add(metric)
    assert per_uid == {"n1": {"m1", "m2", "m3", "m4"}, "n2": {"m1", "m2", "m3", "m4"}}


def _consensus_row(uid: str, candidates: list[str], *, match: str | None) -> dict:
    passes = match is not None
    states = {
        seed: {
            "top_metric_id": match or candidates[0],
            "top_predicted_relation": "EXACT" if passes else "REJECT",
            "top_exact_probability": 0.99 if passes else 0.01,
            "second_exact_probability": 0.01,
            "top_exact_margin": 0.98 if passes else 0.0,
            "score_threshold": 0.8,
            "top_margin_threshold": 0.1,
            "passes_frozen_gate": passes,
        }
        for seed in ("s1", "s2")
    }
    return {
        "schema_version": CONSENSUS_SCHEMA,
        "task": "humor",
        "corpus": "jokes",
        "norm_uid": uid,
        "source_group": f"group-{uid}",
        "split": "dev",
        "decision": "MATCH" if passes else "ROUTE_TO_ADJUDICATION",
        "routing_category": "MATCH" if passes else "CE_REJECT_BOTH",
        "automatic_match": passes,
        "metric_id": match,
        "candidate_count": len(candidates),
        "seed_decisions": states,
        "candidates": [{"metric_id": value} for value in candidates],
        "provisional_routing_only": not passes,
        "human_abstention_subtype_assigned": False,
    }


def _consensus_artifacts(
    root: Path, name: str, rows: list[dict]
) -> tuple[Path, Path]:
    output = root / f"{name}.jsonl"
    _jsonl(output, rows)
    report = root / f"{name}.report.json"
    _json(
        report,
        {
            "schema_version": CONSENSUS_REPORT_SCHEMA,
            "status": "COMPLETE",
            "output_sha256": sha256_file(output),
            "norm_count": len(rows),
            "seeds": [
                {
                    "seed_id": seed,
                    "training_report_sha256": ("a" if seed == "s1" else "b") * 64,
                    "frozen_gate": {"provenance": "checkpoint.dev"},
                }
                for seed in ("s1", "s2")
            ],
            "validation": {
                "all_thresholds_from_checkpoint_dev": True,
                "test_threshold_tuning_performed": False,
                "all_norms_preserved": True,
                "seed_norm_candidate_source_split_universes_identical": True,
            },
        },
    )
    return output, report


def test_dev_policy_authorizes_only_simultaneously_precise_stable_exit(tmp_path: Path) -> None:
    truth = tmp_path / "truth.jsonl"
    rows = [
        {
            "task": "humor",
            "norm_uid": f"n{i:03d}",
            "source_group": f"group-n{i:03d}",
            "split": "dev",
            "collection_role": "dev",
            "training_eligible": False,
            "blind_evaluation_only": False,
            "decision": "MATCH",
            "metric_id": "m1",
        }
        for i in range(200)
    ]
    _jsonl(truth, rows)
    early, early_report = _consensus_artifacts(
        tmp_path,
        "early",
        [_consensus_row(row["norm_uid"], ["m1"], match="m1") for row in rows],
    )
    terminal, terminal_report = _consensus_artifacts(
        tmp_path,
        "terminal",
        [_consensus_row(row["norm_uid"], ["m1", "m2"], match="m1") for row in rows],
    )
    policy = freeze_policy(
        task="humor",
        truth_path=truth,
        trials=[
            ("component-union-d1", early, early_report),
            ("fullbank-k2-rescue", terminal, terminal_report),
        ],
        output_path=tmp_path / "policy.json",
    )
    assert policy["authorized_early_stop_trials"] == ["component-union-d1"]
    assert policy["trial_audits"][0]["exact_truth_error_upper_simultaneous"] < 0.05
    assert policy["estimated_compute"]["estimated_pair_evaluation_reduction_rate"] > 0
    assert policy["safety"]["test_or_blind_labels_read"] is False


def test_dev_policy_rejects_early_matches_that_terminal_bank_displaces(
    tmp_path: Path,
) -> None:
    truth = tmp_path / "truth.jsonl"
    truth_rows = [
        {
            "task": "humor",
            "norm_uid": f"n{i:03d}",
            "source_group": f"group-n{i:03d}",
            "split": "dev",
            "collection_role": "dev",
            "training_eligible": False,
            "blind_evaluation_only": False,
            "decision": "MATCH",
            "metric_id": "m1",
        }
        for i in range(200)
    ]
    _jsonl(truth, truth_rows)
    early, early_report = _consensus_artifacts(
        tmp_path,
        "unstable-early",
        [
            _consensus_row(row["norm_uid"], ["m1"], match="m1")
            for row in truth_rows
        ],
    )
    terminal_rows = []
    for index, row in enumerate(truth_rows):
        terminal_rows.append(
            _consensus_row(
                row["norm_uid"],
                ["m1", "m2"],
                match="m2" if index < 20 else "m1",
            )
        )
    terminal, terminal_report = _consensus_artifacts(
        tmp_path, "unstable-terminal", terminal_rows
    )
    policy = freeze_policy(
        task="humor",
        truth_path=truth,
        trials=[
            ("component-union-d1", early, early_report),
            ("fullbank-k2-rescue", terminal, terminal_report),
        ],
        output_path=tmp_path / "unstable-policy.json",
    )
    assert policy["authorized_early_stop_trials"] == []
    assert policy["trial_audits"][0]["terminal_decision_instability_count"] == 20
    assert policy["trial_audits"][0]["authorized_for_early_stop"] is False


def test_active_pair_filter_is_immutable_and_unauthorized_matches_continue(tmp_path: Path) -> None:
    source = tmp_path / "tier.jsonl"
    source_rows = []
    for uid in ("n1", "n2"):
        for metric in ("m1", "m2"):
            source_rows.append(
                {
                    **_pair(uid, metric, 1, 2),
                    "progressive_trial_id": "t1",
                }
            )
    _jsonl(source, source_rows)
    active_source = tmp_path / "active.jsonl"
    _jsonl(
        active_source,
        [
            {
                "schema_version": UNIVERSE_SCHEMA,
                "task": "humor",
                "corpus": "jokes",
                "norm_uid": "n2",
                "source_group": "group-n2",
                "split": "production",
            }
        ],
    )
    trial = {
        "trial_id": "t1",
        "runtime_root": str(tmp_path / "runtime"),
        "pairs": {"path": str(source), "sha256": sha256_file(source)},
        "ordinal": 1,
        "early_stop_authorized": False,
    }
    plan = {"task": "humor", "execution": {"num_shards_per_seed": 2}}
    output, meta = _materialize_active_pairs(
        plan, trial, {"n2"}, active_source
    )
    assert meta["pair_count"] == 2
    assert {row["norm_uid"] for row in map(json.loads, output.read_text().splitlines())} == {"n2"}
    assert _materialize_active_pairs(plan, trial, {"n2"}, active_source)[1] == meta

    consensus = tmp_path / "consensus.jsonl"
    row = _consensus_row("n2", ["m1", "m2"], match="m1")
    row["split"] = "production"
    _jsonl(consensus, [row])
    accepted, continued, accepted_count, continued_count = _partition_trial(
        trial, consensus, terminal=False
    )
    assert (accepted_count, continued_count) == (0, 1)
    assert accepted.read_text() == ""
    assert next(read for read in map(json.loads, continued.read_text().splitlines()))[
        "automatic_match"
    ] is True


def test_final_merger_emits_exactly_one_decision_per_norm(tmp_path: Path) -> None:
    universe = tmp_path / "universe.jsonl"
    _jsonl(
        universe,
        [
            {
                "schema_version": UNIVERSE_SCHEMA,
                "task": "humor",
                "corpus": "jokes",
                "norm_uid": uid,
                "source_group": f"g-{uid}",
                "split": "production",
            }
            for uid in ("n1", "n2", "n3")
        ],
    )
    early_accepted = tmp_path / "early.accepted.jsonl"
    terminal_accepted = tmp_path / "terminal.accepted.jsonl"
    terminal_continue = tmp_path / "terminal.continue.jsonl"
    rows = {
        "n1": _consensus_row("n1", ["m1"], match="m1"),
        "n2": _consensus_row("n2", ["m1", "m2"], match="m2"),
        "n3": _consensus_row("n3", ["m1", "m2"], match=None),
    }
    for row in rows.values():
        row["split"] = "production"
    _jsonl(early_accepted, [rows["n1"]])
    _jsonl(terminal_accepted, [rows["n2"]])
    _jsonl(terminal_continue, [rows["n3"]])
    empty = tmp_path / "empty.jsonl"
    _jsonl(empty, [])
    stage1 = tmp_path / "stage1.json"
    stage2 = tmp_path / "stage2.json"
    _json(stage1, {"complete": True})
    _json(stage2, {"complete": True})
    manifest = tmp_path / "progressive.json"
    _json(
        manifest,
        {
            "coverage_contract": {
                "worst_case_two_seed_pair_evaluations": 12,
            }
        },
    )
    records = [
        {
            "terminal": False,
            "accepted": _artifact(early_accepted, count=1),
            "continued": _artifact(empty, count=0),
            "new_pair_count_one_seed": 3,
        },
        {
            "terminal": True,
            "accepted": _artifact(terminal_accepted, count=1),
            "continued": _artifact(terminal_continue, count=1),
            "new_pair_count_one_seed": 2,
        },
    ]
    plan = {
        "task": "humor",
        "norm_count": 3,
        "norm_universe": _artifact(universe, count=3),
        "progressive_pairs_manifest": _artifact(manifest),
        "dev_stop_policy": _artifact(stage1),
        "trials": [
            {"stage_record": str(stage1)},
            {"stage_record": str(stage2)},
        ],
        "seeds": [
            {
                "seed_id": seed,
                "checkpoint": f"/{seed}",
                "training_report": {"path": f"/{seed}.json", "sha256": seed * 64},
                "checkpoint_contract": {
                    "score_threshold": 0.8,
                    "top_margin_threshold": 0.1,
                },
            }
            for seed in ("a", "b")
        ],
        "outputs": {
            "progressive_consensus": str(tmp_path / "final.jsonl"),
            "progressive_consensus_report": str(tmp_path / "final.report.json"),
        },
    }
    output, _, report = _merge_final(plan, records)
    final_rows = list(map(json.loads, output.read_text().splitlines()))
    assert [row["norm_uid"] for row in final_rows] == ["n1", "n2", "n3"]
    assert report["validation"]["one_terminal_ce_decision_per_norm"] is True
    assert report["compute"]["realized_pair_evaluation_reduction_rate"] == pytest.approx(1 / 6)
