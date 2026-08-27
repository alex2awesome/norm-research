import json
from pathlib import Path

from scripts.tools.silver_match_v3.common import read_jsonl, write_jsonl
from scripts.tools.silver_match_v3.prepare_false_abstention_audit import prepare


def test_false_abstention_samples_are_uniform_scoped_and_blind(tmp_path):
    banks = {}
    corpora = {}
    final_rows = []
    for task_index, task in enumerate(("t1", "t2")):
        bank = tmp_path / f"{task}.bank.json"
        bank.write_text(
            json.dumps(
                {"metrics": [{"metric_id": "a0", "name": "metric", "description": "d"}]}
            )
        )
        banks[task] = {"path": str(bank), "source_sha256": f"sha-{task}"}
        corpus = f"c{task_index}"
        norms = []
        for row_index in range(6):
            uid = f"{task_index}{row_index}".ljust(64, "0")
            norms.append(
                {
                    "norm_uid": uid,
                    "row": row_index,
                    "corpus": corpus,
                    "task": task,
                    "norm": f"norm {row_index}",
                    "context": f"context {row_index}",
                }
            )
            final_rows.append(
                {
                    "norm_uid": uid,
                    "row": row_index,
                    "corpus": corpus,
                    "task": task,
                    "decision": "MATCH" if row_index == 0 else "NO_CANDIDATE_FITS",
                    "metric_id": "a0" if row_index == 0 else None,
                    "confidence": "high",
                    "reason": "hidden system reason",
                }
            )
        norm_path = tmp_path / f"{corpus}.jsonl"
        write_jsonl(norm_path, norms)
        corpora[corpus] = {"task": task, "count": len(norms), "path": str(norm_path)}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"corpora": corpora, "banks": banks}))
    final = tmp_path / "final.jsonl"
    write_jsonl(final, final_rows)

    report = prepare(
        manifest_path=manifest,
        final_paths=[final],
        output_root=tmp_path / "audit",
        global_n=4,
        per_task_n=3,
        seed="fixed",
    )
    assert report["population_decisions"] == 10
    assert report["outputs"]["global"]["sample_n"] == 4
    assert report["outputs"]["task:t1"]["sample_n"] == 3
    blind = list(read_jsonl(tmp_path / "audit" / "global.blind.jsonl"))
    key = list(read_jsonl(tmp_path / "audit" / "global.key.jsonl"))
    assert all(row["decision"] is None and row["reason"] is None for row in blind)
    assert all("system_decision" not in row for row in blind)
    assert all(row["system_decision"] == "NO_CANDIDATE_FITS" for row in key)
    assert {row["norm_uid"] for row in blind} == {row["norm_uid"] for row in key}
    task_pack_ref = report["outputs"]["task:t1"]["label_pack_validation"]
    task_pack = json.loads(Path(task_pack_ref["path"]).read_text())
    assert task_pack["schema_version"] == "silver-match-v3-final-risk-label-pack-v1"
    assert task_pack["truth_hidden"] is True
    assert task_pack["system_key_excluded_from_label_pack"] is True
    assert task_pack["count"] == 3

    match_report = prepare(
        manifest_path=manifest,
        final_paths=[final],
        output_root=tmp_path / "match-audit",
        global_n=2,
        per_task_n=1,
        seed="fixed-match",
        sample_kind="match",
    )
    assert match_report["sample_kind"] == "match"
    assert match_report["population_decisions"] == 2
    match_key = list(read_jsonl(tmp_path / "match-audit" / "global.key.jsonl"))
    assert all(row["system_decision"] == "MATCH" for row in match_key)

    excluded_uid = "01".ljust(64, "0")
    exclusion = tmp_path / "analysis-exclusion.jsonl"
    write_jsonl(exclusion, [{"norm_uid": excluded_uid}])
    eligible_report = prepare(
        manifest_path=manifest,
        final_paths=[final],
        output_root=tmp_path / "eligible-audit",
        global_n=4,
        per_task_n=3,
        seed="fixed-eligible",
        exclude_paths=[exclusion],
    )
    assert eligible_report["population_decisions"] == 9
    assert eligible_report["analysis_exclusions"]["count"] == 1
    assert eligible_report["analysis_exclusions"]["excluded_final_rows_seen"] == 1
    eligible_blind = list(
        read_jsonl(tmp_path / "eligible-audit" / "global.blind.jsonl")
    )
    assert excluded_uid not in {row["norm_uid"] for row in eligible_blind}
