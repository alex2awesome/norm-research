import json
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.derive_explicit_role_activation import derive
from scripts.tools.silver_match_v3.common import sha256_file


def _release(path, task, role):
    path.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-clean-gepa-exact-truth-release-v2",
                "status": "FROZEN_EXACT_TRUTH_RELEASE_AUDITED",
                "task": task,
                "role": role,
                "scientific_contract": {
                    "strict_transcript_pass_required_for_every_consensus_pass": True,
                    "cross_workspace_artifacts_hash_equivalent": True,
                    "legacy_transcripts_allowed": False,
                },
            }
        )
    )


def _fixture(tmp_path):
    source = tmp_path / "activation_v1.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-task-local-gepa-predeclaration-v1",
                "status": "FROZEN_AND_EXECUTION_AUTHORIZED",
                "tasks": {
                    task: {
                        "execution_evidence": {
                            "optimize_truth_release_sha256": "old-optimize",
                            "select_truth_release_sha256": "old-select",
                            "complete_exclusion_union_sha256": f"union-{task}",
                        }
                    }
                    for task in ("code-review", "math-stackexchange")
                },
            }
        )
    )
    paths = {}
    for task, short in (("code-review", "code"), ("math-stackexchange", "math")):
        for role in ("optimize", "select"):
            path = tmp_path / f"{short}-{role}.json"
            _release(path, task, role)
            paths[f"{short}_{role}_release"] = str(path)
    return Namespace(
        source_lock=str(source),
        output=str(tmp_path / "activation_v2.json"),
        **paths,
    )


def test_derives_only_new_truth_release_evidence(tmp_path):
    args = _fixture(tmp_path)
    result = derive(args)
    output = json.loads(Path(args.output).read_text())
    source = json.loads(Path(args.source_lock).read_text())
    for task, short in (("code-review", "code"), ("math-stackexchange", "math")):
        evidence = output["tasks"][task]["execution_evidence"]
        assert evidence["complete_exclusion_union_sha256"] == f"union-{task}"
        assert evidence["optimize_truth_release_sha256"] == sha256_file(
            Path(getattr(args, f"{short}_optimize_release"))
        )
        assert evidence["select_truth_release_sha256"] == sha256_file(
            Path(getattr(args, f"{short}_select_release"))
        )
    assert result["sha256"] == sha256_file(Path(args.output))
    assert output["schema_version"] == source["schema_version"]


def test_refuses_overwrite(tmp_path):
    args = _fixture(tmp_path)
    Path(args.output).write_text("existing\n")
    with pytest.raises(FileExistsError):
        derive(args)
