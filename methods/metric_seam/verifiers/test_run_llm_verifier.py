from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import subprocess

import pytest

from methods.metric_seam.verifiers import run_llm_verifier as runner
from methods.metric_seam.verifiers.llm_contract import UnitContract, compile_request


MODEL = "claude-test-pinned"
DIFF = """diff --git a/a.py b/a.py
index 1111111..2222222 100644
--- a/a.py
+++ b/a.py
@@ -1 +1,2 @@
 x = 1
+print(x)
"""
CONTRACT = UnitContract(
    "u1", "Observability", "Avoid bare debug output.", "llm", "node-1"
)
VALID = '{"applies":false,"violated":false,"witnesses":[]}'
FENCED_VALID = f"```json\n{VALID}\n```"


def _write_bundle(
    path: Path, count: int = 12, *, split: str = "compiler_train"
) -> list[dict]:
    requests = [
        compile_request(
            contract=CONTRACT,
            item_key=f"item-{index:02d}",
            ctext=DIFF,
            pass_index=1 + index % 2,
            model=MODEL,
            split=split,
        )
        for index in range(count)
    ]
    path.write_text(
        "".join(json.dumps(request, sort_keys=True) + "\n" for request in requests),
        encoding="utf-8",
    )
    return requests


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_dry_run_plans_exact_smoke_without_subprocess_or_output(tmp_path: Path) -> None:
    bundle = tmp_path / "requests.jsonl"
    output = tmp_path / "responses.jsonl"
    _write_bundle(bundle)

    def forbidden_invoker(argv, timeout):  # pragma: no cover - failure sentinel
        raise AssertionError("dry run invoked a subprocess")

    summary = runner.run_bundle(
        bundle_path=bundle,
        output_path=output,
        phase="smoke",
        model=MODEL,
        max_concurrency=4,
        dry_run=True,
        invoker=forbidden_invoker,
    )

    assert summary["planned_request_count"] == 10
    assert summary["attempts_appended"] == 0
    assert not output.exists()


def test_mocked_subprocess_fenced_smoke_then_production_never_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "requests.jsonl"
    output = tmp_path / "responses.jsonl"
    _write_bundle(bundle)
    calls: list[tuple[str, ...]] = []

    def fake_subprocess_run(argv, **kwargs):
        calls.append(tuple(argv))
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert "-p" in argv
        assert "--resume" not in argv
        assert argv[argv.index("--model") + 1] == MODEL
        return subprocess.CompletedProcess(argv, 0, FENCED_VALID, "")

    monkeypatch.setattr(runner.subprocess, "run", fake_subprocess_run)
    smoke = runner.run_bundle(
        bundle_path=bundle,
        output_path=output,
        phase="smoke",
        model=MODEL,
        max_concurrency=4,
        max_attempts=1,
    )
    assert smoke["smoke_passed"] is True
    assert smoke["attempts_appended"] == 10

    production = runner.run_bundle(
        bundle_path=bundle,
        output_path=output,
        phase="production",
        model=MODEL,
        max_concurrency=4,
        max_attempts=1,
    )
    assert production["attempts_appended"] == 2
    assert len(calls) == 12

    repeated = runner.run_bundle(
        bundle_path=bundle,
        output_path=output,
        phase="production",
        model=MODEL,
        max_concurrency=4,
        max_attempts=1,
    )
    assert repeated["planned_request_count"] == 0
    assert repeated["attempts_appended"] == 0
    assert len(calls) == 12
    rows = _rows(output)
    assert len(rows) == 12
    assert {row["status"] for row in rows} == {"valid"}
    assert {
        row["validated_response"]["parse_mode"] for row in rows
    } == {"fence_unwrapped"}


def test_smoke_rejects_retries_that_could_hide_contract_instability(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "requests.jsonl"
    output = tmp_path / "responses.jsonl"
    _write_bundle(bundle, count=10)
    with pytest.raises(runner.HarnessError, match="exactly one attempt"):
        runner.run_bundle(
            bundle_path=bundle,
            output_path=output,
            phase="smoke",
            model=MODEL,
            max_attempts=2,
            invoker=lambda argv, timeout: runner.InvocationResult(0, VALID, ""),
        )
    assert not output.exists()


def test_production_stops_before_invocation_when_smoke_is_not_10_of_10(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "requests.jsonl"
    output = tmp_path / "responses.jsonl"
    _write_bundle(bundle)
    invocation_count = 0

    def one_bad_smoke(argv, timeout):
        nonlocal invocation_count
        invocation_count += 1
        raw = VALID if invocation_count <= 9 else "invalid"
        return runner.InvocationResult(0, raw, "")

    smoke = runner.run_bundle(
        bundle_path=bundle,
        output_path=output,
        phase="smoke",
        model=MODEL,
        max_concurrency=1,
        max_attempts=1,
        invoker=one_bad_smoke,
    )
    assert smoke["smoke_passed"] is False
    before = output.read_bytes()

    def forbidden_production(argv, timeout):  # pragma: no cover - failure sentinel
        raise AssertionError("STOP gate allowed a subprocess")

    with pytest.raises(runner.StopProduction, match="observed 9/10"):
        runner.run_bundle(
            bundle_path=bundle,
            output_path=output,
            phase="production",
            model=MODEL,
            max_attempts=1,
            invoker=forbidden_production,
        )
    assert output.read_bytes() == before
    assert invocation_count == 10


def test_command_rejects_resumed_sessions_and_concurrency_above_four(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "requests.jsonl"
    output = tmp_path / "responses.jsonl"
    request = _write_bundle(bundle, count=10)[0]
    template = runner.parse_command_template(
        "claude --model {model} --system-prompt {system_prompt} "
        "--resume old-session -p {user_prompt}"
    )
    with pytest.raises(runner.HarnessError, match="session-resuming"):
        runner.build_command(template, request)
    with pytest.raises(runner.HarnessError, match="between 1 and 4"):
        runner.run_bundle(
            bundle_path=bundle,
            output_path=output,
            phase="smoke",
            model=MODEL,
            max_concurrency=5,
            dry_run=True,
        )


def _write_freeze_receipt(tmp_path: Path, bundle: Path) -> Path:
    artifacts = []
    for role in (
        "v_ast_implementation",
        "train_gate_readout",
        "llm_contract_source",
    ):
        path = tmp_path / f"{role}.txt"
        path.write_text(f"frozen {role}\n", encoding="utf-8")
        artifacts.append(
            {"role": role, "path": path.name, "sha256": runner._file_sha256(path)}
        )
    receipt = tmp_path / "freeze_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": runner.FREEZE_RECEIPT_SCHEMA,
                "status": "frozen_before_sealed_heldout",
                "model": MODEL,
                "heldout_bundle_sha256": runner._file_sha256(bundle),
                "frozen_artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    return receipt


def test_train_runner_rejects_heldout_and_finalizer_is_one_shot(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "heldout.jsonl"
    output = tmp_path / "heldout_responses.jsonl"
    _write_bundle(bundle, count=10, split="sealed_heldout")
    receipt = _write_freeze_receipt(tmp_path, bundle)

    with pytest.raises(runner.HarnessError, match="compiler_train"):
        runner.run_bundle(
            bundle_path=bundle,
            output_path=output,
            phase="production",
            model=MODEL,
            dry_run=True,
        )

    finalized = runner.run_bundle(
        bundle_path=bundle,
        output_path=output,
        phase="heldout_finalize",
        model=MODEL,
        freeze_receipt_path=receipt,
        max_concurrency=2,
        invoker=lambda argv, timeout: runner.InvocationResult(0, VALID, ""),
    )
    assert finalized["final_valid_count"] == 10
    assert {row["phase"] for row in _rows(output)} == {"heldout_finalize"}
    assert {row["split"] for row in _rows(output)} == {"sealed_heldout"}
    assert {row["freeze_receipt_sha256"] for row in _rows(output)} == {
        runner._file_sha256(receipt)
    }

    with pytest.raises(runner.HarnessError, match="second execution is forbidden"):
        runner.run_bundle(
            bundle_path=bundle,
            output_path=output,
            phase="heldout_finalize",
            model=MODEL,
            freeze_receipt_path=receipt,
            invoker=lambda argv, timeout: (_ for _ in ()).throw(
                AssertionError("second heldout invocation")
            ),
        )


def test_heldout_finalizer_requires_unchanged_freeze_artifacts(tmp_path: Path) -> None:
    bundle = tmp_path / "heldout.jsonl"
    output = tmp_path / "heldout_responses.jsonl"
    _write_bundle(bundle, count=10, split="sealed_heldout")
    receipt = _write_freeze_receipt(tmp_path, bundle)
    (tmp_path / "train_gate_readout.txt").write_text("drift\n", encoding="utf-8")
    with pytest.raises(runner.HarnessError, match="changed or is missing"):
        runner.run_bundle(
            bundle_path=bundle,
            output_path=output,
            phase="heldout_finalize",
            model=MODEL,
            freeze_receipt_path=receipt,
            dry_run=True,
        )
