import pytest
import subprocess

from scripts.tools.silver_match_v3.gpu_host_policy import (
    filter_gpu_rows_for_host,
    is_sk3_host,
    validate_gpu_indices_for_host,
    validate_launch_gpus,
)


def test_sk3_aliases_are_recognized() -> None:
    assert is_sk3_host("sk3")
    assert is_sk3_host("skampere3")
    assert is_sk3_host("skampere3.stanford.edu")
    assert not is_sk3_host("skampere2")


def test_sk3_selection_fails_closed_on_prohibited_gpu() -> None:
    with pytest.raises(ValueError, match="sk3 GPU policy violation"):
        validate_gpu_indices_for_host([0, 2], hostname="skampere3")


def test_sk3_selection_accepts_only_allowlisted_devices() -> None:
    assert validate_gpu_indices_for_host(
        [0, 5, 6, 7], hostname="skampere3"
    ) == (0, 5, 6, 7)


def test_sk2_selection_has_no_gpu_count_cap() -> None:
    assert validate_gpu_indices_for_host(
        list(range(8)), hostname="skampere2"
    ) == tuple(range(8))


def test_sk3_dynamic_candidates_are_filtered() -> None:
    rows = [{"index": value} for value in range(8)]
    assert [
        row["index"]
        for row in filter_gpu_rows_for_host(rows, hostname="skampere3")
    ] == [0, 5, 6, 7]


def test_sk2_launch_guard_has_no_count_or_owner_gate(monkeypatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *_args, **_kwargs: "".join(
            f"{index}, GPU-{index}, 4, 0\n" for index in range(8)
        ),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, "", ""),
    )
    result = validate_launch_gpus(range(8), hostname="skampere2")
    assert result["selected_gpu_indices"] == list(range(8))
    assert result["gpu_count_gate_applied"] is False
    assert result["projected_owner_count_check_applied"] is False


@pytest.mark.parametrize(
    ("gpu_row", "process_rows"),
    [
        ("0, GPU-0, 4, 0\n", "GPU-0, 123\n"),
        ("0, GPU-0, 4096, 0\n", ""),
        ("0, GPU-0, 4, 1\n", ""),
    ],
)
def test_launch_guard_retains_process_memory_and_utilization_safety(
    monkeypatch, gpu_row: str, process_rows: str
) -> None:
    monkeypatch.setattr(
        subprocess, "check_output", lambda *_args, **_kwargs: gpu_row
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, process_rows, ""
        ),
    )
    with pytest.raises(RuntimeError, match="not genuinely idle"):
        validate_launch_gpus([0], hostname="skampere2")
