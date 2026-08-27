from __future__ import annotations

import gzip
import io
import json
import urllib.error
from pathlib import Path

from methods.metric_seam.run_hierarchy_prompt_jobs import (
    execute_one,
    preflight_jobs,
    run_jobs,
)


CHANNEL = "implementation_disclosed"


def _job(cell: int, item: int, pass_id: int, channel: str = CHANNEL) -> dict:
    request_id = f"cell-{cell:02d}::{channel}::p{pass_id}::item-{item:03d}"
    return {
        "request_id": request_id,
        "request": {"system": "system prompt", "user": f"user {cell}/{item}"},
        "executor_metadata": {"code_score": "must-not-leak"},
        "audit_metadata": {
            "channel": channel,
            "cell_id": f"cell-{cell:02d}",
            "item_key": f"item-{item:03d}",
            "pass_id": pass_id,
            "source_path": "/secret/source.py",
            "code_vector_id": "secret-vector",
            "outcome": "secret-outcome",
        },
    }


def _write_bundle(path: Path, extra_channel_rows: int = 0) -> list[dict]:
    rows = [
        _job(cell, item, pass_id)
        for cell in range(18)
        for item in range(125)
        for pass_id in (1, 2)
    ]
    rows.extend(
        _job(0, item, 1, "source_only_whole_construct")
        for item in range(extra_channel_rows)
    )
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return rows


def _envelope(text: str, model: str = "glm-5.2") -> bytes:
    return json.dumps(
        {
            "model": model,
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 10, "output_tokens": 8},
        }
    ).encode()


def _valid_text(score: float = 0.75) -> str:
    return json.dumps(
        {
            "measurement_status": "scored",
            "score": score,
            "evidence": ["snippet"],
            "rationale": "supported",
        }
    )


def test_exact_4500_job_filtering(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs.jsonl.gz"
    _write_bundle(jobs, extra_channel_rows=17)

    summary = preflight_jobs(jobs, channel=CHANNEL, expected_jobs=4500)

    assert summary.selected_jobs == 4500
    assert len(summary.cell_ids) == 18
    assert len(summary.request_ids) == 4500


def test_metadata_exclusion_and_valid_invalid_parsing() -> None:
    row = _job(0, 0, 1)
    bodies: list[bytes] = []

    def valid_post(body: bytes, key: str, timeout: float) -> bytes:
        bodies.append(body)
        return _envelope(_valid_text())

    valid = execute_one(
        row,
        key="credential",
        provider="zai_anthropic",
        model="glm-5.2",
        temperature=0.2,
        max_tokens=1024,
        post=valid_post,
        sleep=lambda _: None,
    )
    sent = json.loads(bodies[0])
    serialized = bodies[0].decode()
    assert set(sent) == {"model", "max_tokens", "temperature", "system", "messages"}
    assert sent["system"] == row["request"]["system"]
    assert sent["messages"] == [{"role": "user", "content": row["request"]["user"]}]
    assert "audit_metadata" not in serialized
    assert "secret-vector" not in serialized
    assert "/secret/source.py" not in serialized
    assert "secret-outcome" not in serialized
    assert valid["status"] == "valid"
    assert valid["parsed_response"]["score"] == 0.75

    calls = 0

    def fenced_post(body: bytes, key: str, timeout: float) -> bytes:
        nonlocal calls
        calls += 1
        return _envelope("```json\n" + _valid_text() + "\n```")

    # A Markdown fence is transport framing, not model content.  A fenced valid
    # object carries the same meaning as an unfenced one and must validate.  The
    # raw text is still retained verbatim, fence included.
    fenced = execute_one(
        row,
        key="credential",
        provider="zai_anthropic",
        model="glm-5.2",
        temperature=0.2,
        max_tokens=1024,
        post=fenced_post,
        sleep=lambda _: None,
    )
    assert fenced["status"] == "valid"
    assert fenced["parsed_response"]["score"] == 0.75
    assert fenced["raw_response"].startswith("```json")
    assert calls == 1

    # Evidence spans quote tab-indented source.  A strict parse would reject these
    # non-randomly, selecting against tab-indented languages.
    tabbed = execute_one(
        row,
        key="credential",
        provider="zai_anthropic",
        model="glm-5.2",
        temperature=0.2,
        max_tokens=1024,
        post=lambda body, key, timeout: _envelope(
            '{"measurement_status":"scored","score":0.5,'
            '"evidence":["+\tif err != nil {"],"rationale":"tab-indented Go"}'
        ),
        sleep=lambda _: None,
    )
    assert tabbed["status"] == "valid"
    assert tabbed["parsed_response"]["evidence"] == ["+\tif err != nil {"]

    # Deserialization unwraps framing; it never repairs a payload.  A schema
    # violation inside a well-formed fence is still a contract error.
    violating = execute_one(
        row,
        key="credential",
        provider="zai_anthropic",
        model="glm-5.2",
        temperature=0.2,
        max_tokens=1024,
        post=lambda body, key, timeout: _envelope(
            '```json\n{"measurement_status":"not_applicable","score":0.5,'
            '"evidence":[],"rationale":"score forbidden unless scored"}\n```'
        ),
        sleep=lambda _: None,
    )
    assert violating["status"] == "contract_error"
    assert violating["parsed_response"] == {}


def test_transport_retry_but_no_contract_retry() -> None:
    row = _job(0, 0, 1)
    bodies: list[bytes] = []

    def flaky_post(body: bytes, key: str, timeout: float) -> bytes:
        bodies.append(body)
        if len(bodies) == 1:
            raise urllib.error.HTTPError(
                "https://provider.invalid", 429, "rate limited", {}, io.BytesIO()
            )
        if len(bodies) == 2:
            raise TimeoutError("timed out")
        return _envelope(_valid_text(0.25))

    sleeps: list[float] = []
    result = execute_one(
        row,
        key="credential",
        provider="zai_anthropic",
        model="glm-5.2",
        temperature=0.2,
        max_tokens=1024,
        post=flaky_post,
        sleep=sleeps.append,
    )
    assert result["status"] == "valid"
    assert result["attempts"] == 3
    assert bodies[0] == bodies[1] == bodies[2]
    assert sleeps == [1.0, 2.0]

    calls = 0

    def contract_post(body: bytes, key: str, timeout: float) -> bytes:
        nonlocal calls
        calls += 1
        return _envelope('{"measurement_status":"scored"}')

    contract = execute_one(
        row,
        key="credential",
        provider="zai_anthropic",
        model="glm-5.2",
        temperature=0.2,
        max_tokens=1024,
        post=contract_post,
        sleep=lambda _: None,
    )
    assert contract["status"] == "contract_error"
    assert contract["attempts"] == 1
    assert calls == 1


def test_resume_executes_only_missing_request_ids(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs.jsonl.gz"
    rows = _write_bundle(jobs)
    output = tmp_path / "responses.jsonl"
    existing = {
        "request_id": rows[0]["request_id"],
        "status": "transport_error",
    }
    output.write_text(json.dumps(existing) + "\n", encoding="utf-8")
    seen_users: list[str] = []

    def post(body: bytes, key: str, timeout: float) -> bytes:
        sent = json.loads(body)
        seen_users.append(sent["messages"][0]["content"])
        return _envelope(_valid_text())

    summary = run_jobs(
        jobs_path=jobs,
        channel=CHANNEL,
        backend="zai_anthropic",
        model="glm-5.2",
        temperature=0.2,
        max_tokens=1024,
        concurrency=1,
        expected_jobs=4500,
        output_path=output,
        limit=1,
        key="credential",
        post=post,
        sleep=lambda _: None,
    )

    emitted = [json.loads(line) for line in output.read_text().splitlines()]
    assert summary["previously_completed"] == 1
    assert summary["written"] == 1
    assert len(emitted) == 2
    assert emitted[1]["request_id"] == rows[1]["request_id"]
    assert seen_users == [rows[1]["request"]["user"]]
