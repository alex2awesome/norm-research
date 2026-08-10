import json

import pytest

from methods.metric_seam.verifiers.llm_contract import (
    PARSER_VERSION,
    UnitContract,
    compile_request,
    parse_response,
    smoke_passes,
    validate_response_envelope,
)
from methods.metric_seam.verifiers.schema import SchemaError


DIFF = """diff --git a/a.py b/a.py
index 1111111..2222222 100644
--- a/a.py
+++ b/a.py
@@ -1 +1,2 @@
 x = 1
+print(x)
"""
CONTRACT = UnitContract("u1", "Observability", "Avoid bare debug output.", "llama8b", "0")


def test_request_is_deterministic_and_pass_specific():
    one = compile_request(
        contract=CONTRACT, item_key="i1", ctext=DIFF, pass_index=1,
        model="glm-5.2", split="compiler_train"
    )
    again = compile_request(
        contract=CONTRACT, item_key="i1", ctext=DIFF, pass_index=1,
        model="glm-5.2", split="compiler_train"
    )
    two = compile_request(
        contract=CONTRACT, item_key="i1", ctext=DIFF, pass_index=2,
        model="glm-5.2", split="compiler_train"
    )
    assert one == again
    assert one["request_sha256"] != two["request_sha256"]
    assert one["split"] == "compiler_train"
    assert '"confidence"' not in one["system_prompt"]


@pytest.mark.parametrize(
    ("raw", "mode"),
    [
        ('{"applies":true,"violated":true,"witnesses":[{"path":"a.py","start_line":2,"end_line":2}]}', "strict_json"),
        ('```json\n{"applies":true,"violated":true,"witnesses":[{"path":"a.py","start_line":2,"end_line":2}]}\n```', "fence_unwrapped"),
    ],
)
def test_parser_records_recovery_mode(raw, mode):
    parsed = parse_response(raw, ctext=DIFF)
    assert parsed.parser_version == PARSER_VERSION
    assert parsed.parse_mode == mode
    assert parsed.verdict.violated


def test_parser_rejects_floats_extra_keys_and_unbound_witnesses():
    with pytest.raises(SchemaError):
        parse_response('{"applies":true,"violated":false,"witnesses":[],"confidence":0.5}', ctext=DIFF)
    with pytest.raises(SchemaError):
        parse_response('{"applies":true,"violated":true,"witnesses":[{"path":"b.py","start_line":2,"end_line":2}]}', ctext=DIFF)
    with pytest.raises(SchemaError):
        parse_response('{"applies":true,"violated":false,"witnesses":[]}', ctext=DIFF)


def test_envelope_binding_and_smoke_stop():
    request = compile_request(
        contract=CONTRACT, item_key="i1", ctext=DIFF, pass_index=1,
        model="glm-5.2", split="compiler_train"
    )
    raw = json.dumps({
        "applies": True,
        "violated": True,
        "witnesses": [{"path": "a.py", "start_line": 2, "end_line": 2}],
    })
    row = validate_response_envelope(
        {"request_sha256": request["request_sha256"], "raw_response": raw}, request
    )
    assert row["parse_mode"] == "strict_json"
    assert smoke_passes([{"status": "valid"}] * 10)
    assert not smoke_passes([{"status": "valid"}] * 9)
    assert not smoke_passes([{"status": "valid"}] * 9 + [{"status": "contract_error"}])
