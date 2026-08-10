import hashlib
import json

from scripts.tools.silver_match_v3.seal_verifier_pair import CONFIG_KEYS, validate_pair


def test_seals_matching_order_pair(tmp_path):
    source = tmp_path / "verify.py"
    source.write_text("pass\n")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    outputs = []
    common = {key: "value" for key in CONFIG_KEYS}
    common["prompt_sha256"] = "prompt"
    for order in ("hashed", "reverse"):
        path = tmp_path / f"{order}.jsonl"
        path.write_text(json.dumps({"norm_uid": "u"}) + "\n")
        meta = {
            **common,
            "order_mode": order,
            "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        path.with_suffix(".jsonl.meta.json").write_text(json.dumps(meta))
        outputs.append(path)
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"chosen": {"prompt_sha256": "prompt"}}))
    result = validate_pair(
        source, outputs[0], outputs[1], selection,
        expected_source_sha256=source_sha,
    )
    assert result["count"] == 1
    assert result["orders"] == ["hashed", "reverse"]
