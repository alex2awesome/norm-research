import json

from scripts.tools.silver_match_v3.common import sha256_file, write_jsonl
from scripts.tools.silver_match_v3.materialize_task_local_inference_manifest import materialize


def test_materializes_only_hidden_pack_inputs(tmp_path):
    pack = tmp_path / "pack"
    items = pack / "items.jsonl"
    bank = pack / "bank.json"
    write_jsonl(items, [{"norm_uid": "u", "corpus": "c", "task": "t"}])
    bank.write_text(json.dumps({"task": "t", "source_sha256": "source", "metrics": []}))
    (pack / "validation.json").write_text(
        json.dumps(
            {
                "task": "t",
                "truth_hidden": True,
                "bank_source_sha256": "source",
                "outputs": {
                    "items": {"sha256": sha256_file(items)},
                    "bank": {"sha256": sha256_file(bank)},
                },
            }
        )
    )
    output = tmp_path / "manifest.json"
    result = materialize(pack, output)
    payload = json.loads(output.read_text())
    assert result["count"] == 1
    assert payload["truth_or_label_fields_in_manifest"] is False
    assert payload["corpora"]["c"]["path"] == str(items.resolve())
