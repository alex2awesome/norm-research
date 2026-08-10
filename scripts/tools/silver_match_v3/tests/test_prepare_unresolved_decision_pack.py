import json

from scripts.tools.silver_match_v3.common import read_jsonl
from scripts.tools.silver_match_v3.prepare_unresolved_decision_pack import prepare


def test_prepare_hides_system_reason_and_renders_full_bank(tmp_path):
    bank = tmp_path / "bank.json"
    bank.write_text(
        json.dumps(
            {
                "metrics": [
                    {"metric_id": "m1", "name": "one", "description": "first"},
                    {"metric_id": "m2", "name": "two", "description": "second"},
                ]
            }
        )
    )
    norms = tmp_path / "norms.jsonl"
    norms.write_text(
        json.dumps(
            {
                "norm_uid": "u",
                "corpus": "c",
                "task": "t",
                "row": 3,
                "norm": "be clearer",
                "context": "The author should be clearer.",
                "kind": "suggestion",
                "polarity": "negative",
            }
        )
        + "\n"
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "banks": {"t": {"path": str(bank), "source_sha256": "banksha"}},
                "corpora": {"c": {"path": str(norms), "task": "t", "count": 1}},
            }
        )
    )
    unresolved = tmp_path / "unresolved.jsonl"
    unresolved.write_text(
        json.dumps(
            {
                "norm_uid": "u",
                "corpus": "c",
                "task": "t",
                "row": 3,
                "source": "typed_abstention",
                "unresolved_reason": "possible_exact_bank_match",
            }
        )
        + "\n"
    )
    output = tmp_path / "pack"
    report = prepare(
        manifest_path=manifest,
        unresolved_path=unresolved,
        output_root=output,
        chunk_size=20,
        seed=7,
    )
    blind = list(read_jsonl(output / "t" / "items.blind.jsonl"))[0]
    key = list(read_jsonl(output / "t" / "items.key.jsonl"))[0]
    rendered_bank = json.loads((output / "t" / "bank.blind.json").read_text())
    label_pack = output / "t" / "label_pack"
    label_item = list(read_jsonl(label_pack / "items.jsonl"))[0]
    label_validation = json.loads((label_pack / "validation.json").read_text())
    assert "unresolved_reason" not in blind and "source" not in blind
    assert key["unresolved_reason"] == "possible_exact_bank_match"
    assert {row["metric_id"] for row in rendered_bank["metrics"]} == {"m1", "m2"}
    assert report["system_reasons_hidden_from_items"] is True
    assert label_item["norm"] == "be clearer"
    assert label_item["context"] == "The author should be clearer."
    assert label_item["permanently_excluded_from_gradients"] is True
    assert not (label_pack / "items.key.jsonl").exists()
    assert label_validation["system_key_excluded_from_label_pack"] is True
    assert report["outputs"]["t"]["label_pack_validation"]["sha256"]
