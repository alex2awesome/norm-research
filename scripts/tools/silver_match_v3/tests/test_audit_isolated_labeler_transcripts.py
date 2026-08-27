import json
from pathlib import Path

from scripts.tools.silver_match_v3.audit_isolated_labeler_transcripts import audit


def _fixture(tmp_path: Path, command: str) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    pack = repo / "outputs" / "panel" / "pass_a"
    guide = repo / "scripts" / "guide.md"
    guide.parent.mkdir(parents=True)
    guide.write_text("guide\n")
    (pack / "chunks").mkdir(parents=True)
    (pack / "logs").mkdir()
    (pack / "raw_labels").mkdir()
    (pack / "bank.json").write_text("{}\n")
    (pack / "items.jsonl").write_text('{"norm_uid":"u"}\n')
    (pack / "validation.json").write_text("{}\n")
    (pack / "chunks" / "part-000.jsonl").write_text('{"norm_uid":"u"}\n')
    (pack / "raw_labels" / "part-000.json").write_text("{}\n")
    (pack / "logs" / "part-000.log").write_text(
        "approval: never\nsandbox: read-only\n"
        f"prompt {guide.relative_to(repo)} {pack.relative_to(repo)}/bank.json "
        f"{pack.relative_to(repo)}/chunks/part-000.jsonl\n"
        f"exec\n{command} in {repo}\n succeeded\n"
    )
    return repo, pack, guide


def test_accepts_only_frozen_inputs(tmp_path: Path) -> None:
    repo, pack, guide = _fixture(
        tmp_path,
        "/bin/zsh -lc \"sed -n '1,40p' scripts/guide.md && cat "
        'outputs/panel/pass_a/bank.json outputs/panel/pass_a/chunks/part-000.jsonl"',
    )
    result = audit(pack, [guide], repo)
    assert result["status"] == "PASS"
    assert result["complete"] is True
    assert result["full_pack_artifact_binding"] is True
    assert result["items"]["path"].endswith("items.jsonl")
    assert result["pack_validation"]["path"].endswith("validation.json")


def test_rejects_candidate_or_discovery_access(tmp_path: Path) -> None:
    repo, pack, guide = _fixture(
        tmp_path,
        '/bin/zsh -lc "rg metric outputs/panel/pass_a/bank.json; '
        'cat outputs/secret/candidates.jsonl"',
    )
    result = audit(pack, [guide], repo)
    assert result["status"] == "FAIL"
    details = json.dumps(result["violations"])
    assert "outputs/secret/candidates.jsonl" in details


def test_accepts_targeted_rg_and_cwd_relative_pack_paths(tmp_path: Path) -> None:
    repo, pack, guide = _fixture(
        tmp_path,
        "/bin/zsh -lc \"sed -n '1,40p' scripts/guide.md; "
        "rg -n 'clarity|access|tone' pack/bank.json; sed -n '1,40p' "
        'pack/chunks/part-000.jsonl"',
    )
    # Match the production labeler cwd, where the pack is a direct child.
    relocated = pack.parent.parent.parent / "isolated" / "pack"
    relocated.parent.mkdir(parents=True)
    pack.rename(relocated)
    result = audit(relocated, [guide], repo)
    assert result["status"] == "PASS"


def test_rejects_untargeted_rg(tmp_path: Path) -> None:
    repo, pack, guide = _fixture(
        tmp_path,
        '/bin/zsh -lc "cat scripts/guide.md '
        "outputs/panel/pass_a/bank.json outputs/panel/pass_a/chunks/part-000.jsonl; "
        'rg -n secret"',
    )
    result = audit(pack, [guide], repo)
    assert result["status"] == "FAIL"
    assert "repository discovery command" in json.dumps(result["violations"])


def test_accepts_rg_filtering_allowlisted_pipeline_stdin(tmp_path: Path) -> None:
    repo, pack, guide = _fixture(
        tmp_path,
        "/bin/zsh -lc \"sed -n '1,40p' scripts/guide.md "
        "outputs/panel/pass_a/chunks/part-000.jsonl; "
        "jq -r '.metrics[]' outputs/panel/pass_a/bank.json | rg -i 'tone|clarity'\"",
    )
    result = audit(pack, [guide], repo)
    assert result["status"] == "PASS"


def test_rejects_rg_pipeline_with_extra_file_operand(tmp_path: Path) -> None:
    repo, pack, guide = _fixture(
        tmp_path,
        "/bin/zsh -lc \"sed -n '1,40p' scripts/guide.md "
        "outputs/panel/pass_a/chunks/part-000.jsonl; "
        "jq -r '.metrics[]' outputs/panel/pass_a/bank.json | "
        "rg -i 'tone|clarity' other.json\"",
    )
    result = audit(pack, [guide], repo)
    assert result["status"] == "FAIL"
    assert "repository discovery command" in json.dumps(result["violations"])
