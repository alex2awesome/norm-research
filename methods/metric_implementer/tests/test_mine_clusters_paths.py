from pathlib import Path

from methods.metric_implementer.experiments import mine_clusters


def test_mining_inputs_are_repo_anchored_outside_caller_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    repo_root = Path(mine_clusters.__file__).resolve().parents[3]
    assert Path(mine_clusters._HIER_DIR) == repo_root / "outputs" / "hierarchy"
    assert Path(mine_clusters._STRUCT_DIR) == repo_root / "outputs" / "analyses" / "structural_metrics"
    assert Path(mine_clusters._CANON) == repo_root / "outputs" / "analyses" / "canon_all_real_forms.jsonl"
