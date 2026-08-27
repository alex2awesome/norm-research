"""Content-only guard + bank-quality residual guard + reliability instrument."""

import numpy as np
import pytest

from metrics_tree_infilling.config import InfillConfig
from metrics_tree_infilling.generators import Proposal, _drop_surface, is_surface_only
from metrics_tree_infilling.global_infill import _global_contrast


# ---- surface detector -------------------------------------------------------------------

@pytest.mark.parametrize("name,rubric,surface", [
    ("Manual Markdown Formatting", "Score 1.0 if the text contains markdown characters like * or #.", True),
    ("Word Count", "YES if the text has more than 500 words.", True),
    ("Paragraph Number", "YES if the text has at least three paragraphs.", True),
    ("First-Person Pronoun Usage", "1.0 if the text uses first-person pronouns.", True),
    ("Bulleted Contributions", "Score 1.0 if the text uses bullet points to list contributions.", True),
    # content metrics that MENTION a surface token but are about content -> survive
    ("Subtextual Dialogue", "YES if dialogue conveys meaning beyond the literal words (subtext).", False),
    ("Empirical Rigor", "YES if claims are supported by concrete evidence and analysis.", False),
    ("Narrative Voice", "YES if the prose sustains a distinctive character voice and tone.", False),
    # a content metric about STRUCTURE of argument, not markdown structure
    ("Argument Structure", "YES if the argument builds premises to a conclusion coherently.", False),
])
def test_is_surface_only(name, rubric, surface):
    assert is_surface_only(name, rubric) is surface


def test_drop_surface_respects_flag():
    props = [Proposal("Word Count", "d", "YES if over 500 words.", "unconditional"),
             Proposal("Evidence Depth", "d", "YES if claims cite concrete evidence.", "unconditional")]
    off = InfillConfig(content_only_guard=False)
    on = InfillConfig(content_only_guard=True)
    assert len(_drop_surface(props, off)) == 2                 # guard off: keep all
    kept = _drop_surface(props, on)
    assert [p.name for p in kept] == ["Evidence Depth"]        # guard on: surface dropped


def test_guard_appends_instruction():
    from metrics_tree_infilling.generators import _guard, CONTENT_ONLY_INSTRUCTION
    assert _guard("base", InfillConfig(content_only_guard=True)).endswith(CONTENT_ONLY_INSTRUCTION)
    assert _guard("base", InfillConfig(content_only_guard=False)) == "base"


# ---- bank-quality residual guard --------------------------------------------------------

def test_residual_contrast_skipped_on_garbage_bank():
    rng = np.random.default_rng(0)
    n = 200
    y = (rng.uniform(size=n) < 0.5).astype(int)
    X_garbage = rng.uniform(size=(n, 3))                       # bank independent of y -> AUC ~0.5
    texts = [f"text {i}" for i in range(n)]
    cfg_off = InfillConfig(min_bank_auc_for_residual=0.0, text_column="text")
    cfg_on = InfillConfig(min_bank_auc_for_residual=0.60, text_column="text")
    # guard off: forms a contrast (or None only from degeneracy), guard on: None on garbage bank
    assert _global_contrast(X_garbage, y, texts, cfg_on, rng) is None
    # informative bank -> contrast forms even with the guard on
    signal = y + rng.normal(0, 0.3, size=n)
    X_good = np.column_stack([signal, rng.uniform(size=n)])
    c = _global_contrast(X_good, y, texts, cfg_on, rng)
    assert c is not None and c.n_wrong > 0


# ---- reliability instrument (offline path stubbed) --------------------------------------

def test_judge_test_retest_separates_reliable_from_noise(monkeypatch):
    import metrics_tree_infilling.reliability as R
    from metrics_tree_infilling.io_metrics import MetricSpec
    rng = np.random.default_rng(0)
    texts = [f"doc {i}" for i in range(60)]
    latent = rng.uniform(size=len(texts))

    def reliable_scorer(cfg, metric, sample, temperature, salt):
        # judge reads the SAME latent both passes + tiny noise -> high retest
        idx = [int(t.split()[1]) for t in sample]
        base = latent[idx]
        noise = rng.normal(0, 0.02, size=len(sample))
        lv = np.clip(base + noise, 0, 1)
        return lv, np.ones(len(sample), bool)

    def noise_scorer(cfg, metric, sample, temperature, salt):
        lv = rng.uniform(size=len(sample))                    # independent each pass -> ~0 retest
        return lv, np.ones(len(sample), bool)

    cfg = InfillConfig(materialize_backend="vllm_offline", min_reliability=0.5)
    m = MetricSpec(metric_id="x", name="m", description="d", kind="judge", guidance="rubric")

    monkeypatch.setattr(R, "_score_offline_once", reliable_scorer)
    good = R.judge_test_retest(m, texts, cfg, n_sample=40, seed=1)
    assert good["retest_spearman"] > 0.8 and not good["attenuation_flag"]

    monkeypatch.setattr(R, "_score_offline_once", noise_scorer)
    bad = R.judge_test_retest(m, texts, cfg, n_sample=40, seed=1)
    assert bad["retest_spearman"] < 0.5 and bad["attenuation_flag"]


# ---- residual-arm parser robustness (offline 70B array-format bug, 2026-07-05) ----------

def test_residual_parser_handles_offline_array_format():
    from metrics_tree_infilling.feature_gen import _parse_json_candidates
    # the offline 70B returns a top-level array; the old slicer mangled it to invalid JSON
    arr = '[{"name":"A","rubric":"r1"},{"name":"B","rubric":"r2"}]'
    assert [c["name"] for c in _parse_json_candidates(arr)] == ["A", "B"]
    assert len(_parse_json_candidates('```json\n[{"name":"A","rubric":"r"}]\n```')) == 1
    assert len(_parse_json_candidates('{"candidates":[{"name":"A","rubric":"r"}]}')) == 1
    assert len(_parse_json_candidates('{"name":"A","rubric":"r"}')) == 1     # legacy single obj
    assert len(_parse_json_candidates('prose\n[{"name":"A","rubric":"r"}]\nmore')) == 1
    assert _parse_json_candidates("no json here") == []
