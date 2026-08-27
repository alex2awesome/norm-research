"""Instrument-ASSEMBLY tests — the prompt-construction layer (passes.py + runner helpers).

The GLM calibration cycles validated instrument WORDINGS against mechanical oracles; these
tests freeze the assembly MECHANICS: what string is actually shown to the model for each
variant, how the two-stage prompts are spliced, how answers/confidences are parsed, and
that the acceptance gate actually rejects a drifted scoring path. Wording constants change
only via dated prereg addendum — the freeze test below fails on ANY drift by design.
"""
import json

import numpy as np
import pytest

from methods.tacit_channels.battery.passes import (
    COMPOSED_WRAPPER, CONFIDENCE_QUESTION, EXCLUSION_FIXED_QUESTION, EXCLUSION_PREFIX,
    HOLISTIC_GRADED_TEMPLATE, HOLISTIC_PROMPT, NEG_FX_QUESTION, NEG_FX_WRAPPER,
    NEGATED_WRAPPER, REASON_FIRST_INSTR, build_single_stage_rows, confidence_prompt,
    plan_summary,
)
from methods.tacit_channels.battery.run_reason_first_pass import (
    ANSWER_INSTR, parse_confidence, reason_first_generation_prompt,
    reason_first_tf_prompt, tf_answers_from_grid,
)
from methods.tacit_channels.battery.run_variant_pass import run_acceptance

TEMPLATE = ("Criterion:\n{rubric}\n\nText:\n{text}\n\n"
            "Answer with exactly one word: YES or NO.")


def _cells():
    def cell(tag):
        return {"arms": [{"id": "name", "forms": [
            {"id": "canonical", "prompt": f"{tag} canonical"},
            {"id": "question", "prompt": f"{tag} question"}]}]}
    return {"c1": cell("C1"), "c2": cell("C2")}


def test_single_stage_row_algebra():
    """Row counts follow the planner law (cells x forms per variant, pairs x forms for
    composed) and each variant's content is EXACTLY the declared transform of the base."""
    rows = build_single_stage_rows(_cells(), ("tf", "exclusion", "negated", "composed"),
                                   composed_pairs=(("c1", "c2"),),
                                   forms=("canonical", "question"))
    assert plan_summary(rows) == {"tf": 4, "exclusion": 4, "negated": 4, "composed": 2}
    by = {(r["cell_id"], r["variant"], r["form"]): r for r in rows}
    base = "C1 canonical"
    assert by[("c1", "tf", "canonical")]["content"] == base
    assert by[("c1", "exclusion", "canonical")]["content"] == EXCLUSION_PREFIX + base
    assert (by[("c1", "negated", "canonical")]["content"]
            == NEGATED_WRAPPER.format(content=base))
    comp = by[("c1&&c2", "composed", "question")]
    assert comp["content"] == COMPOSED_WRAPPER.format(content_a="C1 question",
                                                      content_b="C2 question")
    assert comp["pair"] == ["c1", "c2"]


def test_missing_forms_skipped_never_fabricated():
    cells = _cells()
    cells["c2"]["arms"][0]["forms"] = [{"id": "canonical", "prompt": "C2 canonical"}]
    rows = build_single_stage_rows(cells, ("tf", "composed"),
                                   composed_pairs=(("c1", "c2"),),
                                   forms=("canonical", "question"))
    got = {(r["cell_id"], r["form"]) for r in rows}
    assert ("c2", "question") not in got          # no fabricated form
    assert ("c1&&c2", "question") not in got      # pair needs BOTH members' form
    assert ("c1&&c2", "canonical") in got


def test_wordings_frozen_and_answer_mapped():
    """Freeze-guard: any edit to an instrument wording fails this test — wordings change
    only via dated prereg addendum + recalibration (then this sha is updated IN the same
    commit as the addendum). Also asserts the certified forms carry an EXPLICIT answer
    mapping (the property that fixed the cycle-1 defects)."""
    import hashlib
    blob = "\x1e".join([EXCLUSION_PREFIX, NEGATED_WRAPPER, COMPOSED_WRAPPER,
                        REASON_FIRST_INSTR, CONFIDENCE_QUESTION, HOLISTIC_PROMPT,
                        EXCLUSION_FIXED_QUESTION, NEG_FX_WRAPPER, NEG_FX_QUESTION,
                        HOLISTIC_GRADED_TEMPLATE])
    assert (hashlib.sha256(blob.encode()).hexdigest()
            == "459ce46f4603fa408761d9b5129e3ac0e622ebb1319cbf960ffb9dc466d5468f")
    # certified exclusion: maps YES->FAILS, and never re-asks the straight question
    assert "FAILS" in EXCLUSION_FIXED_QUESTION and "YES" in EXCLUSION_FIXED_QUESTION
    assert "Does the text satisfy" not in EXCLUSION_FIXED_QUESTION
    # certified negation: both directions of the mapping are explicit
    assert "ABSENCE" in NEG_FX_WRAPPER
    assert "YES if the property is absent" in NEG_FX_QUESTION
    assert "NO if" in NEG_FX_QUESTION
    # templates format cleanly with their declared slots
    assert "P!" in NEG_FX_WRAPPER.format(content="P!")
    filled = HOLISTIC_GRADED_TEMPLATE.format(rubric="R", text="T")
    assert "R" in filled and "T" in filled and "0-10" in filled


def test_reason_first_two_stage_assembly():
    row = {"content": "the criterion"}
    gen = reason_first_generation_prompt(TEMPLATE, row, "some text", 4000)
    assert REASON_FIRST_INSTR in gen and ANSWER_INSTR not in gen
    tf = reason_first_tf_prompt(TEMPLATE, row, "some text",
                                "  because   reasons\nacross lines  ", 4000)
    assert tf.count(ANSWER_INSTR) == 1            # question restored exactly once
    assert "Reasoning: because reasons across lines" in tf   # whitespace-normalized
    assert tf.index("Reasoning:") < tf.index(ANSWER_INSTR)   # rationale BEFORE answer
    capped = reason_first_tf_prompt(TEMPLATE, row, "t", "x" * 2000, 4000,
                                    max_rationale_chars=100)
    assert "x" * 100 in capped and "x" * 101 not in capped


def test_frozen_readout_template_contains_the_splice_sentinel():
    """The two-stage splice is a str.replace on ANSWER_INSTR — silently a no-op if the
    deployed template ever drops that exact sentence. Bind the invariant to the frozen
    template on disk."""
    from pathlib import Path
    tpl = Path("outputs/tacit_channels/exp_gtk1/readout_template.txt")
    if not tpl.exists():
        pytest.skip("frozen readout template not present on this machine")
    assert tpl.read_text().count(ANSWER_INSTR) == 1


def test_confidence_prompt_and_parser():
    p = confidence_prompt("JUDGMENT PROMPT", "YES")
    assert p.startswith("JUDGMENT PROMPT")
    assert "Your answer was: YES" in p and CONFIDENCE_QUESTION in p
    assert "Your answer was: NO" in confidence_prompt("j", "NO")
    # parser: first integer IN RANGE wins; out-of-range and junk never crash
    assert parse_confidence("I am 110% sure, call it 95") == 95.0
    assert parse_confidence("confidence: 0") == 0.0
    assert parse_confidence("100") == 100.0
    assert np.isnan(parse_confidence("one hundred"))
    assert np.isnan(parse_confidence(""))
    assert np.isnan(parse_confidence(None))


def test_acceptance_gate_passes_and_fails(tmp_path):
    """The gate must PASS on an identical path and FAIL CLOSED on a decorrelated row or
    an empty match set (the stale-log watcher incident makes this worth encoding)."""
    rng = np.random.default_rng(0)
    ref_scores = rng.uniform(size=(3, 60))
    ref_meta = np.array([json.dumps({"cell_id": f"c{i}", "arm_id": "name",
                                     "form": "canonical"}) for i in range(3)],
                        dtype=object)
    ref = tmp_path / "ref.npz"
    np.savez(ref, scores=ref_scores, meta=ref_meta)
    meta = [{"cell_id": f"c{i}", "variant": "tf", "arm_id": "name", "form": "canonical"}
            for i in range(3)]
    run_acceptance(ref_scores.copy(), meta, str(ref), 0.999)   # identical -> passes
    drifted = ref_scores.copy()
    drifted[1] = rng.uniform(size=60)
    with pytest.raises(SystemExit):
        run_acceptance(drifted, meta, str(ref), 0.999)
    with pytest.raises(SystemExit):                            # zero matched tf rows
        run_acceptance(ref_scores, [{"cell_id": "cX", "variant": "exclusion",
                                     "arm_id": "name_exclusion", "form": "canonical"}],
                       str(ref), 0.999)


def test_tf_answers_from_grid_filters_and_thresholds(tmp_path):
    scores = np.array([[0.2, 0.5, 0.8],     # tf, right domain
                       [0.9, 0.9, 0.9],     # exclusion row -> ignored
                       [0.1, 0.1, 0.1]])    # tf, WRONG domain -> ignored
    meta = np.array([
        json.dumps({"cell_id": "c1", "form": "canonical", "variant": "tf",
                    "domain": "humor"}),
        json.dumps({"cell_id": "c1", "form": "canonical", "variant": "exclusion",
                    "domain": "humor"}),
        json.dumps({"cell_id": "c2", "form": "canonical", "variant": "tf",
                    "domain": "math"})], dtype=object)
    path = tmp_path / "grid.npz"
    np.savez(path, scores=scores, meta=meta)
    ans = tf_answers_from_grid(str(path), "humor")
    assert set(ans) == {("c1", "canonical")}
    assert ans[("c1", "canonical")].tolist() == [False, True, True]   # >= .5 inclusive
