"""Adversarial tests for the tf-baseline / pass-planner / variant-runner PLUMBING
(passes.py + run_variant_pass.py) — the substrate every W1 instrument (exclusion,
negation, composition, holistic) rides on top of. Each test plants a synthetic
cell/bank/grid with a KNOWN profile that a naive reimplementation of this plumbing
would misread: missing arms, self-referential pairs, brace-bearing content, string-vs-
tuple type confusion, floor-boundary strictness, and NaN propagation through the
acceptance gate. None of these angles duplicate test_battery.py, test_stats.py, or
test_instruments.py — see those files for the row-algebra, wording-freeze, and
confidence-parsing coverage.
"""
import json

import numpy as np
import pytest

from methods.tacit_channels.battery.passes import (
    build_single_stage_rows, name_form_prompt, plan_summary,
)
from methods.tacit_channels.battery.run_variant_pass import (
    load_composed_pairs, run_acceptance, score_rows,
)
from methods.tacit_channels.channels.common import spearman

TEMPLATE = ("Criterion:\n{rubric}\n\nText:\n{text}\n\n"
            "Answer with exactly one word: YES or NO.")


def _cell(tag):
    return {"arms": [{"id": "name", "forms": [
        {"id": "canonical", "prompt": f"{tag} canonical"}]}]}


# 1 -------------------------------------------------------------------------
def test_missing_name_arm_yields_zero_rows_never_fabricated():
    """A cell whose arms list has NO id=="name" entry (only e.g. "source_definition")
    must contribute zero tf/exclusion/negated rows and must not crash the planner, and
    a composed pair naming it must be silently dropped (a-or-b-is-None guard) rather
    than fabricating a prompt from some other arm.
    kills: code that assumes arms[0] IS the name arm, or that falls back to any
    present arm when "name" is absent."""
    cells = {
        "c1": _cell("C1"),
        "c_no_name": {"arms": [{"id": "source_definition",
                                "forms": [{"id": "canonical", "prompt": "definition text"}]}]},
    }
    rows = build_single_stage_rows(cells, ("tf", "exclusion", "negated", "composed"),
                                   composed_pairs=(("c1", "c_no_name"),),
                                   forms=("canonical",))
    cell_ids = {r["cell_id"] for r in rows}
    assert "c_no_name" not in cell_ids            # zero rows fabricated for it
    assert "c1&&c_no_name" not in cell_ids        # composed pair silently dropped
    assert plan_summary(rows) == {"tf": 1, "exclusion": 1, "negated": 1}  # c1 only
    assert name_form_prompt(cells["c_no_name"], "canonical") is None


# 2 -------------------------------------------------------------------------
def test_self_pair_composed_current_behavior():
    """composed_pairs containing a SELF-pair ("c1","c1") is not rejected: the planner
    emits a "c1&&c1" row whose content is the construct AND-ed with itself and whose
    "pair" field lists the same cell_id twice. Documents current (permissive) behavior
    so a future dedup/validation pass doesn't silently change row identity.
    kills: an accidental dedup of (a,b) pairs by set() that would drop self-pairs, or
    a crash from double-keying cells[cell_a] == cells[cell_b]."""
    from methods.tacit_channels.battery.passes import COMPOSED_WRAPPER
    cells = {"c1": _cell("C1")}
    rows = build_single_stage_rows(cells, ("composed",), composed_pairs=(("c1", "c1"),),
                                   forms=("canonical",))
    assert len(rows) == 1
    row = rows[0]
    assert row["cell_id"] == "c1&&c1" and row["pair"] == ["c1", "c1"]
    base = "C1 canonical"
    assert row["content"] == COMPOSED_WRAPPER.format(content_a=base, content_b=base)


# 3 -------------------------------------------------------------------------
def test_literal_brace_content_survives_planner_and_scorer_roundtrip():
    """A construct prompt containing LITERAL "{rubric}"/"{text}" substrings (e.g. a
    construct that is itself about templating) must survive build_single_stage_rows'
    NEGATED_WRAPPER.format(content=base) call AND the downstream
    template.format(rubric=row["content"], text=...) in score_rows without KeyError
    and without the braces being re-interpreted as placeholders (str.format never
    rescans a substituted VALUE for further fields, but a switch to nested formatting
    or %-substitution would break this).
    kills: a refactor to recursive/nested template rendering, or to `%`-style
    formatting, either of which would crash or double-substitute on stray braces."""
    tricky = "Contains literal {rubric} and {text} sequences."
    cells = {"c1": {"arms": [{"id": "name", "forms": [{"id": "canonical", "prompt": tricky}]}]}}
    rows = build_single_stage_rows(cells, ("tf", "negated"), forms=("canonical",))
    tf_row = next(r for r in rows if r["variant"] == "tf")
    neg_row = next(r for r in rows if r["variant"] == "negated")
    assert tf_row["content"] == tricky                     # untouched
    assert tricky in neg_row["content"]                     # embedded literally, once
    assert neg_row["content"].count("{rubric}") == 1 and neg_row["content"].count("{text}") == 1

    captured = []
    def stub(_b, prompts, pos, neg, expected_token_ids, seed):
        captured.append(list(prompts))
        return [0.5] * len(prompts)
    score_rows(None, [tf_row], ["ITEM"], TEMPLATE, 100, {"YES": 1, "NO": 2}, "humor",
              score_fn=stub)
    expected = ("Criterion:\nContains literal {rubric} and {text} sequences."
                "\n\nText:\nITEM\n\nAnswer with exactly one word: YES or NO.")
    assert captured[0] == [expected]                        # no KeyError, no double-sub


# 4 -------------------------------------------------------------------------
def test_acceptance_floor_boundary_strict_less_than():
    """run_acceptance's gate is `if np.nanmin(rhos) < floor: raise` (strict). Setting
    floor to EXACTLY the achieved min rho must PASS; floor a hair above it must FAIL.
    kills: a `<` -> `<=` mutation (would wrongly reject an exact-equality run) and the
    reverse `<=` -> `<` mutation elsewhere that would wrongly accept a just-below run."""
    rng = np.random.default_rng(3)
    row_scores = rng.normal(size=50)
    ref_scores = rng.normal(size=50)
    rho = spearman(row_scores, ref_scores)
    assert not np.isnan(rho)
    meta = [{"cell_id": "c1", "variant": "tf", "arm_id": "name", "form": "canonical"}]
    import tempfile, os
    tmp = tempfile.mkdtemp()
    ref_path = os.path.join(tmp, "ref.npz")
    np.savez(ref_path, scores=ref_scores[None, :],
             meta=np.array([json.dumps({"cell_id": "c1", "arm_id": "name",
                                        "form": "canonical"})], dtype=object))
    run_acceptance(row_scores[None, :], meta, ref_path, floor=rho)            # == floor -> PASS
    with pytest.raises(SystemExit):
        run_acceptance(row_scores[None, :], meta, ref_path, floor=rho + 1e-9)  # epsilon below


# 5 -------------------------------------------------------------------------
def test_acceptance_nan_rhos_fail_closed():
    """FIXED 2026-07-25 (was a documented hazard): when a matched tf row is
    zero-variance, spearman returns nan and `nan < floor` is False — the gate used to
    print ACCEPTANCE PASSED on a completely degenerate scoring path. run_acceptance now
    rejects ANY NaN rho explicitly (a NaN row cannot demonstrate rho >= floor).
    kills: removing the n_nan guard, or reverting to the bare nanmin comparison."""
    const_row = np.full(50, 0.5)
    const_ref = np.full(50, 0.7)
    meta = [{"cell_id": "c1", "variant": "tf", "arm_id": "name", "form": "canonical"}]
    import tempfile, os
    tmp = tempfile.mkdtemp()
    ref_path = os.path.join(tmp, "ref.npz")
    np.savez(ref_path, scores=const_ref[None, :],
             meta=np.array([json.dumps({"cell_id": "c1", "arm_id": "name",
                                        "form": "canonical"})], dtype=object))
    with pytest.raises(SystemExit, match="NaN-rho"):
        run_acceptance(const_row[None, :], meta, ref_path, floor=0.999)


# 6 -------------------------------------------------------------------------
def test_composed_row_order_follows_pair_list_not_sorted():
    """Per-cell tf/exclusion/negated rows are emitted in `sorted(cells.items())` order
    (alphabetical by cell_id), but composed rows are emitted in the ORDER GIVEN by
    composed_pairs — a completely different, unsorted iteration. Verified with a pair
    list that is neither alpha-sorted nor first-element-sorted.
    kills: a change that runs `sorted(composed_pairs)` (or sorts by first member) for
    "consistency" with the per-cell loop, silently reordering pair rows relative to
    whatever the caller's w1_composed_pairs.json declared."""
    cells = {"c1": _cell("C1"), "c2": _cell("C2"), "c3": _cell("C3")}
    pairs = (("c3", "c1"), ("c2", "c1"))   # deliberately not sorted either way
    rows = build_single_stage_rows(cells, ("composed",), composed_pairs=pairs,
                                   forms=("canonical",))
    assert [r["cell_id"] for r in rows] == ["c3&&c1", "c2&&c1"]
    assert sorted(pairs) != list(pairs)  # sanity: the input really was unsorted


# 7 -------------------------------------------------------------------------
def test_duplicate_name_arm_falls_through_per_form_not_wholesale():
    """A malformed cell with TWO arms both id=="name" is not rejected. name_form_prompt
    has no `break`/return-on-miss: for a given form_id it returns from the FIRST "name"
    arm that HAS that form_id, falling through to a LATER "name" arm if an earlier one
    lacks it — it does NOT take the first "name" arm wholesale, nor merge them. This is
    the exact profile a naive "first match wins" mental model gets wrong.
    kills: a rewrite that breaks after the first arm.id=="name" regardless of whether
    the requested form was found in it (would wrongly return None for "question" here
    instead of falling through to the second "name" arm)."""
    cell = {"arms": [
        {"id": "name", "forms": [{"id": "canonical", "prompt": "FIRST-C"}]},
        {"id": "name", "forms": [{"id": "canonical", "prompt": "SECOND-C"},
                                 {"id": "question", "prompt": "SECOND-Q"}]},
    ]}
    assert name_form_prompt(cell, "canonical") == "FIRST-C"   # first arm has it -> wins
    assert name_form_prompt(cell, "question") == "SECOND-Q"   # falls through to 2nd arm


# 8 -------------------------------------------------------------------------
def test_variants_as_bare_string_substring_footgun():
    """DOCUMENTED HAZARD (passes.py:94,97,100,103): `variants` is type-hinted `tuple`
    but the body only ever does `"tf" in variants` / `"exclusion" in variants` / etc.
    Python's `in` on a bare STRING is substring containment, not membership. Passing a
    plain string that happens to CONTAIN "tf" as a substring (e.g. "outfile", verified:
    "tf" in "outfile" == True since outfile[2:4] == "tf") silently activates the tf
    variant even though no valid variant token was ever requested and no exception is
    raised. No isinstance/type guard exists anywhere in build_single_stage_rows.
    kills: nothing currently (freezes the footgun); would catch a future "fix" that
    validates variants strictly and should therefore start raising here."""
    cells = {"c1": _cell("C1")}
    rows = build_single_stage_rows(cells, "outfile", forms=("canonical",))
    assert plan_summary(rows) == {"tf": 1}                     # spuriously activated
    assert rows[0]["content"] == "C1 canonical"
    assert not build_single_stage_rows(cells, "zzz-not-a-variant", forms=("canonical",))


# 9 -------------------------------------------------------------------------
def test_score_rows_empty_items_degenerates_cleanly():
    """texts=[] (a domain packet with zero surviving items after filtering) must not
    crash score_rows: every prompt list is empty, np.vstack of N zero-length vectors
    gives a (N, 0) matrix, and meta is still populated one entry per row (content_sha256
    included) so the caller can distinguish "ran with no items" from "row was skipped".
    kills: code that indexes texts[0] to size something, or that treats an empty score
    vector as a sentinel for "row failed" and drops its meta entry."""
    cells = {"c1": _cell("C1"), "c2": _cell("C2")}
    rows = build_single_stage_rows(cells, ("tf",), forms=("canonical",))
    def stub(_b, prompts, pos, neg, expected_token_ids, seed):
        return []
    matrix, meta = score_rows(None, rows, [], TEMPLATE, 100, {"YES": 1, "NO": 2}, "humor",
                              score_fn=stub)
    assert matrix.shape == (2, 0)
    assert len(meta) == 2
    assert all("content_sha256" in m for m in meta)


# 10 ------------------------------------------------------------------------
def test_load_composed_pairs_explicit_json_null_raises_typeerror(tmp_path):
    """DOCUMENTED HAZARD (run_variant_pass.py:49): `d.get("pairs_a_x_a", [])` only
    substitutes the [] default when the KEY IS ABSENT — a well-formed JSON file that
    explicitly sets one list to `null` (as opposed to omitting the key, or `[]`) makes
    .get() return None, and `None + d.get("pairs_non_a", [])` raises an uncaught
    TypeError instead of a clean, actionable error. Distinguishing "key missing" from
    "key present but null" is exactly what naive JSON-config handling gets wrong.
    Suggested fix (not applied): `d.get("pairs_a_x_a") or []`.
    kills: nothing currently (freezes the crash); documents the gap between "missing
    key" (handled) and "present-but-null key" (not handled) for load_composed_pairs."""
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps({"pairs_a_x_a": None, "pairs_non_a": [["c1", "c2"]]}))
    with pytest.raises(TypeError):
        load_composed_pairs(str(path))
