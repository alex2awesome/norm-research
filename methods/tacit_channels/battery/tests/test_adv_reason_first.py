"""Adversarial tests for the W1b reason-first / two-stage probe (run_reason_first_pass.py +
the reason-first pieces of passes.py). test_battery.py and test_instruments.py already cover
the HAPPY-PATH assembly (whitespace-normalized splice + rationale cap, splice-sentinel-on-disk
check, tf_answers_from_grid domain/variant filter + 0.5 threshold, and the w1b helper stubs);
this file goes after injection strings, alignment/type attacks, and silent-drop paths that
those tests do not exercise. Deterministic; no network/GPU -- every backend call goes through
an injected score_fn/gen_fn recording stub with backend=None.

Findings are asserted as CURRENT behavior with a "DOCUMENTED HAZARD" comment where the code
under test is genuinely fragile (per task: this file does not patch run_reason_first_pass.py
or passes.py -- W1b already ran with this code, so these are provenance notes, not bug
reports awaiting a fix)."""
import json

import numpy as np
import pytest

from methods.tacit_channels.battery.passes import REASON_FIRST_INSTR, assemble_reason_first_tf
from methods.tacit_channels.battery.run_reason_first_pass import (
    ANSWER_INSTR, reason_first_generation_prompt, reason_first_tf_prompt, run_confidence,
    run_reason_first, tf_answers_from_grid,
)

TEMPLATE = ("Criterion:\n{rubric}\n\nText:\n{text}\n\n"
            "Answer with exactly one word: YES or NO.")


# ---- 1. candidate angle: item text carrying the sentinel verbatim -----------------------


def test_item_text_containing_sentinel_gets_double_replaced():
    """kills: any assumption that only the template's OWN trailing instruction line changes
    under reason_first_generation_prompt.
    DOCUMENTED HAZARD (run_reason_first_pass.py:46): `base.replace(ANSWER_INSTR,
    REASON_FIRST_INSTR)` matches ALL occurrences of the sentinel in the fully-formatted
    prompt, including any copy that happens to live INSIDE the item text (e.g. an essay
    quoting the readout instructions verbatim, or an adversarial item designed to do so).
    The item's quoted copy is silently rewritten into REASON_FIRST_INSTR too -- stage 1 ends
    up asking the model to "explain" TWICE, once for real and once inside the quoted item."""
    row = {"content": "Wit"}
    text = f'The essay says: "{ANSWER_INSTR}" and nothing else.'
    base_would_be = TEMPLATE.format(rubric=row["content"], text=text[:4000])
    assert base_would_be.count(ANSWER_INSTR) == 2      # template tail + the item's own quote
    gen = reason_first_generation_prompt(TEMPLATE, row, text, 4000)
    assert gen.count(ANSWER_INSTR) == 0
    assert gen.count(REASON_FIRST_INSTR) == 2          # BOTH occurrences silently rewritten


# ---- 2. candidate angle: rationale carrying the sentinel back --------------------------


def test_rationale_containing_sentinel_duplicates_the_answer_instruction():
    """kills: generalizing the clean-rationale invariant in
    test_instruments.py::test_reason_first_two_stage_assembly (`tf.count(ANSWER_INSTR) == 1`)
    to ALL rationales.
    DOCUMENTED HAZARD (run_reason_first_pass.py:53-54): if the model's own stage-1 rationale
    happens to quote the answer-instruction sentinel back (plausible -- models often restate
    instructions while reasoning), the splice's replacement text contains ANSWER_INSTR twice
    (once inside the injected "Reasoning: ..." block, once as the real trailing question), and
    since base only had ONE occurrence to replace, the final teacher-forced prompt ends up
    with two copies of the sentinel, not one."""
    row = {"content": "Wit"}
    rationale = f"I checked the rule ({ANSWER_INSTR}) and it is satisfied here."
    tf = reason_first_tf_prompt(TEMPLATE, row, "a joke", rationale, 4000)
    assert tf.count(ANSWER_INSTR) == 2
    assert tf.endswith(ANSWER_INSTR)                   # the real trailing question survives


# ---- 3. candidate angle: rationale carrying format-braces -------------------------------


def test_rationale_containing_format_braces_does_not_crash():
    """kills: a refactor that pipes the rationale through `.format(...)` instead of
    `.replace(...)` for the splice (which would raise KeyError/IndexError on unmatched braces,
    or silently substitute the wrong values). The splice is a literal string replace, so
    brace-like content in the rationale must pass through verbatim, uninterpreted."""
    row = {"content": "Wit"}
    rationale = "This depends on {rubric} and {text} and {0} and {}."
    tf = reason_first_tf_prompt(TEMPLATE, row, "a joke", rationale, 4000)
    assert "Reasoning: This depends on {rubric} and {text} and {0} and {}." in tf


# ---- 4. candidate angle: run_confidence's zip(texts, ans) truncates silently -------------


def test_run_confidence_zip_truncates_silently_on_short_answers_vector():
    """kills: an assumption that the confidence matrix's item axis always equals len(texts).
    DOCUMENTED HAZARD (run_reason_first_pass.py:122 `for t, a in zip(texts, ans)`): if the
    answers vector for a (cell_id, form) key is SHORTER than the current items list (e.g. a
    stale W1a tf grid scored against fewer items before the packet grew), zip silently stops
    at the shorter length instead of raising -- the returned confidence row is narrower than
    len(texts) with no error or warning anywhere in the call chain."""
    row = {"cell_id": "c1", "form": "canonical", "content": "Wit", "domain": "humor"}
    texts = ["t0", "t1", "t2", "t3", "t4"]                      # 5 items
    answers = {("c1", "canonical"): np.array([True, False, True])}   # only 3 answers
    gen = lambda prompts, seed: ["50"] * len(prompts)
    conf, meta, rates = run_confidence(None, [row], texts, TEMPLATE, 4000, answers, gen_fn=gen)
    assert conf.shape == (1, 3)                # NOT (1, 5) -- silently truncated to len(ans)
    assert rates == [1.0] and meta[0]["arm_id"] == "name_confidence"


# ---- 5. candidate angle: rationale cap applies AFTER whitespace normalization ------------


def test_rationale_cap_applies_after_whitespace_normalization():
    """kills: a refactor that caps the RAW rationale length before collapsing whitespace,
    which would silently truncate mostly into padding on rationales with leading whitespace
    and lose real content. Current order of operations (run_reason_first_pass.py:52):
    strip -> split/join (normalize all internal whitespace to single spaces) -> slice to
    max_rationale_chars -- the cap bites into the NORMALIZED string, not the raw one."""
    row = {"content": "Wit"}
    rationale = " " * 20 + "IMPORTANTWORD"       # 20 leading spaces, then real content
    tf = reason_first_tf_prompt(TEMPLATE, row, "t", rationale, 4000, max_rationale_chars=5)
    assert "Reasoning: IMPOR" in tf              # cap bites into the REAL content...
    assert "Reasoning:      " not in tf          # ...never into the stripped leading padding


# ---- 6. fresh angle: NaN score silently reads as a confident "NO" -----------------------


def test_tf_answers_from_grid_nan_score_fails_closed(tmp_path):
    """FIXED 2026-07-25 (was a documented hazard): a NaN score compared False against
    0.5, silently indistinguishable from a confident NO — the confidence stage would ask
    "your answer was NO" for an item that never produced a real score. The loader now
    refuses non-finite tf rows with an informative ValueError; finite grids unaffected.
    kills: removing the isfinite guard (reverting NaN -> False)."""
    scores = np.array([[0.9, np.nan, 0.1]])
    meta = np.array([json.dumps({"cell_id": "c1", "form": "canonical", "variant": "tf",
                                 "domain": "humor"})], dtype=object)
    path = tmp_path / "grid.npz"
    np.savez(path, scores=scores, meta=meta)
    with pytest.raises(ValueError, match="non-finite"):
        tf_answers_from_grid(str(path), "humor")
    ok = np.array([[0.9, 0.2, 0.1]])                       # finite grid still loads
    np.savez(tmp_path / "ok.npz", scores=ok, meta=meta)
    ans = tf_answers_from_grid(str(tmp_path / "ok.npz"), "humor")
    assert ans[("c1", "canonical")].tolist() == [True, False, False]


# ---- 7. fresh angle: negative max_text_chars is a valid (wrong) slice bound --------------


def test_negative_max_text_chars_slices_from_the_end_not_a_length_cap():
    """kills: assuming max_text_chars always behaves as "keep at most N characters from the
    start". DOCUMENTED HAZARD (run_reason_first_pass.py:45,51 `text[:max_text_chars]`): a
    negative value (a misparsed CLI arg, or an off-by-one subtraction upstream) is a
    perfectly valid Python slice bound, not a length -- it silently chops characters off the
    END of the text instead of raising or truncating from the start."""
    row = {"content": "Wit"}
    text = "0123456789"
    gen = reason_first_generation_prompt(TEMPLATE, row, text, -3)
    assert "0123456" in gen                # first 7 chars kept
    assert "789" not in gen                # last 3 silently dropped, not the FIRST 3
    assert "0123456789" not in gen


# ---- 8. fresh angle: `o or ""` is a falsy guard, not a type guard ------------------------


def test_non_string_generation_output_crashes_run_reason_first():
    """kills: assuming `o or ""` in run_reason_first's list comprehensions (lines 95, 106) is
    a type-safe string guard for generation outputs.
    DOCUMENTED HAZARD: it is a FALSY guard, not a type guard -- it only substitutes "" for
    None/"" and passes any other TRUTHY value straight through unchanged. A generation
    backend that returns a non-string truthy object for one item (e.g. an int, if a decoding
    layer ever returns a token count instead of text) crashes deep inside
    reason_first_tf_prompt's `rationale.strip()` with an AttributeError, not a clean
    validation error at the boundary."""
    row = {"content": "Wit", "domain": "humor"}
    texts = ["t0", "t1"]
    gen = lambda prompts, seed: ["a real rationale", 42]     # second item: non-str truthy
    score = lambda *a, **k: [0.5] * len(a[1]) if len(a) > 1 else [0.5]
    with pytest.raises(AttributeError):
        run_reason_first(None, [row], texts, TEMPLATE, 4000, {"YES": 1, "NO": 2},
                         gen_max_tokens=16, score_fn=score, gen_fn=gen)


# ---- 9. fresh angle: two independent splice reimplementations have already drifted ------


def test_two_splice_implementations_diverge_on_whitespace_and_length_cap():
    """kills: assuming battery/passes.py::assemble_reason_first_tf and
    run_reason_first_pass.py::reason_first_tf_prompt are interchangeable implementations of
    "stage 2: splice the rationale into the tf prompt".
    DOCUMENTED HAZARD: they are two INDEPENDENT reimplementations that have already drifted.
    assemble_reason_first_tf does a bare `.strip()` (keeps internal newlines/repeated spaces,
    no length cap); reason_first_tf_prompt collapses ALL internal whitespace to single spaces
    AND caps to 800 chars by default. The exact same (template, row, text, rationale) input
    produces two DIFFERENT teacher-forced prompts depending on which module's function is
    called -- run_reason_first_pass.py does not import or reuse assemble_reason_first_tf at
    all, it reimplements the splice from scratch."""
    row = {"content": "Wit"}
    rationale = "line one\nline two   has   extra   spaces"
    a = assemble_reason_first_tf(TEMPLATE, row, "a joke", rationale, 4000)
    b = reason_first_tf_prompt(TEMPLATE, row, "a joke", rationale, 4000)
    assert "line one\nline two   has   extra   spaces" in a     # untouched by assemble_*
    assert "line one\nline two   has   extra   spaces" not in b
    assert "line one line two has extra spaces" in b            # collapsed by reason_first_tf_prompt
    assert a != b


# ---- 10. fresh angle: the injection hazard from test #1 is truncation-boundary-sensitive -


def test_max_text_chars_boundary_flips_the_item_text_injection_hazard():
    """kills: treating the injection hazard demonstrated in
    test_item_text_containing_sentinel_gets_double_replaced as a fixed property of the item
    text alone. It is actually a joint property of (item text, max_text_chars): the sentinel
    is matched by exact string equality AFTER truncation, so moving max_text_chars by a
    handful of characters silently flips the very same item between "clean" (1 real
    occurrence rewritten) and "double-injected" (2 occurrences rewritten) with no error
    either way -- the hazard is not deterministic per-item, it depends on an unrelated CLI
    flag."""
    row = {"content": "Wit"}
    prefix = "Essay: "
    text = prefix + ANSWER_INSTR + " -- end of quote."
    full_cut = len(prefix) + len(ANSWER_INSTR)     # keeps the WHOLE sentinel intact
    short_cut = full_cut - 5                        # cuts the sentinel 5 chars short

    gen_full = reason_first_generation_prompt(TEMPLATE, row, text, full_cut)
    gen_short = reason_first_generation_prompt(TEMPLATE, row, text, short_cut)

    assert gen_full.count(REASON_FIRST_INSTR) == 2 and gen_full.count(ANSWER_INSTR) == 0
    assert gen_short.count(REASON_FIRST_INSTR) == 1 and gen_short.count(ANSWER_INSTR) == 0
