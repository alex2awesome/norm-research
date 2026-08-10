"""Adversarial tests for the verbalized-confidence arm (Dienes zero-correlation criterion):
confidence_scale_valid + conf_acc_stats (battery/stats.py), parse_confidence
(battery/run_reason_first_pass.py), confidence_prompt (battery/passes.py).

These do NOT duplicate methods/tacit_channels/battery/tests/test_stats.py (scale-gate
constant-85 / binary-{0,100} degeneracies, unique-boundary-7, sparse rows, Dienes
tracking/independent/degenerate agents, chance anchoring, small-n -> None, NaN-half
handling), test_instruments.py (basic parser cases, prompt composition), or
test_battery.py (end-to-end run_confidence/run_reason_first wiring).

Every test targets an angle a naive or buggy re-implementation would misread. Genuine
current-behavior defects are marked "# DOCUMENTED HAZARD:" and asserted AS-IS — never
patched here. W1b grids were already scored with the CURRENT parser (run_reason_first_pass.
parse_confidence); changing its regex would silently break the provenance of those scored
grids, so any parser hazard below is documented, not fixed.
"""
from __future__ import annotations

import numpy as np
import pytest

from methods.tacit_channels.battery.passes import CONFIDENCE_QUESTION, confidence_prompt
from methods.tacit_channels.battery.run_reason_first_pass import parse_confidence
from methods.tacit_channels.battery.stats import (
    ITEM_AGREEMENT_CHANCE, conf_acc_stats, confidence_scale_valid,
)


def test_parser_echo_of_confidence_question_hijacks_first_integer():
    """A model that echoes back its own eliciting question (a common instruction-echo
    tic) before giving its real number gets misread: the "0-100" INSIDE the frozen
    CONFIDENCE_QUESTION constant itself is the first in-range integer, so the parser
    reports "0" no matter what the model actually meant to say.

    kills: any assumption that parse_confidence extracts the MODEL's stated number; a
    naive "first integer in range" implementation is provably biased toward the
    question's OWN scale bounds whenever the generation echoes the prompt.
    """
    echoed = CONFIDENCE_QUESTION + " I would say 90."
    # DOCUMENTED HAZARD: the frozen CONFIDENCE_QUESTION wording ("...integer 0-100.")
    # contains an in-range integer ("0") that the parser prefers over the model's real
    # answer ("90") whenever the generation echoes the question before answering.
    assert parse_confidence(echoed) == 0.0
    assert parse_confidence(echoed) != 90.0
    # a shorter paraphrase of the same echo tic reproduces it too -- not specific to
    # the exact frozen wording, it's the "0-100" pattern generically.
    assert parse_confidence("Reply with a single integer 0-100: 90") == 0.0


def test_parser_decimal_and_alternate_rating_scale_truncation():
    """Decimal or non-0-100-scale answers are silently mangled into a DIFFERENT
    integer on the 0-100 scale rather than rejected.

    kills: any downstream reader that treats a parsed value as a genuine 0-100
    percentage confidence; a model that answers "87.5" or drifts onto a 0-10 rubric
    ("9.5 out of 10" == "95% confident") gets silently remapped near the FLOOR of the
    scale instead of NaN'd out or rescaled.
    """
    # DOCUMENTED HAZARD: "\b(\d{1,3})\b" grabs only the integer part before the decimal
    # point -- "87.5" truncates to 87 (a small but real information loss: 87 vs 88).
    assert parse_confidence("87.5") == 87.0
    # DOCUMENTED HAZARD: a 0-10 rating-scale answer of "9.5 out of 10" (i.e. == 95 on a
    # 0-100 scale) is read as "9" -- catastrophically reinterpreted near the bottom of
    # the intended 0-100 range, with no scale-mismatch detection anywhere in the path.
    assert parse_confidence("9.5 out of 10") == 9.0


def test_parse_confidence_negative_sign_silently_dropped():
    """The regex has no sign handling: a minus sign is not a \\d character, so it is
    simply skipped and the magnitude is read as a positive in-range confidence.

    kills: any assumption that an explicitly negative or "no answer" style expression
    ("-5", "confidence: -1") is rejected (NaN) rather than silently flipped positive --
    a future channel that lets the model express "-10% confident" (sarcasm / below-
    floor refusal) would be miscoded as genuine positive confidence.
    """
    # DOCUMENTED HAZARD: sign is dropped, not consumed as part of a failed match.
    assert parse_confidence("-5") == 5.0
    assert parse_confidence("confidence: -1") == 1.0
    # the magnitude still has to land in [0, 100] after the sign is dropped -- a
    # large negative number is naturally rejected via ITS magnitude, not its sign.
    assert np.isnan(parse_confidence("about -250"))


def test_parse_confidence_long_digit_run_vanishes_to_nan_uncapped():
    """A confidence value rendered with a stray extra digit (a token-merge glitch, or a
    model literally typing "1000" meaning "100.0") does not clamp/truncate to a valid
    reading -- the \\b...\\b boundary can't match ANY 1-3-digit sub-run of a longer
    contiguous digit block, so the whole value disappears with no error raised.

    kills: an assumption that oversized numerals degrade gracefully (e.g. "1000" ->
    100, or "1000" -> the substring "100"); the actual behavior is a silent total
    parse-rate loss for that row with nothing distinguishing it from a genuinely
    unparseable non-numeric reply.
    """
    # DOCUMENTED HAZARD: 4+ contiguous digits never match \b(\d{1,3})\b at all -- not
    # even a "100" prefix -- because no interior boundary exists inside a digit run.
    assert np.isnan(parse_confidence("1000"))
    assert np.isnan(parse_confidence("confidence 1000"))
    assert np.isnan(parse_confidence("10000"))
    # a 4-digit YEAR mention earlier in the same text is likewise invisible to the
    # parser (neither wrongly consumed NOR the source of a false match) -- the real,
    # separately-spaced 2-3 digit answer later in the string is what gets found.
    assert parse_confidence("in 2024 I would say 85") == 85.0


def test_conf_acc_stats_fifty_pair_cliff():
    """conf_acc_stats requires ok.sum() >= 50 with a hard cutoff, not a graduated
    confidence/warning at low n.

    kills: an off-by-one loosening of the floor, or any "degrade gracefully near the
    threshold" assumption -- one fewer usable (finite, finite) pair flips the return
    from a full stats dict to bare None with no partial-information fallback.
    """
    rng = np.random.default_rng(4001)
    n = 70
    agree = np.clip(rng.normal(0.75, 0.1, n), 0, 1)
    conf = agree * 100 + rng.normal(0, 2, n)

    conf50 = conf.copy()
    conf50[50:] = np.nan          # indices 0..49 finite -> exactly 50 usable pairs
    r50 = conf_acc_stats(conf50, agree)
    assert r50 is not None and r50["degenerate_confidence"] is False

    conf49 = conf.copy()
    conf49[49:] = np.nan          # indices 0..48 finite -> exactly 49 usable pairs
    assert conf_acc_stats(conf49, agree) is None


def test_conf_acc_stats_constant_agreement_flagged_not_nan():
    """FIXED 2026-07-25 (was a documented hazard): a constant AGREEMENT vector sailed
    past the confidence-only degeneracy guard and left conf_acc_corr as an unguarded NaN
    float, silently poisoning any cross-cell mean. conf_acc_stats now returns
    conf_acc_corr=None with degenerate_agreement=True, so tallies can filter on the flag.

    kills: removing the a.std()==0 branch (reverting to the bare spearman NaN).
    """
    rng = np.random.default_rng(4002)
    n = 300
    conf = rng.uniform(0, 100, n)          # confidence DOES vary
    agree_const = np.full(n, 1.0)          # agreement does NOT vary
    r = conf_acc_stats(conf, agree_const)
    assert r["degenerate_confidence"] is False
    assert r["degenerate_agreement"] is True
    assert r["conf_acc_corr"] is None      # None, never a bare NaN float
    # flag-filtered aggregation stays finite:
    cells = [r, {"conf_acc_corr": 0.6, "degenerate_agreement": False},
             {"conf_acc_corr": 0.4, "degenerate_agreement": False}]
    usable = [c["conf_acc_corr"] for c in cells if not c["degenerate_agreement"]]
    assert np.isfinite(np.mean(usable))


def test_conf_acc_stats_inverse_dienes_agent_exposes_above_mean_guess_agreement():
    """The real-world P-B5 finding this instrument must be able to surface: an agent
    whose verbalized confidence tracks accuracy NEGATIVELY. Its bottom-quartile-
    confidence ("guessing") trials are, counterintuitively, ABOVE the sample's mean
    item agreement -- the opposite of the naive Dienes-null expectation that
    low-confidence trials always look worse.

    kills: any implementation that assumes a large-negative conf_acc_corr must
    co-occur with a LOWER guess_agreement than average (e.g. a version that clips or
    re-signs guess_agreement based on the sign of the correlation); the statistic must
    report this inversion undistorted, since it's the entire point of the P-B5 arm.
    """
    rng = np.random.default_rng(4003)
    n = 400
    agreement = np.clip(rng.normal(0.75, 0.1, n), 0, 1)
    # confidence is HIGH exactly when agreement is LOW (inverse tracker)
    conf_inverse = (1 - agreement) * 100 + rng.normal(0, 2, n)
    r = conf_acc_stats(conf_inverse, agreement)
    assert r["conf_acc_corr"] < -0.8
    assert r["guess_agreement"] > agreement.mean()
    assert r["guess_agreement_minus_chance"] > (agreement.mean() - ITEM_AGREEMENT_CHANCE)


def test_confidence_scale_valid_transposed_axes_silently_flip_the_verdict():
    """confidence_scale_valid enforces no shape/axis contract: it assumes rows are
    prompts/cells and columns are items, purely by caller convention. A cell whose
    confidence is anchored entirely by WHICH PROMPT it is (constant within a cell
    across items, but varying between cells -- i.e. no real within-cell discrimination
    at all) is correctly flagged invalid in the intended orientation, but a mere
    transpose (e.g. an upstream reshape/loader bug) launders it into "valid".

    kills: any assumption that the gate is orientation-robust or self-checking; a
    silent row/column swap flips the verdict with no error, no shape assertion, and no
    warning anywhere in confidence_scale_valid.
    """
    rng = np.random.default_rng(4004)
    n_cells, n_items = 90, 400
    per_cell_anchor = np.linspace(0, 100, n_cells)
    conf = per_cell_anchor[:, None] + rng.normal(0, 1.0, (n_cells, n_items))

    correct = confidence_scale_valid(conf)              # rows=cells (intended)
    assert correct["valid"] is False                    # no real within-cell spread

    # DOCUMENTED HAZARD: transposing (rows=items, cols=cells) flips the verdict to
    # "valid" -- the function cannot tell it was handed the wrong axis.
    transposed = confidence_scale_valid(conf.T)
    assert transposed["valid"] is True
    assert transposed["median_cell_std"] > 10 * correct["median_cell_std"]


def test_confidence_scale_valid_float_noise_inflates_n_unique_diagnostic():
    """n_unique is computed on raw float equality (np.unique). Grid loading elsewhere
    in this codebase averages multiple reps (methods/tacit_channels/channels/common.py
    load_grid: np.mean(v, axis=0)); a model that ALWAYS answers the same integer can
    still emerge from rep-averaging with sub-ULP float jitter, which np.unique counts
    as thousands of distinct values.

    kills: any report/dashboard step that reads `n_unique` in isolation as "the model
    used N distinct confidence levels" -- here n_unique balloons from a true value of 1
    to tens of thousands on a de-facto-constant load. The final `valid` verdict happens
    to stay correct only because median_cell_std (checked jointly, with its own
    min_cell_std=5.0 floor) is unmoved by noise at this scale -- n_unique alone would
    mislead.
    """
    rng = np.random.default_rng(4005)
    n_cells, n_items = 90, 400
    conf = np.full((n_cells, n_items), 85.0) + rng.normal(0, 1e-9, (n_cells, n_items))
    r = confidence_scale_valid(conf)
    assert r["valid"] is False                    # correct final verdict...
    assert r["median_cell_std"] < 1e-6             # ...for the right reason (std floor)
    # DOCUMENTED HAZARD: ...but the diagnostic alongside it is wildly overstated.
    assert r["n_unique"] > 1000


def test_confidence_prompt_performs_no_answer_validation():
    """confidence_prompt splices its `answer` argument in verbatim with an f-string --
    it never checks that answer is one of the two canonical strings the rest of the
    pipeline assumes ("YES"/"NO").

    kills: any assumption that a caller-side bug (wrong argument order, an empty
    string from an upstream failure, or accidentally passing a raw score/rationale
    instead of the mapped label) is caught here -- it is not; the composed prompt is
    produced without error regardless of what garbage `answer` is, and the CONFIDENCE_
    QUESTION suffix is appended exactly the same way either way.
    """
    # DOCUMENTED HAZARD: no validation -- garbage in, silently-composed prompt out.
    empty = confidence_prompt("JUDGMENT", "")
    assert "Your answer was: \n\n" + CONFIDENCE_QUESTION in empty
    garbled = confidence_prompt("JUDGMENT", "MAYBE")
    assert "Your answer was: MAYBE" in garbled
    # an accidentally-numeric "answer" (e.g. an argument-order bug feeding a score
    # string where the YES/NO label belongs) is embedded just as silently.
    numeric_leak = confidence_prompt("JUDGMENT", "85")
    assert "Your answer was: 85" in numeric_leak
    assert numeric_leak.endswith(CONFIDENCE_QUESTION)
