"""Smoke tests for ops_capability.py (E2L capability-ops library, VERSION 'e2l-v1').

This is a standalone historical battery, not a pytest suite.  It runs every op against
real corpus text (via battery_common.load_ctx, the same loader every E2L crew uses) plus
the specific constructed kill-case probes named in the E2L pre-registration
(notes/2026-07-10__seam-agentic-program-runbook.md), and prints PASS/FAIL per check. Exit
code is nonzero if anything fails.  Pytest collection skips this module; the canonical
CPU runner executes it separately so the 43 checks still run.

Usage: python -m methods.metric_seam.hybrids.test_ops_capability
"""
import sys
import time
import pathlib

# This legacy battery intentionally executes at module scope and terminates with
# ``sys.exit``.  Fail closed at pytest collection before any of those side effects.  It
# is exercised as a standalone subprocess by ``methods.metric_seam.run_cpu_tests``.
if __name__ != "__main__" and "pytest" in sys.modules:
    import pytest

    pytest.skip(
        "standalone 43-check historical battery; run by the canonical CPU test runner",
        allow_module_level=True,
    )

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "battery"))

_PASS, _FAIL = 0, 0


def check(name, cond, detail=""):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  PASS  {name}")
    else:
        _FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def section(title):
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# 0. import time + VERSION/CAPABILITIES freeze
# ---------------------------------------------------------------------------
section("0. import + freeze")
_t0 = time.time()
import ops_capability as oc  # noqa: E402
_import_s = time.time() - _t0
check("bare `import ops_capability` under 5s", _import_s < 5.0, f"took {_import_s:.3f}s")
check("VERSION is frozen 'e2l-v1'", oc.VERSION == "e2l-v1", oc.VERSION)
check("CAPABILITIES lists all 14 ops",
      set(oc.CAPABILITIES) == {
          "attributions", "self_attributed", "parse_math", "licensing_does_work",
          "restates_definition", "stat_consistency", "number_consistency",
          "date_chain", "deadline_satisfied", "sentence_graph", "is_refrain",
          "discourse_position", "fact_density", "entities_with_evidence"},
      sorted(oc.CAPABILITIES))
check("CapabilityOps exposes every op as a bound method",
      all(hasattr(oc.CapabilityOps, k) for k in oc.CAPABILITIES))

# ---------------------------------------------------------------------------
# garbage-input never-crash pass (every op, degenerate inputs)
# ---------------------------------------------------------------------------
section("0b. never-raise on garbage input")
_garbage = [None, "", 123, [], {}, "   ", "\n\n\n"]
_crashed = []
for g in _garbage:
    try:
        oc.attributions(g)
        oc.self_attributed("some text", g if not isinstance(g, str) else "x")
        oc.parse_math(g)
        oc.licensing_does_work(g, g)
        oc.restates_definition(g, [g] if not isinstance(g, list) else g)
        oc.stat_consistency(g)
        oc.number_consistency(g)
        oc.date_chain(g)
        if not isinstance(g, (list, dict)):
            oc.deadline_satisfied(g, g, 5)
        oc.sentence_graph(g)
        oc.is_refrain(g)
        oc.discourse_position(g, "x")
        oc.fact_density(g)
        oc.entities_with_evidence(g)
    except Exception as e:  # pragma: no cover - the whole point is this never fires
        _crashed.append((repr(g)[:30], type(e).__name__, str(e)[:60]))
check("no op raises on any garbage input", not _crashed, _crashed)

# ---------------------------------------------------------------------------
# 1. ATTRIBUTION -- a31 kill-case axis
# ---------------------------------------------------------------------------
section("1. ATTRIBUTION (press_releases/a31 axis)")

# 1a. the exact a31 kill-case construction: a bare reporting verb with NO governed
# clause ("believes in borders", a PP not a ccomp) sits one sentence away from a
# genuine self-voiced overclaim. A window-search implementation (the rejected
# candidate's _ATTRIB_RE) falsely discounts the overclaim as third-party-attributed.
kill_a31_text = (
    "Nothing will improve until there is a public servant leading the Department of "
    "Homeland Security who believes in borders--and secure ones at that. "
    "This is a stunning, unbelievable failure of leadership that has never happened before."
)
overclaim_sentence = ("This is a stunning, unbelievable failure of leadership that has "
                      "never happened before.")
attrs_kill = oc.attributions(kill_a31_text)
check("bare 'believes in borders' (no ccomp) produces NO attribution record",
      all(a["verb"] != "believe" for a in attrs_kill), attrs_kill)
sa_kill = oc.self_attributed(kill_a31_text, overclaim_sentence)
check("neighboring overclaim resolves SELF-VOICED (True), not falsely attributed",
      sa_kill is True, sa_kill)

# 1b. genuine third-party attribution still detected (positive control)
pos_text = 'Jane Doe, an independent analyst, said the merger "raises serious concerns."'
attrs_pos = oc.attributions(pos_text)
check("genuine third-party quote IS detected",
      any(a["verb"] == "say" and not a["speaker_is_first_person_org"] for a in attrs_pos),
      attrs_pos)

# 1c. first-person-org self-quote (the "own scientist" generalization: NOT a fixed
# title-vocabulary match, decided from the document's own most-frequent ORG entity)
self_text = ('"We are proud of this result," said Maria Chen, a senior researcher at '
            "Acme Biotech. Acme Biotech has led the field for a decade.")
attrs_self = oc.attributions(self_text)
own_hit = [a for a in attrs_self if a["verb"] == "say"]
check("own-researcher quote resolves speaker_is_first_person_org=True "
      "(generalizes beyond a fixed exec-title list)",
      bool(own_hit) and own_hit[0]["speaker_is_first_person_org"] is True, attrs_self)

# ---------------------------------------------------------------------------
# 2. MATH ENTAILMENT -- a150/a30 vacuity axis
# ---------------------------------------------------------------------------
section("2. MATH ENTAILMENT (math/a150, math/a30 vacuity axis)")

# 2a. the exact vacuity failure mode: a content-free "note that... since..." filler
# must NEVER parse as checkable math (this is the bug the prose guard exists for --
# sympy's raw lark grammar silently reads prose as a monomial of single letters).
vacuous = "note that the result follows since n exists in the given set"
vac_parsed = oc.parse_math(vacuous)
check("vacuous filler prose does NOT parse as math (checkable=False, not a guess)",
      vac_parsed is None, vac_parsed)

premise = oc.parse_math("2x + 3 = 7")
lic_vacuous = oc.licensing_does_work(premise, vac_parsed)
check("licensing_does_work(genuine premise, vacuous filler) -> checkable=False",
      lic_vacuous["checkable"] is False, lic_vacuous)

# 2b. genuine algebraic rewrite DOES get licensed
consequent_right = oc.parse_math("x = 2")
lic_right = oc.licensing_does_work(premise, consequent_right)
check("genuine rearrangement (2x+3=7 -> x=2) is checkable AND follows=True",
      lic_right["checkable"] and lic_right["follows"] is True, lic_right)

# 2c. syntactically-math but WRONG consequent is caught (not rubber-stamped)
consequent_wrong = oc.parse_math("x = 99")
lic_wrong = oc.licensing_does_work(premise, consequent_wrong)
check("wrong consequent (x=99) is checkable but follows=False",
      lic_wrong["checkable"] and lic_wrong["follows"] is False, lic_wrong)

# 2d. math/a30's "citation-name stacking with zero new content" pattern: repeating the
# SAME licensed fact under a different surface form must not look like new licensing
# work -- restates_definition finds the SYMBOLIC match (order/rearrangement-independent
# equivalence, not a string comparison), so a candidate can de-duplicate by symbol
# rather than by counting distinct citation surface forms.
rd = oc.restates_definition("y = x^2 + 1", ["y = 1 + x^2", "y = 2x"])
check("restates_definition finds the symbolic match across a reordered surface form",
      rd["checkable"] and rd["match_index"] == 0, rd)
rd_prose = oc.restates_definition("as noted above and clearly established", ["y = 2x"])
check("restates_definition on a prose (non-math) clause -> checkable=False",
      rd_prose["checkable"] is False, rd_prose)

# ---------------------------------------------------------------------------
# 3. CONSISTENCY RECOMPUTATION -- peer_review axis
# ---------------------------------------------------------------------------
section("3. CONSISTENCY RECOMPUTATION (statcheck-style)")
stat_text = ("The groups differed significantly, t(28) = 2.05, p = .05. "
            "We also found a decision-inconsistent result: t(100) = 0.50, p < .05.")
sc = oc.stat_consistency(stat_text)
row_consistent = next((r for r in sc if r["test"] == "t" and r["df"] == 28.0), None)
row_bad = next((r for r in sc if r["test"] == "t" and r["df"] == 100.0), None)
check("t(28)=2.05,p=.05 recomputes to a consistent p (~.0498)",
      row_consistent and row_consistent["checkable"] and row_consistent["numeric_consistent"],
      row_consistent)
check("t(100)=0.50,p<.05 is flagged decision_inconsistent "
      "(recomputed p~.62, nowhere near significant)",
      row_bad and row_bad["checkable"] and row_bad["decision_inconsistent"] is True,
      row_bad)

nc = oc.number_consistency("Of 200 patients, 40 out of 200 (20%) reported side effects.")
check("40/200 (20%) count-pct arithmetic verified consistent",
      nc and nc[0]["consistent"] is True, nc)
nc_bad = oc.number_consistency("30 out of 200 (25%) reported issues.")
check("30/200 (25%, actually 15%) flagged INconsistent",
      nc_bad and nc_bad[0]["consistent"] is False, nc_bad)

# ---------------------------------------------------------------------------
# 4. PROCEDURAL DATES -- legal axis
# ---------------------------------------------------------------------------
section("4. PROCEDURAL DATES")
date_text = ("Richard Harper began working on January 29, 2018. He filed his EEOC "
            "charge on 2019-03-15. The employer terminated him on 04/02/2019.")
dc = oc.date_chain(date_text)
check("date_chain extracts 3 dates in document order",
      len(dc) == 3 and dc[0]["date"] == "2018-01-29" and dc[2]["date"] == "2019-04-02",
      dc)
check("deadline_satisfied: filing 45 days after event, 90-day window -> True",
      oc.deadline_satisfied("2019-01-01", "2019-02-15", 90) is True)
check("deadline_satisfied: filing 45 days after event, 30-day window -> False",
      oc.deadline_satisfied("2019-01-01", "2019-02-15", 30) is False)
check("deadline_satisfied: unparseable date -> None (honest, not a guess)",
      oc.deadline_satisfied("not a date", "2019-02-15", 30) is None)

# ---------------------------------------------------------------------------
# 5. DISCOURSE/STRUCTURE -- a117/a315 humor axis
# ---------------------------------------------------------------------------
section("5. DISCOURSE/STRUCTURE (humor/a117 repetition-function axis)")

# 5a. constructed padding (no progression between repeats) -> NOT a refrain
padding_text = ("This release contains forward-looking statements about our plans. "
                "This release contains forward-looking statements about our plans. "
                "This release contains forward-looking statements about our plans.")
refr_pad = oc.is_refrain(padding_text)
check("adjacent verbatim copy-paste padding is NOT flagged as refrain/craft",
      refr_pad and refr_pad[0]["is_refrain"] is False, refr_pad)

# 5b. constructed alternating-boilerplate padding (a HARDER padding case: the gap
# between repeats is itself a second static refrain, not genuine new content)
alt_padding = ("Our commitment to quality drives everything we do. "
              "We strive for excellence in every product. "
              "Our commitment to quality drives everything we do. "
              "We strive for excellence in every product. "
              "Our commitment to quality drives everything we do. "
              "We strive for excellence in every product.")
refr_alt = oc.is_refrain(alt_padding)
check("interleaved A/B static boilerplate (two refrains, no progression) -> NOT craft",
      refr_alt and all(not r["is_refrain"] for r in refr_alt), refr_alt)

# 5c. discourse_position: a fixed trailing-% cut can slice through the final beat;
# position here is sentence-bounded.
struct_text = "Opening line. Middle beat one. Middle beat two. This is the final coda."
check("discourse_position: first sentence -> opening",
      oc.discourse_position(struct_text, "Opening line.") == "opening")
check("discourse_position: last sentence -> coda",
      oc.discourse_position(struct_text, "This is the final coda.") == "coda")
check("discourse_position: interior sentence -> middle",
      oc.discourse_position(struct_text, "Middle beat one.") == "middle")

# ---------------------------------------------------------------------------
# 6. NER FACTS
# ---------------------------------------------------------------------------
section("6. NER FACTS")
ner_text = ("Dr. Jane Smith, CEO of Acme Corp, announced on March 3, 2023 that revenue "
           "grew 20%. See https://acme.com/report for details.")
fd = oc.fact_density(ner_text)
check("fact_density finds >=1 entity type with nonzero total", fd["total"] > 0, fd)
ew = oc.entities_with_evidence(ner_text)
check("entities_with_evidence finds Acme Corp evidenced (number/date/url nearby)",
      any(e["label"] == "ORG" and e["evidenced"] for e in ew), ew)

# ---------------------------------------------------------------------------
# 7. real corpus smoke tests, 3+ train texts per relevant task
#    (battery_common.load_ctx -- the same loader every E2L crew imports)
# ---------------------------------------------------------------------------
section("7. real-corpus smoke tests (battery_common.load_ctx)")
try:
    from battery_common import load_ctx
except Exception as e:
    check("battery_common importable", False, e)
    load_ctx = None

if load_ctx is not None:
    for task, groups in [
        ("press_releases", ("attribution", "dates", "ner")),
        ("math", ("math",)),
        ("humor", ("discourse",)),
        ("peer_review", ("consistency", "ner")),
    ]:
        try:
            ctx = load_ctx(task)
        except Exception as e:
            check(f"load_ctx({task!r})", False, e)
            continue
        train_ids = sorted(ctx["train"])[:5]
        texts = [ctx["items"][d] for d in train_ids if d in ctx["items"]][:3]
        check(f"load_ctx({task!r}) yields >=3 train texts", len(texts) >= 3,
              f"got {len(texts)}")
        ok = True
        try:
            for t in texts:
                if "attribution" in groups:
                    oc.attributions(t)
                if "dates" in groups:
                    oc.date_chain(t)
                if "ner" in groups:
                    oc.fact_density(t)
                    oc.entities_with_evidence(t)
                if "math" in groups:
                    # math items are prose-with-LaTeX; extract inline spans via the
                    # task's own MathOps (already on sys.path via load_ctx) and try
                    # parse_math on each -- most won't be "checkable" (full answers,
                    # not bare equations), and that is the CORRECT honest outcome.
                    spans = ctx["ops"].extract_math_spans(t) if hasattr(ctx["ops"], "extract_math_spans") else []
                    for _kind, content in spans[:5]:
                        oc.parse_math(content)
                if "discourse" in groups:
                    oc.is_refrain(t)
                    oc.sentence_graph(t)
                    oc.discourse_position(t, t[:30]) if len(t) > 30 else None
                if "consistency" in groups:
                    oc.stat_consistency(t)
                    oc.number_consistency(t)
        except Exception as e:
            ok = False
            check(f"{task}: ops run without exception on real train text", False, e)
        if ok:
            check(f"{task}: ops run without exception on real train text ({len(texts)} docs)",
                  True)

    # legal (bonus, not in the required-4 list, but the DATE ops' named target axis)
    try:
        ctx_legal = load_ctx("legal_title_vii")
        train_ids = sorted(ctx_legal["train"])[:5]
        texts = [ctx_legal["items"][d] for d in train_ids if d in ctx_legal["items"]][:3]
        any_dates = False
        for t in texts:
            dc = oc.date_chain(t)
            any_dates = any_dates or bool(dc)
        check("legal_title_vii (bonus): date_chain finds real dates in >=1 of 3 docs",
              any_dates, "no dates found")
    except Exception as e:
        check("legal_title_vii bonus check", False, e)

# ---------------------------------------------------------------------------
# 8. real a31 kill-case documents (press_releases d00964, d01806)
# ---------------------------------------------------------------------------
section("8. real a31 kill-case documents (d00964, d01806)")
if load_ctx is not None:
    try:
        ctx_pr = load_ctx("press_releases")
        items = ctx_pr["items"]
        if "d00964" in items:
            a964 = oc.attributions(items["d00964"])
            # decisive_reason: "FAIR believes that..." must resolve self-attributed
            # (org's own voice), NOT be discounted as third-party via the bare-verb
            # window-leak the rejected candidate exhibited.
            fair_hit = [a for a in a964 if a["verb"] == "believe"
                       and "FAIR" in a["speaker_span"]]
            check("d00964: 'FAIR believes that...' resolves speaker_is_first_person_org=True",
                  bool(fair_hit) and fair_hit[0]["speaker_is_first_person_org"] is True,
                  a964)
        else:
            check("d00964 present in press_releases items", False, "missing id")
        if "d01806" in items:
            a806 = oc.attributions(items["d01806"])
            # decisive_reason: h0's _EXEC_TITLE_RE (closed CEO/CFO/... vocabulary)
            # wrongly treated the company's own scientist as an independent
            # third-party. This op must generalize via home-org detection instead.
            sci_hit = [a for a in a806 if "scientist" in a["speaker_span"].lower()]
            check("d01806: 'principal scientist at GE Research' resolves "
                  "speaker_is_first_person_org=True (generalizes beyond exec-title list)",
                  bool(sci_hit) and sci_hit[0]["speaker_is_first_person_org"] is True,
                  a806)
        else:
            check("d01806 present in press_releases items", False, "missing id")
    except Exception as e:
        check("real a31 kill-case documents", False, e)

# ---------------------------------------------------------------------------
# 9. real a117 kill-case document (humor d01575, the banana-joke escalating refrain)
# ---------------------------------------------------------------------------
section("9. real a117 kill-case document (d01575)")
if load_ctx is not None:
    try:
        ctx_h = load_ctx("humor")
        if "d01575" in ctx_h["items"]:
            banana = ctx_h["items"]["d01575"]
            refr = oc.is_refrain(banana)
            hit = [r for r in refr if len(r["occurrences"]) >= 3]
            check("d01575: 'Do you have a banana?' escalating rule-of-three IS flagged "
                  "as refrain/craft (not penalized as near-dup padding, the a117 kill)",
                  bool(hit) and hit[0]["is_refrain"] is True, refr)
            check("d01575: punchline ('Then, do you have a banana?') resolves coda",
                  oc.discourse_position(banana, "Then, do you have a banana?") == "coda")
        else:
            check("d01575 present in humor items", False, "missing id")
    except Exception as e:
        check("real a117 kill-case document", False, e)

# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print(f"TOTAL: {_PASS} passed, {_FAIL} failed  (import time {_import_s:.3f}s)")
sys.exit(1 if _FAIL else 0)
