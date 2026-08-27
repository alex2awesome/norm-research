"""Smoke test for the round-2 notice-and-comment hybrid programs (_h1.py).

For each aspect: strong / mid / weak synthetic comment text, extracted={} (LLM fields not
run — this only exercises the code channel + LLM-blend with empty extraction, plus a
hand-filled "ideal extraction" pass to sanity-check the LLM channel), stub ops (None and a
minimal real Ops-like object). Asserts:
  - score(strong) > score(weak) for both extracted={} and hand-filled extraction
  - every score in [0, 1]
  - ops=None never crashes (exercises the try/except None-safety contract)

Run: python test_h1_smoke.py
"""
import importlib.util
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
HYBRIDS = HERE.parent
sys.path.insert(0, str(HYBRIDS))
from ops import Ops  # noqa: E402


def load_module(name):
    spec = importlib.util.spec_from_file_location(f"nc_{name}", HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _StubOps:
    """Minimal ops stub: only normalize (identity) + sent_stats (dummy), no corpus."""
    @staticmethod
    def normalize(text):
        return text

    @staticmethod
    def sent_stats(text):
        return (1, 10.0, 0.1)


REAL_OPS = Ops()  # no corpus_path -> retrieve_similar unused but normalize/sent_stats real


# ---------------------------------------------------------------------------
# Synthetic comments + hand-filled extractions per aspect
# ---------------------------------------------------------------------------

CASES = {
    "stance_alignment_h1": {
        "strong": (
            "We support the proposed rule as proposed. The agency correctly identified the "
            "compliance gap and we support the proposed timeline and requirements without "
            "reservation. We support this approach and support finalizing it as proposed."
        ),
        "mid": (
            "The rule addresses a real problem, though we have some questions about the "
            "compliance timeline that the agency should clarify before finalizing."
        ),
        "weak": (
            "We strongly oppose this rule and object to nearly every provision in it. The "
            "agency must withdraw this rule immediately and rescind the entire proposal. "
            "We reject the underlying premise of the rule entirely."
        ),
        "extracted_strong": {"overall_position": "support"},
        "extracted_weak": {"overall_position": "oppose_fully"},
    },
    "ask_modesty_h1": {
        "strong": (
            "We ask the agency to clarify the definition of 'small entity' in this section. "
            "This is a minor technical correction and we request a narrow exemption for "
            "seasonal operators, along with a short phase-in period."
        ),
        "mid": (
            "We ask the agency to revise the reporting requirement and adjust the threshold "
            "used to determine applicability, and to reconsider the frequency of reporting."
        ),
        "weak": (
            "We demand the agency withdraw this rule entirely and start over from scratch. "
            "The entire regulatory framework must be completely overhauled and rewritten "
            "industry-wide, for all facilities, with no exceptions."
        ),
        "extracted_strong": {"main_ask": "clarify the definition of small entity",
                              "ask_category": "clarification"},
        "extracted_weak": {"main_ask": "withdraw the rule entirely",
                            "ask_category": "withdrawal"},
    },
    "deference_tone_h1": {
        "strong": (
            "We respectfully submit these comments and we appreciate the opportunity to "
            "comment on this proposed rule. Thank you for considering our perspective. We "
            "believe the agency has done thoughtful work here, and we would suggest one "
            "small clarification."
        ),
        "mid": (
            "This comment addresses the proposed reporting requirements in section 3. The "
            "requirements as written would apply to roughly 200 facilities nationwide."
        ),
        "weak": (
            "YOUR AGENCY IS COMPLETELY INCOMPETENT AND THIS RULE IS RIDICULOUS!!! You people "
            "clearly don't understand the industry at all! STOP this outrageous, shameful "
            "proposal immediately!!! Fix this now!!!"
        ),
        "extracted_strong": {},
        "extracted_weak": {},
    },
    "technical_precision_h1": {
        "strong": (
            "As discussed in the preamble at page 12, the agency estimates compliance costs "
            "of $2 million annually. Under proposed § 63.42(a)(3), the term \"affected "
            "source\" is defined as any facility exceeding the threshold. See Docket No. "
            "EPA-HQ-OAR-2025-0001, Exhibit 4, for the underlying data."
        ),
        "mid": (
            "The proposed rule would create new obligations for regulated entities and we "
            "think the agency should reconsider some of the requirements described in the "
            "proposal."
        ),
        "weak": (
            "This is a bad idea and the government should not be doing this kind of thing at "
            "all. We are against more regulation in general and think this rule is wrong."
        ),
        "extracted_strong": {"engaged_element": "proposed § 63.42(a)(3)"},
        "extracted_weak": {"engaged_element": "NONE"},
    },
}


def main():
    failures = []
    for name, c in CASES.items():
        mod = load_module(name)
        print(f"\n=== {name} ===  LLM_FIELDS={list(mod.LLM_FIELDS.keys())}")

        # (a) extracted={} for all three tiers, with real Ops
        scores_empty = {}
        for tier in ("strong", "mid", "weak"):
            s = mod.score(c[tier], {}, REAL_OPS)
            scores_empty[tier] = s
            if not (0.0 <= s <= 1.0):
                failures.append(f"{name}[{tier}] extracted={{}} out of [0,1]: {s}")
        print(f"  extracted={{}}      strong={scores_empty['strong']:.3f} "
              f"mid={scores_empty['mid']:.3f} weak={scores_empty['weak']:.3f}")
        if not (scores_empty["strong"] > scores_empty["weak"]):
            failures.append(f"{name}: strong <= weak with extracted={{}} "
                             f"({scores_empty['strong']:.3f} <= {scores_empty['weak']:.3f})")

        # (b) hand-filled extraction (only meaningful when LLM_FIELDS non-empty)
        if mod.LLM_FIELDS:
            s_strong = mod.score(c["strong"], c.get("extracted_strong", {}), REAL_OPS)
            s_weak = mod.score(c["weak"], c.get("extracted_weak", {}), REAL_OPS)
            print(f"  hand-filled extract strong={s_strong:.3f} weak={s_weak:.3f}")
            for label, s in (("strong", s_strong), ("weak", s_weak)):
                if not (0.0 <= s <= 1.0):
                    failures.append(f"{name}[{label}] hand-filled out of [0,1]: {s}")
            if not (s_strong > s_weak):
                failures.append(f"{name}: strong <= weak with hand-filled extraction "
                                 f"({s_strong:.3f} <= {s_weak:.3f})")

        # (c) ops=None must never crash (try/except 0.5 fallback contract)
        for tier in ("strong", "mid", "weak"):
            try:
                s_none = mod.score(c[tier], {}, None)
            except Exception as e:  # pragma: no cover - contract violation
                failures.append(f"{name}[{tier}] ops=None RAISED: {e!r}")
                continue
            if not (0.0 <= s_none <= 1.0):
                failures.append(f"{name}[{tier}] ops=None out of [0,1]: {s_none}")
        print(f"  ops=None            all tiers scored without raising, in [0,1]: OK")

        # (d) stub ops (has normalize/sent_stats but no corpus) must also work
        for tier in ("strong", "mid", "weak"):
            try:
                s_stub = mod.score(c[tier], {}, _StubOps())
            except Exception as e:  # pragma: no cover
                failures.append(f"{name}[{tier}] stub ops RAISED: {e!r}")
                continue
            if not (0.0 <= s_stub <= 1.0):
                failures.append(f"{name}[{tier}] stub ops out of [0,1]: {s_stub}")

        # (e) empty text must never crash
        try:
            mod.score("", {}, REAL_OPS)
            mod.score(None, {}, REAL_OPS)
        except Exception as e:
            failures.append(f"{name}: empty/None text RAISED: {e!r}")

    print("\n" + "=" * 60)
    if failures:
        print(f"FAIL ({len(failures)} issue(s)):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("ALL PASS: strong > weak for every aspect, all scores in [0,1], "
              "ops=None/stub/empty-text never crash.")
        print("SMOKE_DONE")


if __name__ == "__main__":
    main()
