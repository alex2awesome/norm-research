"""Smoke test for the round-3 (VERIFICATION-tier) notice-and-comment hybrid programs (_h2.py).

For each aspect: strong (verified-coherent) / mid (nothing to verify) / weak (internally
inconsistent / fabricated) synthetic comment text. Asserts:
  - score(strong) > score(weak) for both extracted={} and hand-filled extraction
  - every score in [0, 1]
  - ops=None never crashes (exercises the try/except None-safety contract)
  - numeric_consistency: a comment whose numbers DON'T add up scores lower than one whose
    numbers DO add up
  - authority_lookup: a real CFR citation (7 CFR 56) beats a fabricated one (7 CFR 9999)

Run: python test_h2_smoke.py
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
    @staticmethod
    def normalize(text):
        return text

    @staticmethod
    def sent_stats(text):
        return (1, 10.0, 0.1)


REAL_OPS = Ops()

CASES = {
    "numeric_consistency_h2": {
        "strong": (
            "Our facility's annual compliance cost totals $600,000. This total is made up of "
            "$350,000 in labor costs and $250,000 in equipment costs. That works out to $50 "
            "per worker across our 12,000 affected workers nationwide. Compliance costs "
            "increased from $500,000 to $650,000, a 30% increase over the prior baseline."
        ),
        "mid": (
            "The rule will affect approximately 5,000 small businesses nationwide. We estimate "
            "this could increase compliance costs by 10% for many operators, though exact "
            "figures will vary by facility size and region."
        ),
        "weak": (
            "Our facility's annual compliance cost totals $600,000. This total is made up of "
            "$200,000 in labor costs and $150,000 in equipment costs. Elsewhere, compliance "
            "costs are $2 million per year. Later in this same comment, compliance costs are "
            "described as $5 million, which is unsustainable for small operators."
        ),
        "extracted_strong": {
            "central_quant_claim": "compliance costs total $600,000 annually ($50/worker x "
                                    "12,000 workers)",
            "supporting_figures": "$350,000 labor, $250,000 equipment, $50 per worker, "
                                   "12,000 workers",
        },
        "extracted_weak": {
            "central_quant_claim": "compliance costs are $5 million",
            "supporting_figures": "NONE",
        },
    },
    "authority_lookup_h2": {
        "strong": (
            "This comment relies on 7 CFR 56.32, which sets the applicable grading standard "
            "for the product at issue. We also note 40 CFR Part 60 governs emission limits "
            "for this source category, and the agency should apply both consistently."
        ),
        "mid": (
            "We believe Congress intended a different outcome and cite general principles of "
            "administrative law in support of our position, without reference to a specific "
            "section."
        ),
        "weak": (
            "This comment relies on 7 CFR 9999, which the agency must follow. We also cite "
            "40 CFR Part 88888 as controlling authority for this rule."
        ),
        "extracted_strong": {"authority_relied_on": "7 CFR 56.32"},
        "extracted_weak": {"authority_relied_on": "7 CFR 9999"},
    },
}


def main():
    failures = []
    for name, c in CASES.items():
        mod = load_module(name)
        print(f"\n=== {name} ===  LLM_FIELDS={list(mod.LLM_FIELDS.keys())}")

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

        for tier in ("strong", "mid", "weak"):
            try:
                s_none = mod.score(c[tier], {}, None)
            except Exception as e:  # pragma: no cover
                failures.append(f"{name}[{tier}] ops=None RAISED: {e!r}")
                continue
            if not (0.0 <= s_none <= 1.0):
                failures.append(f"{name}[{tier}] ops=None out of [0,1]: {s_none}")
        print("  ops=None            all tiers scored without raising, in [0,1]: OK")

        for tier in ("strong", "mid", "weak"):
            try:
                s_stub = mod.score(c[tier], {}, _StubOps())
            except Exception as e:  # pragma: no cover
                failures.append(f"{name}[{tier}] stub ops RAISED: {e!r}")
                continue
            if not (0.0 <= s_stub <= 1.0):
                failures.append(f"{name}[{tier}] stub ops out of [0,1]: {s_stub}")

        try:
            mod.score("", {}, REAL_OPS)
            mod.score(None, {}, REAL_OPS)
        except Exception as e:
            failures.append(f"{name}: empty/None text RAISED: {e!r}")

    # aspect-specific extra assertions from the task spec
    nc = load_module("numeric_consistency_h2")
    coherent = CASES["numeric_consistency_h2"]["strong"]
    contradictory = CASES["numeric_consistency_h2"]["weak"]
    s_coh = nc.score(coherent, {}, REAL_OPS)
    s_bad = nc.score(contradictory, {}, REAL_OPS)
    print(f"\n[numeric_consistency] coherent={s_coh:.3f} vs contradictory={s_bad:.3f}")
    if not (s_coh > s_bad):
        failures.append(f"numeric_consistency: coherent <= contradictory "
                         f"({s_coh:.3f} <= {s_bad:.3f})")

    al = load_module("authority_lookup_h2")
    real_cite = "This comment relies on 7 CFR 56, which is directly on point."
    fake_cite = "This comment relies on 7 CFR 9999, which is directly on point."
    s_real = al.score(real_cite, {}, REAL_OPS)
    s_fake = al.score(fake_cite, {}, REAL_OPS)
    print(f"[authority_lookup] real (7 CFR 56)={s_real:.3f} vs "
          f"fabricated (7 CFR 9999)={s_fake:.3f}")
    if not (s_real > s_fake):
        failures.append(f"authority_lookup: real <= fabricated ({s_real:.3f} <= {s_fake:.3f})")
    print(f"[authority_lookup] CFR index loaded: {al._CFR_INDEX is not None} "
          f"({len(al._CFR_INDEX) if al._CFR_INDEX else 0} titles)")

    print("\n" + "=" * 60)
    if failures:
        print(f"FAIL ({len(failures)} issue(s)):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("ALL PASS: strong > weak for every aspect, all scores in [0,1], "
              "ops=None/stub/empty-text never crash, coherent>contradictory, real>fabricated.")
        print("SMOKE_DONE")


if __name__ == "__main__":
    main()
