"""a128 — Automated testing strategy and adequacy (hybrid, exec-op enabled).

Construct: is the PR's testing ADEQUATE for what it changes (right kinds, right coverage),
not merely present? Code reads the diff's test/code balance and framework markers; the LLM
names the test kinds and the most obviously UNCOVERED changed behavior; the mocked transplant
runner supplies the adequacy ground: vacuous mass = inadequate by construction, pinned =
adequate for the changed behavior.
"""
import re

LLM_FIELDS = {
    "test_kinds": "list the kinds of automated tests this PR adds or modifies "
                  "(unit, integration, e2e, none — comma-separated)",
    "untested_change": "name one behavior this PR changes that has NO test coverage in this "
                       "diff; answer NONE if everything changed is covered",
}

_ANY_PATH = re.compile(r"^diff --git a/(\S+)", re.M)
_UNIT = re.compile(r"(^|/)tests?(/|_)|_test\.|test_|\.spec\.", re.I)
_INTEG = re.compile(r"(integration|e2e|functional|acceptance)", re.I)
_MOCK = re.compile(r"^\+.*?(mock|patch\(|stub|fake[A-Z_]|monkeypatch)", re.M)


def score(text, extracted, ops, dpid=None):
    try:
        t = ops.normalize(text or "")
        if not t.strip():
            return 0.0
        paths = _ANY_PATH.findall(t)
        tpaths = [p for p in paths if _UNIT.search(p)]
        cpaths = [p for p in paths if p not in tpaths]

        # --- CODE layer: balance + breadth -------------------------------------------
        if not paths:
            s = 0.1
        elif not cpaths:                                  # tests-only PR
            s = 0.55
        elif not tpaths:
            s = 0.18                                      # code change, zero test change
        else:
            balance = len(tpaths) / max(1, len(cpaths))
            s = 0.45 + 0.2 * min(1.0, balance)
        if any(_INTEG.search(p) for p in tpaths):
            s = min(1.0, s + 0.08)                        # beyond-unit layer present
        if _MOCK.search(t):
            s = min(1.0, s + 0.04)

        # --- TOOL layer: mocked transplant runner ------------------------------------
        ev = None
        try:
            ev = ops.test_transition(dpid) if dpid else None
        except Exception:
            ev = None
        if ev:
            lab = (ev.get("label") or "")
            if lab == "pinned":
                s = max(s, 0.8)
            elif lab == "partial_pinned":
                s = max(s, 0.62)
            elif lab == "vacuous":
                s = min(s, 0.4)                           # tests pass without the patch
            elif lab == "none" and cpaths:
                s = min(s, 0.2)
            tbr = ev.get("test_byte_ratio")
            if tbr is not None:
                if tbr < 0.05 and cpaths:
                    s -= 0.08                             # trivial test mass vs change mass
                elif 0.15 <= tbr <= 0.9:
                    s += 0.05
            nv = ev.get("n_vacuous_pass") or 0
            if nv > 2:
                s -= 0.07

        # --- LLM layer ---------------------------------------------------------------
        kinds = (extracted or {}).get("test_kinds", "").strip().lower()
        gap = (extracted or {}).get("untested_change", "").strip()
        if kinds and kinds != "none":
            n_kinds = len({k.strip() for k in kinds.split(",") if k.strip()} - {"none"})
            if tpaths:
                s = min(1.0, s + 0.05 * min(2, n_kinds))
        elif kinds == "none" and cpaths:
            s = min(s, 0.3)
        if gap and gap.upper() != "NONE":
            s -= 0.1                                      # extractor names an uncovered behavior
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.3
