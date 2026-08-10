"""a104 — Automated tests: presence and design quality (hybrid, exec-op enabled).

Construct: does the PR ship automated tests that actually verify its change, and are they
well-designed? Three-layer channel:
  CODE   parse the diff: which test files are touched, how many assertions added, is the
         change test-only or code+tests together.
  LLM    thick extraction: name ONE added/changed test that verifies THIS change (grounded
         against the diff), and classify the change kind (docs/config changes cap the need).
  TOOL   ops.test_transition(dpid) — the MOCKED transplant runner: 'pinned' means the
         transplanted tests fail on base & pass with the patch (they demonstrably verify the
         change); 'vacuous' means they pass without the patch (presence without verification);
         'none' = no usable tests.
"""
import re

LLM_FIELDS = {
    "test_evidence": "quote the name of ONE test function/method/case that this PR adds or "
                     "modifies to verify its main change; answer NONE if no test changes",
    "change_kind": "one word for what this PR mainly changes: feature, bugfix, refactor, "
                   "docs, config, tests, or other",
}

_TEST_PATH = re.compile(
    r"^diff --git .*?(test[s]?/|_test\.(go|py|rb|js|ts)|test_[\w./-]+\.py|Test\w*\.java"
    r"|\.spec\.(js|ts)|/spec/)", re.M | re.I)
_ANY_PATH = re.compile(r"^diff --git a/(\S+)", re.M)
_ASSERT_ADD = re.compile(
    r"^\+.*?(assert|require\.|expect\(|t\.Error|t\.Fatal|assertEquals|assertTrue"
    r"|\.should\.|pytest\.raises|@Test)", re.M)
_DOCS_ONLY = re.compile(r"\.(md|rst|txt|adoc)$")


def _diff_stats(t):
    paths = _ANY_PATH.findall(t)
    testish = [p for p in paths if re.search(
        r"(^|/)tests?(/|_)|_test\.|test_|Test\w*\.java|\.spec\.", p, re.I)]
    return paths, testish


def score(text, extracted, ops, dpid=None):
    try:
        t = ops.normalize(text or "")
        if not t.strip():
            return 0.0
        paths, testish = _diff_stats(t)
        n_assert = len(_ASSERT_ADD.findall(t))
        docs_only = bool(paths) and all(_DOCS_ONLY.search(p) for p in paths)

        # --- CODE layer: presence + in-diff design signals -------------------------
        if not paths:
            presence = 0.1
        elif not testish:
            presence = 0.12
        else:
            frac = len(testish) / len(paths)
            presence = 0.45 + 0.2 * min(1.0, frac * 2) + min(0.2, 0.03 * n_assert)
        if docs_only:
            presence = 0.55   # docs-only PR: tests not expected; mid, not failing

        # --- TOOL layer: mocked transplant runner (evidence op) --------------------
        ev = None
        try:
            ev = ops.test_transition(dpid) if dpid else None
        except Exception:
            ev = None
        if ev:
            lab = (ev.get("label") or "")
            if lab == "pinned":
                presence = max(presence, 0.85)          # tests PIN the change (F2P-causal-like)
            elif lab == "partial_pinned":
                presence = max(presence, 0.7)
            elif lab == "vacuous":
                presence = min(presence, 0.45) * 0.8    # tests exist but verify nothing
            elif lab == "none" and not testish:
                presence = min(presence, 0.15)
            nv = ev.get("n_vacuous_pass") or 0
            if nv and nv > 2:
                presence *= 0.85
            tor = ev.get("test_only_ratio")
            if tor is not None and 0 < tor < 1 and lab in ("pinned", "partial_pinned"):
                presence = min(1.0, presence + 0.05)    # code+tests co-change, verified

        # --- LLM layer: grounded thick extraction ----------------------------------
        fx = (extracted or {}).get("test_evidence", "").strip()
        kind = (extracted or {}).get("change_kind", "").strip().lower()
        if fx and fx.upper() != "NONE":
            token = re.sub(r"[^\w]", "", fx.split("(")[0])[-40:]
            grounded = token and token.lower() in re.sub(r"[^\w]", "", t.lower())
            if grounded and testish:
                presence = min(1.0, presence + 0.1)
            elif not grounded:
                presence *= 0.9                          # ungrounded claim: don't reward
        elif "test_evidence" in (extracted or {}) and testish:
            presence *= 0.85                             # extractor saw no real test change
        if kind in ("docs", "config") and not testish:
            presence = max(presence, 0.5)
        return max(0.0, min(1.0, presence))
    except Exception:
        return 0.3
