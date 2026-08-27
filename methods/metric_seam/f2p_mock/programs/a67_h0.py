"""a67 — Functional correctness and defect risk (hybrid, exec-op enabled).

Construct: how confident can a reviewer be that this PR is functionally correct / low
defect-risk? The exec op is the natural evidence: a 'pinned' transplant outcome means the
change's behavior is demonstrably exercised by tests that fail without it. Text-side signals
(scope discipline, error handling, fix framing) modulate; the LLM extracts the claimed defect
and a visible risk line, both validated against the diff.
"""
import re

LLM_FIELDS = {
    "claimed_fix": "in at most 15 words, what defect or incorrect behavior does this PR claim "
                   "to fix; answer NONE if it is not a fix",
    "risk_line": "quote one added line from the diff most likely to change existing behavior "
                 "unintentionally; answer NONE if none is visible",
}

_ANY_PATH = re.compile(r"^diff --git a/(\S+)", re.M)
_ADD = re.compile(r"^\+(?!\+\+)", re.M)
_DEL = re.compile(r"^-(?!--)", re.M)
_ERRH = re.compile(r"^\+.*?(try:|except |if err != nil|catch\s*\(|\.catch\(|raise |errors\.)",
                   re.M)
_TODO = re.compile(r"^\+.*?(TODO|FIXME|XXX|HACK)\b", re.M)
_FIXKW = re.compile(r"\b(fix(es|ed)?|bug|regression|incorrect|crash|leak|race|off.by.one"
                    r"|overflow|null.?pointer|npe)\b", re.I)


def score(text, extracted, ops, dpid=None):
    try:
        t = ops.normalize(text or "")
        if not t.strip():
            return 0.0
        head = t[:1500]
        n_add, n_del = len(_ADD.findall(t)), len(_DEL.findall(t))
        paths = _ANY_PATH.findall(t)
        size = n_add + n_del

        # --- CODE layer: scope discipline + hygiene ---------------------------------
        s = 0.5
        if size and size <= 80 and len(paths) <= 4:
            s += 0.12                                   # small, focused change
        elif size > 800 or len(paths) > 15:
            s -= 0.12                                   # sprawling change, higher risk
        if _ERRH.search(t):
            s += 0.06
        n_todo = len(_TODO.findall(t))
        s -= min(0.15, 0.05 * n_todo)
        is_fix = bool(_FIXKW.search(head))

        # --- TOOL layer: mocked transplant runner ------------------------------------
        ev = None
        try:
            ev = ops.test_transition(dpid) if dpid else None
        except Exception:
            ev = None
        verified = False
        if ev:
            lab = (ev.get("label") or "")
            if lab == "pinned":
                s += 0.3; verified = True               # behavior change verified by tests
            elif lab == "partial_pinned":
                s += 0.18; verified = True
            elif lab == "vacuous":
                s -= 0.12                               # tests don't catch the change at all
            elif lab == "none":
                s -= 0.05 if size > 200 else 0.0        # big unverified change
            if (ev.get("n_compile_fail") or 0) > 0 or (ev.get("n_setup_fail") or 0) > 2:
                s -= 0.08                               # fragile environment/tests

        # --- LLM layer: claimed defect + visible risk, grounded ----------------------
        cf = (extracted or {}).get("claimed_fix", "").strip()
        rl = (extracted or {}).get("risk_line", "").strip()
        if cf and cf.upper() != "NONE":
            if verified:
                s += 0.08                               # claims a fix AND tests pin it
            elif ev and (ev.get("label") or "") in ("vacuous", "none"):
                s -= 0.08                               # claims a fix, nothing verifies it
        if rl and rl.upper() != "NONE":
            token = re.sub(r"\s+", "", rl)[:60]
            if token and token in re.sub(r"\s+", "", t):
                s -= 0.07                               # grounded visible-risk line
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.3
