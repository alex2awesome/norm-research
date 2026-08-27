"""cv2 hybrid: Does the claim beat baselines? (comparative-claim verification)

Construct: ~1.0 = the abstract claims to beat named baselines by some margin AND those baselines
appear in the results section (the comparison is substantiated); ~0.5 = a comparison is claimed
but baselines/margins only partially appear in results; ~0.0 = no comparative claim, OR a
claimed beating whose baseline appears nowhere in results (unsupported superiority claim).

The LLM EXTRACTS the comparative claim + beaten baselines + margin from the abstract; CODE
checks whether each named baseline actually appears in the results/experiments body. A claim of
"outperforms BERT and GPT-4" where neither appears in the results table is an unsupported boast
code can flag. Margin verification is best-effort (numbers reformatted).

INPUT split: extraction from ABSTRACT; score() verifies against BODY (results+experiments).
"""
import re

LLM_FIELDS = {
    "comparative_claim": (
        "In <=25 words, the abstract's COMPARATIVE claim (what the method beats/outperforms, with "
        "the margin if stated). Answer NONE if the abstract makes no comparative/superiority claim."
    ),
    "beaten_baselines": (
        "Comma-separated named baselines/methods the abstract claims to OUTPERFORM or match "
        "(e.g. 'BERT, GPT-4, ResNet-50, prior work'). Answer NONE if none are named."
    ),
    "claimed_margin": (
        "The stated improvement margin (e.g. '12%', '3.5x faster', '2 points'), or NONE if no margin."
    ),
}

_NONE = {"none", "n/a", "na", "not stated", "not mentioned", "unknown", "null", ""}
_NUM_RE = re.compile(r"\d+\.\d+|\d+")

def _split_names(val):
    if not isinstance(val, str) or val.strip().lower() in _NONE:
        return []
    out = []
    for p in re.split(r"[,;/]| and ", val):
        p = p.strip()
        if p and p.lower() not in _NONE and len(p) >= 2:
            out.append(p)
    return out

def _norm_num(s):
    return re.sub(r"[%xXkKmM,\s]", "", s)

def _code_score(body, extracted):
    body = (body or "")
    blow = body.lower()
    bases = [b for b in _split_names(extracted.get("beaten_baselines"))]
    comp = extracted.get("comparative_claim")
    margin = extracted.get("claimed_margin")

    has_comp = isinstance(comp, str) and comp.strip().lower() not in _NONE
    if not has_comp and not bases:
        return 0.1  # no comparative claim at all -> not applicable (low)

    # how many named baselines appear in the body (results)?
    real = [b for b in bases if b.lower() in blow]
    base_frac = (len(real) / len(bases)) if bases else 0.0

    # margin substantiation: does the margin's number(s) appear in body?
    margin_hit = False
    if isinstance(margin, str) and margin.strip().lower() not in _NONE:
        mnums = [_norm_num(n) for n in _NUM_RE.findall(margin) if len(n) >= 1]
        bnums = {_norm_num(m.group(0)) for m in _NUM_RE.finditer(body)}
        margin_hit = any(n in bnums for n in mnums) if mnums else False

    if not bases:
        # comparative claim but no named baseline -> weak, can't verify the target
        return 0.35 if has_comp else 0.1
    # blend baseline-presence (0.6) + margin (0.2) + has-claim (0.2)
    s = 0.6 * base_frac + 0.2 * (1 if margin_hit else 0) + 0.2
    # a claimed beating where NO baseline appears in body is the red flag
    if base_frac == 0:
        s = min(s, 0.25)
    return max(0.0, min(1.0, s))

def _llm_score(extracted):
    bases = _split_names(extracted.get("beaten_baselines"))
    comp = extracted.get("comparative_claim")
    has_comp = isinstance(comp, str) and comp.strip().lower() not in _NONE
    if not has_comp:
        return 0.1
    return min(0.8, 0.4 + 0.15 * len(bases))

def score(text: str, extracted: dict, ops) -> float:
    try:
        body = ops.normalize(text) if (text and ops) else (text or "")
        return max(0.0, min(1.0, 0.7 * _code_score(body, extracted) + 0.3 * _llm_score(extracted)))
    except Exception:
        return 0.5
