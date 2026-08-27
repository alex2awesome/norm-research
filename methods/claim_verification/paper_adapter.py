"""Peer-review (research paper) domain adapter for the claim-verification framework.

Transports the patents localize-then-verify architecture to papers, apples-to-apples with the
press-release battery (same arm structure: r_* hybrid retrieval, s_* LLM support, null_* placebo
twins), plus two paper-specific legs:
  PERT  — number-perturbation twin: claims with numbers get deterministically perturbed copies;
          FULL-rate must drop on perturbed claims or the verifier isn't reading the numbers.
  PA    — prior-art leg: each claim vs top-K earlier-year ICLR abstracts, with the paper's OWN
          abstract planted as gold (must be ANTICIPATES) and a random foreign-topic candidate
          (must be DISTINCT) — the planted-control discipline from the patents pipeline.

Differences from news that motivated this adapter:
  - head = abstract (the claim source), body = SUBTRACTIVE full text (everything except
    preamble/abstract/references/related-work/acks) — the additive section whitelist left
    18.5% of bodies empty.
  - bodies are 10K+ chars, so verify passages are RETRIEVED per claim (lexical containment,
    same scorer as the F2 arm), not first-N paragraphs.

Pure logic only — no vLLM/network imports; the sk3 runner does the batching.
"""
import hashlib, json, re

from .seam_metrics import _toks, _overlap

# ---------------- prompts ----------------
PAPER_CLAIM_EXTRACT = """You are analyzing the ABSTRACT of a research paper. Extract the ATOMIC \
CLAIMS it asserts. An atomic claim is a single checkable assertion: a quantitative result, a \
comparison against baselines, a novelty assertion (first/new X), a proposed method/technique, or \
a claimed capability. Do NOT extract background statements, motivation, or vague aspirations.

ABSTRACT:
{head}

Return JSON: {{"claims": [{{"claim": "<one-sentence self-contained restatement>", \
"kind": "<quantity|comparative|novelty|method|capability>"}}]}}
At most {max_claims} claims, most central first. Return ONLY the JSON."""

PAPER_LOCALIZE_VERIFY = """You are verifying whether a claim made in a research paper's ABSTRACT \
is substantiated in the paper's BODY.

CLAIM: {claim}

BODY PASSAGES (numbered):
{passages}

Step 1 - LOCALIZE: find the passage(s) that bear on this claim, if any.
Step 2 - VERIFY:
  FULL    = a passage substantiates the claim with specifics (matching numbers, experiment
            results, a proof, a concrete described mechanism)
  PARTIAL = a passage repeats or weakly supports the claim without independent specifics
  NONE    = no passage supports it (the claim lives only in the abstract)
Step 3 - EVIDENCE TYPE (if FULL/PARTIAL): one of
  data_statistic | table_figure | ablation_analysis | theory_proof | restatement_only

Return JSON: {{"verdict": "FULL|PARTIAL|NONE", "passage_idx": <int or null>, \
"span": "<VERBATIM quote (<=200 chars) from the passage that supports, or empty>", \
"evidence_type": "<type or null>", "reason": "<one sentence>"}}
The span MUST be copied verbatim from a passage. Return ONLY the JSON."""

PRIOR_ART_VERIFY = """A research paper claims the following contribution:

CLAIM: {claim}

CANDIDATE PRIOR WORKS (numbered; each is the title+abstract of an EARLIER paper):
{candidates}

For EACH candidate, judge whether that prior work ANTICIPATES the claim:
  ANTICIPATES = the prior work already does/shows substantially the same thing
  PARTIAL     = substantial overlap, but the claim differs in a key aspect
  DISTINCT    = different problem or approach; does not anticipate

Return JSON: {{"verdicts": [{{"idx": 0, "verdict": "<ANTICIPATES|PARTIAL|DISTINCT>"}}, \
{{"idx": 1, "verdict": "..."}}, ... one entry per candidate, each echoing its idx], \
"best_idx": <idx of the single closest candidate, or null>, \
"reason": "<one sentence on the closest candidate>"}}
Echo each candidate's idx explicitly. Return ONLY the JSON."""

GRADED_VERIFY = """You are verifying whether a claim made in a research paper's ABSTRACT is \
substantiated in the paper's BODY.

CLAIM: {claim}

BODY PASSAGES (numbered):
{passages}

Rate the SUPPORT the passages give the claim on a 0-4 scale:
  0 = no passage bears on the claim at all
  1 = a passage merely restates the claim without evidence
  2 = weak/partial support: some relevant specifics but incomplete
  3 = substantiated: a passage supports the claim with concrete specifics (results, mechanism, proof)
  4 = fully substantiated: the specifics directly match the claim (same numbers, completed proof,
      direct experimental result)

Return JSON: {{"support": <0-4>, "passage_idx": <int or null>, \
"span": "<VERBATIM quote (<=200 chars) from the passage, or empty>", "reason": "<one sentence>"}}
The span MUST be copied verbatim from a passage. Return ONLY the JSON."""

PRIOR_ART_VERIFY_FT = """A research paper claims the following contribution:

CLAIM: {claim}

CANDIDATE PRIOR WORK (an EARLIER paper; title + excerpts from its full text):
TITLE: {title}
EXCERPTS:
{excerpts}

Does this prior work ANTICIPATE the claim?
  ANTICIPATES = the prior work already does/shows substantially the same thing
  PARTIAL     = substantial overlap, but the claim differs in a key aspect
  DISTINCT    = different problem or approach; does not anticipate

Return JSON: {{"verdict": "ANTICIPATES|PARTIAL|DISTINCT", \
"span": "<VERBATIM quote from an excerpt that best evidences your verdict, or empty>", \
"reason": "<one sentence>"}}
Return ONLY the JSON."""

# ---------------- body construction (subtractive) ----------------
EXCLUDE_SECS = {"preamble", "abstract", "references", "reference", "bibliography",
                "acknowledgements", "acknowledgement", "related work", "acknowledgments"}
_REFS_CUT_RE = re.compile(r"\n\s*(references|bibliography)\s*\n", re.I)

def subtractive_body(sections_json, full_text, cap=60000):
    """body = every section EXCEPT preamble/abstract/references/related-work/acks.
    Falls back to stripped full_text when sections are missing/degenerate.
    Returns (body, src) with src in {sections, fulltext_fallback, none}."""
    body = ""
    if sections_json:
        try:
            d = json.loads(sections_json)
            if isinstance(d, dict):
                parts = [v for k, v in d.items()
                         if isinstance(v, str) and k.lower().strip() not in EXCLUDE_SECS]
                body = "\n\n".join(p.strip() for p in parts if p.strip())
        except Exception:
            body = ""
    if len(body) >= 2000:
        return body[:cap], "sections"
    if full_text and len(full_text) > 4000:
        t = full_text
        # cut trailing references (search the last 60% of the doc)
        tail = list(_REFS_CUT_RE.finditer(t))
        for m in reversed(tail):
            if m.start() > 0.4 * len(t):
                t = t[:m.start()]
                break
        # crude head strip: preamble+abstract live in the first ~1500 chars
        return t[1500:cap + 1500], "fulltext_fallback"
    return "", "none"

def paragraphs(body, window=700, max_paras=400):
    """Paragraph-split; PDF text often lacks blank lines, so fall back to fixed windows
    cut at sentence-ish boundaries when the split is degenerate."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if len(p.strip()) > 60]
    if len(paras) >= 8 and sorted(len(p) for p in paras)[len(paras) // 2] >= 150:
        return paras[:max_paras]
    out, i = [], 0
    while i < len(body) and len(out) < max_paras:
        chunk = body[i:i + window]
        if i + window < len(body):
            cut = max(chunk.rfind(". "), chunk.rfind(".\n"))
            if cut > window // 2:
                chunk = chunk[:cut + 1]
        if len(chunk.strip()) > 60:
            out.append(chunk.strip())
        i += max(len(chunk), window // 2)
    return out

def select_passages(claim, paras, k=8, max_chars=700):
    """Top-k passages by lexical containment of the claim (same scorer as the F2 arm)."""
    ct = _toks(claim)
    scored = sorted(((_overlap(ct, _toks(p)), i) for i, p in enumerate(paras)), reverse=True)
    return [paras[i][:max_chars] for _, i in scored[:k]]

# ---------------- planted controls ----------------
_NUM_RE = re.compile(r"\d+\.\d+|\d+")

def perturb_numbers(claim):
    """Deterministically perturb every number in the claim (~+37%, same decimal places).
    Returns the perturbed claim, or None if the claim has no numbers."""
    if not _NUM_RE.search(claim or ""):
        return None
    def rep(m):
        s = m.group(0)
        if "." in s:
            dec = len(s.split(".")[1])
            v = float(s) * 1.37 + 0.11
            out = f"{v:.{dec}f}"
        else:
            v = int(s)
            out = str(v + max(1, int(v * 0.37)))
        return out if out != s else out + "1"
    return _NUM_RE.sub(rep, claim)

def stable_pos(key, k):
    """Deterministic planted-gold position (stable-hash discipline, no RNG)."""
    return int(hashlib.sha1(key.encode()).hexdigest(), 16) % k

# ---------------- parsing + verdict handling ----------------
_OBJ_RE = re.compile(r"\{[\s\S]*\}")

def parse_json(raw):
    m = _OBJ_RE.search(raw or "")
    if not m:
        return None
    for fix in (lambda s: s, lambda s: re.sub(r",\s*([}\]])", r"\1", s)):
        try:
            return json.loads(fix(m.group(0)))
        except Exception:
            continue
    return None

def parse_claims(raw, max_claims=5):
    obj = parse_json(raw) or {}
    out = []
    for c in obj.get("claims", []):
        if isinstance(c, dict) and isinstance(c.get("claim"), str) and len(c["claim"].strip()) > 15:
            out.append({"claim": c["claim"].strip(),
                        "kind": str(c.get("kind", "?")).lower()[:16]})
    return out[:max_claims]

def parse_verify(raw, passages):
    """FULL/PARTIAL/NONE + verbatim-grounding demotion (patents discipline)."""
    obj = parse_json(raw) or {}
    verdict = str(obj.get("verdict", "NONE")).upper()
    if verdict not in ("FULL", "PARTIAL", "NONE"):
        verdict = "NONE"
    span = (obj.get("span") or "").strip()
    grounded = bool(span) and any(span[:120].lower() in p.lower() for p in passages)
    ungrounded = verdict != "NONE" and not grounded
    if ungrounded and verdict == "FULL":
        verdict = "PARTIAL"
    return {"verdict": verdict, "evidence_type": obj.get("evidence_type"),
            "span": span[:200], "grounded": grounded, "ungrounded": ungrounded,
            "reason": (obj.get("reason") or "")[:200], "parsed": bool(obj)}

def parse_graded(raw, passages):
    """0-4 support scale + verbatim-grounding demotion (>=3 without a grounded span -> 2)."""
    obj = parse_json(raw) or {}
    try:
        support = max(0, min(4, int(obj.get("support"))))
    except (TypeError, ValueError):
        support = 0
    span = (obj.get("span") or "").strip()
    grounded = bool(span) and any(span[:120].lower() in p.lower() for p in passages)
    ungrounded = support >= 3 and not grounded
    if ungrounded:
        support = 2
    return {"support": support, "span": span[:200], "grounded": grounded,
            "ungrounded": ungrounded, "reason": (obj.get("reason") or "")[:200],
            "parsed": bool(obj)}


def graded_metrics(rows, prefix="g_"):
    n = len(rows)
    if not n:
        return {prefix + "mean_support": float("nan"), prefix + "frac_ge3": float("nan"),
                prefix + "frac_ge2": float("nan"), prefix + "grounded_rate": float("nan")}
    return {prefix + "mean_support": sum(r["support"] for r in rows) / (4.0 * n),
            prefix + "frac_ge3": sum(r["support"] >= 3 for r in rows) / n,
            prefix + "frac_ge2": sum(r["support"] >= 2 for r in rows) / n,
            prefix + "grounded_rate": sum(r["grounded"] for r in rows) / n}


PA_VERDICTS = ("ANTICIPATES", "PARTIAL", "DISTINCT")


def parse_pa_single(raw):
    """Per-ref verdict (one candidate per call — no array alignment to get wrong)."""
    obj = parse_json(raw) or {}
    v = str(obj.get("verdict", "DISTINCT")).upper()
    return {"verdict": v if v in PA_VERDICTS else "DISTINCT",
            "span": (obj.get("span") or "")[:200],
            "reason": (obj.get("reason") or "")[:200], "parsed": bool(obj)}


def ft_pa_metrics(claim_rows):
    """claim_rows: per-claim {real: [verdicts], self: verdict|None, foreign: verdict|None}."""
    if not claim_rows:
        return {"ft_anticipated_rate": float("nan"), "ft_partial_rate": float("nan"),
                "ft_self_detect": float("nan"), "ft_foreign_distinct": float("nan")}
    ant = sum(1 for r in claim_rows if any(v == "ANTICIPATES" for v in r["real"]))
    part = sum(1 for r in claim_rows
               if not any(v == "ANTICIPATES" for v in r["real"])
               and any(v == "PARTIAL" for v in r["real"]))
    selfs = [r["self"] for r in claim_rows if r.get("self")]
    fors = [r["foreign"] for r in claim_rows if r.get("foreign")]
    n = len(claim_rows)
    return {"ft_anticipated_rate": ant / n, "ft_partial_rate": part / n,
            "ft_self_detect": (sum(v == "ANTICIPATES" for v in selfs) / len(selfs)) if selfs else float("nan"),
            "ft_foreign_distinct": (sum(v == "DISTINCT" for v in fors) / len(fors)) if fors else float("nan")}

def parse_prior_art(raw, k):
    """Echo-indexed verdicts ({"idx": i, "verdict": ...}) keyed by the model's own idx —
    positional arrays proved misaligned in the pilot (self-detect .48 positional vs .86 by the
    judge's own best_idx). Falls back to positional for plain-string arrays."""
    obj = parse_json(raw) or {}
    vs = obj.get("verdicts") or []
    out = ["DISTINCT"] * k
    for i, item in enumerate(vs):
        if isinstance(item, dict):
            try:
                idx = int(item.get("idx"))
            except (TypeError, ValueError):
                continue
            v = str(item.get("verdict", "")).upper()
        else:
            idx, v = i, str(item).upper()
        if 0 <= idx < k and v in PA_VERDICTS:
            out[idx] = v
    return {"verdicts": out, "best_idx": obj.get("best_idx"),
            "reason": (obj.get("reason") or "")[:200], "parsed": bool(obj)}

# ---------------- per-paper aggregation ----------------
def support_metrics(verdicts, prefix="s_"):
    n = len(verdicts)
    if not n:
        return {prefix + "support_rate": float("nan"), prefix + "partial_rate": float("nan"),
                prefix + "none_rate": float("nan"), prefix + "grounded_rate": float("nan")}
    full = sum(1 for v in verdicts if v["verdict"] == "FULL")
    part = sum(1 for v in verdicts if v["verdict"] == "PARTIAL")
    return {prefix + "support_rate": full / n, prefix + "partial_rate": part / n,
            prefix + "none_rate": (n - full - part) / n,
            prefix + "grounded_rate": sum(1 for v in verdicts if v["grounded"]) / n}

def retrieval_metrics(claims, paras, thresh=0.5, prefix="r_"):
    """F2 arm on the full paragraph pool (not just the selected passages)."""
    if not claims or not paras:
        return {prefix + "top1_overlap": float("nan"), prefix + "mean_top1": float("nan"),
                prefix + "support_coverage": float("nan"), prefix + "margin": float("nan")}
    ptoks = [_toks(p) for p in paras]
    top1s, margins = [], []
    for c in claims:
        ct = _toks(c)
        scores = sorted((_overlap(ct, pt) for pt in ptoks), reverse=True)
        top1s.append(scores[0])
        margins.append(scores[0] - (scores[1] if len(scores) > 1 else 0.0))
    return {prefix + "top1_overlap": max(top1s),
            prefix + "mean_top1": sum(top1s) / len(top1s),
            prefix + "support_coverage": sum(s >= thresh for s in top1s) / len(top1s),
            prefix + "margin": sum(margins) / len(margins)}

def prior_art_metrics(pa_rows):
    """pa_rows: per-claim dicts {verdicts, self_idx, foreign_idx}. Planted candidates are
    EXCLUDED from the real novelty readout and scored separately as instrument controls."""
    if not pa_rows:
        return {"pa_anticipated_rate": float("nan"), "pa_partial_rate": float("nan"),
                "pa_self_detect": float("nan"), "pa_foreign_distinct": float("nan")}
    ant, part, self_ok, foreign_ok, self_n, foreign_n = 0, 0, 0, 0, 0, 0
    for r in pa_rows:
        planted = {r.get("self_idx"), r.get("foreign_idx")}
        real = [v for i, v in enumerate(r["verdicts"]) if i not in planted]
        if any(v == "ANTICIPATES" for v in real):
            ant += 1
        elif any(v == "PARTIAL" for v in real):
            part += 1
        if r.get("self_idx") is not None:
            self_n += 1
            self_ok += r["verdicts"][r["self_idx"]] == "ANTICIPATES"
        if r.get("foreign_idx") is not None:
            foreign_n += 1
            foreign_ok += r["verdicts"][r["foreign_idx"]] == "DISTINCT"
    n = len(pa_rows)
    return {"pa_anticipated_rate": ant / n, "pa_partial_rate": part / n,
            "pa_self_detect": (self_ok / self_n) if self_n else float("nan"),
            "pa_foreign_distinct": (foreign_ok / foreign_n) if foreign_n else float("nan")}
