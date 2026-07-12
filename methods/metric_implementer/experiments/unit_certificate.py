"""Certified Unit Framework (CUF) — construction, identification and analysis of Ω units.

Implements notes/2026-07-04__unit-certification-theory.md (Defs 1-9, U1-U5):
a "unit" is a species of addresses whose ablation fingerprint is (U1) detectable against a
sham-ablation null under the context measure H = Φ⊗Λ⊗𝒞, (U2) form-stable in identity,
(U3) context-robust (band charges ε_id/ε_ctx, house ε_form idiom), (U4) minimal (ATOM/COMPOSITE),
(U5) executor-scoped. Two effect arms per user decision D3: target-free δ^free and M_ω-relative δ^M.

Architecture: the statistical core is score_fn-agnostic — `certify_host` takes
    score_fn(prompts: list[str]) -> np.ndarray of shape (len(prompts), n_probes)
so CPU tests run with synthetic executors; the driver wires batched vLLM `score_binary`.
All prompt variants for a host are built first, dedup'd, and scored in ONE score_fn call
(house rule: thousands of prompts per vLLM call).
"""
from __future__ import annotations

import re
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------------
# Def 4 — address lattice (deterministic segmentation; determinism is unit-tested)
# --------------------------------------------------------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")
_CLAUSE_SPLIT = re.compile(r",\s+(?:and|or|but|while|whereas|which|that)\s+|;\s+|\s+—\s+")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


@dataclass
class Node:
    """A lattice node: a candidate address inside the host."""
    node_id: int
    level: int                 # 1 = sentence/item, 2 = clause
    span: str                  # the address text
    parent: Optional[int]      # node_id of the level-1 parent (None for level-1 nodes)
    sent_idx: int              # index of the level-1 segment this node lives in


def _segment_sentences(text: str) -> List[str]:
    """Level-1 segmentation: bullet items if present, else sentence-ish spans."""
    lines = [l for l in text.split("\n") if l.strip()]
    if sum(bool(_BULLET.match(l)) for l in lines) >= 2:      # checklist host
        return [_BULLET.sub("", l).strip() for l in lines if _BULLET.sub("", l).strip()]
    segs = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    return segs


def address_lattice(host: str, depth: int = 2, min_words: int = 3) -> List[Node]:
    """Deterministic segmentation lattice. Level 1 = sentences/items; level 2 = clause splits.
    Nodes shorter than min_words words are not emitted (un-ablatable fragments)."""
    nodes: List[Node] = []
    nid = 0
    for si, sent in enumerate(_segment_sentences(host)):
        if len(sent.split()) >= min_words:
            nodes.append(Node(nid, 1, sent, None, si)); parent_id = nid; nid += 1
        else:
            continue
        if depth >= 2:
            parts = [p.strip(" ,") for p in _CLAUSE_SPLIT.split(sent) if p.strip(" ,")]
            if len(parts) >= 2:
                for p in parts:
                    if len(p.split()) >= min_words:
                        nodes.append(Node(nid, 2, p, parent_id, si)); nid += 1
    return nodes


# --------------------------------------------------------------------------------------------
# Def 4 — ablation operators + inert filler bank (length-matched neutral-replace is PRIMARY)
# --------------------------------------------------------------------------------------------
# Graded-length inert fillers: generic discourse glue with no evaluative/task content.
FILLER_BANK = [
    "as noted",
    "as noted in passing here",
    "as has been noted in passing at this point",
    "as has been noted in passing at this point in the present document",
    "as has been noted in passing at this point in the present document for completeness of the record",
    ("as has been noted in passing at this point in the present document for completeness of the "
     "record and without further elaboration on the matter"),
]


def _match_filler(span: str) -> str:
    """Pick the filler whose word count is closest to the span's (length control, Def 4)."""
    w = len(span.split())
    return min(FILLER_BANK, key=lambda f: abs(len(f.split()) - w))


def ablate(host: str, span: str, mode: str = "neutral") -> str:
    """Return the ablated twin of `host` with `span` removed per `mode`.
    neutral  -> replace with length-matched inert filler (PRIMARY; controls length/position)
    delete   -> remove outright (secondary robustness check)
    """
    if span not in host:
        raise ValueError("span not found verbatim in host")
    if mode == "neutral":
        return host.replace(span, _match_filler(span), 1)
    if mode == "delete":
        out = host.replace(span, "", 1)
        return re.sub(r"\s{2,}", " ", out)
    raise ValueError(mode)


# --------------------------------------------------------------------------------------------
# Def 3 — context measure H = Φ (host forms) ⊗ Λ (slots) ⊗ 𝒞 (company)
# --------------------------------------------------------------------------------------------
_ORBIT_PREFIX = [
    "",                                                        # canonical
    "Please consider the following. ",                          # boilerplate prefix
    "Instructions: ",                                           # header form
]
_ORBIT_SUFFIX = ["", " Consider the above carefully."]


@dataclass
class CtxDraw:
    form_id: int      # index into Φ forms (prefix/suffix combos)
    slot_id: int      # 0=front, 1=native, 2=end  (Λ)
    company_seed: int  # seed for 𝒞 sibling subsample (-1 = full company)


def sample_contexts(n_ctx: int, seed: int = 0, use_company: bool = True) -> List[CtxDraw]:
    """Factorial-ish sample from H (forms × slots × company seeds), deterministic in seed."""
    rng = np.random.default_rng(seed)
    n_forms = len(_ORBIT_PREFIX) * len(_ORBIT_SUFFIX)
    draws = [CtxDraw(0, 1, -1)]                                # canonical draw always included
    while len(draws) < n_ctx:
        draws.append(CtxDraw(int(rng.integers(n_forms)), int(rng.integers(3)),
                             int(rng.integers(10_000)) if use_company else -1))
    return draws[:n_ctx]


def _apply_form(text: str, form_id: int) -> str:
    pre = _ORBIT_PREFIX[form_id % len(_ORBIT_PREFIX)]
    suf = _ORBIT_SUFFIX[form_id // len(_ORBIT_PREFIX) % len(_ORBIT_SUFFIX)]
    return f"{pre}{text}{suf}"


def render_host(host: str, node: Node, nodes: Sequence[Node], ctx: CtxDraw,
                with_address: bool, mode: str = "neutral") -> str:
    """Def 4 installation map ι: rebuild the host under context draw ctx, with the address
    present (with_address=True) or replaced by inert filler. Company subsampling (𝒞) drops a
    random ~30% of OTHER level-1 segments; slot moves (Λ) relocate the address's segment."""
    sents = _segment_sentences(host)
    keep = list(range(len(sents)))
    if ctx.company_seed >= 0 and len(sents) >= 4:
        rng = np.random.default_rng(ctx.company_seed)
        others = [i for i in keep if i != node.sent_idx]
        drop = set(rng.choice(others, size=max(1, len(others) // 3), replace=False).tolist())
        keep = [i for i in keep if i not in drop]
    # the address's own segment, with or without the address span
    seg = sents[node.sent_idx]
    seg_out = seg if with_address else _swap_span(seg, node.span, mode)
    body = [seg_out if i == node.sent_idx else sents[i] for i in keep]
    # Λ: relocate the address segment front/native/end
    if ctx.slot_id != 1 and node.sent_idx in keep:
        pos = body.index(seg_out) if seg_out in body else None
        if pos is not None:
            body.pop(pos)
            body.insert(0 if ctx.slot_id == 0 else len(body), seg_out)
    return _apply_form(" ".join(body), ctx.form_id)


def _swap_span(segment: str, span: str, mode: str) -> str:
    if span == segment:
        return _match_filler(span) if mode == "neutral" else ""
    if span in segment:
        return (segment.replace(span, _match_filler(span), 1) if mode == "neutral"
                else re.sub(r"\s{2,}", " ", segment.replace(span, "", 1)))
    return segment   # segmentation drift under reorder — segment unchanged (conservative)


# --------------------------------------------------------------------------------------------
# Def 6 — sham (inert-edit) ensemble: the U1 null
# --------------------------------------------------------------------------------------------
_INERT_EDITS: List[Tuple[str, str]] = [
    ("  ", " "), (" ,", ","), ("e.g.", "for example"), ("i.e.", "that is"),
    ("--", "—"), (" .", "."), ("...", "…"),
]


def sham_variants(host: str, k: int, seed: int = 0) -> List[str]:
    """k DISTINCT inert edits of the host (typography/glue + varied micro-inserts).
    These are scored exactly like real ablations; their fingerprints seed the null 𝒩."""
    rng = np.random.default_rng(seed)
    outs: List[str] = []
    edits = [e for e in _INERT_EDITS if e[0] in host]
    for a, b in edits[:k]:
        outs.append(host.replace(a, b, 1))
    sents = _segment_sentences(host)
    j = 0
    while len(outs) < k:
        # varied micro-inserts: tiny filler fragment after a rotating sentence boundary
        frag = FILLER_BANK[j % 2] + ("," if j % 3 else " —")
        pos = j % max(len(sents), 1)
        body = sents[: pos + 1] + [frag] + sents[pos + 1:]
        outs.append(" ".join(body))
        j += 1
        _ = rng.integers(1)
    # dedup while preserving order (distinctness matters for null pool size)
    seen, uniq = set(), []
    for o in outs:
        if o not in seen and o != host:
            seen.add(o); uniq.append(o)
    return uniq[:k]


def augment_null(null_fps: List[np.ndarray], n_draws: int, stat_fn, seed: int = 0) -> List[float]:
    """Def 6 estimator note: enlarge the null pool to n_draws via probe-level sign-flip +
    probe-bootstrap of the sham fingerprints (exchangeable under the null), so the attainable
    p floor 1/(n_draws+1) clears the Bonferroni gate α/m (ctree lesson: n_null ≥ max(999, m/α))."""
    rng = np.random.default_rng(seed)
    base = [np.asarray(f, float) for f in null_fps if np.isfinite(np.asarray(f, float)).any()]
    if not base:
        return []
    n = base[0].size
    out = [float(stat_fn(f)) for f in base]
    while len(out) < n_draws:
        f = base[int(rng.integers(len(base)))]
        idx = rng.integers(0, n, size=n)                      # probe bootstrap
        signs = rng.choice([-1.0, 1.0], size=n)               # sign flips
        out.append(float(stat_fn(f[idx] * signs)))
    return out[:n_draws]


# --------------------------------------------------------------------------------------------
# statistics: fingerprints, identity, nulls, charges, verdicts
# --------------------------------------------------------------------------------------------
def fingerprint(sig_with: np.ndarray, sig_without: np.ndarray) -> np.ndarray:
    """Def 5: signed per-probe behavior shift. Inputs (n_ctx, n_probes) or (n_probes,)."""
    d = np.asarray(sig_with, float) - np.asarray(sig_without, float)
    return d.mean(axis=0) if d.ndim == 2 else d


def delta_free(fp: np.ndarray) -> float:
    return float(np.nanmean(np.abs(fp)))


def delta_M(sig_with: np.ndarray, sig_without: np.ndarray, m_bar: np.ndarray) -> float:
    """Def 5 metric-relative arm: mean_H [ corr(with, m̄) − corr(without, m̄) ]."""
    sw = np.atleast_2d(sig_with); so = np.atleast_2d(sig_without)
    vals = []
    for w, o in zip(sw, so):
        vals.append(_safe_corr(w, m_bar) - _safe_corr(o, m_bar))
    return float(np.nanmean(vals))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 3 or np.std(a[ok]) < 1e-9 or np.std(b[ok]) < 1e-9:
        return 0.0
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def identity_corr(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    """Def 7 similarity kernel ρ_id."""
    return _safe_corr(fp_a, fp_b)


def calibrate_r_star(self_corrs: Sequence[float], q: float = 0.05) -> float:
    """Def 7: r* = low quantile of within-paraphrase-orbit self-similarities."""
    v = np.asarray([c for c in self_corrs if np.isfinite(c)], float)
    return float(np.quantile(v, q)) if v.size else 0.5


def perm_p(stat_obs: float, null_stats: Sequence[float], one_sided: bool = False) -> float:
    """Permutation-style p from the sham-null pool (add-one convention)."""
    ns = np.asarray(null_stats, float)
    if one_sided:
        return float((np.sum(ns >= stat_obs) + 1) / (ns.size + 1))
    return float((np.sum(np.abs(ns) >= abs(stat_obs)) + 1) / (ns.size + 1))


def context_stats(per_ctx_deltas: np.ndarray) -> Dict[str, float]:
    """U3 statistics from the per-context effect draws."""
    d = np.asarray(per_ctx_deltas, float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return {"sign_stability": float("nan"), "kappa": float("nan")}
    dom = np.sign(np.nanmean(d)) or 1.0
    return {"sign_stability": float(np.mean(np.sign(d) == dom)),
            "kappa": float(np.var(d) / max(np.mean(d) ** 2, 1e-12))}


def variance_decomposition(deltas: Sequence[float], ctxs: Sequence[CtxDraw]) -> Dict[str, float]:
    """Prop 1: crude ANOVA main-effect shares over H's factors (form / slot / company)."""
    d = np.asarray(deltas, float)
    out = {}
    tot = np.var(d) if np.var(d) > 0 else 1e-12
    for name, key in [("form", lambda c: c.form_id), ("slot", lambda c: c.slot_id),
                      ("company", lambda c: 0 if c.company_seed < 0 else 1)]:
        levels = {}
        for x, c in zip(d, ctxs):
            levels.setdefault(key(c), []).append(x)
        between = np.var([np.mean(v) for v in levels.values()]) if len(levels) > 1 else 0.0
        out[name] = float(between / tot)
    return out


def decide_unit(row: Dict, alpha: float, m: int, beta: float = 0.2, kappa_star: float = 4.0,
                mode: str = "band", delta_min: float = 0.0,
                delta_min_M: Optional[float] = None) -> Dict:
    """Band-mode decision (user D1): ε_id/ε_ctx charges; categorical only at extremes.
    Detection is TWO-gate (pilot lesson, 70B placebo): statistical (p ≤ α/m) AND material
    (δ ≥ δ_min) — a hyper-precise executor makes everything statistically detectable, so
    materiality is part of the declared tuple 𝔗. δ_min_M defaults to δ_min/2 (corr units)."""
    d = row["delta_free"]; ci = row.get("ci_half", 0.0)
    if delta_min_M is None:
        delta_min_M = delta_min / 2.0
    p_gate = alpha / max(m, 1)
    detect_free = (row["p_free"] <= p_gate) and (d >= delta_min)
    detect_M = ((row.get("p_M") is not None) and (row["p_M"] <= p_gate)
                and abs(row.get("delta_M") or 0.0) >= delta_min_M)
    if not detect_free and not detect_M:
        v = "UNDERSAMPLED" if ci > max(d, 1e-9) else "SUBTHRESHOLD"
        return {**row, "verdict": v, "eps_id": 0.0, "eps_ctx": 0.0, "certified_lo": 0.0,
                "detect_free": False, "detect_M": False}
    r_self = row.get("r_self")
    eps_id = 0.0 if r_self is None else float((1.0 - max(r_self, 0.0)) * d)
    kappa = row.get("kappa", 0.0) or 0.0
    eps_ctx = float(min(kappa, kappa_star) / kappa_star * d) if np.isfinite(kappa) else 0.0
    verdict = "CERTIFIED-UNIT"
    if mode == "band":
        if r_self is not None and r_self < 0:
            verdict = "FORM-FRAGILE"
        elif row.get("sign_stability", 1.0) < 1.0 - beta - 0.3:      # extreme sign instability
            verdict = "CONTEXT-CONDITIONAL"
    certified_lo = max(d - eps_id - eps_ctx - ci, 0.0)
    return {**row, "verdict": verdict, "eps_id": eps_id, "eps_ctx": eps_ctx,
            "certified_lo": certified_lo, "detect_free": bool(detect_free),
            "detect_M": bool(detect_M)}


def atom_status(fp_whole: np.ndarray, part_fps: List[np.ndarray], part_detect: List[bool],
                eta: float = 0.5) -> Dict:
    """Def 8: ATOM vs COMPOSITE via additive reconstruction of the whole's fingerprint."""
    if not part_fps:
        return {"atom": "ATOM", "recon_deficiency": None}
    recon = np.nansum(np.stack(part_fps), axis=0)
    denom = np.nansum(np.abs(fp_whole))
    deficiency = float(np.nansum(np.abs(fp_whole - recon)) / max(denom, 1e-12))
    if any(part_detect) and deficiency <= eta:
        return {"atom": "COMPOSITE", "recon_deficiency": deficiency}
    return {"atom": "ATOM", "recon_deficiency": deficiency}


# --------------------------------------------------------------------------------------------
# orchestrator — build every variant, score once, assemble certificate rows
# --------------------------------------------------------------------------------------------
def host_species(fps: Dict[int, np.ndarray], node_ids: Sequence[int], r_star: float = 0.8) -> Dict[int, int]:
    """Within-host Def-7 species merge (Tier-2): single-linkage on fingerprint corr >= r_star.
    Use SOLO fingerprints where available — identity is the unit's intrinsic function, not its
    residual-in-company. Returns node_id -> species_id."""
    ids = [i for i in node_ids if i in fps]
    lab = {i: k for k, i in enumerate(ids)}
    for x in range(len(ids)):
        for y in range(x + 1, len(ids)):
            if identity_corr(fps[ids[x]], fps[ids[y]]) >= r_star:
                a, b = lab[ids[x]], lab[ids[y]]
                for j in lab:                       # union
                    if lab[j] == b:
                        lab[j] = a
    # compact ids
    uniq = {v: k for k, v in enumerate(dict.fromkeys(lab.values()))}
    return {i: uniq[lab[i]] for i in ids}


def certify_host(host: str,
                 score_fn: Callable[[List[str]], np.ndarray],
                 n_ctx: int = 8,
                 n_sham: int = 10,
                 alpha: float = 0.05,
                 depth: int = 2,
                 m_bar: Optional[np.ndarray] = None,
                 paraphrases: Optional[Dict[int, List[str]]] = None,
                 mode: str = "neutral",
                 delta_min: float = 0.0,
                 company_profile: bool = False,
                 r_star_merge: float = 0.8,
                 seed: int = 0) -> Dict:
    """Run U1-U4 for every lattice node of `host` under one executor (Def 1 scope: the caller's
    score_fn IS the executor). Returns {rows, r_star, null, meta}. U5 = run per executor and
    compare fingerprints with identity_corr across runs."""
    nodes = address_lattice(host, depth=depth)
    m = len(nodes)
    ctxs = sample_contexts(n_ctx, seed=seed)

    # ---- build all prompt variants with bookkeeping, dedup, one scoring call ----
    prompts: List[str] = []
    index: Dict[str, int] = {}

    def _add(p: str) -> int:
        if p not in index:
            index[p] = len(prompts); prompts.append(p)
        return index[p]

    node_pairs: Dict[int, List[Tuple[int, int]]] = {}          # node_id -> [(with_i, without_i)]
    for nd in nodes:
        node_pairs[nd.node_id] = [( _add(render_host(host, nd, nodes, c, True, mode)),
                                    _add(render_host(host, nd, nodes, c, False, mode)) )
                                  for c in ctxs]
    sham_pairs: List[Tuple[int, int]] = []
    base_idx = _add(_apply_form(host, 0))
    for sv in sham_variants(host, n_sham, seed=seed):
        sham_pairs.append((_add(sv), base_idx))

    # Tier-2 company profile: SOLO variants (address's segment alone, forms only) + solo shams
    solo_pairs: Dict[int, List[Tuple[int, int]]] = {}
    solo_sham_pairs: List[Tuple[int, int]] = []
    if company_profile:
        sents = _segment_sentences(host)
        n_forms = len(_ORBIT_PREFIX) * len(_ORBIT_SUFFIX)
        for nd in nodes:
            seg = sents[nd.sent_idx]
            pl = []
            for f in range(n_forms):
                pl.append((_add(_apply_form(seg, f)),
                           _add(_apply_form(_swap_span(seg, nd.span, mode), f))))
            solo_pairs[nd.node_id] = pl
        for f in range(n_forms):                      # solo-scale null: filler-vs-filler segments
            for fa, fb in [(0, 2), (1, 3), (2, 4)]:
                solo_sham_pairs.append((_add(_apply_form(FILLER_BANK[fa] + ".", f)),
                                        _add(_apply_form(FILLER_BANK[fb] + ".", f))))
    para_pairs: Dict[int, List[List[Tuple[int, int]]]] = {}
    for nd in nodes:
        for para in (paraphrases or {}).get(nd.node_id, []):
            h2 = host.replace(nd.span, para, 1)
            if h2 == host:
                continue
            nd2 = Node(nd.node_id, nd.level, para, nd.parent, nd.sent_idx)
            para_pairs.setdefault(nd.node_id, []).append(
                [(_add(render_host(h2, nd2, nodes, c, True, mode)),
                  _add(render_host(h2, nd2, nodes, c, False, mode))) for c in ctxs])

    S = np.asarray(score_fn(prompts), float)                    # (n_prompts, n_probes)

    # ---- null ensemble (Def 6): sham fingerprints, augmented to clear the Bonferroni gate ----
    null_fp = [S[i] - S[j] for i, j in sham_pairs]
    n_draws = max(999, int(np.ceil(m / alpha)))
    null_free = augment_null(null_fp, n_draws, delta_free, seed=seed)
    null_M = (augment_null(null_fp, n_draws,
                           lambda f: delta_M((S[base_idx] + f)[None], S[base_idx][None], m_bar),
                           seed=seed + 1)
              if m_bar is not None else None)

    # ---- per-node certificate rows ----
    rows: List[Dict] = []
    fps: Dict[int, np.ndarray] = {}
    for nd in nodes:
        pw = np.stack([S[i] for i, _ in node_pairs[nd.node_id]])
        po = np.stack([S[j] for _, j in node_pairs[nd.node_id]])
        fp = fingerprint(pw, po); fps[nd.node_id] = fp
        per_ctx = [delta_free(w - o) for w, o in zip(pw, po)]
        row = {"node_id": nd.node_id, "level": nd.level, "span": nd.span, "parent": nd.parent,
               "delta_free": delta_free(fp),
               "p_free": perm_p(delta_free(fp), null_free),
               "ci_half": float(1.96 * np.std(per_ctx) / max(np.sqrt(len(per_ctx)), 1)),
               **context_stats(np.asarray(per_ctx)),
               "var_decomp": variance_decomposition(per_ctx, ctxs)}
        if m_bar is not None:
            row["delta_M"] = delta_M(pw, po, m_bar)
            row["p_M"] = perm_p(row["delta_M"], null_M, one_sided=True)
        else:
            row["delta_M"], row["p_M"] = None, None
        # U2: paraphrase-identity self-similarity
        selfc = []
        for pair_list in para_pairs.get(nd.node_id, []):
            fpp = fingerprint(np.stack([S[i] for i, _ in pair_list]),
                              np.stack([S[j] for _, j in pair_list]))
            selfc.append(identity_corr(fp, fpp))
        row["r_self"] = float(np.median(selfc)) if selfc else None
        row["self_corrs"] = selfc
        rows.append(row)

    r_star = calibrate_r_star([c for r in rows for c in r["self_corrs"]]) if paraphrases else None

    # ---- Tier-2: solo effects (company-profile bracket) + within-host species merge ----
    solo_fps: Dict[int, np.ndarray] = {}
    if company_profile:
        null_solo_fp = [S[i] - S[j] for i, j in solo_sham_pairs]
        null_solo = augment_null(null_solo_fp, n_draws, delta_free, seed=seed + 2)
        for r0 in rows:
            pl = solo_pairs[r0["node_id"]]
            sw = np.stack([S[i] for i, _ in pl]); so = np.stack([S[j] for _, j in pl])
            fp_s = fingerprint(sw, so); solo_fps[r0["node_id"]] = fp_s
            r0["delta_free_solo"] = delta_free(fp_s)
            r0["p_free_solo"] = perm_p(r0["delta_free_solo"], null_solo)
            if m_bar is not None:
                r0["delta_M_solo"] = delta_M(sw, so, m_bar)
            r0["delta_bracket"] = [r0["delta_free"], r0["delta_free_solo"]]

    # ---- decisions + U4 atomicity ----
    rows = [decide_unit(r, alpha=alpha, m=m, delta_min=delta_min) for r in rows]
    if company_profile:
        p_gate = alpha / max(m, 1)
        for r0 in rows:
            solo_det = (r0.get("p_free_solo", 1.0) <= p_gate and
                        r0.get("delta_free_solo", 0.0) >= delta_min)
            r0["detect_solo"] = bool(solo_det)
            if r0.get("detect_free") and solo_det:
                r0["company_verdict"] = "UNIT"
            elif solo_det:
                r0["company_verdict"] = "UNIT-IN-COMPANY"     # substitutability: LOO~0, solo real
            elif r0.get("detect_free"):
                r0["company_verdict"] = "LOO-ONLY"            # anomalous; flag for inspection
            else:
                r0["company_verdict"] = "DEAD"                # null at all company levels
        # identity from SOLO fingerprints (intrinsic function)
        sp = host_species(solo_fps, [r0["node_id"] for r0 in rows if r0["level"] == 1],
                          r_star=r_star_merge)
        for r0 in rows:
            r0["species_id"] = sp.get(r0["node_id"])
    by_id = {r["node_id"]: r for r in rows}
    for r in rows:
        if r["level"] == 1:
            kids = [x for x in rows if x["parent"] == r["node_id"]]
            r.update(atom_status(fps[r["node_id"]], [fps[k["node_id"]] for k in kids],
                                 [bool(k.get("detect_free")) for k in kids]))
    return {"rows": rows, "r_star": r_star,
            "fingerprints": {r["node_id"]: fps[r["node_id"]].tolist() for r in rows},
            "null": {"free": null_free, "M": null_M, "n": len(null_free)},
            "meta": {"m_nodes": m, "n_ctx": n_ctx, "n_sham": n_sham, "alpha": alpha,
                     "n_prompts_scored": len(prompts), "mode": mode, "seed": seed}}


def cross_executor_scope(fp_by_exec: Dict[str, np.ndarray], detect_by_exec: Dict[str, bool],
                         ladder: Sequence[str], r_star: float = 0.5) -> str:
    """Def 9 / U5: type a unit along an executor ladder (ordered weak -> strong)."""
    det = [e for e in ladder if detect_by_exec.get(e)]
    if not det:
        return "NOT-DETECTED"
    if len(det) == 1:
        return f"E-SPECIFIC({det[0]})"
    ref = det[-1]

    def _same(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        if np.std(a) < 1e-9 and np.std(b) < 1e-9:              # constant fingerprints:
            return float(np.mean(np.abs(a - b))) < 0.05        # compare levels directly
        return identity_corr(a, b) >= r_star

    matched = all(_same(fp_by_exec[e], fp_by_exec[ref]) for e in det[:-1])
    if len(det) == len(ladder):
        return "E-SHARED" if matched else "E-DRIFT"
    first = ladder.index(det[0])
    if all(detect_by_exec.get(e) for e in ladder[first:]):
        return f"E-EMERGENT(>={det[0]})"
    return f"E-PARTIAL({','.join(det)})"
