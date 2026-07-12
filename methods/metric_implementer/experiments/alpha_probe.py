"""§12.1a ALPHA-PROBE — the minimal runnable front half of `B_E-ATLAS` (theory
`notes/2026-06-18__prompt-optimality-theory.md` §12.1a).

Pipeline: freeze → breadth-sample → collide → estimate the Heaps exponent **α** + coverage →
**GO** (run the full ATLAS, trust saturation-based stopping) / **NO-GO** (`T` is the only global
optimality statement available) / **AMBIGUOUS** (add families or probes, re-run).

What this module IS: the species-accounting layer (B1–B7) — pure numpy/scipy statistics over a
frozen breadth sample of behaviors. Value / active / atlas stages (§12.1 stages 4–6) are out of
scope; this only answers "is the reachable behavior space low-dimensional enough to cover?"

THE §12.1a DISCIPLINE (the flaw NOT to inherit, theory §12.1a D): the breadth stream is **frozen
iid** from a fixed proposal (NO GEPA, NO reflection) and the discovery estimators consume its
``f_j`` spectrum with **rarefaction averaging** — NEVER the GEPA trajectory.
``discovery_scaling.part_a`` accumulates distinct criteria in GEPA-version order, which front-loads
easy/high-value criteria, saturates too fast, and yields **α too small → false "coverable"** (the
dangerous, anti-conservative direction). Every estimator below takes the frozen ``f_j`` only.

Reuse (theory §12.1a B, audited 2026-06-25):
  * soft P(YES) readout        — recon_channel._pyes                       (recon_channel.py:98)
  * breadth-proposal πₖ        — run_real_test._free_generate (no GEPA)    (run_real_test.py:150)
  * CMI behavioral equivalence — orthogonalize.orthogonalization_filter /
                                 shannon_cmi_surrogate                      (orthogonalize.py:91,63)
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.special import gammaln
    _HAVE_SCIPY = True
except Exception:                                             # pragma: no cover
    _HAVE_SCIPY = False


# Prompt-FORM options for the YES/NO readout. The default (rubric-first, in recon_channel) is what the
# rest of the pipeline uses. The TEXT-FIRST form puts the probe text at the token prefix so vLLM
# prefix-caching reuses it across all criteria (efficiency lever #1) — the same 300 texts are prefilled
# ONCE per run instead of once per criterion. B_E/species invariance to this reorder is validated by
# ``prompt_ordering_check`` (the template-level prompt-ordering permutation test).
_YESNO_TEXTFIRST = (
    "Text:\n{text}\n\n"
    "You are evaluating the above text on ONE specific criterion.\n"
    "Criterion:\n{rubric}\n\n"
    "Does the text satisfy the criterion? Answer with exactly one word: YES or NO."
)


# --------------------------------------------------------------------------------------------
# B1 — signature  (σ(p) = soft P(YES) over the frozen probe set)
# --------------------------------------------------------------------------------------------

def signature(executor, prompt: str, probe_texts: Sequence[str], max_chars: int, *,
              template: Optional[str] = None) -> np.ndarray:
    """σ(p) = (P(YES|X_1), …, P(YES|X_n)) ∈ [0,1]^n via the continuous 1-token logprob readout.

    Thin wrap of ``recon_channel._pyes`` (the prompt IS the rubric). NaN marks probe items whose
    readout failed (neither YES nor NO in the top logprobs) — masked downstream. The readout is the
    SOFT one (guard G3): a hard-sampled bit would make ``R<T`` a rounding artifact. ``template``
    overrides the default rubric-first form (pass ``_YESNO_TEXTFIRST`` for the prefix-caching form)."""
    from ..recon_channel import _pyes
    return np.asarray(_pyes(executor, prompt, list(probe_texts), max_chars, template=template),
                      dtype=float)


def metric_verdict(executor, rubric: str, probe_texts: Sequence[str], max_chars: int, *,
                   template: Optional[str] = None) -> np.ndarray:
    """**M_i** — the metric's OWN verdict on the probe items: the executor's soft P(YES) using the metric's
    rubric (the R2 cluster ``merged_description``). This is the **anchor-free reconstruction TARGET** for the
    §12.3 value census — value is each criterion's contribution to recovering M_i, **never the aggregate Y**.
    Mechanically identical to ``signature`` (the rubric IS M_i's prompt); named separately to flag it as the
    value-census target. (Full-strength recovery uses ``recon_channel.run_metric`` → ``iv_transmission``;
    this is the closed-form readout the value census weights criteria against.)"""
    return signature(executor, rubric, probe_texts, max_chars, template=template)


def orbit_metric_verdict(executor, rubric: str, probe_texts: Sequence[str], max_chars: int, *,
                         template: Optional[str] = None, n_forms: int = 4) -> dict:
    """**m̄_ω** — the Φ-orbit-averaged soft metric verdict (theory §12.6.2). Scores the metric rubric under
    the canonical form PLUS up to ``n_forms−1`` deterministic reformulations (``_reformulations``: question /
    boilerplate / suffix / clause-reorder) and averages the soft P(YES) — making the target Φ-invariant BY
    CONSTRUCTION. The average of deterministic-given-X readouts is still deterministic given X, so the DPI
    chain and T = I(m̄_ω; X) survive unchanged (guards G3/GV8 intact).

    Returns ``m_bar`` (the target — use in place of ``metric_verdict``'s single-form readout), the per-form
    matrix, ``var_phi_mean`` (the instrument-error bar to report next to every T/R), and ``flip_rate``
    (mean fraction of probes whose binarized verdict under a single form disagrees with binarized m̄_ω —
    the form-sensitivity of the TARGET itself, the §12.6.2 FORM-DOMINATED signal at the metric level)."""
    forms = [("canonical", str(rubric))] + list(_reformulations(rubric))[: max(0, int(n_forms) - 1)]
    mat = np.vstack([signature(executor, txt, probe_texts, max_chars, template=template)
                     for _, txt in forms])                     # (n_forms, n_probes)
    with np.errstate(invalid="ignore"):
        m_bar = np.nanmean(mat, axis=0)
        var_phi = np.nanvar(mat, axis=0)
    b_bar = (np.nan_to_num(m_bar, nan=0.5) > 0.5).astype(int)
    flips = []
    for r in range(mat.shape[0]):
        row = mat[r]
        mask = np.isfinite(row) & np.isfinite(m_bar)
        if mask.any():
            flips.append(float(((np.nan_to_num(row, nan=0.5) > 0.5).astype(int)[mask]
                                != b_bar[mask]).mean()))
    return {"m_bar": m_bar, "per_form": mat, "form_names": [k for k, _ in forms],
            "var_phi_mean": float(np.nanmean(var_phi)),
            "flip_rate": float(np.mean(flips)) if flips else float("nan"),
            "n_forms": int(mat.shape[0])}


# --------------------------------------------------------------------------------------------
# B3 — noise floor τ  (rescore-based)
# --------------------------------------------------------------------------------------------

def noise_floor_tau(executor, prompt: str, probe_texts: Sequence[str], max_chars: int, *,
                    reps: int = 4, c: float = 2.5, template: Optional[str] = None) -> float:
    """τ₀ = c · mean_i std_r[ P(YES|X_i) over r rescorings of the SAME prompt ].

    The within-prompt readout noise floor: two signatures closer than τ are statistically
    indistinguishable. The vLLM forced-token logprob readout is near-deterministic, so this is
    typically a small floor (batch/numerical jitter); τ still must be ≥ τ₀, and B4 reports α over a
    τ-sweep so the certificate is not brittle to one threshold choice (guard G2)."""
    sigs = np.vstack([signature(executor, prompt, probe_texts, max_chars, template=template)
                      for _ in range(reps)])
    per_item_std = np.nanstd(sigs, axis=0)                    # (n,)
    return float(c * np.nanmean(per_item_std)) if np.isfinite(per_item_std).any() else 1e-9


def signature_diagnostics(sigs: np.ndarray, *, max_pairs: int = 400, seed: int = 0) -> dict:
    """How well does the executor DISCRIMINATE across the breadth sample? Reported alongside α so a
    low α can be correctly ATTRIBUTED: a near-constant executor (tiny between-signature L1, near-0
    per-probe std) collapses every criterion into a few species → α artificially low — an
    executor-capacity artifact, NOT a property of the task's reachable space. A genuinely low-dim
    task shows low α WITH a non-trivial between-sig L1 (criteria are distinct but redundant).

      * ``mean_between_sig_l1`` — mean mean-L1 between signatures (the quantity τ acts on in collide);
        small relative to τ ⇒ collide merges everything ⇒ α is a readout-collapse artifact.
      * ``mean_per_probe_std`` — mean over probes of std-across-criteria (how much each probe item
        separates the criteria; ≈0 ⇒ the probe set itself adds no separation).
      * ``mean_per_sig_std`` / ``frac_near_constant_sig`` — per-criterion spread across probes; a high
        near-constant fraction ⇒ the executor returns the same P(YES) regardless of criterion.
      * ``pyes_global_mean`` / ``pyes_global_std`` — the overall soft-verdict distribution."""
    S = np.asarray(sigs, float)
    with np.errstate(invalid="ignore"):
        per_probe_std = np.nanstd(S, axis=0)
        per_sig_std = np.nanstd(S, axis=1)
    n = S.shape[0]
    idx = np.arange(n)
    if n > max_pairs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_pairs, replace=False)
    sub = S[idx]
    subfin = np.isfinite(sub)
    k = len(sub)
    D = np.zeros((k, k))
    for i in range(k):
        diff = np.abs(sub[i][None, :] - sub)
        mask = subfin[i][None, :] & subfin
        num = np.where(mask, diff, 0.0).sum(axis=1)
        den = mask.sum(axis=1).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            D[i] = np.where(den > 0, num / den, np.inf)
    off = D[~np.eye(k, dtype=bool)]
    off = off[np.isfinite(off)]
    return {
        "mean_between_sig_l1": float(off.mean()) if off.size else float("nan"),
        "median_between_sig_l1": float(np.median(off)) if off.size else float("nan"),
        "mean_per_probe_std": float(np.nanmean(per_probe_std)),
        "mean_per_sig_std": float(np.nanmean(per_sig_std)),
        "frac_near_constant_sig": float(np.mean(per_sig_std < 0.02)),
        "pyes_global_mean": float(np.nanmean(S)),
        "pyes_global_std": float(np.nanstd(S)),
        "n_signatures_sampled_for_l1": int(k),
    }


# --------------------------------------------------------------------------------------------
# B2 — breadth_sample  (frozen iid, the capture stream)  [+ tail-tilted novelty list, R9]
# --------------------------------------------------------------------------------------------

def _novelty_generate(backend, task: str, noun: str, k: int) -> List[str]:
    """The tail-tilted breadth proposer (theory R9 / §12.2.2 "+1"): a FIXED proposal distribution biased
    toward UNUSUAL, easily-overlooked criteria, so its support differs from the vanilla families and it
    captures species their shared blind spot misses — the "one novelty-tilted list ≫ k correlated ones"
    lever, cheaper than cranking N when a pass is AMBIGUOUS. It is STILL frozen iid (``existing=[]``,
    fixed prompt, no failure conditioning), so it is a VALID capture list (enters f_j / the K-way table /
    Δ_div); only the proposal *distribution* is tilted. (Implemented here rather than via a backend-level
    system prompt because ``_free_generate`` overrides ``system``.) Mirrors ``_free_generate``'s parse.

    NOT to be confused with the adaptive residual-hunting depth stream (a proposer CONDITIONED on the
    species found so far) — that is non-iid and must NEVER enter the f_j / coverage accounting."""
    import re as _re
    sys = (f"You propose UNUSUAL, niche, easily-overlooked evaluation criteria for {task} items — the "
           f"kind a careful domain expert notices but a generic evaluator misses.")
    prompt = (f"Task: evaluating {task} items ({noun}).\n\nPropose up to {k} UNCOMMON, NON-OBVIOUS, "
              f"independently-checkable YES/NO criteria — favor edge-case, domain-specific, or "
              f"rarely-stated dimensions over generic quality dimensions. Each one a positive sentence on "
              f"a single dimension the {noun} could be judged on. One criterion per line, no numbering, "
              f"no preamble.")
    raw = backend.generate(prompt, system=sys, max_tokens=500, temperature=0.9)
    lines = [_re.sub(r"^[\-\*\d\.\)\s]+", "", ln).strip() for ln in (raw or "").splitlines()]
    return [ln for ln in lines if len(ln) > 10][:k]


def breadth_sample(families: Dict[str, object], executor, task: str, noun: str,
                   probe_texts: Sequence[str], max_chars: int, *, M: int,
                   per_call: int = 10,
                   novelty_families: Optional[Sequence[str]] = None,
                   ) -> Tuple[np.ndarray, List[str], List[str]]:
    """The frozen breadth sample (theory §12.1a B2/stage-1). Each of K diverse proposer families draws
    criteria via ``run_real_test._free_generate`` with ``existing=[]`` (NO GEPA, NO reflection — guard
    G1: the stream must be iid from a fixed proposal or the f_j statistics are invalid), and the ONE
    frozen executor scores each draw's signature via B1. Returns ``(sigs, family_tags, prompts)``.

    Families named in ``novelty_families`` draw via ``_novelty_generate`` (the tail-tilted "+1" list)
    instead of ``_free_generate``; they are still frozen iid and enter the same accounting, only with a
    proposal distribution tilted toward rarely-stated criteria so their support differs from the vanilla
    families. The adaptive residual-hunting depth stream is a DIFFERENT object and must NOT enter here."""
    from .run_real_test import _free_generate
    nov = set(novelty_families or ())
    K = len(families)
    per_family = max(1, M // K)
    sigs, tags, prompts = [], [], []
    for name, be in families.items():
        drawn = 0
        while drawn < per_family:
            k = min(per_call, per_family - drawn)
            batch = (_novelty_generate(be, task, noun, k) if name in nov
                     else _free_generate(be, task, existing=[], noun=noun, k=k))
            if not batch:
                break
            for p in batch:
                sg = signature(executor, p, probe_texts, max_chars)
                if np.isfinite(sg).any():                 # drop fully-failed readouts
                    sigs.append(sg)
                    tags.append(name)
                    prompts.append(p)
                drawn += 1
    if not sigs:
        raise RuntimeError("breadth_sample produced no valid signatures (all readouts failed?)")
    return np.vstack(sigs), tags, prompts


# --------------------------------------------------------------------------------------------
# B2-metric — breadth_sample METRIC-scoped  (Ω = children + GEPA + within-metric free-gen)
# §12.1a at the METRIC level (per R2 cluster), NOT the task level. See feedback_alpha_probe_is_metric_level.
# --------------------------------------------------------------------------------------------

def _free_generate_metric(backend, task: str, metric_name: str, metric_desc: str,
                          noun: str, k: int) -> List[str]:
    """Within-metric free-gen: novel ATOMIC criteria for ONE metric, seeded by its description — the
    frozen-iid capture stream for the metric-scoped α-probe. Mirrors run_real_test._free_generate's parse
    but restricts the proposal to criteria that test THIS metric (not other metrics)."""
    import re as _re
    sys = ("You propose NOVEL, independently-checkable atomic YES/NO criteria for ONE specific "
           "evaluation metric — fine-grained checks that each probe one aspect of that metric.")
    prompt = (f"Task: evaluating {task} items ({noun}).\n\nThe metric is:\n  {metric_name}"
              f"{(' — ' + metric_desc) if metric_desc else ''}\n\nPropose up to {k} NOVEL, DISTINCT, "
              f"atomic YES/NO criteria that each test ONE fine-grained, checkable aspect of THIS metric "
              f"({metric_name}) — NOT criteria for other metrics. Each a single positive sentence naming a "
              f"concrete check on the {noun}. One criterion per line, no numbering, no preamble.")
    raw = backend.generate(prompt, system=sys, max_tokens=500, temperature=0.7)
    lines = [_re.sub(r"^[\-\*\d\.\)\s]+", "", ln).strip() for ln in (raw or "").splitlines()]
    return [ln for ln in lines if len(ln) > 10][:k]


def breadth_sample_metric(families: Dict[str, object], executor, task: str, noun: str,
                          metric_name: str, metric_desc: str,
                          children: Sequence[str], gepa_criteria: Sequence[str],
                          probe_texts: Sequence[str], max_chars: int, *,
                          M_freegen: int, per_call: int = 10,
                          template: Optional[str] = None,
                          ) -> Tuple[np.ndarray, List[str], List[str]]:
    """The METRIC-scoped breadth sample (§12.1a B2 at the per-metric level). Ω for ONE metric (R2
    cluster) = its atomic units/criteria from three sources, all scored by the ONE frozen executor over
    the TASK probe items (a metric is a function on task items — read its behavior space there):

      * ``children``  — the cluster's own leaves (r2_children); the curated atomic backbone.
      * ``gepa_criteria`` — criteria mined from the metric's GEPA-optimized prompt versions (source (a)),
        INCLUDED when a registry artifact exists for the metric (deferred otherwise — GEPA Phase A is
        GLM-gated). Per [[feedback_alpha_probe_is_metric_level]] GEPA is a named Ω source.
      * within-metric free-gen from the K diverse proposer families — the frozen-iid capture stream
        (the only source that licenses the coverage bounds; children/GEPA enrich Ω but are not iid).

    Family tags = ``children`` / ``gepa`` / proposer-name, so multi-list capture-recapture and Δ_div
    still read cleanly. Returns ``(sigs, family_tags, prompts)`` — same shape as ``breadth_sample``, so
    the rest of the pipeline (collide → spectrum → rarefaction → α) is UNCHANGED."""
    sigs, tags, prompts = [], [], []

    def _add(crit: str, tag: str):
        if not crit or len(str(crit)) < 5:
            return
        sg = signature(executor, crit, probe_texts, max_chars, template=template)
        if np.isfinite(sg).any():
            sigs.append(sg); tags.append(tag); prompts.append(crit)

    for c in children:                                # (c) the cluster's atomic backbone
        _add(c, "children")
    for c in gepa_criteria:                           # (a) GEPA-mined (if a registry artifact exists)
        _add(c, "gepa")
    K = len(families)
    per_family = max(1, M_freegen // K) if K else 0   # (d) within-metric free-gen, frozen iid
    for name, be in families.items():
        drawn = 0
        while drawn < per_family:
            k = min(per_call, per_family - drawn)
            batch = _free_generate_metric(be, task, metric_name, metric_desc, noun, k)
            if not batch:
                break
            for p in batch:
                _add(p, name)
                drawn += 1
    if not sigs:
        raise RuntimeError("breadth_sample_metric produced no valid signatures "
                           "(all readouts failed? empty Ω?)")
    return np.vstack(sigs), tags, prompts


# --------------------------------------------------------------------------------------------
# B4 — collide  (single-linkage species clustering + CMI cross-check)
# --------------------------------------------------------------------------------------------

def _pairwise_mean_l1(sigs: np.ndarray) -> np.ndarray:
    """Pairwise mean |σ_i − σ_j| over CO-FINITE probe items (NaN positions masked per pair). n² but
    n is the breadth-prompt count (hundreds), so fine."""
    S = np.asarray(sigs, float)
    n = S.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        fi = np.isfinite(S[i])
        diff = np.abs(S[i][None, :] - S)                       # (n, nprobes)
        mask = fi[None, :] & np.isfinite(S)
        num = np.where(mask, diff, 0.0).sum(axis=1)
        den = mask.sum(axis=1).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            D[i] = np.where(den > 0, num / den, np.inf)
    return D


def _union_find(n: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = [find(i) for i in range(n)]
    relabel = {r: k for k, r in enumerate(sorted(set(roots)))}
    return np.array([relabel[r] for r in roots])


def collide(sigs: np.ndarray, tau: float) -> np.ndarray:
    """Single-linkage species labels: p ~ p' iff mean |σ_p − σ_p'| ≤ τ over co-finite probes.

    Connected components of the within-τ graph (NaN-masked distance). Conservative in the species
    sense — any real behavioral difference (on a deterministic logprob readout) splits species, so a
    small τ yields MORE species, which is the anti-"false-coverable" direction (guard for §D)."""
    D = _pairwise_mean_l1(sigs)
    edges = [(i, j) for i in range(D.shape[0]) for j in range(i + 1, D.shape[1]) if D[i, j] <= tau]
    return _union_find(D.shape[0], edges)


def cmi_agreement(sigs: np.ndarray, labels: np.ndarray, *, seed: int = 0,
                  max_pairs: int = 600) -> dict:
    """The CMI cross-check (theory §12.1a A / §B): verify single-linkage species agree with the
    behavioral-equivalence test. Binarize σ at 0.5 (the recovery-relevant partition) and measure,
    via ``shannon_cmi_surrogate``, the mean WITHIN-species MI (members of one species induce the SAME
    partition ⇒ each perfectly predicts the other ⇒ HIGH MI) vs the mean BETWEEN-species MI
    (different species ⇒ ~independent ⇒ LOW MI). A behaviorally meaningful cut has within ≫ between;
    ``coherent`` flags that.

    SCALING: ``shannon_cmi_surrogate`` is a 5-fold LogisticRegression OOF fit (×2 per pair), so the
    full n² loop is infeasible at breadth scale (n=1200 ⇒ 720k pairs ⇒ many hours). We SUBSAMPLE up
    to ``max_pairs`` within- and between-species pairs — the within/between MEAN (the only thing the
    ``coherent`` flag reads) is well-estimated from a few hundred pairs. Within-pairs are usually few
    anyway (singleton-heavy samples ⇒ few multi-member species), so all available within-pairs are
    kept (capped); between-pairs are randomly sampled. Reported counts let you see the subsample."""
    from collections import defaultdict
    from ..experiments.orthogonalize import shannon_cmi_surrogate
    labels = np.asarray(labels)
    n = len(labels)
    rng = np.random.default_rng(seed)
    B = ((np.asarray(sigs, float) > 0.5).astype(int)).T          # (n_probes, n_prompts) cols = prompts

    groups: Dict[int, List[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        groups[int(lab)].append(i)
    within_idx: List[Tuple[int, int]] = []
    for members in groups.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                within_idx.append((members[a], members[b]))
    if len(within_idx) > max_pairs:                              # cap (rare: one huge species)
        sel = rng.choice(len(within_idx), max_pairs, replace=False)
        within_idx = [within_idx[i] for i in sel]
    between_idx: List[Tuple[int, int]] = []
    seen, guard = set(), 0
    while len(between_idx) < max_pairs and guard < 50 * max_pairs and len(groups) >= 2:
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        guard += 1
        if a == b:
            continue
        i, j = (a, b) if a < b else (b, a)
        if (i, j) in seen:
            continue
        seen.add((i, j))
        if labels[i] != labels[j]:
            between_idx.append((i, j))

    within = [float(shannon_cmi_surrogate(B[:, a], np.empty((B.shape[0], 0)),
                                          x_new=B[:, b], seed=seed))
              for a, b in within_idx]
    between = [float(shannon_cmi_surrogate(B[:, a], np.empty((B.shape[0], 0)),
                                           x_new=B[:, b], seed=seed))
               for a, b in between_idx]
    mw = float(np.mean(within)) if within else float("nan")
    mb = float(np.mean(between)) if between else float("nan")
    return {"within_species_cmi": mw, "between_species_cmi": mb,
            "n_within_pairs": len(within), "n_between_pairs": len(between),
            "coherent": bool(np.isfinite(mw) and np.isfinite(mb) and mw > mb)}


# --------------------------------------------------------------------------------------------
# B5 — spectrum  (f_j + the K-way capture contingency)
# --------------------------------------------------------------------------------------------

def spectrum(species_labels: Sequence[int], family_tags: Sequence) -> dict:
    """``f_j`` = # species seen exactly j times (j=1,2,…); the K-way capture table records, for each
    non-empty family-subset S, ``n_S`` = # species captured by exactly that subset of families. This
    table is the input to the Fienberg log-linear coverage bound (B6)."""
    sp_to_count: Dict[int, int] = Counter(int(x) for x in species_labels)
    f: Dict[int, int] = Counter(sp_to_count.values())          # multiplicity j -> # species
    sp_to_fams: Dict[int, set] = {}
    for sp, fam in zip((int(x) for x in species_labels), family_tags):
        sp_to_fams.setdefault(sp, set()).add(str(fam))
    capture_table: Dict[frozenset, int] = Counter()
    for sp, fams in sp_to_fams.items():
        capture_table[frozenset(fams)] += 1
    return {"f": dict(f), "N": len(species_labels), "D": len(sp_to_count),
            "capture_table": {",".join(sorted(k)): v for k, v in capture_table.items()},
            "sp_to_fams": {sp: sorted(fs) for sp, fs in sp_to_fams.items()},
            "families": sorted({str(x) for x in family_tags})}


# --------------------------------------------------------------------------------------------
# B6 — estimators  (~150 LOC pure numpy/scipy species accounting)
# --------------------------------------------------------------------------------------------

def good_turing_coverage(f: Dict[int, int], N: int) -> float:
    """Ĉ = 1 − f_1/N. Independence-OPTIMISTIC point estimate (Good–Turing). For intuition only —
    positive capture correlation over-claims saturation, so this is NOT the certificate."""
    return 1.0 - f.get(1, 0) / max(N, 1)


def missing_mass_bound(f: Dict[int, int], N: int, *, delta: float = 0.05) -> dict:
    """M̂₀ = f_1/N with the Berend–Kontorovich concentration ``± sqrt(log(1/δ)/N)`` (theory §12.1a A
    line 1550). This is an ASSUMPTION-FREE, distribution-free upper bound on missing mass — valid for
    ANY dependence between families — which is why the coverage lower bound built on it (B6
    ``coverage_interval``) cannot be invalidated by shared-LM-bias correlation."""
    m0 = f.get(1, 0) / max(N, 1)
    eps = float(np.sqrt(np.log(1.0 / delta) / max(N, 1)))
    return {"m0_hat": m0, "ci_eps": eps, "m0_ci_hi": min(m0 + eps, 1.0)}


def chao1(f: Dict[int, int], D: int) -> float:
    """Chao1 richness: D + f_1²/(2 f_2) (f_2>0), else D + f_1(f_1−1)/2 (the singleton-only branch).
    A LOWER bound on true richness (underestimates when many rare species) → an OPTIMISTIC coverage
    denominator; reported as the rarity diagnostic, NOT the conservative certificate."""
    f1, f2 = f.get(1, 0), f.get(2, 0)
    if f2 > 0:
        return D + f1 * f1 / (2.0 * f2)
    return D + f1 * (f1 - 1) / 2.0


def _rarefy(f: Dict[int, int], N: int, m: int) -> float:
    """E[S(m)] = D − Σ_j f_j · C(N−j, m)/C(N, m), the de-noised (rarefied) expected species count in
    a size-m subsample. Computed via log-gamma for numerical stability (C(300,150) ≈ 1e89)."""
    D = sum(f.values())
    if m > N:
        return float(D)
    lg = gammaln
    log_CN_m = lg(N + 1) - lg(m + 1) - lg(N - m + 1)            # log C(N, m)
    deduct = 0.0
    for j, fj in f.items():
        if N - j < m or fj == 0:
            continue
        log_ratio = (lg(N - j + 1) - lg(m + 1) - lg(N - j - m + 1)) - log_CN_m
        deduct += fj * np.exp(log_ratio)
    return float(D - deduct)


def rarefaction(f: Dict[int, int], N: int, *, n_points: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    """The rarefaction curve E[S(m)] for m = 1..N, evaluated at ``n_points`` log-spaced points. This
    is the DE-NOISED discovery curve that ``heaps_alpha`` differentiates — NEVER the raw GEPA-version
    accumulation (theory §D), which is anti-conservative."""
    ms = np.unique(np.linspace(1, N, min(n_points, N)).astype(int))
    S = np.array([_rarefy(f, N, int(m)) for m in ms])
    return ms, S


def heaps_alpha(ms: np.ndarray, S: np.ndarray) -> np.ndarray:
    """α(m) = d log E[S(m)] / d log m — the LOCAL Heaps exponent, returned as a TRAJECTORY (not a
    single number). The terminal slope ``α(N)`` is the manifold-dimension estimate and the go/no-go
    signal: α ≪ 1 ⇒ low-dim, fast saturation (coverable); α ~ 1 ⇒ inexhaustible."""
    ms = np.asarray(ms, float)
    S = np.maximum(np.asarray(S, float), 1e-9)
    return np.gradient(np.log(S), np.log(ms))


def diversity_gap(pooled_f: Dict[int, int], pooled_N: int,
                  per_family_f: Dict[str, Dict[int, int]], per_family_N: Dict[str, int],
                  *, M: Optional[int] = None) -> dict:
    """Δ_div = D_pooled|_M − max_k D_family_k|_M: how many MORE species the pooled families have than
    the richest single family, both rarefied DOWN to M = min family draw-count for fairness. A large,
    still-growing Δ_div ⇒ the families are finding DIFFERENT species ⇒ keep adding families; closing
    ⇒ saturation. If any family has < 2 draws, Δ_div is undefined (NaN).

    NOTE: the pooled spectrum ``pooled_f`` MUST be computed from the GLOBAL species labels (species
    identity preserved), NOT by summing per-family multiplicity spectra — a species seen 6× in family
    A and a species seen 6× in family B are different species, and summing multiplicities would
    invent phantom species. The caller passes the global ``f``/``N``."""
    Ns = [n for n in per_family_N.values() if n > 0]
    if len(Ns) < 2 or min(Ns) < 2:
        return {"delta_div": float("nan"), "M": None, "pooled_at_M": float("nan"),
                "max_family_at_M": float("nan"), "reason": "need ≥2 families with ≥2 draws"}
    M = M or min(Ns)
    fam_at_M = [_rarefy(per_family_f[k], per_family_N[k], M) for k in per_family_f]
    pool_at_M = _rarefy(pooled_f, pooled_N, M)
    return {"delta_div": float(pool_at_M - max(fam_at_M)), "M": int(M),
            "pooled_at_M": float(pool_at_M), "max_family_at_M": float(max(fam_at_M))}


def coverage_interval(f: Dict[int, int], N: int, *, delta: float = 0.05) -> dict:
    """Coverage interval [C_lo, C_hat].

    * ``C_hat`` = Good–Turing (independence-optimistic; intuition only).
    * ``C_lo``  = 1 − (f_1/N + sqrt(log(1/δ)/N)) — the **assumption-free** lower bound on coverage,
      built on the Berend–Kontorovich missing-mass concentration. It is valid under ANY
      family-dependence (shared LM biases cannot invalidate a distribution-free bound), which is why
      it is the reported CERTIFICATE, not C_hat.

    Multi-list tightening (theory §12.1a C): a Fienberg log-linear fit on the K-way capture table
    would give a dependence-CORRECTED n_∅; the all-zero cell is not point-identified without an
    interaction-strength bound (max positive interaction inflates it without limit), so the
    assumption-free C_lo is the safe choice and the log-linear value is a reported sensitivity point
    (``coverage_pairwise_petersen``) rather than the certificate."""
    mm = missing_mass_bound(f, N, delta=delta)
    return {"C_hat": float(good_turing_coverage(f, N)),
            "C_lo": float(max(0.0, 1.0 - mm["m0_ci_hi"])),
            "m0_hat": mm["m0_hat"], "m0_ci_eps": mm["ci_eps"], "delta": delta}


def coverage_pairwise_petersen(spec: dict, *, positive_only: bool = True) -> float:
    """A dependence-aware coverage SENSITIVITY point from the K-way capture table: average the
    2-list Lincoln–Petersen coverage D·a/(n_1·n_2) over all family pairs. Positive list-correlation
    (shared biases) biases Petersen DOWNWARD in N (over-claims coverage); ``positive_only`` keeps
    only pairs whose observed overlap a ≥ independence, flagging where dependence bites. Reported for
    intuition alongside the assumption-free certificate — NOT the headline bound."""
    sp_fams = spec.get("sp_to_fams", {})
    if not sp_fams:
        return float("nan")
    fams = sorted({fam for fs in sp_fams.values() for fam in fs})
    if len(fams) < 2:
        return float("nan")
    D = spec["D"]
    vals = []
    for i in range(len(fams)):
        for j in range(i + 1, len(fams)):
            fi, fj = fams[i], fams[j]
            ni = sum(1 for fs in sp_fams.values() if fi in fs)          # list-i capture count
            nj = sum(1 for fs in sp_fams.values() if fj in fs)
            a = sum(1 for fs in sp_fams.values() if fi in fs and fj in fs)  # both-list overlap
            if ni == 0 or nj == 0 or a == 0:
                continue
            a_ind = ni * nj / D                                          # independence-expected overlap
            if positive_only and a < a_ind - 1e-9:
                continue                                                 # not positively associated
            c = D * a / (ni * nj)                                        # Petersen coverage (optimistic)
            vals.append(c)
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------------------------
# B6+ — Fienberg multi-list coverage (log-linear on the K-way capture table)
# --------------------------------------------------------------------------------------------

def _poisson_glm_fit(X: np.ndarray, y: np.ndarray, *, n_iter: int = 300,
                     ridge: float = 1e-8) -> np.ndarray:
    """Poisson log-linear GLM via Newton (IRLS), pure numpy. Returns β (log link). The 1-token readout
    of log-linear CRC: fit on the observed cells, predict the all-empty cell from exp(β_0)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    _, p = X.shape
    beta = np.zeros(p)
    beta[0] = np.log(max(y.mean(), 1e-3))                       # intercept init at the mean count
    for _ in range(n_iter):
        eta = np.clip(X @ beta, -30, 30)
        mu = np.exp(eta)
        XtWX = X.T @ (X * mu[:, None]) + ridge * np.eye(p)
        grad = X.T @ (y - mu)
        try:
            step = np.linalg.solve(XtWX, grad)
        except np.linalg.LinAlgError:                           # saturated/degenerate → ridge lstsq
            step = np.linalg.lstsq(XtWX, grad, rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta


def coverage_fienberg(spec: dict) -> dict:
    """The Fienberg multi-list capture-recapture coverage POINT estimates from the K-way capture table
    (theory §12.1a C). Fit a Poisson log-linear model to the ``2^K − 1`` OBSERVED capture-pattern cells
    (the all-empty cell is unobserved and the prediction target) and read the empty-cell count as
    ``exp(β_0)``; coverage = ``D / (D + n_empty)``.

      * **independence** (main effects) — the multi-list generalization of Lincoln–Petersen; the tight
        multi-list POINT estimate, but optimistic under positive list-correlation (shared LM biases
        inflate the observed overlap ⇒ under-estimates n_empty).
      * **pairwise-interactions** (all 2-way) — the dependence-CORRECTED estimate: absorbing observed
        positive co-capture into the pairwise terms raises n_empty ⇒ lower, more conservative coverage.
        Identified only when the table has residual df: ``2^K − 1 > 1 + K + C(K,2)`` ⇔ **K ≥ 4**; for
        K ≤ 3 the pairwise model is saturated and the empty cell is unidentifiable (returned as NaN).

    HONEST CAVEAT (why this is a POINT, not the §C certificate): the §12.1a-C "maximum positive
    interaction consistent with observed margins" bound is **unbounded** — the pairwise interaction
    can be driven to infinity, inflating n_empty without limit, because the all-empty cell is not
    point-identified without an external interaction-strength assumption. So there is NO finite
    Fréchet/LP lower bound on coverage; the MLE above is the tightest FINITE dependence-aware estimate,
    and the assumption-free ``coverage_interval`` ``C_lo`` remains the only bound VALID under arbitrary
    dependence. Reported as a sensitivity ladder, not the certificate."""
    from itertools import combinations
    sp_fams = spec.get("sp_to_fams", {})
    K = len(spec.get("families") or [])
    res = {"K": K, "C_independence": float("nan"), "C_pairwise": float("nan"),
           "n_empty_independence": float("nan"), "n_empty_pairwise": float("nan"),
           "pairwise_identified": False}
    if K < 2 or not sp_fams:
        res["reason"] = "need ≥2 lists"
        return res
    fam_idx = {f: i for i, f in enumerate(sorted(spec["families"]))}
    ct: Dict[tuple, float] = {}
    for fs in sp_fams.values():
        idxs = tuple(sorted(fam_idx[f] for f in fs if f in fam_idx))
        ct[idxs] = ct.get(idxs, 0.0) + 1.0
    patterns = [combo for r in range(1, K + 1) for combo in combinations(range(K), r)]
    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    D = spec["D"]

    def row(combo, with_pairs):
        x = [1.0] + [1.0 if k in combo else 0.0 for k in range(K)]
        if with_pairs:
            x += [1.0 if (i in combo and j in combo) else 0.0 for (i, j) in pairs]
        return x

    y = np.array([ct.get(c, 0.0) for c in patterns], float)
    X_ind = np.array([row(c, False) for c in patterns])
    b_ind = _poisson_glm_fit(X_ind, y)
    n_empty_ind = float(np.exp(np.clip(b_ind[0], -30, 30)))
    res["n_empty_independence"] = n_empty_ind
    res["C_independence"] = float(D / (D + n_empty_ind)) if (D + n_empty_ind) > 0 else float("nan")

    # pairwise model is identified only with residual degrees of freedom (K ≥ 4)
    n_params_pair = 1 + K + len(pairs)
    if len(patterns) > n_params_pair:
        X_pair = np.array([row(c, True) for c in patterns])
        b_pair = _poisson_glm_fit(X_pair, y)
        n_empty_pair = float(np.exp(np.clip(b_pair[0], -30, 30)))
        res["n_empty_pairwise"] = n_empty_pair
        res["C_pairwise"] = float(D / (D + n_empty_pair)) if (D + n_empty_pair) > 0 else float("nan")
        res["pairwise_identified"] = True
    else:
        res["reason"] = (f"pairwise model saturated ({len(patterns)} cells = {n_params_pair} params); "
                         f"need K≥4 to identify the empty cell under pairwise interactions")
    return res


# --------------------------------------------------------------------------------------------
# B4c — CONDITIONAL (non-overlapping) species + capture-recapture upper/lower bounds
# --------------------------------------------------------------------------------------------
# The f(x+omega)=f(y+omega) species definition: a candidate is the SAME species as an existing core
# unit iff that unit's verdict already explains >= cmi_thresh of the candidate's entropy. Greedy, so
# the partition is NON-OVERLAPPING by construction — paraphrases collapse regardless of marginal L1,
# which is what makes B_E identifiable (marginal `collide` leaves ~95% singletons -> Chao1 diverges;
# conditional species give few singletons -> finite Chao1, Chao1≈Lincoln-Petersen, ~90% coverage).
# Capture-recapture bounds: B_E (richness) = UPPER bound on the reachable space; D_obs = LOWER bound
# (definitely-seen); coverage = D_obs/B_E spans them. ORDER + FORM permutation is a STANDARD validity
# gate (order_stability, form_invariance) — see feedback to bake permutation testing into the procedure.

def _h_bits(p: float) -> float:
    p = float(p)
    return 0.0 if p <= 0 or p >= 1 else float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def _i_binary(x, y) -> float:
    """Exact Shannon MI I(x; y) in bits for two binary vectors — self-contained (avoids the
    value_census circular import). 0 if either vector is constant."""
    x = np.asarray(x, int); y = np.asarray(y, int); n = len(x)
    if n == 0 or len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
        return 0.0
    hy = _h_bits(float(y.mean())); h = 0.0
    for xv in (0, 1):
        m = x == xv
        if m.any():
            h += float(m.mean()) * _h_bits(float(y[m].mean()))
    return max(0.0, hy - h)


def conditional_species(sigs: np.ndarray, *, cmi_thresh: float = 0.15,
                        order: Optional[Sequence[int]] = None) -> np.ndarray:
    """Non-overlapping species labels via conditional behavioral equivalence. Candidate e joins an
    existing core unit k iff I(X_k; X_e) ≥ cmi_thresh·H(X_e) (exact binary MI); otherwise it founds a
    NEW species. Greedy over `order` (default: most-balanced first). Returns int labels (-1 = constant
    criterion dropped). This is the dedup that stops paraphrase inflation; the headline B_E prefers the
    non-circular `two_list_crc` (cores built on disjoint iid halves), with this as the within-sample read.
    """
    S = np.asarray(sigs, float); n = len(S)
    B = (np.nan_to_num(S, nan=0.5) > 0.5).astype(int)
    h = np.array([_h_bits(float(B[e].mean())) for e in range(n)])
    if order is None:
        order = sorted(range(n), key=lambda e: (abs(float(B[e].mean()) - 0.5), e))
    core: List[int] = []; labels = np.full(n, -1, int)
    for e in order:
        if h[e] < 1e-9:
            continue
        if not core:
            core.append(e); labels[e] = 0; continue
        mis = np.array([_i_binary(B[c], B[e]) for c in core])
        if mis.size and mis.max() >= cmi_thresh * h[e]:
            labels[e] = int(mis.argmax())
        else:
            labels[e] = len(core); core.append(e)
    return labels


def _core_of(idx, B: np.ndarray, h: np.ndarray, th: float) -> List[int]:
    c: List[int] = []
    for e in idx:
        if h[e] < 1e-9:
            continue
        if not c:
            c.append(e); continue
        mx = max((_i_binary(B[k], B[e]) for k in c), default=0.0)
        if mx < th * h[e]:
            c.append(e)
    return c


def two_list_crc(sigs: np.ndarray, list_a: Sequence[int], list_b: Sequence[int], *,
                 cmi_thresh: float = 0.15) -> dict:
    """Two-list held-out Lincoln-Petersen B_E — the NON-circular capture-recapture estimate. Build
    species cores on iid half A, recapture on iid half B (independent streams, so species identity is
    not defined by the data being counted). Chapman-corrected for small m. Returns B_E (upper bound),
    D_obs (lower bound), coverage, and the recapture match count. Agreement with within-sample Chao1
    is the validation that saturation is real, not a clustering artifact."""
    S = np.asarray(sigs, float)
    B = (np.nan_to_num(S, nan=0.5) > 0.5).astype(int)
    h = np.array([_h_bits(float(B[e].mean())) for e in range(len(B))])
    th = cmi_thresh
    cA = _core_of(list_a, B, h, th); cB = _core_of(list_b, B, h, th)
    if not cA or not cB:
        return {"B_E": float("nan"), "coverage": float("nan"), "D_A": 0, "D_B": 0,
                "matched": 0, "D_obs": 0}
    used = set(); m = 0
    for a in cA:
        mis = np.array([_i_binary(B[a], B[b]) for b in cB])
        for ob in np.argsort(-mis):
            if mis[ob] >= th * h[a] and ob not in used:
                m += 1; used.add(int(ob)); break
    DA, DB = len(cA), len(cB); Dobs = DA + DB - m
    chap = (DA + 1) * (DB + 1) / (m + 1) - 1
    return {"B_E": float(chap), "coverage": float(Dobs / chap) if chap > 0 else float("nan"),
            "D_A": DA, "D_B": DB, "matched": m, "D_obs": Dobs}


def crc_bootstrap(sigs: np.ndarray, iid_idx: Sequence[int], *, cmi_thresh: float = 0.15,
                  n_splits: int = 15, seed: int = 0) -> dict:
    """Error bars on two_list_crc B_E by resampling the A/B split `n_splits` times. The within-metric
    B_E std vs the cross-metric spread is the test of whether metrics genuinely differ (a noise/signal
    ratio < 1 means the heterogeneity is real, not small-sample LP variance)."""
    rng = np.random.default_rng(seed); iid = list(iid_idx); bes = []; covs = []
    for _ in range(n_splits):
        rng.shuffle(iid); A = iid[: len(iid) // 2]; B = iid[len(iid) // 2:]
        r = two_list_crc(sigs, A, B, cmi_thresh=cmi_thresh)
        if np.isfinite(r["B_E"]):
            bes.append(r["B_E"]); covs.append(r["coverage"])
    if not bes:
        return {"B_E_mean": float("nan"), "B_E_std": float("nan"),
                "coverage_mean": float("nan"), "n_splits_ok": 0}
    return {"B_E_mean": float(np.mean(bes)), "B_E_std": float(np.std(bes)),
            "coverage_mean": float(np.mean(covs)), "n_splits_ok": len(bes)}


# --------------------------------------------------------------------------------------------
# B4q — QUOTIENT species (§12.6.2: semantic-judge MERGE / behavioral SPLIT / judge never splits)
# --------------------------------------------------------------------------------------------

def _shingle_jaccard(a: str, b: str, *, k: int = 3) -> float:
    """Word-shingle Jaccard similarity of two criterion strings (k-word shingles; word-set fallback for
    short strings). EMBEDDING-FREE lexical similarity — used ONLY to nominate judge candidates (the
    beh-DIFF & sem-SAME cell has low MI, so MI alone cannot nominate it), NEVER to merge."""
    import re as _re

    def sh(t):
        w = _re.findall(r"[a-z0-9']+", str(t).lower())
        return set(w) if len(w) < k else {" ".join(w[i:i + k]) for i in range(len(w) - k + 1)}

    A, B = sh(a), sh(b)
    return len(A & B) / len(A | B) if A and B else 0.0


def make_glm_semantic_judge(model: str = "glm-4.7", *, backend: str = "zai_anthropic",
                            max_tokens: int = 5):
    """Build a SAME/DIFFERENT pair judge (the `semantic_behavioral` prompt) as a
    ``judge_fn(pairs) -> List[bool]`` for ``quotient_species``. **Be sparing — GLM quota is binding**:
    the caller prunes candidates hard (``max_judge_pairs``) before this is ever invoked; one batched
    call per quotient. Lazy imports so offline tests never touch backends."""
    from .. import backends as _backends, config as _cfgmod
    cfg = _cfgmod.ImplementerConfig(backend=backend)
    be = _backends.LLMBackend(model, "quotient_judge", cfg)
    sysp = ("You decide whether two evaluation criteria test the same thing. Consider only whether they "
            "evaluate the SAME dimension/aspect of the item, ignoring wording style or specificity level.")

    def judge_fn(pairs):
        import re as _re
        qs = [f"Criterion A: {a}\n\nCriterion B: {b}\n\nDo A and B evaluate the SAME aspect of the item "
              f"(one is a paraphrase of the other), or DIFFERENT aspects? Reply with one word: SAME or "
              f"DIFFERENT." for a, b in pairs]
        outs = be.generate_batch(qs, system=sysp, max_tokens=max_tokens, temperature=0.0, seed=0)
        return [bool(_re.search(r"\bSAME\b", (o or "").strip().upper()))
                and not _re.search(r"\bDIFFERENT\b", (o or "").strip().upper()) for o in outs]

    return judge_fn


def quotient_species(sigs: np.ndarray, prompts: Optional[Sequence[str]] = None, *,
                     judge_fn=None, cmi_thresh: float = 0.15, hi_nmi: float = 0.5,
                     lexical_jaccard: float = 0.45, max_judge_pairs: int = 300,
                     order: Optional[Sequence[int]] = None, seed: int = 0
                     ) -> Tuple[np.ndarray, dict]:
    """The §12.6.2 QUOTIENT partition: **semantic-judge MERGE / behavioral SPLIT / judge never splits.**

    1. **Strict behavioral partition** — ``conditional_species`` at identity-level ``hi_nmi`` (merge only
       when a core already explains ≥ hi_nmi of the candidate's entropy). This is the SPLIT side: it
       fixes the over-merge failure of the 0.15-significance partition (merge-precision 0.28 — a
       significance test is not an identity test).
    2. **Judge-nominated MERGES across strict species** — candidate pairs are (a) the MI *gray zone*
       (significant at ``cmi_thresh`` but below ``hi_nmi``: behaviorally related, not identical) and
       (b) *lexically similar* rep prompts (``_shingle_jaccard`` ≥ ``lexical_jaccard`` — catches the
       form-fragile paraphrases whose MI is LOW, the beh-DIFF & sem-SAME cell). Ranked (gray zone by
       NMI desc, then lexical by Jaccard desc), capped at ``max_judge_pairs`` (quota), judged
       SAME/DIFFERENT in ONE batch; SAME ⇒ union-find merge of the two species. A DIFFERENT verdict
       never splits a behavioral collision (asymmetry: over-split is flux-SAFE, over-merge is not —
       §12.6.2), and with ``judge_fn=None`` no merges happen and the audit reports the unresolved mass.

    Returns ``(labels, audit)``; labels shaped like ``conditional_species``'s (−1 = dropped constant).
    Population statistics (spectrum → f_j → flux w_j) should consume THESE labels; the head of the
    §12.6 certificate is partition-free and unaffected."""
    S = np.asarray(sigs, float)
    n = len(S)
    labels = conditional_species(S, cmi_thresh=hi_nmi, order=order).copy()
    B = (np.nan_to_num(S, nan=0.5) > 0.5).astype(int)
    h = np.array([_h_bits(float(B[e].mean())) for e in range(n)])

    members: Dict[int, List[int]] = {}
    for i, l in enumerate(labels):
        if int(l) >= 0:
            members.setdefault(int(l), []).append(i)
    species = sorted(members)
    n_strict = len(species)
    audit = {"n_strict": n_strict, "hi_nmi": float(hi_nmi), "cmi_thresh": float(cmi_thresh),
             "lexical_jaccard": float(lexical_jaccard), "n_candidates_gray": 0,
             "n_candidates_lex": 0, "n_judged": 0, "n_merged": 0, "n_unresolved": 0,
             "n_quotient": n_strict}
    if n_strict < 2:
        return labels, audit

    # rep per species = highest-entropy member (most informative verdict pattern)
    rep = {s: max(ms, key=lambda i: h[i]) for s, ms in members.items()}
    gray, lex = [], []
    for a_i in range(len(species)):
        for b_i in range(a_i + 1, len(species)):
            sa, sb = species[a_i], species[b_i]
            ra, rb = rep[sa], rep[sb]
            hmin = min(h[ra], h[rb])
            nmi = (_i_binary(B[ra], B[rb]) / hmin) if hmin > 1e-9 else 0.0
            if cmi_thresh <= nmi < hi_nmi:
                gray.append((nmi, sa, sb))
            elif prompts is not None and nmi < cmi_thresh:
                # k=2 shingles: nomination is RECALL-oriented (the judge decides, wrong candidates only
                # cost quota) and word-order paraphrases shred trigrams — the exact cell this path catches
                jac = _shingle_jaccard(prompts[ra], prompts[rb], k=2)
                if jac >= lexical_jaccard:
                    lex.append((jac, sa, sb))
    gray.sort(key=lambda t: -t[0])
    lex.sort(key=lambda t: -t[0])
    cands = [(sa, sb) for _, sa, sb in gray] + [(sa, sb) for _, sa, sb in lex]
    audit["n_candidates_gray"], audit["n_candidates_lex"] = len(gray), len(lex)
    judged = cands[: max_judge_pairs]
    audit["n_unresolved"] = max(0, len(cands) - len(judged))
    if judge_fn is None or not judged or prompts is None:
        audit["n_unresolved"] = len(cands)
        return labels, audit

    same = judge_fn([(str(prompts[rep[sa]]), str(prompts[rep[sb]])) for sa, sb in judged])
    audit["n_judged"] = len(judged)
    parent = {s: s for s in species}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (sa, sb), sm in zip(judged, same):
        if sm:
            ra_, rb_ = find(sa), find(sb)
            if ra_ != rb_:
                parent[ra_] = rb_
                audit["n_merged"] += 1
    remap = {}
    for s in species:
        r = find(s)
        remap.setdefault(r, len(remap))
    out = labels.copy()
    for i, l in enumerate(labels):
        if int(l) >= 0:
            out[i] = remap[find(int(l))]
    audit["n_quotient"] = len(remap)
    return out, audit


def order_stability(sigs: np.ndarray, family_tags: Sequence, *, cmi_thresh: float = 0.15,
                    n_orders: int = 8, delta: float = 0.05, seed: int = 0) -> dict:
    """STANDARD validity gate (theory §6.8): permute the greedy core-build (candidate) order
    `n_orders` times and report mean±std of the within-sample CRC estimates. CRC reads multiplicities,
    not which criterion represents a species, so the estimates must be order-invariant even though core
    identities shuffle. Pass (std ≪ mean) => the conditional partition is a population property and the
    f_j spectrum is trustworthy. This is the permutation test on the ORDER of elements in Ω."""
    S = np.asarray(sigs, float); n = len(S); rng = np.random.default_rng(seed)
    Ds, f1s, chas, cls, f1ns = [], [], [], [], []
    for _ in range(n_orders):
        lab = conditional_species(S, cmi_thresh=cmi_thresh, order=list(rng.permutation(n)))
        v = lab >= 0
        sp = spectrum(lab[v], [family_tags[i] for i in range(n) if v[i]])
        f, N, D = sp["f"], sp["N"], sp["D"]
        Ds.append(D); f1s.append(f.get(1, 0)); chas.append(chao1(f, D))
        cls.append(coverage_interval(f, N, delta=delta)["C_lo"])
        f1ns.append(f.get(1, 0) / max(N, 1))
    return {"D_mean": float(np.mean(Ds)), "D_std": float(np.std(Ds)),
            "D_min": float(np.min(Ds)), "D_max": float(np.max(Ds)),
            "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
            "f1_over_N_mean": float(np.mean(f1ns)),
            "chao1_mean": float(np.mean(chas)), "chao1_std": float(np.std(chas)),
            "chao1_min": float(np.min(chas)), "chao1_max": float(np.max(chas)),
            "C_lo_mean": float(np.mean(cls)), "C_lo_std": float(np.std(cls)),
            "n_orders": n_orders}


def _reformulations(prompt: str):
    """Deterministic prompt-FORM reformulations (no API): isolate how phrasing / format / boilerplate
    affect the executor readout. Each should preserve meaning; only surface form changes."""
    import re as _re
    p = str(prompt).strip()
    outs = [("question", f"Does the item meet this criterion? {p} Answer YES or NO."),
            ("boilerplate", "You are an expert evaluator. Evaluate strictly. " + p),
            ("suffix", p + " Respond with YES if satisfied, otherwise NO.")]
    parts = [s.strip() for s in _re.split(r"[,;]|\band\b|\bor\b", p) if len(s.strip()) > 3]
    if len(parts) >= 2:                                   # clause reorder (multi-clause criteria only)
        outs.append(("reorder", ". ".join(parts[::-1])))
    return outs


def form_invariance(executor, prompts: Sequence[str], probe_texts: Sequence[str], max_chars: int, *,
                    tau0: float, max_n: int = 40, seed: int = 0,
                    sigs: Optional[np.ndarray] = None, template: Optional[str] = None) -> dict:
    """STANDARD validity test: is σ_c robust to HOW c is phrased/formatted in the prompt? For a sample
    of criteria, build deterministic reformulations (question / boilerplate prefix / clause reorder /
    suffix instruction), re-score each, and measure signature drift mean|σ − σ'|. Drift ≈ τ₀ ⇒
    form-invariant (the species partition is NOT built on prompt-formulation noise); drift ≫ τ₀ ⇒ form
    is a confound and B_E is unstable to rewording. `tau0` is the within-prompt readout noise floor
    (``noise_floor_tau``); a form-invariant metric has median drift ≲ 2·τ₀."""
    rng = np.random.default_rng(seed)
    idx = list(rng.choice(len(prompts), size=min(max_n, len(prompts)), replace=False))
    drifts, flips = [], []
    for i in idx:
        # #6: the canonical signature was already scored in breadth_sample — reuse it instead of a
        # redundant rescore (the reformulations below are the only NEW scores). `sigs` must have been
        # built with the same `template` so the drift comparison is apples-to-apples.
        sig = (np.asarray(sigs[i], float) if sigs is not None
               else np.asarray(signature(executor, prompts[i], probe_texts, max_chars,
                                         template=template), float))
        for _kind, txt in _reformulations(prompts[i]):
            sig2 = np.asarray(signature(executor, txt, probe_texts, max_chars, template=template),
                              float)
            mask = np.isfinite(sig) & np.isfinite(sig2)
            if mask.any():
                drifts.append(float(np.abs(sig[mask] - sig2[mask]).mean()))
                b1 = (np.nan_to_num(sig, nan=0.5) > 0.5).astype(int)
                b2 = (np.nan_to_num(sig2, nan=0.5) > 0.5).astype(int)
                flips.append(float((b1[mask] != b2[mask]).mean()))   # fraction of probes whose verdict flips
    d = np.asarray(drifts) if drifts else np.asarray([float("nan")])
    fl = np.asarray(flips) if flips else np.asarray([float("nan")])
    return {"median_drift": float(np.median(d)), "mean_drift": float(np.mean(d)),
            "p95_drift": float(np.percentile(d, 95)), "frac_over_2tau0": float(np.mean(d > 2 * tau0)),
            "binary_flip_rate": float(np.median(fl)),     # the species-relevant metric: how often >0.5 flips
            "n_pairs": int(len(drifts)), "tau0": float(tau0),
            "form_invariant": bool(np.nanmedian(fl) <= 0.10)}   # <=10% probe flips = roughly form-robust


def conditional_crc_report(sigs: np.ndarray, family_tags: Sequence, *, cmi_thresh: float = 0.15,
                           delta: float = 0.05, n_orders: int = 8, n_splits: int = 15,
                           seed: int = 0) -> dict:
    """The standard per-metric capture-recapture certificate (upper + lower bounds + validity gates),
    all CPU post-hoc on a frozen signature checkpoint:

      * within-sample: Chao1 + Good–Turing C_lo on the conditional partition (order_stability-checked);
      * two-list held-out: Lincoln-Petersen B_E ± bootstrap error bar (the non-circular upper bound);
      * lower bound D_obs (observed non-overlapping species); coverage = D_obs/B_E.
    Agreement of Chao1 and B_E is the saturation check; order_stability is the Ω-order permutation
    gate. (form_invariance is run separately in the driver — it needs the live executor.)"""
    S = np.asarray(sigs, float); tags = list(family_tags)
    iid = [i for i, t in enumerate(tags) if str(t) not in ("children", "gepa")]
    ost = order_stability(S, tags, cmi_thresh=cmi_thresh, n_orders=n_orders, delta=delta, seed=seed)
    if len(iid) >= 40:
        boot = crc_bootstrap(S, iid, cmi_thresh=cmi_thresh, n_splits=n_splits, seed=seed)
        A = iid[: len(iid) // 2]; B = iid[len(iid) // 2:]
        lp = two_list_crc(S, A, B, cmi_thresh=cmi_thresh)
    else:
        boot = {"B_E_mean": float("nan"), "B_E_std": float("nan"), "coverage_mean": float("nan"),
                "n_splits_ok": 0}
        lp = {"B_E": float("nan"), "coverage": float("nan"), "D_obs": ost["D_mean"],
              "matched": 0, "D_A": 0, "D_B": 0}
    return {"cmi_thresh": float(cmi_thresh),
            "B_E_upper": boot["B_E_mean"], "B_E_upper_std": boot["B_E_std"],
            "chao1_within": ost["chao1_mean"], "D_obs_lower": ost["D_mean"],
            "coverage": boot["coverage_mean"],
            "f1_over_N": ost["f1_over_N_mean"],               # Lemma 12.6.0 gate input
            "lp_single": lp,
            "order_stability": ost,
            "iid_n": len(iid),
            "two_method_agree": bool(np.isfinite(boot["B_E_mean"]) and np.isfinite(ost["chao1_mean"])
                                     and abs(boot["B_E_mean"] - ost["chao1_mean"])
                                     <= 0.5 * max(boot["B_E_mean"], ost["chao1_mean"], 1e-9))}


# --------------------------------------------------------------------------------------------
# B7 — orchestrator + decision rule
# --------------------------------------------------------------------------------------------

def _terminal_alpha(ms: np.ndarray, alpha_traj: np.ndarray, *, frac: float = 0.25) -> float:
    """Mean of the last ``frac`` of the α(m) trajectory — the terminal slope the GO/NO-GO rule reads."""
    if len(alpha_traj) == 0:
        return float("nan")
    k = max(1, int(len(alpha_traj) * frac))
    return float(np.mean(alpha_traj[-k:]))


def decide_census(report: dict, *, eps: float = 0.05, alpha_go: float = 0.3) -> str:
    """The ORIGINAL §12.1a-F count-axis rule — retained as the **behavior-census / vocabulary read**,
    NO LONGER the depth verdict (superseded by ``decide``, theory §12.6.6; count exponents are pinned
    at 1 mechanically when the spectrum is singleton-dominated — Lemma 12.6.0):
      GO      : terminal α < ~0.3  AND  C_lo ≥ 1−ε  AND  D/Chao1 → 1  AND  Δ_div closing
      NO-GO   : α ~ 1 (not decreasing)  OR  Δ_div large & still growing
      AMBIGUOUS: otherwise (add families/probes, re-run)."""
    a = report["alpha_terminal"]
    c_lo = report["coverage"]["C_lo"]
    chao = report["chao1"]
    d = report["D"]
    ddiv = report["diversity_gap"]["delta_div"]
    sat_chao = (np.isfinite(chao) and chao > 0 and d / chao >= 1 - eps)
    if (np.isfinite(ddiv) and not np.isnan(ddiv)) and ddiv > 0.15 * max(d, 1):
        return "NO-GO"                                          # families still finding new species
    if np.isfinite(a) and a >= 0.9:                            # α ~ 1, not decreasing
        return "NO-GO"
    if (np.isfinite(a) and a < alpha_go and c_lo >= 1 - eps and sat_chao
            and (np.isnan(ddiv) or ddiv <= 0.15 * max(d, 1))):
        return "GO"
    return "AMBIGUOUS"


def eps_form_bits(form_invariance: Optional[dict], certificate: Optional[dict]) -> dict:
    """Form-fragility as a NON-NEGATIVE bits charge that WIDENS the census ceiling in the residual
    `Δ(E) = lowerCI(C_dense) − [OPT_Ω + ε + ε_form]` (the 2026-07-03 gate redesign: fragility is
    quantified against the ceiling, NOT used to eject the metric via a categorical verdict).
    Deliberately kept OUT of the CODIFIABLE gate — that gate reads the census ε only; form
    stability is an orthogonal reported axis.

    EXACT (theory §1): `ε_form := max_form OPT_Ω(form) − OPT_Ω(canonical)` — the extra census value
    a form-robust rephrasing could surface. Needs a per-form re-census (GPU: reformulated
    signatures), delivered when the certificate carries ``opt_omega_by_form`` {form: bits}.
    PROXY (CPU, when only the cheap drift summary exists): `q · OPT_Ω`, linear in the median
    per-probe flip rate `q` — a proportionate, clearly-labeled placeholder, NOT the certified
    number. Returns {bits, source, q}; ``source`` ∈ {exact, proxy, none} so callers never mistake
    a proxy for the real quantity."""
    opt0 = (certificate or {}).get("opt_omega_bits")
    opt0 = float(opt0) if opt0 is not None and np.isfinite(opt0) else None
    by_form = (certificate or {}).get("opt_omega_by_form")
    if isinstance(by_form, dict) and by_form and opt0 is not None:
        best = max(v for v in by_form.values() if np.isfinite(v))
        return {"bits": float(max(0.0, best - opt0)), "source": "exact", "q": None}
    q = (form_invariance or {}).get("binary_flip_rate")
    if form_invariance is None or q is None or not np.isfinite(q) or q <= 0 or opt0 is None:
        return {"bits": 0.0, "source": "none", "q": None}
    q = float(min(q, 0.5))
    return {"bits": q * opt0, "source": "proxy", "q": q}


def decide(report: dict, *, form_invariance: Optional[dict] = None,
           certificate: Optional[dict] = None, f1_hi: float = 0.8, eps0_frac: float = 0.05,
           deep_tail_frac: float = 0.25, deep_hill: float = 1.0,
           form_mode: str = "verdict") -> str:
    """The §12.6.6 verdict rule (supersedes the count-axis GO/NO-GO for the value/depth question).

    The f₁/N gate is ASYMMETRIC by design (§12.6.2/§12.6.6): over-split fake singletons carry
    v|S_g ≈ 0, so they can only INFLATE ε — an ε-small certificate is therefore conservative and
    CODIFIABLE may be issued from inside the singleton regime. DEEP may NOT: in an undersampled pool
    the greedy gains are spread thin because the head criteria haven't arrived yet, not because value
    is inherently spread — the theory table requires ``f₁/N ≪ 1`` for DEEP. Lemma 12.6.0 kills the
    EXPONENT readout at f₁/N → 1, never the value certificate.

      FORM-DOMINATED   [``form_mode='verdict'`` — original behavior] form gates fail
                       (``form_invariance['form_invariant'] is False``): meaning unstable under
                       rephrasing — a linguistic finding; certificate valid only at the adverse
                       orbit end, which this function cannot verify — so it preempts everything.
                       [``form_mode='band'`` — 2026-07-03 redesign] NOT a verdict: the metric keeps
                       its census verdict (CODIFIABLE/DEEP/…) and the form boolean survives only as
                       a reported diagnostic; fragility is quantified separately as ``ε_form`` bits
                       (``eps_form_bits``) that widen the ceiling in the residual Δ, NOT in this
                       categorical gate (form stability is an orthogonal axis). Keeps every metric
                       in the value analysis instead of ejecting it.
      with certificate (from ``value_certificate.certificate``):
        CODIFIABLE     a validated upper-bound payload ∧ ε ≤ ``eps0_frac``·H(M) ∧ explicit
                       adversarial-saturation pass ∧ head-concentrated
                       (tail_frac < ``deep_tail_frac``): a SHORT rubric certified within ε of the
                       process-horizon optimum. Issued at ANY f₁/N (see above).
        DEEP           greedy value tail heavy (tail_frac ≥ ``deep_tail_frac`` or Hill ξ ≥ ``deep_hill``)
                       ∧ f₁/N < ``f1_hi``: measured inexhaustible VALUE depth — claimable only with a
                       recapturing stream and a passing planted positive control alongside (§12.6.7).
        UNDERSAMPLED   neither verdict certifiable and the cure is more frozen-iid draws: either the
                       singleton regime blocks DEEP (f₁/N ≥ ``f1_hi``) or ε has not resolved
                       (ε > ``eps0_frac``·H — Ĝ+slack still shrink with N).
        INDETERMINATE  otherwise (e.g. ε small but adversarial saturation FAILED with a light tail —
                       synergy escape; or non-finite certificate fields).
      without certificate:
        UNDERSAMPLED   f₁/N ≥ ``f1_hi`` — exponents pinned at 1 MECHANICALLY (Lemma 12.6.0) and no
                       value instrument supplied; fix the quotient / draw more first.
        NEEDS-CERTIFICATE  otherwise — run ``experiments.value_certificate`` (the census alone cannot
                       issue a depth verdict).

    Count-α / Chao1 / C_lo never enter — they remain descriptive (``decide_census``)."""
    if form_mode not in ("verdict", "band"):
        raise ValueError(f"form_mode must be 'verdict' or 'band', got {form_mode!r}")
    f1n = report.get("f1_over_N")
    f1_high = f1n is not None and np.isfinite(f1n) and f1n >= f1_hi
    if (form_mode == "verdict" and form_invariance is not None
            and form_invariance.get("form_invariant") is False):
        return "FORM-DOMINATED"
    if certificate is None:
        return "UNDERSAMPLED" if f1_high else "NEEDS-CERTIFICATE"
    H = certificate.get("H_M")
    H = float(H) if H is not None and np.isfinite(H) else 1.0
    # ε is read at the ORDER-ADVERSE end when the certificate carries the permutation band
    # (§12.6.6: partition-dependent quantities are reported at their adverse end, like the orbit)
    eps_b = certificate.get("eps_bits_adv", certificate.get("eps_bits"))
    eps_b = float(eps_b) if eps_b is not None else float("nan")
    tail = certificate.get("tail_frac")
    tail = float(tail) if tail is not None else float("nan")
    hill = certificate.get("hill")
    hill = float(hill) if hill is not None else float("nan")
    adv = certificate.get("adv_saturated")
    adv_ok = adv is True
    bound_ok = certificate.get("upper_bound_valid") is True
    eps_small = np.isfinite(eps_b) and eps_b <= eps0_frac * max(H, 1e-9)
    tail_heavy = (np.isfinite(tail) and tail >= deep_tail_frac) or (np.isfinite(hill) and hill >= deep_hill)
    if bound_ok and eps_small and adv_ok and np.isfinite(tail) and tail < deep_tail_frac:
        return "CODIFIABLE"                     # conservative at any f₁/N: over-split only inflates ε
    if tail_heavy:
        return "DEEP" if not f1_high else "UNDERSAMPLED"   # DEEP needs a recapturing stream (f₁/N ≪ 1)
    if np.isfinite(eps_b) and not eps_small:
        return "UNDERSAMPLED"                   # ε unresolved: Ĝ + slack still shrink with more draws
    return "INDETERMINATE"


def alpha_probe(sigs: np.ndarray, family_tags: Sequence, *, tau: float,
                delta: float = 0.05, cmi_seed: int = 0, compute_cmi: bool = True) -> dict:
    """Run the full front-half pipeline on a frozen breadth sample and return the report + decision.

    Inputs: ``sigs`` (n_prompts, n_probes) soft signatures from B2's frozen stream;
    ``family_tags`` (n_prompts,) source family per prompt; ``tau`` the collide threshold (B3). Stages
    4–6 (value annotation, active bracket, full atlas) are NOT run — this only estimates α + coverage
    for the GO/NO-GO call.

    ``compute_cmi``: the CMI cross-check is the only O(n²)-with-model-fit stage (see
    ``cmi_agreement``); it is τ-irrelevant as a DIAGNOSTIC, so the driver's τ-sweep passes
    ``compute_cmi=False`` to skip recomputing it at every τ (the main call computes it once)."""
    sigs = np.asarray(sigs, float)
    labels = collide(sigs, tau)
    spec = spectrum(labels, family_tags)
    f, N, D = spec["f"], spec["N"], spec["D"]
    ms, S = rarefaction(f, N)
    alpha_traj = heaps_alpha(ms, S)
    per_family_f: Dict[str, Dict[int, int]] = {}
    per_family_N: Dict[str, int] = {}
    for sp, fam in zip((int(x) for x in labels), family_tags):
        per_family_N[str(fam)] = per_family_N.get(str(fam), 0) + 1
    for fam in per_family_N:
        labs = [int(x) for x, ff in zip(labels, family_tags) if str(ff) == fam]
        cc = Counter(labs)
        per_family_f[fam] = dict(Counter(cc.values()))
    report = {
        "N": N, "D": D, "tau": float(tau),
        "alpha_trajectory": [float(x) for x in alpha_traj],
        "alpha_terminal": _terminal_alpha(ms, alpha_traj),
        "f1_over_N": float(f.get(1, 0) / max(N, 1)),        # Lemma 12.6.0: ≈1 ⇒ α, α_V pinned mechanically
        "rarefaction_m": [int(x) for x in ms],
        "rarefaction_S": [float(x) for x in S],
        "coverage": coverage_interval(f, N, delta=delta),
        "coverage_fienberg": coverage_fienberg(spec),
        "coverage_pairwise_petersen": coverage_pairwise_petersen(spec),
        "chao1": float(chao1(f, D)),
        "diversity_gap": diversity_gap(f, N, per_family_f, per_family_N),
        "cmi_check": (cmi_agreement(sigs, labels, seed=cmi_seed) if compute_cmi else None),
        "capture_table": spec["capture_table"],
        "families": spec["families"],
        "f_spectrum": {str(k): v for k, v in f.items()},
        "signature_diagnostics": signature_diagnostics(sigs),
    }
    report["decision_census"] = decide_census(report)        # the count-axis vocabulary read (descriptive)
    report["decision"] = decide(report)                      # §12.6.6 (UNDERSAMPLED / NEEDS-CERTIFICATE here)
    return report
