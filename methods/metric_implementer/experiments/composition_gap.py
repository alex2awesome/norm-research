"""Δ_comp — the COMPOSITION-GAP arm (§12.6.4 composition escape; user-approved 2026-07-01).

The §12.6 certificate bounds CHECKLIST-articulability: the best selection/weighting of
separately-executed criteria under a named combiner F. A single COMPOSED prompt (the criteria
stated together, in an order, a phrasing, possibly a persona frame) is **not a function of the unit
verdicts** — the executor's joint reading is a different channel, no DPI applies, and prompt-space
performance is not monotone in criteria. This module makes that escape a MEASURED quantity instead
of a hidden assumption:

  Δ_comp_total  = max over composed variants of I(exec(composed); M) − OPT_Ω
                  (can be negative — interference/dilution is real)
  Δ_comp_beyond = max over variants of I(exec(composed); M | S_g)
                  (the value the composed channel adds BEYOND the certified head — the sharp read)

Interpretation discipline: Δ_comp ≈ 0 ⇒ the unit model is empirically adequate for this metric;
Δ_comp > 0 on taste metrics with ≈ 0 on craft metrics ⇒ part of the tacit content lives in the
SAYING, not the said — an anthropological finding, not an instrument failure. Report per
metric × tier; never fold Δ_comp into ε (they bound different channels).

This module is CPU/offline: it BUILDS composed-prompt texts and MEASURES verdict columns; executing
the texts is the driver's job (same executor, same probes, same frozen setup as the certificate).
``holistic_probe_prompts`` feeds ``orthogonalize.adversarial_saturation`` so the saturation
backstop covers the composition escape too (GEPA-optimized whole prompts are the strongest
attacker — wire ``gepa_pr`` outputs in via ``extra_prompts``)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .value_census import i_binary
from .value_certificate import _cmi_block


# --------------------------------------------------------------------------------------------
# composed-prompt builders (deterministic, no API — execution happens in the driver)
# --------------------------------------------------------------------------------------------

def compose_checklist_prompts(criteria: Sequence[str], name: str = "", *,
                              seed: int = 0) -> List[Dict[str, str]]:
    """Deterministic composed variants of the head criteria: orders × phrasings × ±persona frame.
    Six variants — enough to expose order/phrasing/gestalt effects without exploding the grid.
    Returns [{"id": …, "text": …}, …]; the driver executes each text per probe (YES/NO verdict)."""
    crits = [str(c).strip().rstrip(".") for c in criteria if str(c).strip()]
    rng = np.random.default_rng(seed)
    shuf = list(rng.permutation(len(crits)))
    orders = {"given": list(range(len(crits))), "reversed": list(range(len(crits)))[::-1],
              "shuffled": shuf}

    def numbered(ix):
        return "\n".join(f"{k + 1}. {crits[i]}." for k, i in enumerate(ix))

    def prose(ix):
        return " ".join(f"{crits[i]}." for i in ix)

    head = (f"Evaluate the item on '{name}'. " if name else "Evaluate the item. ")
    ask = "\nDoes the item satisfy the criteria above, taken together? Answer YES or NO."
    persona = ("You are a seasoned editor with deep experience of this craft; judge as that "
               "editor would, weighing the criteria holistically rather than mechanically.\n")
    out = [
        {"id": "numbered_given", "text": head + "Criteria:\n" + numbered(orders["given"]) + ask},
        {"id": "numbered_reversed", "text": head + "Criteria:\n" + numbered(orders["reversed"]) + ask},
        {"id": "prose_given", "text": head + prose(orders["given"]) + ask},
        {"id": "prose_shuffled", "text": head + prose(orders["shuffled"]) + ask},
        {"id": "terse_given", "text": head + "; ".join(crits[i] for i in orders["given"]) + ask},
        {"id": "persona_prose", "text": persona + head + prose(orders["given"]) + ask},
    ]
    return out


def holistic_probe_prompts(criteria: Sequence[str], name: str = "", *,
                           extra_prompts: Optional[Sequence[str]] = None,
                           seed: int = 0) -> List[Dict[str, str]]:
    """Holistic/pointer probe prompts for the ADVERSARIAL saturation set (§12.6.4): personas,
    gestalt instructions, and referring expressions that NAME rather than state the concept — the
    prompt modes a unit pool cannot represent. Pass their executed verdict columns to
    ``orthogonalize.adversarial_saturation`` with ``probe_kinds`` including 'holistic'; append
    GEPA-optimized whole prompts via ``extra_prompts`` (the strongest attacker). Not exhaustive over
    prompt space — nothing is (§12.2.4) — but it converts "assume no composition synergy" into "our
    best attacker found none"."""
    nm = name or "overall quality"
    out = [
        {"id": "pointer_name", "text": f"Judge the item purely on '{nm}', as an expert in this "
                                       f"community would understand that term. Answer YES if it "
                                       f"meets the bar, NO otherwise."},
        {"id": "persona_gestalt", "text": "You are a veteran practitioner. Form a holistic, gut-level "
                                          "judgement of the item the way the community's best judges "
                                          "would — do not enumerate criteria. Answer YES or NO."},
        {"id": "community_norm", "text": f"Would the community that cares about '{nm}' accept this "
                                         f"item as meeting its standard? Answer YES or NO."},
    ]
    out += compose_checklist_prompts(criteria, nm, seed=seed)[:2]      # composed variants as probes
    for k, t in enumerate(extra_prompts or []):
        out.append({"id": f"gepa_{k}", "text": str(t)})
    return out


# --------------------------------------------------------------------------------------------
# the measurement (given executed verdict columns)
# --------------------------------------------------------------------------------------------

def delta_comp(composed_sigs: np.ndarray, M: np.ndarray, S_cols: np.ndarray,
               opt_omega_bits: float, *, variant_ids: Optional[Sequence[str]] = None,
               thresh: float = 0.5, beyond_floor: float = 0.02, seed: int = 0) -> dict:
    """Measure the composition gap from executed composed-variant verdicts.

    ``composed_sigs`` (n_variants, n_probes) soft P(YES) of each composed prompt on the SAME frozen
    probes; ``M`` the binarized target; ``S_cols`` the certificate head's conditioning matrix
    (``certificate()['…']``/``greedy_head()['S_cols']``); ``opt_omega_bits`` the certified head value.

    Returns per-variant {v_total (plug-in MI with M), v_beyond (CE-drop | S_g)} and the two
    headline numbers (both at the ADVERSE = max end, matching the certificate's reporting
    convention): ``delta_comp_total`` = max v_total − OPT_Ω and ``delta_comp_beyond`` = max
    v_beyond, plus ``composition_carries_value`` = beyond > ``beyond_floor`` bits. The BEYOND read
    is the sharp one: it is exactly the quantity the checklist certificate cannot bound."""
    C = np.asarray(composed_sigs, float)
    if C.ndim == 1:
        C = C[None, :]
    M = (np.asarray(M, float) > 0.5).astype(int)
    S_cols = np.asarray(S_cols, float)
    if S_cols.ndim == 1:
        S_cols = S_cols[:, None]
    ids = list(variant_ids) if variant_ids is not None else [f"v{k}" for k in range(len(C))]
    per = []
    for k in range(len(C)):
        col = (np.nan_to_num(C[k], nan=0.5) > thresh).astype(int)
        per.append({"id": ids[k] if k < len(ids) else f"v{k}",
                    "v_total": float(i_binary(M, col)),
                    "v_beyond": float(_cmi_block(M, S_cols, col, seed=seed))})
    v_tot = max((p["v_total"] for p in per), default=float("nan"))
    v_bey = max((p["v_beyond"] for p in per), default=float("nan"))
    return {"per_variant": per,
            "delta_comp_total": float(v_tot - float(opt_omega_bits)),
            "delta_comp_beyond": float(v_bey),
            "best_variant_total": max(per, key=lambda p: p["v_total"])["id"] if per else None,
            "best_variant_beyond": max(per, key=lambda p: p["v_beyond"])["id"] if per else None,
            "composition_carries_value": bool(np.isfinite(v_bey) and v_bey > beyond_floor),
            "beyond_floor": float(beyond_floor), "n_variants": int(len(C))}
