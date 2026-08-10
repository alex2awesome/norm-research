"""Channel-core probes: the estimand itself as profile rows + the exemplar arm."""
from __future__ import annotations

from methods.tacit_channels.battery.registry import ProbeSpec, register
from methods.tacit_channels.channels.common import cell_stats, is_conditioned_rescue


def compute_channel_core(ctx) -> list[dict]:
    rows = []
    tgt, _tm = ctx.grid(ctx.target_job)
    for rung in ctx.executors():
        exe, em = ctx.grid(rung)
        for cell in ctx.cells():
            s = cell_stats(tgt, exe, em, cell)
            if s is None:
                continue
            base = {"construct": cell, "rung": rung, "domain": ctx.domain,
                    "probe": "P-CHAN-core"}
            for stat, val in (("name_rho", s["exec_name_rho"]),
                              ("best_articulation_rho", s["best_rho"]),
                              ("articulation_gain", s["gain"]),
                              ("native_gap", s["gap"]),
                              ("conditioned_rescue", float(is_conditioned_rescue(s)))):
                if val is not None:
                    rows.append({**base, "statistic": stat, "value": float(val)})
    return rows


register(ProbeSpec(
    id="P-CHAN-core",
    title="Estimand core: name/articulation reconstruction, gain, gap, conditioned rescue",
    cluster="channels", catalog_refs=("B16", "C29"),
    tacitness_direction="low articulation_gain despite native_gap = tacit-candidate",
    requires=("target_grid", "exec_grid"), wave=0, cost_class="free",
    falsifier="articulation gain matching the gap everywhere = nothing tacit anywhere",
    compute=compute_channel_core,
    gates=("turner_phrasing", "tier_stamp"),
))

register(ProbeSpec(
    id="P-CHAN-Aprime",
    title="Exemplar-in-context arm (worked examples, no statements) vs criteria-only",
    cluster="channels", catalog_refs=("B22",),
    tacitness_direction="exemplars>criteria on near transfer = exemplar-primacy (Kuhn); "
                        "criteria>exemplars on far = categorization-literature counter",
    requires=("exec_grid", "exemplar_rows"), wave=1, cost_class="score-rows",
    falsifier="exemplar-only arm generalizes WORSE than criteria-only on far/OOD items "
              "(Nosofsky counter-prediction) kills Kuhn's exemplar-primacy here",
    compute=None,
    notes="rows: k-shot target-judgment exemplars in-context, no rule text; needs pass planner",
))
