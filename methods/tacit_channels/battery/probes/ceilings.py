"""Ceilings cluster: subspace cap, asymptotic closure (tier ladder), hand-extraction
positive control, uninvention."""
from __future__ import annotations

import numpy as np

from methods.tacit_channels.battery.registry import ProbeSpec, register
from methods.tacit_channels.channels.common import stable_split
from methods.tacit_channels.channels.eval.prompt_subspace_cap import cap_for_cell


def compute_subspace_cap(ctx) -> list[dict]:
    rows = []
    for rung in ctx.executors():
        exe, _em = ctx.grid(rung)
        for cell in ctx.cells():
            target = ctx.target_name_vector(cell)
            prompt_rows = [v for (c, a, f), v in exe.items() if c == cell]
            if target is None or len(prompt_rows) < 8:
                continue
            n = len(target)
            halves = [stable_split(f"{cell}::item{i}", 0.5, salt="exp_gtk1")
                      for i in range(n)]
            half1 = np.array([i for i, h in enumerate(halves) if h == "train"])
            half2 = np.array([i for i, h in enumerate(halves) if h != "train"])
            res = cap_for_cell(np.vstack(prompt_rows), target, half1, half2)
            base = {"construct": cell, "rung": rung, "domain": ctx.domain,
                    "probe": "P-CEIL-1"}
            for stat in ("cap_oos", "eff_rank_90", "best_single_rho"):
                if res.get(stat) is not None:
                    rows.append({**base, "statistic": stat, "value": float(res[stat]),
                                 "saturated": bool(res["saturated"])})
    return rows


register(ProbeSpec(
    id="P-CEIL-1",
    title="Prompt-subspace cap (out-of-sample ceiling of the observed articulation channel)",
    cluster="ceilings", catalog_refs=(),
    tacitness_direction="weight-channel rho > cap_oos (saturated) = per-cell tacit residual "
                        "(Tier ~1.9)",
    requires=("target_grid", "exec_grid"), wave=0, cost_class="free",
    falsifier="new prompts routinely escape the saturated subspace",
    compute=compute_subspace_cap, gates=("tier_stamp",),
))

register(ProbeSpec(
    id="P-CEIL-2",
    title="Asymptotic-closure test (tier ladder: GEPA-optimized search vs certified cap) — "
          "THE camp-separating confirmatory test",
    cluster="ceilings", catalog_refs=("B19", "B28"),
    tacitness_direction="gap persists under best-effort text optimization against a cap = "
                        "anti-intellectualism; closes = Stanley/Fodor vindicated",
    requires=("gepa_runs", "exec_grid"), wave=1, cost_class="score-rows",
    falsifier="symmetric by design: either outcome falsifies one camp",
    compute=None, gates=("tier_stamp",),
))

register(ProbeSpec(
    id="P-CEIL-3",
    title="Hand-extraction positive control (chicken-sexing arm)",
    cluster="ceilings", catalog_refs=("A12",),
    tacitness_direction="one analyst-distilled minimal instruction ~ fine-tuning => construct "
                        "was under-articulated, NOT tacit (mixture-separator per construct)",
    requires=("elicitation", "exec_grid"), wave=3, cost_class="elicit",
    falsifier="n/a (calibration arm); success generalizing everywhere would collapse the "
              "tacit class entirely",
    compute=None,
))

register(ProbeSpec(
    id="P-CEIL-4",
    title="Uninvention / source-retirement reconstitution",
    cluster="ceilings", catalog_refs=("C32",),
    tacitness_direction="documentation-only reconstitution fails while prior-direct-exposure "
                        "models sustain agreement = tacit residue",
    requires=("elicitation", "adapter_grid"), wave=4, cost_class="score-rows",
    falsifier="full reconstitution from extracted documentation alone",
    compute=None,
))
