"""The Tacitness Battery probe registry.

Every operationalization from the catalog (notes/2026-07-22__tacit-knowledge-
operationalization-catalog.md, entries A1-A14 / B15-B28 / C29-C42) is accounted for in exactly
one of: a ProbeSpec's catalog_refs, gates.GATES, or FRAMINGS below. A unit test enforces zero
orphans against the catalog file.

A probe NEVER runs its own GPU pass: it declares artifact requirements; the pass planner
(passes.py, W1+) bundles all row-needs per engine key. compute(ctx) is a pure function over
artifacts already on disk, returning long-format stat rows for the profile store.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# --- stat row convention (long format, cells_v1 philosophy) -----------------------------
# {"construct": cell_id, "rung": exec_job, "domain": d, "probe": probe_id,
#  "statistic": name, "value": float, ...extra provenance keys}


@dataclass
class ProbeSpec:
    id: str                      # e.g. "P-STAT-1"
    title: str
    cluster: str                 # statability | interference | internalization | ...
    catalog_refs: tuple          # ("A3", "A4") — catalog entry ids this probe implements
    tacitness_direction: str     # how the primary statistic maps to "more tacit"
    requires: tuple              # artifact kinds: "target_grid","exec_grid","adapter_grid",
                                 # "elicitation","annotation","checkpoints","target_composed",
                                 # "item_embeddings","judge_batch"
    wave: int                    # 0 = existing data; 1..4 per plan
    cost_class: str              # "free" | "score-rows" | "elicit" | "judge" | "train" | "build"
    falsifier: str               # from the lit reviews — what result kills the source claim
    compute: Callable | None = None   # ctx -> list[stat rows]; None = registered, not yet runnable
    gates: tuple = ()            # gate ids from gates.py applied to this probe's verdicts
    notes: str = ""


PROBES: dict[str, ProbeSpec] = {}


def register(spec: ProbeSpec) -> ProbeSpec:
    if spec.id in PROBES:
        raise ValueError(f"duplicate probe id {spec.id}")
    PROBES[spec.id] = spec
    return spec


# Catalog entries that are FRAMINGS (paper-level positioning, not executable probes or gates).
FRAMINGS = {
    "B16": "transmission-channel asymmetry IS the program (Polanyi/Oakeshott/Stanley-concession)",
    "C29": "Collins transmission test IS the program's channel comparison",
    "C33": "our channel-(a) rho = first quantitative SECI externalization-fidelity test",
    "C36": "our estimand = Sternberg profile-similarity scoring (psychometric precedent)",
}


def all_probes(wave: int | None = None, runnable_only: bool = False) -> list[ProbeSpec]:
    import methods.tacit_channels.battery.probes  # noqa: F401  (imports register everything)
    specs = sorted(PROBES.values(), key=lambda s: s.id)
    if wave is not None:
        specs = [s for s in specs if s.wave <= wave]
    if runnable_only:
        specs = [s for s in specs if s.compute is not None]
    return specs


def covered_catalog_ids() -> set:
    import methods.tacit_channels.battery.probes  # noqa: F401
    from methods.tacit_channels.battery.gates import GATES
    ids = set(FRAMINGS)
    for spec in PROBES.values():
        ids.update(spec.catalog_refs)
    for gate in GATES.values():
        ids.update(gate.get("catalog_refs", ()))
    return ids
