"""Append the adjudicated 2026-07-13 metric-seam census update to the report.

The notebook predates the articulability/verifiability terminology correction.
This updater makes small corrections to the historical narrative, then appends
an idempotent, artifact-computed results section.  It does not execute cells.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK = ROOT / "notebooks/2026-07-02__metric-seam-certificates-and-overnight-report.ipynb"
CELL_PREFIX = "seam-20260713-"


def _source(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def _set_code_source(cell: dict[str, Any], text: str) -> bool:
    """Replace code only when it changed, invalidating saved execution then."""

    source = _source(text)
    if cell.get("source") == source:
        return False
    cell["source"] = source
    cell["execution_count"] = None
    cell["outputs"] = []
    return True


def _markdown(cell_id: str, text: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": _source(text),
    }


def _code(cell_id: str, text: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


NEW_CELLS = [
    _markdown(
        CELL_PREFIX + "definitions",
        r"""
## §19 · Adjudicated metric-seam census update (2026-07-13)

This section is the current interpretive layer. Where older sections use **articulable** as a synonym for
"reconstructed by code," this section supersedes that wording. The reconstruction objective remains entirely
unsupervised: the frozen LLM judgment is the reference to be reconstructed, not external ground truth.

| axis | operative meaning | positive evidence | what failure means |
|---|---|---|---|
| **Prompt articulability** | a prompt/LLM implementation of the judgment | stable, specified prompt behavior | bounded failure of that prompt/model/budget |
| **Code verifiability** | an executable program issues a scoped, replayable decision or witness | a passing executable witness on its stated relation | no witness found in the frozen program class/capabilities/budget |
| **Reconstruction agreement** | candidate scores agree with the frozen LLM reference | held-out agreement on common support | candidate/reference mismatch; not automatically either side's error |
| **Isomorphism** | construct, input representation, program channel, and reference instrument align | all four fidelities survive adversarial checks | not certified by correlation alone |
| **Constructive extension** | code resolves a reference disagreement for code-native reasons | independent construct-validity certificate on disagreements | unavailable unless disagreements themselves are adjudicated |

This implements the Collins asymmetry directly: a successful prompt or executable witness can establish a
bounded positive claim; a failed search can establish only bounded non-discovery, never tacitness.

### There is no honest single “% codable” yet

Four denominators answer different questions: (1) authored channel hypotheses, (2) reconstruction by existing
programs, (3) synthetic contract execution on selected cells, and (4) held-out promotion. The cells below report
all four rather than silently substituting one for another.
""",
    ),
    _code(
        CELL_PREFIX + "codability",
        r"""
# Artifact-computed interim codability funnel (CPU-only; no model calls).
sys.path.insert(0, str(ROOT))
from methods.metric_seam.pilot import metric_seam_notebook_stats as seam_stats

domain_labels = {
    "press_releases": "Press releases",
    "creative_writing": "Creative writing",
    "math": "Math",
    "humor": "Humor",
    "legal_title_vii": "Title VII",
    "peer_review": "Peer review",
    "legal_ss_disability": "SS disability",
}

codability = pd.DataFrame(seam_stats.codability_by_domain())
channels = seam_stats.channel_contract_summary()
priors = pd.DataFrame(channels["per_domain"])
domain = codability[codability.task != "ALL"].merge(priors, on="task", suffixes=("", "_tagged"))
domain["domain"] = domain.task.map(domain_labels)

display(
    domain[
        [
            "domain", "n", "tagged_criteria", "tagged_probes", "code_probe_pct", "code_mean", "code_ge_30_pct",
            "code_ge_50_pct", "code_ge_80_pct", "hybrid_mean",
        ]
    ]
    .rename(
        columns={
            "domain": "domain",
            "n": "panel n",
            "tagged_criteria": "tagged criterion n",
            "tagged_probes": "typed probe n",
            "code_probe_pct": "authored CODE probes (%)",
            "code_mean": "shallow-code mean r̃",
            "code_ge_30_pct": "code r̃≥.30 (%)",
            "code_ge_50_pct": "code r̃≥.50 (%)",
            "code_ge_80_pct": "code r̃≥.80 (%)",
            "hybrid_mean": "hybrid mean r̃",
        }
    )
    .style.format(
        {
            "authored CODE probes (%)": "{:.1f}",
            "shallow-code mean r̃": "{:.3f}",
            "code r̃≥.30 (%)": "{:.1f}",
            "code r̃≥.50 (%)": "{:.1f}",
            "code r̃≥.80 (%)": "{:.1f}",
            "hybrid mean r̃": "{:.3f}",
        }
    )
)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), constrained_layout=True)
order = np.arange(len(domain))
axes[0].bar(order, domain.code_probe_pct, color="#2563eb")
axes[0].set(title="Articulated channel prior", ylabel="authored CODE probes (%)", ylim=(0, 100))
axes[1].bar(order - .24, domain.code_ge_30_pct, width=.24, label="r̃ ≥ .30", color="#60a5fa")
axes[1].bar(order, domain.code_ge_50_pct, width=.24, label="r̃ ≥ .50", color="#2563eb")
axes[1].bar(order + .24, domain.code_ge_80_pct, width=.24, label="r̃ ≥ .80", color="#1e3a8a")
axes[1].set(title="Existing shallow-code reconstruction", ylabel="panel criteria above threshold (%)", ylim=(0, 100))
for ax in axes:
    ax.set_xticks(order, domain.domain, rotation=35, ha="right")
axes[1].legend(frameon=False)
plt.show()
""",
    ),
    _markdown(
        CELL_PREFIX + "codability-reading",
        r"""
### What the percentages support

- **Authored sub-relation allocation:** 299/672 typed probes are tagged CODE (**44.5%**) and 373/672 L
  (55.5%). These cover 142/159 criteria; 17 legacy contracts contribute 97 untyped probes and are excluded.
  Of the 142 fully tagged criteria, 130 (91.5%) contain at least one CODE probe, 45 (31.7%) are majority-CODE,
  and only 6 (4.2%) are all-CODE. This is articulated decomposition, not observed success.
- **Existing shallow executable programs on the purposive 159-criterion panel:** mean ceiling-normalized
  reconstruction is **.233** (median .177). **56/159 (35.2%)** reach r̃≥.30, **23/159 (14.5%)** reach
  r̃≥.50, and **1/159 (0.6%)** reaches r̃≥.80. These are transparent sensitivity thresholds, not a
  preregistered binary definition of codability.
- **Hybrid reconstruction is a different estimand:** 118/159 (74.2%) reach r̃≥.30, 77/159 (48.4%) reach
  r̃≥.50, and 13/159 (8.2%) reach r̃≥.80. Because these programs use LLM fields, those percentages must not
  be called pure-code codability or prompt articulability.
- CODE-share rises from the authored floor band (31.1%) through mid (43.1%) to controls (57.6%). Across the
  seven domains its association with observed shallow-code reconstruction is strong (matched-sample Spearman
  ρ=.857), but within 142 criteria it is weak (ρ=.247). Both are post hoc and domain knowledge informed the
  tags. The broad domain ordering is promising; channel tags alone do not validate individual metrics.

The defensible broad claim is therefore **domain-structured, relation-local compilability**, not “44.5% of
metrics are codable.”
""",
    ),
    _code(
        CELL_PREFIX + "census",
        r"""
# Selected census cells, synthetic probes, and the only domain-wide held-out batch.
census_progress = pd.DataFrame(seam_stats.census_progress())
census_outcomes = seam_stats.census_outcome_summary()
probe_replay = seam_stats.census_probe_channel_replay()  # fresh execution of all 43 candidates
cw_heldout = seam_stats.creative_writing_heldout_adjudication()

census_progress["domain"] = census_progress.task.map(domain_labels).fillna("ALL")
display(
    census_progress[["domain", "attempted", "panel_n", "attempted_pct", "train_contract_queue_n"]]
    .rename(columns={
        "domain": "domain", "attempted": "attempted", "panel_n": "panel n",
        "attempted_pct": "attempted (%)", "train_contract_queue_n": "queue n",
    })
    .style.format({"attempted (%)": "{:.1f}"})
)

probe_table = pd.DataFrame(probe_replay["by_authored_channel"]).T.reset_index(names="authored channel")
display(probe_table.style.format({"separation_pct": "{:.1f}"}))

print(
    f"Final train/probe contracts: {census_outcomes['final_contract_passes']}/"
    f"{census_outcomes['attempted_cells']} pass "
    f"({census_outcomes['final_contract_pass_pct']:.1f}%); "
    f"{census_outcomes['final_separations']}/"
    f"{census_outcomes['final_separation_opportunities']} probe pairs separate "
    f"({census_outcomes['final_separation_pct']:.1f}%)."
)
print(
    f"CW held-out: {cw_heldout['exploratory_pairwise_count']}/"
    f"{cw_heldout['unambiguous_count']} raw pairwise wins; "
    f"{cw_heldout['g1_and_pairwise_count']} also clear G1; "
    f"{cw_heldout['bh_survivor_count']}/{cw_heldout['bh_test_count']} survive BH-FDR .10."
)
""",
    ),
    _markdown(
        CELL_PREFIX + "census-reading",
        r"""
### Census adjudication

The current census has attempted **43/159 criteria (27.0%)** and is highly selected: 30 are creative writing;
only 13 come from the other six domains. Fresh replay finds **33/43 (76.7%)** final candidates pass the
train-only code-path contract and **171/212 (80.7%)** synthetic pairs separate. One pass, CW a333 (6/6), is
not in the 32-entry promotion queue; its artifacts do not explicitly record why.

The contract rate is an **upper-biased probe-local engineering result**, not a population codability estimate.
The strongest internal diagnostic is that code separates 73/84 authored-CODE probes (86.9%) but also 98/128
authored-L probes (76.6%). Candidates can construct surface proxies for synthetic L pairs while running with
`extracted={}`; passing does not make the intended L relation generally code-verifiable.

The only domain-wide held-out census batch is creative writing. Of 22 candidates, two have low judge coverage.
Among 20 unambiguous tests, four clear the exploratory raw pairwise rule, only a144 and a90 also clear G1, and
**0/20 survive BH-FDR .10**. Thus “4 promoted” is historical exploratory language; the strict interim
certification count is zero. PR a31's earlier held-out Δρ=+.0145, P=.9105 is likewise an exploratory raw result
because the .90 rule was selected at analysis time and no multiplicity family was applied.

The floor investigations remain theoretically useful: every selected low-rho baseline examined closely has
exposed a concrete instrument defect. That rejects “low reconstruction ⇒ tacit” as an inference; it does not
show that all floor constructs are codable.
""",
    ),
    _markdown(
        CELL_PREFIX + "hierarchy-funnel-heading",
        r"""
## §19A · Technical R1/R2/R3 pre-reconstruction funnels

This is the first canonical 30-metric-per-level technical panel: 90 code-review metrics from the frozen
**expanded-source hierarchy** (30 each at R1, R2, and R3). It is a separate hierarchy lineage from the newer
rebuilt lexicon partition. The code-review funnel stops before any prompt output or reference comparison.
The same cell reports cross-audited retrospective static-witness coverage for math, a full-article science
claim capability, and the narrower patent audit over four historical prior-art hybrids. A subsequent math
stage executes fixed-L conditional slices, and a separately frozen science full-article-section sample executes
the existing strict document-internal verifier. Canonical science hierarchy items and patent candidates remain
unexecuted.
The historical science capability expects full-body evidence, while the current 300 hierarchy item bytes are
abstract-only; this representation mismatch blocks an isomorphic execution on the canonical item panel.
Document-internal science and the patent external-evidence channel are kept distinct from external scientific
truth and pure code, respectively.
""",
    ),
    _code(
        CELL_PREFIX + "hierarchy-funnel",
        r"""
# Frozen code-side funnel plus an unscored prompt manifest. No APIs or models run here.
r123 = seam_stats.code_review_hierarchy_reconstruction_funnel()
code_additive = seam_stats.code_review_additive_unused_program_funnel()

display(
    pd.DataFrame(r123["stages"])
    .rename(columns={
        "stage": "pre-reconstruction stage", "n": "metric mappings",
        "denominator": "panel n", "pct": "panel share (%)",
    })
    .style.format({"panel share (%)": "{:.1f}"})
)

display(
    pd.DataFrame(r123["by_level"])
    .rename(columns={
        "level": "hierarchy level", "panel_n": "panel n",
        "static_relation_local_n": "static relation-local",
        "train_operational_n": "TRAIN operational",
        "heldout_code_score_ready_n": "held-out ≥30 code scores",
        "heldout_pct_of_panel": "ready / panel (%)",
        "heldout_pct_of_static": "ready / static (%)",
    })
    .style.format({"ready / panel (%)": "{:.1f}", "ready / static (%)": "{:.1f}"})
)

display(
    pd.DataFrame(r123["by_depth"])[
        ["depth", "depth_meaning", "static_relation_local_n", "train_operational_n",
         "heldout_code_score_ready_n", "heldout_pct_of_static"]
    ]
    .rename(columns={
        "depth": "relation depth", "depth_meaning": "operative meaning",
        "static_relation_local_n": "static relation-local",
        "train_operational_n": "TRAIN operational",
        "heldout_code_score_ready_n": "held-out ≥30 code scores",
        "heldout_pct_of_static": "ready / static (%)",
    })
    .style.format({"ready / static (%)": "{:.1f}"})
)

display(pd.DataFrame([
    {
        "code-review lane": "canonical corrected static",
        "relation mappings / 90": code_additive[
            "canonical_corrected_static_unchanged"
        ],
        "panel share (%)": 100 * code_additive[
            "canonical_corrected_static_unchanged"
        ] / 90,
        "scientific status": "frozen canonical result",
    },
    {
        "code-review lane": "additive repaired-program static union",
        "relation mappings / 90": code_additive["additive_static_union"],
        "panel share (%)": 100 * code_additive["additive_static_union"] / 90,
        "scientific status": "partial relation-local fidelity",
    },
    {
        "code-review lane": "additive TRAIN-selected",
        "relation mappings / 90": code_additive["train_selected_mappings"],
        "panel share (%)": 100 * code_additive["train_selected_mappings"] / 90,
        "scientific status": "outcome-blind TRAIN gate",
    },
    {
        "code-review lane": "additive held-out nondegenerate",
        "relation mappings / 90": code_additive[
            "heldout_nondegenerate_mappings"
        ],
        "panel share (%)": 100 * code_additive[
            "heldout_nondegenerate_mappings"
        ] / 90,
        "scientific status": "executed; not necessarily confirmatory",
    },
    {
        "code-review lane": "additive held-out confirmatory-ready",
        "relation mappings / 90": code_additive[
            "heldout_confirmatory_mappings"
        ],
        "panel share (%)": 100 * code_additive[
            "heldout_confirmatory_mappings"
        ] / 90,
        "scientific status": "at least 30 paired code scores",
    },
]).style.format({"panel share (%)": "{:.1f}"}))
print(
    "Additive code-review extension: new static mappings by depth =",
    code_additive["new_static_by_depth"],
    "; selected mappings by level =",
    code_additive["train_selected_by_level"],
    "; held-out readiness =",
    code_additive["heldout_readiness_counts"],
    "; axes =", code_additive["axes"],
)

weighted_labels = {
    "retrieved_candidate": "candidate retrieved",
    "relation_local_static_fidelity": "static relation-local fidelity",
    "train_operational_relation_witness": "TRAIN operational witness",
    "heldout_confirmatory_reconstruction_evaluable": "held-out reconstruction-evaluable",
}
weighted_rows = []
for outcome, label in weighted_labels.items():
    row = r123["prevalence"]["pooled"][outcome]
    weighted_rows.append({
        "code-side outcome": label,
        "conditional eligible-record expansion (%)": 100 * row["rate"],
    })
display(
    pd.DataFrame(weighted_rows).style.format(
        {"conditional eligible-record expansion (%)": "{:.1f}"}
    )
)
print(
    "Corrected outcome perturbation ranges recomputed:",
    r123["prevalence"]["corrected_outcome_perturbation_ranges_recomputed"],
    "(historical ranges are not reused).",
)

level_rows = []
for level, values in r123["prevalence"]["by_level"].items():
    level_rows.append({
        "hierarchy level": level,
        "static relation-local (%)": 100 * values["relation_local_static_fidelity"],
        "TRAIN operational (%)": 100 * values["train_operational_relation_witness"],
        "held-out evaluable (%)": 100 * values[
            "heldout_confirmatory_reconstruction_evaluable"
        ],
    })
display(pd.DataFrame(level_rows).style.format({column: "{:.1f}" for column in level_rows[0] if column.endswith("(%)")}))

dependence = r123["prevalence"]["dependence_diagnostics"]
display(pd.DataFrame([
    {
        "diagnostic partition": "cross-level raw support",
        **{key: dependence["cross_level_raw_support"][key] for key in (
            "n_components", "largest_component", "n_cross_level_components"
        )},
    },
    {
        "diagnostic partition": "shared candidate program",
        **{key: dependence["shared_candidate_program"][key] for key in (
            "n_components", "largest_component", "n_cross_level_components"
        )},
    },
    {
        "diagnostic partition": "dependency + raw support + program union",
        **{key: dependence["joint_dependency_raw_program_union"][key] for key in (
            "n_components", "largest_component", "n_cross_level_components"
        )},
    },
]))

print(
    "Prompt batch:", r123["prompt_manifest"]["cells"], "cells;",
    r123["prompt_manifest"]["jobs"], "compiled jobs; status =",
    r123["prompt_manifest"]["status"],
)
print("Axis status:", r123["axes"])

code_representation = seam_stats.code_review_representation_family_sensitivity()
display(
    pd.DataFrame(code_representation["primary_program_macro"])
    .rename(columns={
        "comparison": "code input projections",
        "programs": "unique programs",
        "exact_row_agreement": "exact row agreement",
        "status_agreement": "status agreement",
        "applicability_change_rate": "applicability change rate",
        "exact_value_agreement_on_common_scored": (
            "exact value agreement / common scored"
        ),
    })
    .style.format({
        "exact row agreement": "{:.1%}",
        "status agreement": "{:.1%}",
        "applicability change rate": "{:.1%}",
        "exact value agreement / common scored": "{:.1%}",
    })
)
print(
    "Code representation-family anchor:",
    code_representation["P0_exact_replay_rows"], "frozen prefix rows replayed;",
    code_representation["P0_exact_replay_mismatches"], "mismatches; primary =",
    code_representation["primary_unique_programs"], "unique programs /",
    code_representation["primary_relation_mappings"], "typed mappings;",
    "program depths =", code_representation["primary_program_depth_counts"],
    "; axis status =", code_representation["axes"],
)

math_static = seam_stats.math_hierarchy_static_funnel()
math_rows = []
for frame, values in (
    ("balanced 90-cell panel", math_static["balanced_panel"]),
    (
        "eligible-inventory stratum expansion",
        math_static["eligible_inventory_stratum_expansion"],
    ),
):
    math_rows.append({
        "math retrospective static frame": frame,
        "candidate retrieved (%)": 100 * values["retrieved_candidate"]["rate"],
        "relation-local static witness (%)": (
            100 * values["relation_local_static_fidelity"]["rate"]
        ),
        "exact whole construct (%)": 100 * values["whole_construct_exact"]["rate"],
    })
display(
    pd.DataFrame(math_rows).style.format({
        "candidate retrieved (%)": "{:.4f}",
        "relation-local static witness (%)": "{:.4f}",
        "exact whole construct (%)": "{:.4f}",
    })
)
print(
    "Math retrospective static witnesses by level:", math_static["witnesses_by_level"],
    "; by audited relation depth:", math_static["witnesses_by_audited_depth"],
    "; axis status =", math_static["axes"],
)

math_symbolic = seam_stats.math_hierarchy_symbolic_capability_sensitivity()
display(pd.DataFrame([
    {
        "math symbolic-capability sensitivity frame": "balanced 90-cell panel",
        "canonical relation-local (%)": 100 * math_symbolic["balanced_panel"][
            "canonical_relation_local_unchanged"
        ]["rate"],
        "formal-symbolic relation (%)": 100 * math_symbolic["balanced_panel"][
            "formal_symbolic_relation_local"
        ]["rate"],
        "new coverage from capability (%)": 100 * math_symbolic["balanced_panel"][
            "newly_covered_by_formal_symbolic_relation"
        ]["rate"],
        "additive union (%)": 100 * math_symbolic["balanced_panel"][
            "additive_sensitivity_union_relation_local"
        ]["rate"],
    },
    {
        "math symbolic-capability sensitivity frame": (
            "eligible-inventory stratum expansion"
        ),
        "canonical relation-local (%)": 100 * math_symbolic[
            "eligible_inventory_stratum_expansion"
        ]["canonical_relation_local_unchanged"]["rate"],
        "formal-symbolic relation (%)": 100 * math_symbolic[
            "eligible_inventory_stratum_expansion"
        ]["formal_symbolic_relation_local"]["rate"],
        "new coverage from capability (%)": 100 * math_symbolic[
            "eligible_inventory_stratum_expansion"
        ]["newly_covered_by_formal_symbolic_relation"]["rate"],
        "additive union (%)": 100 * math_symbolic[
            "eligible_inventory_stratum_expansion"
        ]["additive_sensitivity_union_relation_local"]["rate"],
    },
]).style.format({
    "canonical relation-local (%)": "{:.4f}",
    "formal-symbolic relation (%)": "{:.4f}",
    "new coverage from capability (%)": "{:.4f}",
    "additive union (%)": "{:.4f}",
}))
print(
    "Math symbolic capability:", math_symbolic["formal_symbolic_relation_local_cells"],
    "depth-3 presented-step relations;", math_symbolic["newly_covered_cells"],
    "new cells; canonical stays", math_symbolic["canonical_relation_local_cells"],
    "of 90 and additive union is", math_symbolic["additive_union_cells"], "of 90;",
    "axis status =", math_symbolic["axes"],
)

math_operational = seam_stats.math_hierarchy_operational_funnel()
math_operational_rows = []
math_stage_labels = {
    "static_relation_local_witness": "static relation-local witness",
    "train_operational_constant_l_slice": "TRAIN-measurable constant-L slice",
    "heldout_measurable_constant_l_slice": "held-out-measurable frozen slice",
}
for frame, values in (
    ("balanced 90-cell panel", math_operational["balanced_panel"]),
    (
        "eligible-inventory stratum expansion",
        math_operational["eligible_inventory_stratum_expansion"],
    ),
):
    for stage, label in math_stage_labels.items():
        math_operational_rows.append({
            "math constant-L operational frame": frame,
            "stage": label,
            "relation mappings": math_operational["stage_relation_mapping_counts"][stage],
            "rate (%)": 100 * values[stage]["rate"],
        })
display(pd.DataFrame(math_operational_rows).style.format({"rate (%)": "{:.4f}"}))
math_sensitivity = math_operational["sentinel_sensitivity"]["pooled_pair_weighted"]
print(
    "Math constant-L execution:",
    math_operational["compiler_train"]["three_state_totals"], "TRAIN calls;",
    math_operational["heldout_pre_reference"]["three_state_totals"], "held-out calls;",
    "target-free train sentinel agreement =", math_sensitivity,
    "; axis status =", math_operational["axes"],
)
math_prompt = math_operational["prompt_batches"]
print(
    "Math prompt-articulability batches:",
    math_prompt["compiler_train"]["n_jobs"], "TRAIN jobs and",
    math_prompt["heldout_pre_reference"]["n_jobs"], "fixed held-out jobs compiled;",
    "prompt responses =", math_prompt["compiler_train"]["n_prompt_responses"]
    + math_prompt["heldout_pre_reference"]["n_prompt_responses"],
    "; reconstruction estimates =",
    math_prompt["compiler_train"]["n_reconstruction_estimates"]
    + math_prompt["heldout_pre_reference"]["n_reconstruction_estimates"],
    "; isomorphism adjudications =",
    math_prompt["compiler_train"]["n_isomorphism_adjudications"]
    + math_prompt["heldout_pre_reference"]["n_isomorphism_adjudications"],
)

science_static = seam_stats.science_hierarchy_static_funnel()
science_rows = []
for frame, values in (
    ("balanced 90-cell panel", science_static["balanced_panel"]),
    (
        "eligible-inventory stratum expansion",
        science_static["eligible_inventory_stratum_expansion"],
    ),
):
    science_rows.append({
        "science full-article retrospective static frame": frame,
        "candidate retrieved (%)": 100 * values["retrieved_candidate"]["rate"],
        "relation-local static witness (%)": (
            100 * values["relation_local_static_fidelity"]["rate"]
        ),
        "depth-3 document-internal relation (%)": (
            100 * values["depth3_relation_local_static_fidelity"]["rate"]
        ),
        "exact whole construct (%)": 100 * values["whole_construct_exact"]["rate"],
    })
display(
    pd.DataFrame(science_rows).style.format({
        "candidate retrieved (%)": "{:.4f}",
        "relation-local static witness (%)": "{:.4f}",
        "depth-3 document-internal relation (%)": "{:.4f}",
        "exact whole construct (%)": "{:.4f}",
    })
)
print(
    "Science full-article static bank:", science_static["retrieved_candidates"], "retrieved;",
    science_static["relation_local_static_witnesses"], "relation-matched; provenance =",
    science_static["channel_provenance"], "; axis status =", science_static["axes"],
)

science_operational = seam_stats.science_hierarchy_fullarticle_operational_funnel()
science_operational_rows = []
science_stage_labels = {
    "static_relation_local_witness": "static relation-local witness",
    "train_operational_fullarticle_section_verifier": (
        "TRAIN-operational full-article-section verifier"
    ),
    "heldout_measurable_fullarticle_section_verifier": (
        "held-out-measurable frozen verifier"
    ),
}
for frame, values in (
    ("balanced 90-cell relation panel", science_operational["balanced_panel"]),
    (
        "eligible-inventory stratum expansion",
        science_operational["eligible_inventory_stratum_expansion"],
    ),
):
    for stage, label in science_stage_labels.items():
        science_operational_rows.append({
            "science additive full-article frame": frame,
            "stage": label,
            "relation mappings": science_operational[
                "stage_relation_mapping_counts"
            ][stage],
            "rate (%)": 100 * values[stage]["rate"],
        })
display(pd.DataFrame(science_operational_rows).style.format({"rate (%)": "{:.4f}"}))
print(
    "Canonical science execution blocker:",
    science_operational["canonical_representation_blocker"],
)
print(
    "Additive full-article-section CPU execution:",
    science_operational["compiler_train"]["three_state_totals_unique_items"],
    "TRAIN items with",
    science_operational["compiler_train"]["n_relation_certificates"],
    "certificates;",
    science_operational["heldout_pre_reference"]["three_state_totals_unique_items"],
    "held-out items with",
    science_operational["heldout_pre_reference"]["n_relation_certificates"],
    "certificates across",
    science_operational["heldout_pre_reference"]["n_items_with_relation_certificate"],
    "items; axis status =", science_operational["axes"],
)
science_addressed = science_operational["additive_addressed_prompt_overlay"]
science_addressed_prompt = science_addressed["prompt_plane"]
print(
    "Science addressed prompt/code scaffold:",
    science_addressed_prompt["distinct_prepared_unscored_request_records"],
    "distinct prepared requests +",
    science_addressed_prompt["structural_abstentions_without_remote_call"],
    "structural abstentions;",
    science_addressed_prompt["planned_two_pass_prompt_jobs_if_executed"],
    "prospective two-pass jobs; prompt responses =",
    science_addressed_prompt["prompt_responses"],
    "; v9↔hierarchy code agreement =",
    f"{science_addressed['code_replay_agreement']['agree']}/"
    f"{science_addressed['code_replay_agreement']['total']}",
    "; exact ctext =",
    science_addressed["representation_contract"]["same_input_representation"],
)
science_exact = science_operational["exact_ctext_prompt_instrument"]
science_exact_summary = science_exact["summary"]
science_projection = science_exact["future_comparison_target"]
print(
    "Science exact-ctext prompt instrument:",
    science_exact_summary["compiled_prompt_pass_records"],
    "compiled prompt-pass records +",
    science_exact_summary["pass_expanded_structural_no_call_outcomes"],
    "pass-expanded no-call outcomes; exact decoded payload replay =",
    f"{science_exact['validation']['decoded_exact_payload_records']}/"
    f"{science_exact_summary['compiled_prompt_pass_records']}",
    "; prompt responses =", science_exact_summary["prompt_responses"],
    "; raw wire identity claimed =",
    science_exact["representation_contract"][
        "raw_jsonl_or_provider_wire_byte_identity_claimed"
    ],
    "; nonstandard-control records =",
    science_exact["transport_control_inventory"]["compiled_prompt_pass_records"],
    "; target =",
    science_projection["name"],
    "; code projection =",
    science_projection["code_projection_summary"],
    "; reconstruction decisions =",
    science_projection["reconstruction_decisions"],
    "; evidence-link in target =",
    science_projection["evidence_link_in_reconstruction_target"],
)

patent_static = seam_stats.patent_hierarchy_static_funnel()
patent_rows = []
for frame, values in (
    ("balanced 90-cell panel", patent_static["balanced_panel"]),
    ("conditional eligible-inventory expansion", patent_static["conditional_eligible_inventory"]),
):
    patent_rows.append({
        "patent static frame": frame,
        "relation-local witness (%)": 100 * values["relation_local_static_fidelity"]["rate"],
        "depth-3 evidence relation (%)": 100 * values["depth3_evidence_relation"]["rate"],
        "pure-code witness (%)": 100 * values["pure_code_witness"]["rate"],
        "exact whole construct (%)": 100 * values["whole_construct_exact"]["rate"],
    })
display(
    pd.DataFrame(patent_rows).style.format({
        "relation-local witness (%)": "{:.1f}",
        "depth-3 evidence relation (%)": "{:.1f}",
        "pure-code witness (%)": "{:.1f}",
        "exact whole construct (%)": "{:.1f}",
    })
)
print(
    "Patent bank:", patent_static["historical_program_families"], "manual hybrids;",
    patent_static["retrieved_candidates"], "of", patent_static["panel_cells"],
    "cells retrieved and relation-matched; axis status =", patent_static["axes"],
)

patent_claim_structure = seam_stats.patent_claim_structure_hierarchy_static_funnel()
patent_claim_operational = (
    seam_stats.patent_claim_structure_hierarchy_operational_funnel()
)
claim_structure_rows = [
    {
        "pure-code patent claim-structure stage": "conservative static partial relation",
        "cells / 90": patent_claim_structure["relation_local_static_fidelity"],
        "balanced panel (%)": 100 * patent_claim_structure["balanced_panel"]
        ["relation_local_static_fidelity"]["rate"],
        "claim boundary": "relation-local; not whole construct",
    },
    {
        "pure-code patent claim-structure stage": "frozen TRAIN operational gate",
        "cells / 90": patent_claim_operational["train_gate_selected_cells"],
        "balanced panel (%)": 100 * patent_claim_operational["balanced_panel"]
        ["train_gate_selected"]["rate"],
        "claim boundary": "relation-output gate; no reconstruction inference",
    },
    {
        "pure-code patent claim-structure stage": "held-out pre-reference measurable",
        "cells / 90": patent_claim_operational["heldout_relation_measurable_cells"],
        "balanced panel (%)": 100 * patent_claim_operational["balanced_panel"]
        ["heldout_relation_measurable"]["rate"],
        "claim boundary": "positive relation-local code verifiability only",
    },
    {
        "pure-code patent claim-structure stage": "static-only formatter-constant witness",
        "cells / 90": patent_claim_structure["static_only_formatter_constant_cells"],
        "balanced panel (%)": 100 * patent_claim_structure["balanced_panel"]
        ["static_only_formatter_constant"]["rate"],
        "claim boundary": "faithful section-presence relation; non-operational here",
    },
    {
        "pure-code patent claim-structure stage": "exact whole construct",
        "cells / 90": patent_claim_structure["whole_construct_exact"],
        "balanced panel (%)": 0.0,
        "claim boundary": "none",
    },
]
display(pd.DataFrame(claim_structure_rows).style.format({"balanced panel (%)": "{:.2f}"}))
print(
    "Pure-code patent TRAIN receipt:", patent_claim_structure["train"]["status_counts"],
    "over", patent_claim_structure["train"]["items"], "items;",
    patent_claim_structure["train"]["items_at_declared_character_cap"],
    "at the declared cap (finite witnesses only); depths =",
    patent_claim_structure["maximum_matching_relation_depth_counts"],
    "; historical/manual overlap =",
    patent_claim_structure["historical_comparison"]["overlap"],
    "; descriptive provenance-mixed union =",
    patent_claim_structure["historical_comparison"]["descriptive_union_cells"],
    "; held-out pre-reference =",
    patent_claim_operational["heldout_pre_reference"]["status_counts"],
    "over", patent_claim_operational["heldout_pre_reference"]["items"],
    "items; finite certificates =",
    patent_claim_operational["heldout_pre_reference"]["finite_certificate_counts"],
    "; unscored prompt jobs train/held-out =",
    patent_claim_operational["prompt_batches_compiled_unscored"]["compiler_train"]
    ["jobs"],
    "/",
    patent_claim_operational["prompt_batches_compiled_unscored"]
    ["heldout_pre_reference"]["jobs"],
    "; held-out prompt temporal status =",
    patent_claim_operational["prompt_batches_compiled_unscored"]
    ["v3_heldout_temporal_status"],
    "; axis status =", patent_claim_operational["axes"],
)

# These rows share the balanced 90-cell panel, but not the same execution channel.
# They are bounded relation-local coverage readouts, never a pooled codability rate.
technical_coverage = pd.DataFrame([
    {
        "technical lane / readout": "Code review (corrected canonical)",
        "static cells / 90": 50,
        "static panel (%)": 100 * 50 / 90,
        "conditional static expansion (%)": 100 * r123["prevalence"]["pooled"][
            "relation_local_static_fidelity"
        ]["rate"],
        "TRAIN measurable cells": 27,
        "held-out measurable cells": 18,
        "matched relation depth": "d1:25; d2:25",
        "scope": "partial code relations; prompt unscored",
    },
    {
        "technical lane / readout": "Code review (+ repaired unused programs)",
        "static cells / 90": code_additive["additive_static_union"],
        "static panel (%)": 100 * code_additive["additive_static_union"] / 90,
        "conditional static expansion (%)": None,
        "TRAIN measurable cells": code_additive["train_selected_mappings"],
        "held-out measurable cells": code_additive[
            "heldout_nondegenerate_mappings"
        ],
        "matched relation depth": "new d2:4; new d4:5",
        "scope": (
            "additive partial relations; 19/90 confirmatory-ready; prompt unscored"
        ),
    },
    {
        "technical lane / readout": "Math (canonical constant-L slices)",
        "static cells / 90": 33,
        "static panel (%)": 100 * 33 / 90,
        "conditional static expansion (%)": 100 * math_static[
            "eligible_inventory_stratum_expansion"
        ]["relation_local_static_fidelity"]["rate"],
        "TRAIN measurable cells": 33,
        "held-out measurable cells": 33,
        "matched relation depth": "d1:10; d2:23",
        "scope": "conditional code variation; prompt unscored",
    },
    {
        "technical lane / readout": "Math (+ formal-symbolic static sensitivity)",
        "static cells / 90": 38,
        "static panel (%)": 100 * 38 / 90,
        "conditional static expansion (%)": 100 * math_symbolic[
            "eligible_inventory_stratum_expansion"
        ]["additive_sensitivity_union_relation_local"]["rate"],
        "TRAIN measurable cells": None,
        "held-out measurable cells": None,
        "matched relation depth": "adds 7 d3 relations / 5 new cells",
        "scope": "additive source audit; capability not run here",
    },
    {
        "technical lane / readout": "Science (full-article capability map)",
        "static cells / 90": 6,
        "static panel (%)": 100 * 6 / 90,
        "conditional static expansion (%)": 100 * science_static[
            "eligible_inventory_stratum_expansion"
        ]["relation_local_static_fidelity"]["rate"],
        "TRAIN measurable cells": 6,
        "held-out measurable cells": 6,
        "matched relation depth": "d3:6",
        "scope": "additive full-article execution; canonical blocked",
    },
    {
        "technical lane / readout": "Patents (pure-code claim structure)",
        "static cells / 90": 8,
        "static panel (%)": 100 * 8 / 90,
        "conditional static expansion (%)": None,
        "TRAIN measurable cells": 5,
        "held-out measurable cells": 5,
        "matched relation depth": "d1:7; d2:1",
        "scope": "partial claim-graph/surface relations; prompt unscored",
    },
    {
        "technical lane / readout": "Patents (prior-art hybrid bank)",
        "static cells / 90": 6,
        "static panel (%)": 100 * 6 / 90,
        "conditional static expansion (%)": 100 * patent_static[
            "conditional_eligible_inventory"
        ]["relation_local_static_fidelity"]["rate"],
        "TRAIN measurable cells": None,
        "held-out measurable cells": None,
        "matched relation depth": "d1:1; d3:5",
        "scope": "oracle-conditioned model-assisted hybrids",
    },
])
display(technical_coverage.style.format({
    "static panel (%)": "{:.2f}",
    "conditional static expansion (%)": "{:.2f}",
}, na_rep="not run"))
""",
    ),
    _markdown(
        CELL_PREFIX + "hierarchy-funnel-reading",
        r"""
### What has—and has not—been established

- Independent source review covered all 68 retrieved placements and 33 candidate modules. Six apparent
  sub-relation witnesses were construct mismatches and one matched depth was too high. The corrected static
  readout is **50/90 (55.6%)**, R1/R2/R3 = 14/15/21 and relation depth 1/2 = 25/25. Every accepted mapping
  remains partial: **0/90** programs implement an exact whole construct.
- Intersecting the prior TRAIN gate with corrected static fidelity leaves **27/90 (30.0%)** minimally operational
  mappings (R1/R2/R3 = 9/6/12; relation depth 1/2 = 19/8). This propagates the construct audit through frozen
  availability metadata without re-executing programs; it is not a validity result.
- The corrected frozen held-out set is **18/90 (20.0%)** mappings with at least 30 code scores, enough for a later
  comparison (R1/R2/R3 = 7/4/7; depth 1/2 = 12/6). Static-to-ready survival is 48.0% for depth 1 and 24.0%
  for depth 2.
- A provenance-separate extension repairs six previously unused programs and accepts **9 new partial mappings**:
  four at relation depth 2 and five at depth 4, where depth 4 denotes actual formatter/linter/scanner execution.
  The additive static union is **59/90 (65.6%)**, R1/R2/R3 = 17/19/23. A train-only gate selects **35/90**
  mappings (11/10/14), and all 35 execute nondegenerately held out. This is not the same as confirmatory
  readiness: only **19/90** have at least 30 paired code scores; 12 are exploratory-sparse and four have
  insufficient paired support. The canonical 50→27→18 result remains untouched.
- The secondary action-node readout is a **conditional stratum expansion**, not an unconditional prevalence
  estimate. Five outcome-blind deterministically ordered nodes were taken from each of 18 source-kind × breadth × level
  strata. Expanding by the frozen stratum ratios over **1,128 eligible records** gives **41.8%** static
  relation-local fidelity, **23.1%** TRAIN-operational witnesses, and **16.0%** held-out evaluability. The full
  inventory has **1,132** complete records; four R1 records were excluded by the frozen name/description/support
  eligibility rule. These point estimates require treating the deterministic order as pseudo-random or
  exchangeable within strata; sampling-design uncertainty is not estimated.
- The corresponding R1/R2/R3 conditional points are 37.4/51.6/72.0%, 22.9/19.0/41.8%, and
  16.4/12.4/24.7%. Levels have different source compositions, so these differences establish **neither a
  hierarchy-round trend nor an abstraction effect**.
- Corrected dependency/provenance perturbation ranges have not yet been recomputed, and the notebook does not
  recycle the superseded historical ranges. The structural partitions still show substantial dependence: the
  90 sampled mappings reduce to 35 cross-level raw-support components (largest 33), 55 shared-program components
  (largest 5), and 25 components under a dependency/raw-support/program union (largest 49). A corrected multiway
  dependence analysis, source-frame sensitivity, and terminal-frontier audit remain outstanding.
- The corrected prompt manifest contains 18 cells × 125 items × three channels × two stateless passes =
  **13,500 jobs**, all **unscored**. Its channels separate source-only whole constructs, a mechanically
  source-only hierarchy subrelation, and implementation-disclosed relations. The salted subrelation choice was
  compiled post hoc (although code-blind), and the implementation summary omits the full executable contract.
  The 18 mappings reuse only 10 program vectors, so aggregate inference must cluster by program and item. The
  scope correction also re-froze four wrong-relation controls that would otherwise point at removed cells.
  Prompt articulability, reconstruction agreement, codability, and isomorphism remain unestimated.

A separate outcome/reference-inaccessible representation audit now makes the shared-input requirement concrete.
It first reproduces **4,000/4,000 frozen prefix executions with zero mismatches** across all 16 previously
executed programs. The primary readout then macro-averages the **10 unique programs** supporting the corrected
18-mapping family, rather than pretending the repeated criterion mappings are independent. Exact-row agreement
is **70.68%** from first-4,000 prefix to historical head/tail, **55.56%** from head/tail to capped raw diff, and
**45.88%** from prefix to raw diff. Applicability changes average 7.08%, 7.80%, and 14.88%, respectively; all ten
programs are status/applicability-sensitive in every contrast. Thus input projection is a family-wide property
of the executable measurement, not an a104 anomaly. This is code/code representation sensitivity only: no
judgment, reference, prompt response, reconstruction correlation, model, API, or outcome was loaded. It does not
measure prompt articulability, isomorphism, codability, whole-construct verification, or tacitness.

Seven primary programs have matched-relation depth 1 and three depth 2. Depth-2 programs have lower descriptive
exact-row and common-value agreement in every contrast (for prefix→head/tail, 58.13% versus 76.06% exact rows),
but **n=3 versus n=7** is an exploratory mechanism signal, not an estimated depth effect.

One instrument incident is closed: the original TRAIN replay mishandled lazy parser errors on binary diff blocks.
`code_review_train_execution_v1.json` is invalidated and excluded; the additive
`code_review_train_execution_v2.json` replay completed all 30 historically selected programs without item or
contract failures. The independent construct correction excludes three relation mismatches from subsequent
train/held-out/prompt eligibility without rewriting that execution record. It changes the admissible comparison
set, not any reconstruction conclusion—none exists yet.

### §19B · Math hierarchy: static witnesses and constant-L execution

The completed independent cross-audit leaves **33/90 (36.6667%)** balanced math hierarchy cells with a named
relation-local static witness in the historical manual hybrid bank; **0/90** is an exact whole-construct match.
The conditional stratum expansion is **36.1266%** over 1,185 eligible action-node records. It conditions on
the frozen source-kind × breadth × level sampling frame and the same pseudo-random/exchangeability assumption
as the other hierarchy expansions; it is not a randomized-design estimate.

The 33 witnesses split R1/R2/R3 = **12/6/15** and audited relation depth 1/2 = **10/23**. These are maximum
depths for the specifically matched sub-relation, not labels inherited from unrelated program branches. The
level differences do not establish a hierarchy or abstraction trend. The static pass itself loaded no item,
program output, prompt reference, outcome label, or score vector; it is retrospective relation-local coverage.

An additive source-only capability sensitivity then applies the already existing manual SymPy/Lark
presented-step checker. Independent object/relation/polarity/applicability/aggregation review accepts **7/15**
retrieved mappings as narrow rational-equality-preservation relations at depth 3. Five are new panel cells and
two add a formal-symbolic relation to already covered cells, so the static union is **38/90 (42.2222%)** while
the canonical result remains **33/90**. The conditional expansion moves from **36.1266% to 37.6231%**. This is
exactly the capability-relative effect the experiment seeks: a relation-matched algebra system moves a bounded
part of the seam. It is still a manually designed pipeline seed, and this pass executed neither the verifier nor
items; it establishes no whole-proof correctness, prompt articulability, reconstruction, or codability rate.

A separate CPU-only stage then held every declared LLM field fixed and executed the conditional slices
$g_c(x)=f(x,c)$. On compiler-train, all **16/16 programs** completed: **36,000/36,000** calls were measured,
230/240 fixed profiles were nonconstant, ten were constant, and none failed or abstained. A gate using only
train coverage, nonconstancy, and failures selected one profile per program. All **16/16 frozen profiles**
remained nonconstant on held-out text, with **2,400/2,400** measured calls and no failures or abstentions.
Thus all **33/33** static mappings survive train and held-out measurability: **36.6667%** of the balanced panel
and **36.1266%** under the conditional eligible-inventory expansion.

As a target-free seam diagnostic, pairwise rank agreement across the 230 nonconstant train profiles has
pair-weighted median Spearman $\rho=1.000$, minimum $.705609$, and 959/2,285 identical vectors (41.97%). It
was not used for profile selection or held-out decisions; profile pairs are dependent and two-field programs
contribute more pairs. The result says that code-side ordering is often robust to these constant controls. It
does **not** make a sentinel semantically valid, execute the original hybrid, or establish a pure-code rewrite,
whole-construct verifiability, prompt articulability, reconstruction, isomorphism, codability, performance
against a target, or evidence of tacitness.

The corresponding prompt side is now compiled but deliberately **unscored**: **295,200 TRAIN jobs** and a
separately fixed **128,700 held-out jobs** cover all 33 mappings, 16 distinct executable-vector clusters, 150
items per split, and two stateless passes. Source-derived definition/rule arms are paired with matched wrong and
inert controls in canonical, question, and boilerplate forms; implementation disclosure is a separate post-code
arm rather than an independent source articulation. Compiling jobs is not evidence of prompt articulability:
there are still **0 prompt responses, 0 reconstruction estimates, and 0 isomorphism adjudications**. When the
prompt side is eventually run, raw signed held-out Spearman correlation is primary. Any relation-local
isomorphism claim additionally requires audited polarity and cross-form sign stability; an unclear direction
must produce abstention rather than a train-fitted sign rescue.

### §19C · Science hierarchy: static coverage and additive full-article execution

The manually designed full-article science verifier retrieves **9/90** hierarchy cells; **6/90 (6.6667%)**
survive static construct-fidelity review, three are relation mismatches, and **0/90** is an exact
whole-construct match. The conditional eligible-inventory stratum expansion is **5.5407%** over 675 records.
All six accepted mappings reach depth 3 through document-local BM25 retrieval, numeric/comparative extraction,
and a distinct-body-sentence consistency relation. Their R1/R2/R3 counts are 2/2/2, which is descriptive and
does not establish a hierarchy trend.

This historical capability operates on **full presented articles**: it checks abstract claim nodes against
other sentences inside the same article. Its implementation is manual and pure code, with no corpus-wide
or external retrieval. But no verifier was executed and no articles, historical certificates, outputs, prompt
references, outcomes, or external supervision were loaded for this hierarchy pass. Moreover, the canonical
300 hierarchy items are abstract-only; only 12 exact-join to the existing 2,400-paper evidence corpus, and just
6 of those have nonempty body evidence. Static mapping therefore does not establish executable performance, and
the canonical item representation does not currently support a same-byte prompt/code full-body comparison.
The canonical execution is accordingly blocked rather than supplemented with hidden body text.

An additive outcome-blind sample now freezes abstract plus upstream-capped extracted methods/results/evaluation
body into one `ctext`, identical for current code and any future prompt arm. It is a new sample and representation,
not a retroactive repair of the canonical hierarchy items. Its 2,400-paper source corpus was historically
sampled in balanced accept/reject strata, but `y` was masked before decoding and never used for item selection,
gating, or execution; item coverage is therefore conditional on that source frame, not population prevalence
or a supervised reconstruction anchor. The train-only gate retains all six depth-3 mappings.
On 150 TRAIN items the strict verifier measures 118, abstains on 32, fails on 0, and emits 7 document-internal
certificates. On 150 held-out pre-reference items it measures **108**, abstains on **42**, fails on **0**, and
emits **10 certificates across 9 items**. The six static mappings therefore survive train and held-out
measurability (6/90; conditional expansion 5.5407%), while 78.7% TRAIN and 72.0% held-out are item execution
coverage—not codability. Document-internal consistency is not external scientific truth or evidence reliability;
no prompt output, reference, reconstruction, isomorphism, performance target, or evidence of tacitness is present.

The existing addressed V8 prompt scaffold is now mechanically bound to all 300 additive items: TRAIN maps to
124 prepared requests plus 26 structural abstentions, and held-out maps to 111 requests plus 39 abstentions.
That is **235 distinct prepared-but-unscored requests**, 65 no-call structural abstentions, and **470 prospective
jobs** under the preregistered two stateless passes; current prompt responses remain zero. Replaying addressed
V9 code against the hierarchy executions agrees on status and all output counts for **300/300 items**, with
exact aggregate dictionaries on both splits. This strengthens the transport and reproducibility instrument,
but it is only same-evidence/source-addressed: the prompt JSONL is not byte-identical to hierarchy `ctext`, so it
does not license full prompt/code isomorphism. The current subset is exploratory, and a fresh split is still
required for a confirmatory prompt/code claim. The decomposition remains a manually constructed pipeline seed.

An additive exact-payload compiler now closes the narrower serialization gap without calling a model. It emits
**248 TRAIN + 222 exploratory-heldout = 470 unscored jobs** for the 235 body-present items across two stateless
passes, plus 65 unique structural abstentions (**130 pass-expanded local no-call outcomes**). Because all six
approved hierarchy mappings name the same relation scope and consume one shared verifier vector, each item/pass
gets one remote-call job rather than six duplicates; the six-way mapping ledger describes 2,820 downstream
mapping applications if responses are ever obtained. All **470/470 decoded model-visible user messages** contain
the exact frozen hierarchy `ctext` UTF-8 payload once at a recorded byte interval and SHA, with zero mismatches or
multiple occurrences. This is `exact_shared_ctext_payload_with_prompt_scaffolding`: JSON escaping means it is not
raw-JSONL/provider-wire identity, prompt instructions mean it is not whole-request identity, and zero responses
mean semantic isomorphism remains unmeasured. Twenty-two unique inputs contain NUL bytes (44 jobs); they are
preserved exactly, while provider transport compatibility is explicitly untested. The current held-out pack was
compiled after code execution and is therefore exploratory/nonsealed for this compiler; confirmation still
requires a fresh split.

The paired code side is now executable rather than merely prospective. A deterministic numeric/comparative
projection reuses the frozen parser, document-local retrieval, strict predicates, and exact matching on those
same 300 `ctext` payloads. It selects **158 claims**, assigning **17 supported** and **141 insufficient**;
the contradicted state is implemented and metamorphically tested but absent in this sample. Evidence-link
judgments and the archived whole-document vector are excluded from the reconstruction target, so future prompt
agreement will compare exactly the three-valued numeric/comparative relation. These counts measure code-side
verifiability only: the prompt has not run, and prompt articulability, reconstruction, and isomorphism remain zero.

### §19D · Patent hierarchy: capability concentration, not broad codability

The patent lane now has two additive instruments with deliberately separate provenance. First, the four
historical patent programs retrieve only **6/90 (6.7%)** balanced hierarchy cells and all six survive
static relation-local audit; **0/90** is an exact whole-construct or pure-code witness. Under the same conditional
stratum-expansion logic, these six sampled mappings represent **5.18%** of 1,368 eligible action-node records.
Five of the six relation matches use the program's depth-3 prior-art evidence operation (conditional estimate
**5.09%**); the utility/industrial-applicability match is only depth 1 even though the surrounding a35 program
also contains an unrelated depth-3 novelty branch.

This is useful positive structure: the tiny bank lands precisely on novelty, the patentability triad, and
prior-art differentiation, while abstaining on formalities, drawing practice, claim clarity, biotech deposits,
and other unsupported relations. But depth 3 here means a precomputed retrieval-plus-reading-model evidence
channel. The candidate set force-included examiner-cited art, and the historical payload retains duplicate
claim-element rows. It is therefore a **manual, model-assisted, oracle-conditioned pipeline seed**, not
autonomous retrieval or pure code. No patent prompt articulability, executable hierarchy performance,
reconstruction agreement, isomorphism, or domain codability rate has been estimated in this pass.

Second, an additive **pure-code claim-structure parser** operates on exactly the frozen patent `ctext` seen by
future prompt arms. It parses named sections and presented claims, constructs and validates finite dependency
edges, recognizes root/dependent layering, emits positive statutory-category surfaces and spans, routes bounded
functional-language markers, and counts the named abstract. Conservative source-plus-TRAIN adjudication credits
**8/90 (8.89%)** partial relation matches at depths d1=7 and d2=1; **0/90** is a whole-construct match. Five of
those eight relations vary on TRAIN and were frozen into the operational gate. Three
application-section witnesses are genuine narrow relation implementations but formatter-constant (ABSTRACT and
CLAIMS are present in every TRAIN item), so they stay static-only. Four additional plausible mappings are
retained only as sensitivity near-misses and receive no credit: numerical-incidence is not measurement clarity,
claim-number contiguity lacks amendment/status context, and the antecedent heuristic is diagnostic rather than
construct-valid.

The TRAIN execution measured 150 items with no failures; 119 lie exactly at the declared 4,000-character cap.
At-cap outputs therefore permit only finite positive witnesses or local counter-witnesses—never absence,
completeness, global claim-set compliance, legal validity, patentability, or drafting-quality conclusions. The
new eight and historical six do not overlap. Their **14/90 descriptive union** is useful as a capability-library
inventory, but it mixes pure-code internal structure with oracle-conditioned model-assisted prior-art evidence
and is not a pooled certification or codability estimate. Prompt articulability, reference reconstruction,
held-out isomorphism, and constructive overperformance remain unmeasured.

After the gate was fixed, the claim-structure program opened the sealed split exactly once. It measured 27 of
150 items below the cap, measured 122 with possible cap truncation, abstained on one, and failed on none; 123
items contact the cap, including the abstention. All five frozen output contracts have held-out relation-local
support: the dependency channel emits 1,336 finite edge/counter certificates, category emits 168 positive
surface/span certificates, functional-language routing emits 206 positive marker certificates, layering has
positive finite witnesses plus below-cap variation, and the named abstract word count varies exactly. This is a
positive **5/90 (5.56%) held-out relation-local code-verifiability witness** under the frozen program class. It
means 5/8 (**62.5%**) of the conservative static matches survived operational gating; the selected maximum
relation depths are four d1 and one d2 (mean 1.2). It does not convert the five parent criteria into
whole-construct verifications, and cap-contact zeros remain unusable as absence evidence.

The future prompt side is now compiled but deliberately unscored: 7,500 TRAIN jobs and 19,500 held-out jobs.
Every selected cell receives the frozen source name, source definition+rules, exact wrong-definition and
inert-length controls, plus a separately labeled post-code structured prompt restricted to the gate's output
mode. The first prompt pack failed an adversarial contract audit: its prose prohibited cap-contact scalars but
its schemas still accepted them, and several certificate shapes were not code-isomorphic. Those v1 jobs are
preserved but **superseded and non-executable**. V2 added cap-specialized per-item schemas, exact v13 dependency
certificate unions, bounded category/functional vocabularies, semantic response validation, and versioned request
identities. A second read-only audit then found two narrower validator holes—duplicate finite certificates and an
incorrect cap-contact no-marker status. V2 was never executed and is also superseded. V3 rejects duplicates,
requires `applicable_abstain` when capped active claims contain no positive marker, and binds the corrected
validator to new request identities. All prompts embed the same `ctext` bytes exactly once.

Chronology still matters: held-out code execution and its operational summary existed before v2 prompt wording.
Mechanical dataflow checks show that v2 did not consume those item-level code outputs, but absence of human
influence cannot be certified after the fact. The v3 held-out pack is therefore
**fixed-after-TRAIN-gate exploratory pre-reference**, not temporally confirmatory; a fresh split is required for
that stronger claim. These packs make an exploratory articulability/reconstruction comparison executable, but do
not themselves measure prompt articulability, prompt/code agreement, or isomorphism, and no model or GPU was
called.

The final table above is therefore the closest honest cross-domain percentage snapshot: corrected code-review
static sub-relation coverage is 55.56%, canonical Math is 36.67% (42.22% only under the explicitly additive
formal-symbolic sensitivity), Science's full-article map and the historical patent bank are each 6.67%, and the
new pure-code patent claim-structure map is 8.89%. The rows share a 90-cell quota panel but differ in
capability-bank size, channel, and execution status, so they are displayed side by side and never pooled into a
single “percent codable.”
""",
    ),
    _markdown(
        CELL_PREFIX + "depth-definition",
        r"""
## §20 · How deep are the programs?

Depth has to be typed as carefully as codability. Here **node count** is decomposition size; **longest-path
depth** is the number of declared dependency edges from a root to the output; **L-frontier distance** is the
longest remaining path from each minimal L node to the output. None is semantic depth or signal contribution.

The clean structural sample is WS4: nine frozen programs were manually refactored into typed DAGs and checked
bit-exact against their originals. This measures what the discovered programs *contain*, not automatic
discovery and not construct fidelity.
""",
    ),
    _code(
        CELL_PREFIX + "depth",
        r"""
ws4 = seam_stats.ws4_depth_summary()
ws4_rows = pd.DataFrame(ws4["rows"])
family_names = {"press_releases": "Press releases", "patents_pa": "Patents", "legal_title_vii": "Title VII"}
ws4_rows["family"] = ws4_rows.task.map(family_names)
ws4_rows["criterion"] = ws4_rows.cell.str.split("__").str[-1]

family_rows = []
for task, group in ws4_rows.groupby("task", sort=False):
    family_rows.append({
        "family": family_names[task],
        "programs": len(group),
        "mean nodes": group.n_nodes.mean(),
        "mean longest path": group.longest_path_edges.mean(),
        "CODE nodes (%)": 100 * group.n_code_nodes.sum() / group.n_nodes.sum(),
        "L nodes": group.n_l_nodes.sum(),
    })
family_depth = pd.DataFrame(family_rows)

display(family_depth.style.format({"mean nodes": "{:.1f}", "mean longest path": "{:.2f}", "CODE nodes (%)": "{:.1f}"}))
display(
    ws4_rows[["family", "criterion", "n_nodes", "n_code_nodes", "n_l_nodes", "longest_path_edges", "retrieval_nodes", "evidence_nodes"]]
    .rename(columns={
        "n_nodes": "nodes", "n_code_nodes": "C", "n_l_nodes": "L",
        "longest_path_edges": "longest path", "retrieval_nodes": "retrieval",
        "evidence_nodes": "evidence-class",
    })
)

fig, ax = plt.subplots(figsize=(8.5, 4.8))
colors = {"Press releases": "#2563eb", "Patents": "#d97706", "Title VII": "#059669"}
for family, group in ws4_rows.groupby("family"):
    ax.scatter(group.n_nodes, group.longest_path_edges, s=95, label=family, color=colors[family])
    for _, row in group.iterrows():
        ax.annotate(row.criterion, (row.n_nodes, row.longest_path_edges), xytext=(5, 4), textcoords="offset points", fontsize=9)
ax.set(xlabel="typed DAG nodes", ylabel="longest root→output path (edges)", title="WS4 structural program depth (nine manual, bit-exact refactors)")
ax.legend(frameon=False)
plt.show()
""",
    ),
    _markdown(
        CELL_PREFIX + "depth-reading",
        r"""
### Depth result

Across the nine programs there are **145 nodes**: 128 code nodes (88.3%) and 17 L nodes (11.7%). The median
program has 14 nodes; median longest path is 5 edges (range 2–10). All 17 L nodes are graph roots; 15 are
abstraction-level 1 and two level 2. Their mean downstream longest-path distance is 2.06 edges (median 2,
range 1–3). The DAGs contain four retrieval nodes and 24 evidence-class nodes.

The family structure is more informative than a single mean:

- Press releases: mean 21 nodes, path depth 7, 90.5% code nodes.
- Patents: mean 13 nodes, path depth 2.33, 87.2% code nodes.
- Title VII: mean 14.3 nodes, path depth 5, 86.0% code nodes.

**Depth is not itself a signal certificate.** The legal programs have five-edge code chains whose ablations carry little
reconstruction signal. The shallower patent programs are dominated by one retrieval node: it carries
84.0–96.4% of summed absolute node-marginal Δρ. This supports the capability-relation thesis more strongly
than “deeper is better”: the right operation at the right representational level can dominate a long pipeline.
""",
    ),
    _code(
        CELL_PREFIX + "depth-source",
        r"""
# Common syntactic descriptors over the exact Python sources that produced
# the active-code scores. These are not semantic relation-depth measures.
source_structure = seam_stats.active_code_source_structure()
structure_metrics = [
    "nonblank_noncomment_lines", "ast_nodes", "ast_max_depth", "function_defs",
    "control_nodes", "max_control_nesting", "condensed_longest_path_edges",
]
structure_rows = []
for metric in structure_metrics:
    summary = source_structure["paired_summary"][metric]
    structure_rows.append({
        "descriptor": metric,
        "deep median": summary["deep_median"],
        "shallow median": summary["shallow_median"],
        "deep > shallow": f"{summary['deep_greater_count']}/{summary['pair_n']}",
    })
display(
    pd.DataFrame(structure_rows).style.format({
        "deep median": "{:.1f}", "shallow median": "{:.1f}",
    })
)

sensitivity_rows = []
for stratum, row in source_structure["association_sensitivity"].items():
    sensitivity_rows.append({
        "comparison stratum": stratum,
        "n": row["n"],
        "AST-node Δ association": row["metrics"]["ast_nodes"]["spearman_structure_delta_vs_reconstruction_delta"],
        "function Δ association": row["metrics"]["function_defs"]["spearman_structure_delta_vs_reconstruction_delta"],
        "control Δ association": row["metrics"]["control_nodes"]["spearman_structure_delta_vs_reconstruction_delta"],
    })
display(
    pd.DataFrame(sensitivity_rows).style.format({
        "AST-node Δ association": "{:+.3f}", "function Δ association": "{:+.3f}",
        "control Δ association": "{:+.3f}",
    })
)
""",
    ),
    _markdown(
        CELL_PREFIX + "depth-source-reading",
        r"""
### Do the “deep” and “shallow” labels denote real structural differences?

Yes, descriptively. On all 15 paired criteria, the manually engineered entry module is larger in nonblank source lines
(median **317 vs 32**), AST nodes (**1,738 vs 351**), and control nodes (**63 vs 3**). Median maximum control
nesting is **7 vs 2**; the conservative scope-qualified, SCC-condensed local call-graph path is **4 vs 0** edges, with the deep
arm larger on 13/15 pairs. Thus the label is not merely rhetorical: the arms differ by source organization,
control structure, helper decomposition, and local call chains—not just by regex count. These descriptors are
entry-module-local and non-transitive: shared parser and library internals are not counted.

But structural magnitude still is not semantic relation match. The post-hoc direction is subset-sensitive:
AST-node Δ versus reconstruction Δ is ρ=−.544 over all 13 defined comparisons, −.262 over the eight with n≥20,
and +.600 over the four that pass the fixed exploratory support gates. The eligible subset is far too small to
interpret, and the sign reversal means **no directional depth–reconstruction association is supported**. LOC,
AST depth, or call-graph length cannot stand in for successful verifiability.
""",
    ),
    _code(
        CELL_PREFIX + "depth-family",
        r"""
# Full-family active-code retrospective: deep manual/static code versus
# TRAIN-selected shallow prompt-generated executable code.
code_depth = seam_stats.active_code_depth_retrospective()
code_depth_rows = pd.DataFrame(code_depth["rows"])
eligible_depth = code_depth_rows[code_depth_rows.p_value.notna()].copy()

display(
    eligible_depth[
        ["criterion_id", "n_paired", "reference_availability", "deep_rho", "shallow_rho",
         "delta_spearman", "ci_low", "ci_high", "p_value", "bh_q_value"]
    ]
    .rename(columns={
        "criterion_id": "criterion", "n_paired": "common n",
        "reference_availability": "reference availability", "deep_rho": "deep ρ",
        "shallow_rho": "shallow ρ", "delta_spearman": "Δρ",
        "ci_low": "CI low", "ci_high": "CI high", "p_value": "p", "bh_q_value": "BH q",
    })
    .style.format({
        "reference availability": "{:.1%}", "deep ρ": "{:.3f}", "shallow ρ": "{:.3f}",
        "Δρ": "{:+.3f}", "CI low": "{:+.3f}", "CI high": "{:+.3f}",
        "p": "{:.3f}", "BH q": "{:.3f}",
    })
)

fig, ax = plt.subplots(figsize=(7.5, 3.8))
plot_rows = eligible_depth.sort_values("delta_spearman")
ax.errorbar(
    plot_rows.delta_spearman,
    np.arange(len(plot_rows)),
    xerr=[plot_rows.delta_spearman - plot_rows.ci_low,
          plot_rows.ci_high - plot_rows.delta_spearman],
    fmt="o", color="#2563eb", ecolor="#93c5fd", capsize=4,
)
ax.axvline(0, color="#374151", lw=1, ls="--")
ax.set(
    yticks=np.arange(len(plot_rows)), yticklabels=plot_rows.criterion_id,
    xlabel="deep − shallow held-out Spearman (paired 95% bootstrap CI)",
    title="Active code: no support-gated depth comparison survives BH-FDR",
)
plt.show()
""",
    ),
    _markdown(
        CELL_PREFIX + "depth-family-reading",
        r"""
### Full-family depth check

All **18 active-code criteria** were retained. Every criterion has a manually engineered deep/static program;
15 also have a shallow executable comparator selected by TRAIN reconstruction only. Thirteen yield a numerical
held-out comparison and nine of those thirteen have a positive descriptive Δρ. Only four meet the fixed exploratory
support gates (common n≥20 and ≥.90 conditional coverage). **None of the four survives BH-FDR .05**; all adjusted
q-values are .460 and every paired bootstrap interval crosses zero. The largest eligible point estimate is a104
(deep ρ=.650, shallow ρ=.509, Δρ=+.141), but p=.153 and q=.460.

This retires the earlier tendency to quote a104 as a standalone positive depth result. It remains a useful
mechanism case study—its deeper AST/static program measures test presence and organization—but it does not link
tests to implementation or execute them, and this
retrospective family does **not** support “deeper code reconstructs better” as a general conclusion. Program
depth, relation match, coverage, and construct fidelity remain separate measurements.
""",
    ),
    _markdown(
        CELL_PREFIX + "technical-heading",
        r"""
## §21 · Technical-domain witnesses and the Sonnet-work adjudication

The technical lane is intentionally additive. Manually engineered/mock/retrospective mechanisms are treated as
pipeline seeds—what the agentic compiler is supposed to have selected—not erased or redescribed as automatic
discoveries. The table is rebuilt from the current corrected artifacts.
""",
    ),
    _code(
        CELL_PREFIX + "technical-ledger",
        r"""
technical_ledger = seam_stats.technical_evidence_ledger_summary()
ledger_summary = technical_ledger["summary"]
display(
    pd.DataFrame([
        {"evidence stratum": key, "typed records": value}
        for key, value in ledger_summary["by_stratum"].items()
    ])
)

families = technical_ledger["family_summaries"]
family_rows = [
    {
        "bounded family": "Active Code deep > shallow",
        "numerator": families["active_code_depth_family"]["bh_fdr_and_minimum_effect_improvements"]["numerator"],
        "denominator": families["active_code_depth_family"]["bh_fdr_and_minimum_effect_improvements"]["denominator"],
        "gate": "BH-FDR + minimum effect among support-eligible criteria",
    },
    {
        "bounded family": "Blind Math construct fidelity",
        "numerator": families["blind_math_construct_family"]["construct_fidelity_passes"]["numerator"],
        "denominator": families["blind_math_construct_family"]["construct_fidelity_passes"]["denominator"],
        "gate": "adversarial construct-fidelity pass among selected blind criteria",
    },
    {
        "bounded family": "Patent WS3 operation contrast",
        "numerator": families["patent_historical_selected_family"]["bh_fdr_rejections"]["numerator"],
        "denominator": families["patent_historical_selected_family"]["bh_fdr_rejections"]["denominator"],
        "gate": "retrospective BH-FDR rejection among four selected criteria",
    },
    {
        "bounded family": "Patent WS3 precision characterized",
        "numerator": families["patent_historical_selected_family"]["effect_precision_characterized"]["numerator"],
        "denominator": families["patent_historical_selected_family"]["effect_precision_characterized"]["denominator"],
        "gate": "positive contrast with a usable paired effect interval",
    },
]
display(pd.DataFrame(family_rows))
print(
    "Domain codability estimates emitted:",
    ledger_summary["domain_codability_estimates_emitted"],
    "| cross-stratum pooled estimates emitted:",
    ledger_summary["cross_stratum_pooled_estimates_emitted"],
)
""",
    ),
    _markdown(
        CELL_PREFIX + "technical-ledger-reading",
        r"""
### One ledger, three non-poolable evidence strata

The source-bound ledger contains **39 typed records**: 24 criterion-scalar reconstruction records,
8 relation-instance records, and 7 program-structure records. Record counts are inventory counts, not success
denominators. The ledger emits **zero** cross-stratum pooled estimates and **zero** technical-domain codability
percentages because a held-out scalar correlation, an exact SymPy pair witness, and an AST/DAG depth do not
measure the same event.

The family fractions remain useful only with their conditioning attached: Active Code has 0/4 support-eligible
deep-over-shallow improvements after BH-FDR and the minimum-effect gate; two selected blind Math criteria have
0/2 construct-fidelity passes; the retrospective patent operation family has 2/4 BH rejections but just 1/4
with fully characterized positive-effect precision. These are bounded family results—not Math, Code, or Patent
codability rates.
""",
    ),
    _code(
        CELL_PREFIX + "a12-generalization",
        r"""
a12_generalization = seam_stats.math_a12_relation_generalization()
a12_coverage_rows = []
for split in ("train", "heldout"):
    row = a12_generalization[split]
    a12_coverage_rows.append({
        "split": split.upper(),
        "covered rows": row["covered_rows"],
        "rows": row["rows"],
        "coverage": row["coverage"],
        "Wilson 95% low": row["coverage_wilson_95"][0],
        "Wilson 95% high": row["coverage_wilson_95"][1],
        "identity pair classifications": row["identity_classifications"],
        "nonidentity pair classifications": row["nonidentity_classifications"],
    })
display(
    pd.DataFrame(a12_coverage_rows).style.format({
        "coverage": "{:.1%}", "Wilson 95% low": "{:.1%}", "Wilson 95% high": "{:.1%}",
    })
)
print(
    f"Held-out − TRAIN coverage: {a12_generalization['heldout_minus_train_coverage']:+.1%}; "
    f"two-sided Fisher exact p={a12_generalization['coverage_fisher_exact_two_sided_p']:.3f}."
)
""",
    ),
    _code(
        CELL_PREFIX + "a12-projection-depth",
        r"""
a12_projection = seam_stats.math_a12_pair_projection_depth()
attempted = a12_projection["depth_views"]["deepest_attempted"]
contributing = a12_projection["depth_views"]["deepest_decision_contributing"]
positive = a12_projection["depth_views"]["positive_relation_evidence"]
display(
    pd.DataFrame([
        {
            "depth view": "deepest attempted",
            "depth 1 rows": attempted["histogram"].get("1", 0),
            "depth 3 rows": attempted["histogram"].get("3", 0),
            "no positive evidence": 0,
        },
        {
            "depth view": "deepest decision-contributing",
            "depth 1 rows": contributing["histogram"].get("1", 0),
            "depth 3 rows": contributing["histogram"].get("3", 0),
            "no positive evidence": 0,
        },
        {
            "depth view": "positive relation evidence",
            "depth 1 rows": positive["histogram"].get("1", 0),
            "depth 3 rows": positive["histogram"].get("3", 0),
            "no positive evidence": positive["no_positive_evidence_rows"],
        },
    ])
)
print(
    f"Inspectable pair records: {a12_projection['pair_certificate_count']} — "
    f"{a12_projection['pair_status_counts']['verified_rational_identity']} identities, "
    f"{a12_projection['pair_status_counts']['exact_nonidentity_witness']} nonidentities, "
    f"{a12_projection['pair_status_counts']['parse_noncoverage']} parse-noncoverage results."
)
print(
    "Positive relation evidence among rows reaching the formal path: "
    f"{a12_projection['formal_path_positive_evidence_rate']:.1%}."
)
""",
    ),
    _markdown(
        CELL_PREFIX + "a12-generalization-reading",
        r"""
### Math a12: a sealed relation-local generalization result

Executable-pair coverage is **42/150 (28.0%) on TRAIN** and **26/100 (26.0%) held out**. The difference is
−2.0 percentage points (two-sided Fisher exact p=.773; held-out Wilson 95% interval 18.4–35.4%). Observed
coverage is therefore similar across the split, although a nonsignificant difference does not establish
equivalence. The held-out run
produced 65 pair-level classifications across 26 covered rows (11 identity and 54 exact nonidentity), while
abstaining on 74 rows. Identity-witness and nonidentity-witness row counts are 4 and 24 respectively and can
overlap within the 26 covered rows.

This is a generalization result for **relation detection and executable classification**, not “26% of rigor is
codable.” The whole-criterion prompt reference is available on 99/100 rows and is internally stable
(two-pass ρ=.835), but it does not expose the frozen universal-scope field needed to bind the code relation to
document-level rigor. No parent scalar, candidate/reference correlation, or isomorphism estimate was created.
The verifier is a manually/retrospectively selected pipeline seed, and its aggregate TRAIN summary was already
visible before the sealed held-out execution; this is held-out generalization of a fixed mechanism, not blind
automatic discovery.

An additive post-reference, code-only projection now makes all **277 pair attempts** independently inspectable
while reproducing every sealed-v1 row classification and aggregate exactly: 11 exact identities, 54 exact
nonidentities, and 212 parse-noncoverage results. This improves auditability but creates no new blind,
reconstruction, or isomorphism result.
“Exact” here means exact under the frozen SymPy normalization. Simplification can erase a cancelled denominator,
so an empty domain-obligation list does not certify equality over the total mathematical domain.

Depth must be read in multiple views. On the ordinal `metric-seam.relation-depth.v1` scale, **65/100 rows reached
the tier-3 formal operation** and 35 stopped at tier-1 structure parsing; the dependency graph itself is only
one edge. Of those 65 formal-path rows, 39 ended in parse noncoverage and 26 yielded positive relation
evidence (40.0%). Positive-evidence depth is therefore 26 rows at depth 3 and 74 with no positive witness.
Counting formal attempts as successful verification would overstate codability; counting failed formal attempts
as shallow execution would understate program depth.
""",
    ),
    _code(
        CELL_PREFIX + "science-relations",
        r"""
science_relations = seam_stats.science_relation_witness_summary()
science_rate_rows = []
for label, row in [
    ("numeric matched relations", science_relations["relation_witnesses"]["numeric"]),
    ("comparative matched relations", science_relations["relation_witnesses"]["comparative"]),
    ("all matched relation instances", science_relations["all_matched_relations"]),
    ("papers with ≥1 strong witness", science_relations["supported_documents"]),
]:
    science_rate_rows.append({
        "conditioning": label,
        "numerator": row["numerator"],
        "denominator": row["denominator"],
        "rate": row["rate"],
        "Wilson 95% low": row["wilson_95"][0],
        "Wilson 95% high": row["wilson_95"][1],
    })
display(
    pd.DataFrame(science_rate_rows).style.format({
        "rate": "{:.1%}", "Wilson 95% low": "{:.1%}", "Wilson 95% high": "{:.1%}",
    })
)
science_replay = science_relations["representation_replay"]
print(
    "Continuous↔exact-address replay: "
    f"{science_replay['strong_witness_intersection']}/"
    f"{science_replay['strong_witness_continuous']} normalized strong witnesses shared; "
    f"{science_replay['supported_document_intersection']}/"
    f"{science_replay['supported_document_continuous']} supported papers shared; "
    f"{science_replay['paper_status_agreement']}/"
    f"{science_replay['paper_status_total']} paper statuses agree."
)
science_prompt = science_relations["prompt_batch"]
print(
    "Prompt-articulability arm:", science_relations["prompt_articulability_status"],
    ";", science_prompt["compiled_unscored_jobs"], "body-present jobs +",
    science_prompt["structural_abstentions_without_remote_call"],
    "structural abstentions over", science_prompt["corpus_records"], "records;",
    "prompt responses =", science_prompt["prompt_responses"],
    "; temporal status =", science_prompt["temporal_status"],
)
""",
    ),
    _markdown(
        CELL_PREFIX + "science-relations-reading",
        r"""
### Science: full-body relation witnesses and representation robustness

The corrected manually constructed program selects claims from abstracts, retrieves evidence from independent
full-article body text, and executes numeric or directional comparative obligations. It produces **68/561
(12.1%)** strong numeric witnesses and **32/634 (5.0%)** strong comparative witnesses. Across all 4,871 matched
relation instances the strong-witness rate is 100/4,871 (2.1%); 95/2,400 papers (4.0%) contain at least one.
These are relation-local parser-witness rates, not percentages of scientific claims that are true or codable.

Running the same strict executable program over continuous segments and exact A/B source addresses yields
**100/100 whitespace-normalized strong-witness identity**, the identical 95-paper supported set, and 2,396/2,400
paper-status agreement. Only 8/100 match as literal text because one representation preserves line breaks; weak
links are slightly less invariant (429 shared, 5 continuous-only, 1 addressed-only). This licenses a useful
**code/code representation-robustness** claim. It is not prompt↔code semantic isomorphism: the prepared Science
prompt arm contains **1,957 body-present jobs compiled but unscored** plus **443 structural abstentions** over
2,400 records, with zero current-bundle responses. The addressed v9 code output now has a byte-exact CPU replay
receipt binding its input, implementation, and three archived outputs. This closes the prior replay-binding gap
and gives prompt and code a common source-address scaffold, but not a common serialized input: the prompt sees
addressed JSONL while the historical continuous code sees normalized text. The current prompt bundle is
instrument-development exploratory; a fresh split is required for a confirmatory prompt/code claim. The
decomposition remains a manually constructed retrospective seed.

The new receipt establishes reproducible code-side representation behavior, not prompt articulability,
semantic prompt/code isomorphism, a whole-review score, or external scientific truth.
""",
    ),
    _code(
        CELL_PREFIX + "patent-family",
        r"""
patent_family = seam_stats.patent_ws3_family_retrospective()
patent_family_rows = []
for row in patent_family["criteria"]:
    interval = row["paired_bootstrap"]["interval"]
    patent_family_rows.append({
        "criterion": row["criterion_id"],
        "reference reliability": row["reference_two_pass_spearman"],
        "full ρ": row["rho_full_evidence_operation"],
        "null ρ": row["rho_null_operation"],
        "Δρ": row["delta_spearman"],
        "paired 95% CI": "unavailable" if interval is None else f"[{interval[0]:+.3f}, {interval[1]:+.3f}]",
        "p": row["paired_randomization"]["p_value"],
        "BH q": row["bh_q_value"],
        "null modal fraction": row["null_score_modal_fraction"],
        "threshold+FDR screens": row["threshold_and_fdr_screens_met"],
    })
display(
    pd.DataFrame(patent_family_rows).style.format({
        "reference reliability": "{:.3f}", "full ρ": "{:.3f}", "null ρ": "{:.3f}",
        "Δρ": "{:+.3f}", "p": "{:.4f}", "BH q": "{:.4f}",
        "null modal fraction": "{:.0%}",
    })
)
""",
    ),
    _markdown(
        CELL_PREFIX + "patent-family-reading",
        r"""
### Patents: the full four-criterion family narrows the headline

Paired randomization with BH-FDR across all four historical WS3 criteria rejects **2/4** null-operation
contrasts: a34 and a35 (both q=.0002). a26's positive Δρ=+.211 is borderline after family correction
(p=.0426, q=.0568); a60 is neither significant nor a usable reconstruction target (two-pass reliability .197).

The precision diagnostics distinguish the two survivors. a35 has Δρ=+.609 with paired 95% bootstrap interval
[+.400,+.804] and a nondegenerate null. a34 has Δρ=+.661, but its nulled program emits the same score on 99% of
rows, making its bootstrap interval unavailable; the randomization contrast supports an ordering difference,
not a precise effect size. Thus **a35 is the one fully characterized family result**, while a34 remains a strong
but null-degeneracy-limited representation result.

This is a retrospective multiplicity-aware reanalysis, not a confirmatory certification. Examiner-cited art was
force-included upstream, so every result remains oracle-conditioned and cannot establish autonomous retrieval,
patent correctness, or a population patent-codability percentage.
""",
    ),
    _code(
        CELL_PREFIX + "technical",
        r"""
a12 = json.load(open(ROOT / "outputs/metric_seam_pilot/reconstruction_v2/math_a12_symbolic_step_heldout_001/finalization/finalization.json"))
science = json.load(open(ROOT / "outputs/metric_seam_pilot/science_claims_v2_relation_strict_v23/results.json"))["summary"]
a104 = next(row for row in code_depth["rows"] if row["criterion_id"] == "a104")
a104_supplement = seam_stats.active_code_a104_supplemental()
a104_repr = a104_supplement["representation_sensitivity"]
a104_exec = a104_supplement["execution_augmentation"]
code_representation = seam_stats.code_review_representation_family_sensitivity()
a407 = json.load(open(ROOT / "outputs/metric_seam_pilot/reconstruction_v2/a407_sealed_historical_eval_001/evaluation.json"))
patent_by_id = {row["criterion_id"]: row for row in patent_family["criteria"]}
blind_a144 = json.load(open(ROOT / "outputs/metric_seam_pilot/reconstruction_v2/blind_math_a144_001/reconstruction_record.json"))

technical = pd.DataFrame([
    {
        "case": "Math a12 symbolic steps",
        "executable coverage / support": f"{a12['candidate_execution_summary']['rows_with_executable_pair']}/{a12['heldout_count']} held-out rows; prompt reference {a12['prompt_reference']['available_both_passes']}/{a12['heldout_count']} (two-pass ρ={a12['prompt_reference']['two_pass_spearman']:.3f})",
        "result": f"{a12_projection['pair_certificate_count']} inspectable attempts; {a12['candidate_execution_summary']['verified_rational_identity_count']} identity + {a12['candidate_execution_summary']['exact_nonidentity_witness_count']} nonidentity witnesses",
        "adjudication": "sealed relation-instance verification; depth-3 attempted on 65 rows, positive evidence on 26; no parent scalar or isomorphism estimate",
    },
    {
        "case": "Science full-article claim↔body evidence",
        "executable coverage / support": f"{science['certificate_decisions']['supported']} certificates in {science['status_counts']['supported']}/{science['papers']} papers ({100*science['certified_document_rate']:.1f}%)",
        "result": f"{science['certificate_relations']['numeric']} numeric + {science['certificate_relations']['comparative']} comparative",
        "adjudication": "strict corrected witnesses; 100/100 code/code representation replay; prompt arm not run; not a whole-metric scalar",
    },
    {
        "case": "Code a104 test evidence",
        "executable coverage / support": f"n={a104['n_paired']} common held-out",
        "result": f"deep code ρ={a104['deep_rho']:.3f}; shallow code ρ={a104['shallow_rho']:.3f}",
        "adjudication": f"retrospective point Δρ={a104['delta_spearman']:+.3f}; p={a104['p_value']:.3f}, BH q={a104['bh_q_value']:.3f}",
    },
    {
        "case": "Code a104 input-projection sensitivity",
        "executable coverage / support": f"n={a104_repr['common_heldout_n']} fixed common V4 held-out; {a104_repr['hierarchy_prefix_rows_at_cap']}/{a104_repr['hierarchy_rows']} hierarchy rows hit 4k cap",
        "result": f"head/tail code ρ={a104_repr['historical_head_tail_rho']:.3f}; 4k-prefix code ρ={a104_repr['prefix4000_rho']:.3f}; Δ={a104_repr['delta_prefix_minus_head_tail']:+.3f}",
        "adjudication": f"post-hoc one-sided sensitivity; LLM reference remains head/tail; code-vector ρ={a104_repr['code_vector_rho']:.3f}; not a same-input isomorphism test",
    },
    {
        "case": "Code a104 repository-execution augmentation",
        "executable coverage / support": f"{a104_exec['exact_repository_pr_overlap']}/{a104_exec['active_items']} exact item overlaps",
        "result": f"{a104_exec['finite_execution_certificates']} finite depth-{a104_exec['relation_depth']} certificate ({a104_exec['finite_certificate_rate_conditional_overlap']:.1%} conditional on overlap)",
        "adjudication": "stored prior execution; extra repository/environment evidence; not same-input, reconstruction, correctness, or a codability denominator",
    },
    {
        "case": "Code 10-program representation family",
        "executable coverage / support": f"{code_representation['P0_exact_replay_rows']} exact frozen replay rows; {code_representation['primary_unique_programs']} programs / {code_representation['primary_relation_mappings']} typed mappings",
        "result": "exact rows: 4k→head/tail 70.7%; head/tail→raw 55.6%; 4k→raw 45.9%",
        "adjudication": "all 10 status/applicability-sensitive; outcome/reference-inaccessible code/code audit; not reconstruction, isomorphism, or codability",
    },
    {
        "case": "Code a407 name expressiveness",
        "executable coverage / support": f"{a407['primary_code_vs_historical_composite']['code_covered_count']}/{a407['primary_code_vs_historical_composite']['eligible_exact_input_count']} exact-input rows",
        "result": f"structural partial ρ={a407['primary_code_vs_historical_composite']['spearman']:.3f}",
        "adjudication": "semantic-context relation absent; whole construct unavailable",
    },
    {
        "case": "Patents a34 evidence-aware",
        "executable coverage / support": f"n={patent_by_id['a34']['reference_common_n']} held-out",
        "result": f"full ρ={patent_by_id['a34']['rho_full_evidence_operation']:.3f}; null ρ={patent_by_id['a34']['rho_null_operation']:.3f}; Δ={patent_by_id['a34']['delta_spearman']:+.3f}",
        "adjudication": f"BH q={patent_by_id['a34']['bh_q_value']:.4f}; interval unavailable; null 99% modal; oracle-conditioned",
    },
    {
        "case": "Patents a35 patentability triad",
        "executable coverage / support": f"n={patent_by_id['a35']['reference_common_n']} held-out",
        "result": f"full ρ={patent_by_id['a35']['rho_full_evidence_operation']:.3f}; null ρ={patent_by_id['a35']['rho_null_operation']:.3f}; Δ={patent_by_id['a35']['delta_spearman']:+.3f}",
        "adjudication": f"BH q={patent_by_id['a35']['bh_q_value']:.4f}; paired CI [+.400,+.804]; oracle-conditioned",
    },
    {
        "case": "Blind math a144",
        "executable coverage / support": "n=52 reference-common held-out",
        "result": f"candidate ρ={blind_a144['reference_isomorphism']['score']:.3f}; historical hybrid ρ=.483",
        "adjudication": blind_a144["outcome"] + "; sealed negative result",
    },
])
display(technical.style.set_properties(subset=["adjudication"], **{"font-weight": "bold"}))
""",
    ),
    _markdown(
        CELL_PREFIX + "adjudication",
        r"""
### What survives independent audit

| Sonnet-lane result | decision | canonical reading |
|---|---|---|
| Blind math a144 | **verified end-to-end** | Real sealing; candidate ρ=.066 versus historical hybrid .483; adversary and construct gates fail, so outcome is `proxy_mismatch`, never tacitness. |
| Math a12 symbolic relation | **new sealed held-out witness set; projection audited** | A manually selected seed executed before reference loading; 26/100 rows were covered and yielded 65 pair witnesses exact under frozen SymPy normalization. A post-reference replay materializes all 277 attempts and reproduces v1 exactly; it improves inspectability but adds no reconstruction result. |
| WS4 typed DAGs | **verified with narrowing** | 9/9 bit-exact refactors and node/depth counts hold. They are manual structural refactors, not automatic discovery or construct-isomorphism certificates. |
| Patents WS3 | **verified, narrowed, oracle-conditioned** | Full-family reanalysis yields 2/4 BH rejections. a35 is fully characterized (Δρ=+.609, CI +.400 to +.804); a34 rejects but has an unavailable bootstrap interval because the null is 99% modal; a26 misses BH at q=.0568. Examiner-cited art was force-included. |
| Science claims | **corrected and strengthened** | The contaminated “171 strong” count is retired. Current strict abstract-claim↔full-body-evidence result is 100 relation-local certificates in 95/2,400 papers: 68 numeric, 32 comparative. The 100/100 same-program representation replay is robust, while the prompt arm remains unrun. |
| Capability library | **original claim narrowed; remediation verified** | The first 7-case replay missed named defects. Additive v2.1 now passes 17/17 frozen counterexamples; this validates the regression instrument, not a metric outcome. |
| “65 certificates / 800 runs” | **relabel** | Classification of stored prior execution artifacts, not 800 live executions in the replay. “Code” there denotes the old `f2p_mock` prototype, not the active coding census. |
| Whole-criterion v2 axes | **complement, do not replace** | Articulability/verifiability/isomorphism classify outcomes; the census presence→position→function ladder remains the finer per-sub-relation taxonomy. |

Additional corrections carried forward: the retrieval operator now indexes only the allowed split rather than
the sealed corpus; CW has 0/20 multiplicity-controlled promotions; WS3 and WS4 are distinct patent experiments;
and no v2 artifact retroactively changes a runbook-certified result.

### Technical interpretation

- Math a12 and science now provide genuinely executable relation classifications that go deeper than regex-only
  matching, but both abstain
  rather than pretending to score the full construct. Math a12 executed all 100 held-out rows before opening
  the reference, found executable pairs on 26, and emitted 65 exact identity/nonidentity witnesses. A later
  reference-free projection materializes all 277 pair attempts. The depth audit separates 65 rows reaching the
  formal path from 26 rows yielding positive evidence; the projection is auditability, not a new blind result.
  Science operates over full articles: it selects claim nodes from the article abstract and verifies them against
  independently addressed body evidence;
  its implementation still combines regex parsing with BM25 retrieval and structured relation matching.
- Code a104 remains the largest eligible active-code depth point estimate, but the full 18-criterion retrospective
  yields no BH-FDR survivor. It is **code versus code**, not code versus prompt articulability, and it is a
  mechanism case study rather than a certified general depth effect. A new frozen-parser sensitivity shows why
  representation must be an experimental factor: on the fixed 93-row common held-out support, changing only the
  code arm from historical head/tail text to the hierarchy's first-4,000-character prefix lowers observed
  reconstruction $\rho$ from .645 to .514 ($\Delta=-.131$), changes applicability on 12/250 rows, and changes
  118/227 common scored values. This is post-hoc and one-sided—the LLM reference remains head/tail—so it is not a
  same-input prompt/code isomorphism test. A separate exact-item join finds one finite depth-4 execution witness
  among 32/250 overlaps with stored repository telemetry. That is sparse feasibility for non-isomorphic
  environment capability augmentation, not a scalar, correctness negative, or codability rate.
- Code a407 operationalizes the proposed function wall: declaration structure is executable, while semantic
  context fit is declared unavailable and the matched prompt arms have not run. It illustrates the boundary but
  does not establish impossibility or record a failed semantic-code search.
- Patent a35 is the cleanest current evidence-operation result; a34 still shows representation alignment but its
  effect precision is limited by a nearly constant null. Neither licenses an external-truth claim or erases the
  oracle candidate-set caveat.
""",
    ),
    _markdown(
        CELL_PREFIX + "synthesis",
        r"""
## §22 · Interim answer and claim frontier

At this stage, the best compact summary is:

> **Articulated contracts allocate 44.5% of typed sub-relations to code. Existing shallow code reaches at least
> .30 ceiling-normalized reconstruction on 35.2% of the purposive panel, but observed population codability is
> not yet identified. Selected agents can make 76.7% of attempted criteria pass synthetic code-path contracts;
> the only domain-wide held-out census has 0/20 multiplicity-controlled certifications. The strongest positive
> results are relation-local technical witnesses and representation-matched evidence operations. Program depth
> is not itself a signal certificate and has not been shown to predict reconstruction in this retrospective.**

The balanced technical hierarchy adds a different, explicitly pre-reconstruction view. Corrected code review
has 50/90 static relation matches, 27/90 train-operational mappings, and 18/90 held-out-measurable mappings.
Canonical Math has 33/90 at all three stages under constant-L slices; the unexecuted formal-symbolic capability
sensitivity raises only the static union to 38/90. Science has 6/90 depth-3 mappings that execute on a separate
full-article-section representation, while canonical abstract-only execution remains blocked. Patents has 6/90
static hybrid mappings and no hierarchy execution. These denominators share a quota panel but not capability-bank
size or execution channel, so they broaden the empirical map without becoming one pooled codability percentage.

What this broadens responsibly:

1. **From regex to relation-matched algorithms.** SymPy identities/nonidentities, AST declaration-use graphs,
   abstract-claim↔full-body-evidence graphs, and prior-art evidence operations are real executable witnesses.
   A newly mapped SymPy capability adds five static Math cells, and the strict Science verifier now generalizes
   measurability to an additive held-out full-article-section split with 10 certificates across 9 papers.
   Science's 100/100 same-program normalized replay adds representation robustness, not prompt↔code isomorphism.
2. **From taste domains to an institutional spectrum.** Authored CODE share and observed shallow-code
   reconstruction are both higher in PR/legal domains than CW/humor, while code and math show sharply
   decomposable technical sub-relations. The between-domain pattern is descriptive but coherent.
3. **From scalar “codability” to a sub-relation census.** Presence often compiles; position compiles when a
   structural capability matches; semantic function remains the recurring L frontier. This is more specific
   and more falsifiable than a whole-criterion label.
4. **From isomorphism-only to two positive outcomes.** Code may either reconstruct the articulated prompt
   judgment or constructively extend it. Extension is claimed only when code-native disagreement certificates
   establish the relevant relation; mere overperformance in ρ is not enough.
5. **A direct Collins result.** Each positive executable witness establishes bounded verifiability. Every failure
   is typed as non-discovery, proxy mismatch, noncoverage, or instrument failure—never as evidence of tacitness.

The next decisive statistic is not another train/probe pass rate. It is a pre-registered, held-out technical-domain
promotion family (math, active code, science, patents) that reports relation coverage, adversarial construct
fidelity, reconstruction agreement, and code-native disagreement certificates separately. No supervised external
anchor is required for that design.
""",
    ),
    _markdown(
        CELL_PREFIX + "construct-validity-repair-md",
        r"""
## §23 · Proposal-first construct-validity repair (2026-07-14)

The corrected workflow is **PROPOSE → BASE-RATE PROBE → AUTHOR/IMPORT → CONSTRUCT CHALLENGE → PER-NODE
GATE → SELECT → FREEZE → TRANSCRIBE → EVALUATE**. Agreement is now downstream of construct validity.

Three locally executable cases stop for different reasons. Math a12 has natural variation and perfect
conditional polarity agreement, but the old identity checker scores **0/12 contextual construct controls**
correctly. Code review detects **152/160 plants**, but **0/4** units have usable natural violation base rates
on merged PRs. The prospective Patent antecedent proposal passes a detector-blind 32-item prompt probe
(14 satisfied, 18 violated), but the imported binary graph collapses to **1 not-applicable / 1 satisfied /
148 violated** on TRAIN, so selection stops before a 150-call transcription run.

These are construct misconstrual, corpus inadequacy, and code-side degeneracy—not one undifferentiated
“uncodable” outcome. The a34 dead-subtree claim remains audit-reported because no exact local node artifact
was found; it is not counted as independently reproduced here.
""",
    ),
    _code(
        CELL_PREFIX + "construct-validity-repair-code",
        r'''
repair_path = ROOT / "outputs/metric_seam_pilot/verifier_pipeline_v2/construct_validity_repair_summary_v1/readout.json"
repair = json.load(open(repair_path))

rows = []
for unit, result in repair["results"].items():
    rows.append({
        "unit": unit,
        "stop_stage": result["stop_stage"],
        "disposition": result["disposition"],
        "locally_bound": not (unit == "a34" and result.get("local_node_artifact") is None),
    })
display(pd.DataFrame(rows))

patent = repair["results"]["patent_antecedent"]
display(pd.DataFrame([
    {"implementation": "prompt pre-authoring probe", **patent["pre_authoring_prompt_probe"]["state_counts"]},
    {"implementation": "imported binary code", **patent["imported_binary_code_train"]["state_counts"]},
]))
print(repair["headline"])
''',
    ),
]


def _replace(cell: dict[str, Any], old: str, new: str) -> None:
    text = "".join(cell["source"])
    if old not in text:
        if new in text:
            return
        raise ValueError(f"expected notebook text not found: {old[:80]!r}")
    cell["source"] = _source(text.replace(old, new))


def update_notebook() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    prior_generated = {
        str(cell.get("id")): cell
        for cell in notebook["cells"]
        if str(cell.get("id", "")).startswith(CELL_PREFIX)
    }
    notebook["cells"] = [
        cell
        for cell in notebook["cells"]
        if not str(cell.get("id", "")).startswith(CELL_PREFIX)
    ]
    by_id = {cell.get("id"): cell for cell in notebook["cells"]}

    _replace(
        by_id["d7ac9900"],
        "# Metric-Seam Certificates & Survey Report — 2026-07-02, updated 2026-07-07",
        "# Metric-Seam Certificates & Survey Report — 2026-07-02, updated 2026-07-13",
    )
    title_text = "".join(by_id["d7ac9900"]["source"])
    update_note = (
        "\n\n**Updated 2026-07-13:** §§19–§22 add the adjudicated 159-criterion census, "
        "separate prompt articulability from code verifiability and reconstruction agreement, "
        "recompute the codability funnel and WS4 program depths, and integrate the independently "
        "audited reconstruction-v2 technical lane. These sections supersede older overloaded uses "
        "of ‘articulable’ and ‘ground truth.’"
    )
    if "**Updated 2026-07-13:**" not in title_text:
        by_id["d7ac9900"]["source"] = _source(title_text + update_note)

    _replace(
        by_id["money-figure-md"],
        "## §8 — The money figure: cross-task codability brackets + CAM (2026-07-04, roadmap-v2 R1)",
        "## §8 — Historical cross-task reconstruction brackets + CAM (legacy label; 2026-07-04)",
    )
    _replace(
        by_id["money-figure-md"],
        "**Certified Articulable Mass (CAM)** — the area under the survival curve of ceiling-normalized fidelities; a single number summarizing how much of a task's evaluation vocabulary can be compiled into code.",
        "**Historical CAM (legacy name)** — the area under the survival curve of ceiling-normalized reconstruction fidelities. It summarizes reference reconstruction by these frozen implementations; it is not prompt articulability, pure-code codability, or external truth.",
    )
    _replace(
        by_id["humor-legal-md"],
        "**The cross-domain articulability spectrum: law > PR > math > CW ≈ humor.** Articulable mass",
        "**The historical hybrid-reconstruction spectrum: law > PR > math > CW ≈ humor.** Reconstruction mass",
    )

    survey_cell = by_id["dde45571"]
    survey_text = "".join(survey_cell["source"])
    block_markers = (
        "# Optional task survey tables: absence is reported, not imputed.",
        "def _survey_table_path(task):",
        'for task, wave in [("math", "survey"), ("code_review", "survey"), ("patents", "survey"),',
    )
    block_starts = [survey_text.find(marker) for marker in block_markers]
    prefix_end = min(index for index in block_starts if index >= 0)
    suffix_start = survey_text.index("CORPORA =", prefix_end)
    optional_surveys = '''# Optional task survey tables: absence is reported, not imputed.
sys.path.insert(0, str(ROOT))
from methods.metric_seam.pilot import metric_seam_notebook_stats as seam_stats

survey_bundle = seam_stats.survey_task_tables(OUTD)
for r in survey_bundle["rows"]:
    rows.append(dict(task=r["task"], wave=r["wave"], aspect=r["aspect"],
                     name=r.get("name", ""), rel1=r.get("rel1"),
                     ceiling=r.get("ceiling"), rho=r.get("rho"),
                     rho_scoped=r.get("rho_scoped"), verdict=r.get("verdict", "ok")))

survey_availability = pd.DataFrame(survey_bundle["sources"])
display(survey_availability[["task", "wave", "status", "row_count", "source_path"]])
if survey_bundle["unavailable_tasks"]:
    print("Unavailable task surveys (excluded, not counted as zero):",
          ", ".join(survey_bundle["unavailable_tasks"]))

'''
    _set_code_source(
        survey_cell,
        survey_text[:prefix_end] + optional_surveys + survey_text[suffix_start:]
    )

    _set_code_source(by_id["8174418c"], r'''
# --- §4a: available code_review corpora side by side ------------------------------
code_surveys = seam_stats.survey_task_tables(OUTD)
tables = {}
for survey_row in code_surveys["rows"]:
    tables.setdefault(survey_row["task"], {})[survey_row["aspect"]] = survey_row

def ok_row(r):
    return r.get("verdict", "ok") == "ok" and r.get("rho") is not None

rows = []
for label, task in (
    ("comments-only", "code_review"),
    ("full diff", "code_review_diffs"),
    ("competition", "code_competition"),
):
    table = tables.get(task)
    if table is None:
        print(f"{label}: unavailable (missing artifact; excluded, not counted as zero)")
        continue
    usable = [r["rho"] for r in table.values() if ok_row(r)]
    rows.append(dict(
        corpus=label,
        metrics=len(table),
        measurable=len(usable),
        degenerate=len(table) - len(usable),
        med_rho=round(float(np.median(usable)), 3) if usable else np.nan,
        max_rho=max(usable) if usable else np.nan,
        n_ge_03=sum(1 for value in usable if value >= .3),
    ))

comments = tables.get("code_review")
diffs = tables.get("code_review_diffs")
if comments is not None and diffs is not None:
    intersection = [
        aspect for aspect in comments
        if aspect in diffs and ok_row(comments[aspect]) and ok_row(diffs[aspect])
    ]
    if intersection:
        print(f"same-aspect intersection (n={len(intersection)}): comments med "
              f"{np.median([comments[a]['rho'] for a in intersection]):.3f} -> diffs med "
              f"{np.median([diffs[a]['rho'] for a in intersection]):.3f}")
    else:
        print("same-aspect intersection: no jointly measurable rows")
else:
    print("same-aspect intersection: unavailable because one or both artifacts are missing")

pd.DataFrame(rows).rename(columns={"n_ge_03": "aspects rho>=0.3"})
''')

    legal_cell = by_id["humor-legal-code"]
    legal_text = "".join(legal_cell["source"])
    legal_starts = [
        legal_text.find(marker)
        for marker in (
            "legal_table = seam_stats.optional_seam_table(",
            "st = json.load(open(B / 'tasks/legal_title_vii/seam_table.json'))",
        )
    ]
    legal_start = min(index for index in legal_starts if index >= 0)
    legal_optional = r'''legal_table = seam_stats.optional_seam_table(
    B / 'tasks/legal_title_vii/seam_table.json'
)
if legal_table["rows"] is None:
    print('\nlegal_title_vii survey: unavailable (missing artifact; excluded, not counted as zero)')
else:
    legal_rows = sorted(
        legal_table["rows"], key=lambda row: -(row.get('rho_over_ceiling') or -9)
    )
    print('\nlegal_title_vii survey (top 6 / bottom 3 by rho/ceiling):')
    for row in legal_rows[:6] + legal_rows[-3:]:
        print(f"  {row['aspect']:5s} rel1={row.get('rel1'):.2f} "
              f"r~={row.get('rho_over_ceiling'):+.2f}  {row['name'][:48]}")
    values = sorted(
        (row.get('rho_over_ceiling') for row in legal_rows
         if row.get('rho_over_ceiling') is not None),
        reverse=True,
    )
    if values:
        print(f"  median r~ = {values[len(values)//2]:.2f}; "
              f"frac>=.3 = {sum(value >= .3 for value in values)}/{len(values)}")
    else:
        print('  present artifact has no non-null rho/ceiling rows')
'''
    _set_code_source(legal_cell, legal_text[:legal_start] + legal_optional)
    _replace(
        by_id["money-figure-code"],
        "subprocess.run(['python3', str(ROOT/'methods/metric_seam/pilot/money_figure.py')], check=True)",
        "subprocess.run([sys.executable, str(ROOT/'methods/metric_seam/pilot/money_figure.py')], check=True)",
    )
    _replace(
        by_id["money-figure-code"],
        "ROOT = pathlib.Path.cwd().parent",
        "# ROOT is initialized once in the setup cell; do not make it CWD-dependent here.",
    )
    _replace(
        by_id["humor-legal-code"],
        "ROOT = pathlib.Path('..') if pathlib.Path.cwd().name == 'notebooks' else pathlib.Path('.')",
        "# Reuse the canonical absolute ROOT initialized in the setup cell.",
    )
    for cell_id, old, new in (
        (
            "day4-code",
            'BASE = pathlib.Path("..") / "outputs/metric_seam_pilot"',
            'BASE = ROOT / "outputs/metric_seam_pilot"',
        ),
        (
            "9a2ca226",
            'BASE = pathlib.Path("..") / "outputs/metric_seam_pilot/battery"',
            'BASE = ROOT / "outputs/metric_seam_pilot/battery"',
        ),
        (
            "b7edbf5c",
            'BASE = pathlib.Path("..") / "outputs/metric_seam_pilot/battery"',
            'BASE = ROOT / "outputs/metric_seam_pilot/battery"',
        ),
        (
            "3831b036",
            'BASE = pathlib.Path("..") / "outputs/metric_seam_pilot"',
            'BASE = ROOT / "outputs/metric_seam_pilot"',
        ),
        (
            "7b706589",
            'BASE = pathlib.Path("..") / "outputs/metric_seam_pilot/battery"',
            'BASE = ROOT / "outputs/metric_seam_pilot/battery"',
        ),
        (
            "7b706589",
            'MET = pathlib.Path("..") / "methods/metric_seam/battery"',
            'MET = ROOT / "methods/metric_seam/battery"',
        ),
        (
            "addebe17",
            'BASE = pathlib.Path("..") / "outputs/metric_seam_pilot/battery"',
            'BASE = ROOT / "outputs/metric_seam_pilot/battery"',
        ),
    ):
        _replace(by_id[cell_id], old, new)

    patent_text = "".join(by_id["patents-null-md"]["source"])
    patent_addendum = (
        "\n\n**2026-07-13 evidence-aware addendum.** The null above is conditional on a document-only "
        "reference, not a theorem that patent evidence cannot help. WS3 level-matched the evaluator to "
        "the same prior-art representation: held-out a34 moved from ρ=.084 with null evidence to .745 "
        "with evidence (op marginal +.661; a26 +.211, a35 +.609). Examiner-cited art was force-included "
        "in candidate sets, so this is an oracle-conditioned representation result, not autonomous retrieval."
    )
    if "**2026-07-13 evidence-aware addendum.**" not in patent_text:
        by_id["patents-null-md"]["source"] = _source(patent_text + patent_addendum)

    scope_text = "".join(by_id["bd398ddc"]["source"])
    scope_note = (
        "\n\n> **Superseded scope note (2026-07-13):** §§19–§22 are the current claim boundary. "
        "In particular, patents_pa is no longer only a document-reference design null: WS3 adds an "
        "oracle-conditioned evidence-aware result. Census train/probe passes and historical CAM values "
        "are not held-out population codability estimates."
    )
    if "> **Superseded scope note (2026-07-13):**" not in scope_text:
        by_id["bd398ddc"]["source"] = _source(scope_text + scope_note)

    generated = json.loads(json.dumps(NEW_CELLS, ensure_ascii=False))
    for cell in generated:
        prior = prior_generated.get(str(cell.get("id")))
        if (
            prior is not None
            and prior.get("cell_type") == "code"
            and prior.get("source") == cell.get("source")
        ):
            cell["outputs"] = prior.get("outputs", [])
            cell["execution_count"] = prior.get("execution_count")
    notebook["cells"].extend(generated)
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


if __name__ == "__main__":
    update_notebook()
