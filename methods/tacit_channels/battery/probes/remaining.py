"""Interference, internalization, composition, dynamics, stratification clusters.
W1-W3.5 probes: registered with full specs; compute functions land with their passes
(the pass planner supplies the artifacts; compute stays a pure function over them)."""
from __future__ import annotations

from methods.tacit_channels.battery.registry import ProbeSpec, register

# ---- interference -------------------------------------------------------------------
register(ProbeSpec(
    id="P-INTF-1",
    title="Reason-first degradation + fluency-mismatch interaction (destructive analysis / "
          "verbal overshadowing)",
    cluster="interference", catalog_refs=("B15", "A13"),
    tacitness_direction="explain-then-score < score-directly, concentrated where judgment "
                        "quality >> rule-statement quality (the Beilock/Schooler boundary)",
    requires=("reason_first_rows", "exec_grid"), wave=1, cost_class="score-rows",
    falsifier="forced articulation never degrades judgment at any expertise level",
    compute=None,
    notes="OUR SIGN-INVERSION is plausibly this signature — test the interaction, not the "
          "main effect",
))
register(ProbeSpec(
    id="P-TOK-1",
    title="CoT-delta (token-dependence of installed knowledge)",
    cluster="interference", catalog_refs=(),
    tacitness_direction="gain present teacher-forced (token-free) = deliberation-free "
                        "competence; gain only-with-CoT = explicit mechanism",
    requires=("reason_first_rows", "adapter_grid"), wave=1, cost_class="score-rows",
    falsifier="all transfer gains require CoT tokens",
    compute=None,
    notes="same mechanical pass as P-INTF-1, different contrast",
))

# ---- internalization ----------------------------------------------------------------
register(ProbeSpec(
    id="P-INT-1",
    title="Exclusion probe (process dissociation): policy leaks through a suppress "
          "instruction",
    cluster="internalization", catalog_refs=("A7",),
    tacitness_direction="residual target-correlation under judge-AGAINST instruction = "
                        "internalized, not instruction-followed",
    requires=("exclusion_rows", "adapter_grid"), wave=1, cost_class="score-rows",
    falsifier="trained policy fully suppressible on command (pure controlled compliance)",
    compute=None,
    notes="report directionally; R/A algebra is not quantitatively valid",
))
register(ProbeSpec(
    id="P-INT-2",
    title="Contrast-pair flip agreement (policy tracks the feature; memorizer doesn't flip)",
    cluster="internalization", catalog_refs=(),
    tacitness_direction="flip-agreement with the target on minimal pairs = feature-based "
                        "policy, not exemplar lookup",
    requires=("contrast_pairs", "adapter_grid", "target_composed"), wave=2,
    cost_class="judge",
    falsifier="high rho with chance-level flip agreement = memorization masquerade",
    compute=None, gates=("anchors",),
))

# ---- composition --------------------------------------------------------------------
register(ProbeSpec(
    id="P-COMP-1",
    title="Composition ladder AND/OR/NOT (knowing-using gap for judgment policies)",
    cluster="composition", catalog_refs=(),
    tacitness_direction="using-gap = knowing(parts) - using(composition), per channel; "
                        "prediction: channels match on knowing, dissociate on using",
    requires=("target_composed", "adapter_grid", "exec_grid"), wave=1,
    cost_class="score-rows",
    falsifier="separate-SFT composes as well as joint training (no knowing-using gap for "
              "policies -- contradicts 2607.08393 in the judgment domain)",
    compute=None, gates=("controls_required",),
    notes="NOT (anti-construct) = cheapest memorization test; SFT-installer interpretation "
          "guard binding (a composition failure is ambiguous between knowledge-locality and "
          "installer-pointwiseness until the reward/joint arms run)",
))
register(ProbeSpec(
    id="P-COMP-2",
    title="Error-correction under perturbation (Ryle multi-track adaptability)",
    cluster="composition", catalog_refs=("B20",),
    tacitness_direction="adaptive self-correction distinguishes genuine know-how from "
                        "static reproduction",
    requires=("perturbed_items", "adapter_grid"), wave=4, cost_class="build",
    falsifier="all channels equal on error-detection regardless of transfer route",
    compute=None,
))

# ---- dynamics -----------------------------------------------------------------------
register(ProbeSpec(
    id="P-DYN-1",
    title="Compilation trajectory (power-law + verbalization decay across checkpoints)",
    cluster="dynamics", catalog_refs=("A8",),
    tacitness_direction="agreement UP while self-articulation quality DOWN across training = "
                        "proceduralization",
    requires=("checkpoints", "elicitation"), wave=3, cost_class="train",
    falsifier="verbalization quality tracks performance monotonically (explicit-first never "
              "hands off)",
    compute=None,
))
register(ProbeSpec(
    id="P-DYN-2",
    title="Learning-set formation (multi-domain Harlow curve)",
    cluster="dynamics", catalog_refs=("A14",),
    tacitness_direction="one-shot agreement on domain N+1 rising with N = tacit "
                        "learning-to-learn",
    requires=("checkpoints", "adapter_grid"), wave=3, cost_class="train",
    falsifier="flat trial-2 curve; or curve explained by a shallow generic heuristic",
    compute=None, gates=("g_control",),
))
register(ProbeSpec(
    id="P-DYN-3",
    title="Meta-acceleration (exchange-rate curve shift on new constructs)",
    cluster="dynamics", catalog_refs=(),
    tacitness_direction="fewer examples to install the NEXT policy = transferable "
                        "installation efficiency",
    requires=("checkpoints", "adapter_grid"), wave=3, cost_class="train",
    falsifier="N-to-criterion unchanged by prior tacit training",
    compute=None, gates=("controls_required", "g_control"),
))
register(ProbeSpec(
    id="P-DYN-4",
    title="LPP curriculum vs flat-batch (legitimate peripheral participation)",
    cluster="dynamics", catalog_refs=("C40",),
    tacitness_direction="staged low-stakes-first, agreement-gated exposure beats flat batch",
    requires=("checkpoints",), wave=3, cost_class="train",
    falsifier="curriculum <= flat at equal data",
    compute=None,
    notes="COMMITTED: trainer --curriculum staged (clear cases first, |p-.5| high; "
          "agreement-gated admission) vs identical data flat-shuffled",
))

# ---- stratification (one anchored judge pass, multiple axes) --------------------------
register(ProbeSpec(
    id="P-STRAT-1",
    title="RB vs II construct classification (COVIS backbone)",
    cluster="stratification", catalog_refs=("A10",),
    tacitness_direction="articulation works on RB-analogs only; distillation on both; "
                        "CoT helps RB / hurts II",
    requires=("annotation",), wave=2, cost_class="judge",
    falsifier="channel outcomes uncorrelated with RB/II class",
    compute=None, gates=("anchors",),
))
register(ProbeSpec(
    id="P-STRAT-2",
    title="Mimeomorphic vs polimorphic item/construct annotation (Collins & Kusch)",
    cluster="stratification", catalog_refs=("C31",),
    tacitness_direction="channel-(a) deficit concentrates on polimorphic items",
    requires=("annotation",), wave=2, cost_class="judge",
    falsifier="deficit uniform across mimeo/polimorphic strata",
    compute=None, gates=("anchors",),
    notes="replaces the F1-risky domain-level Collins gloss with prereg'd item-level axes",
))
register(ProbeSpec(
    id="P-STRAT-3",
    title="Codifiability gradient (info->skills->judgment->wisdom)",
    cluster="stratification", catalog_refs=("C42",),
    tacitness_direction="monotonic channel-(a) decay along the gradient",
    requires=("annotation",), wave=2, cost_class="judge",
    falsifier="no monotonic ordering",
    compute=None, gates=("anchors",),
))
register(ProbeSpec(
    id="P-STRAT-4",
    title="Weak/medium/strong tacitness pre-classification (Gascoigne & Thornton)",
    cluster="stratification", catalog_refs=("B27",),
    tacitness_direction="prereg'd grade predicts observed channel-gap ordering",
    requires=("annotation",), wave=2, cost_class="judge",
    falsifier="observed gaps uncorrelated with a-priori grades (e.g. 'elegant' shows the "
              "smallest gap)",
    compute=None, gates=("anchors",),
))
register(ProbeSpec(
    id="P-STRAT-5",
    title="Task-type x channel interaction (Lam: individual vs collective knowledge forms)",
    cluster="stratification", catalog_refs=("C39",),
    tacitness_direction="channel ranking REVERSES between individually-exercised and "
                        "collectively-negotiated judgment tasks",
    requires=("annotation", "adapter_grid"), wave=3, cost_class="score-rows",
    falsifier="one channel dominates uniformly across task types",
    compute=None,
    notes="COMMITTED; full version couples to the 5.1 peer-review passes; humor-internal "
          "proxy split (observable-class vs performance constructs) runs in W2",
))
