# Adding tacit knowledge to a model — installation channels

Date: 2026-07-21
Status: **framing note + experiment sketch** for a new sub-line. Not results. Records the user's
question and hypothesis, organizes the surrounding thoughts, and proposes a concrete experiment that
reuses the policy-isomorphism apparatus. No GPU work launched from this note — the RL/FT arms are a
new measurement target and need explicit sign-off before anything runs
(cf. `feedback_check_before_new_approach`).

Relationship to existing lines: this is **"part 3" — additive, NOT a replacement** for the
articulation isomorphism work (parts 1–2,
`notes/2026-07-14__tacit-knowledge-roadmap-to-completion.md`). The articulation program stands on its
own and continues.

**The parts are NOT disjoint (user correction, 2026-07-21).** Part 3 does not operate only on the
cells articulation "can't reach." The right frame is a **2-D exchange rate**: reconstruction ρ as a
function of *both* how much explicit text you supply (the parts-1–2 axis) *and* how much
non-linguistic intervention you apply (FT / RL / patch). You can have explicit text substitute for
scale **and then** ask what interventions **reduce the amount of explicit text needed** — even on
cells where articulation already works (there the question is "how much text can I *save*?"). So:
- Parts 1–2 = the **left column** of that surface (intervention = 0): does text substitute for scale?
- Part 3 = **how an intervention shifts the whole curve** — the exchange rate between *weight-encoded*
  and *text-encoded* policy. "Rescue of the failure set" is just the special case where the left
  column is flat (articulation buys nothing) and an intervention lifts it.

And the two literally share a pipeline (§5): the most defensible form of the RL channel *generates*
the explicit text that parts 1–2 evaluate. They compose; they don't partition.

---

## 1. The question, sharpened

> How do we add tacit knowledge to a model?

The whole isomorphism program has, until now, tried exactly one way: hand a smaller model an
**explicit articulation** (definition / rules / leaf-inventory / dossier) and see whether it can
reconstruct a larger model's name-invoked policy on unseen items. The result to date is a clean
gradient:

- **Procedural / communication** (notice-and-comment 34%, grant, PR): articulation *works*.
- **Subjective judgment** (humor 7%, creative-writing, peer-review, code-review): articulation
  *mostly fails*.
- **Verifiable reasoning** (math 0%): floor — nothing helps.
- **Below the capacity floor**: articulation *hurts* — the richer/more procedural the text, the more
  it degrades a sub-capacity executor (Qwen-3B n&c: median gain −.151, 68/90 cells hurt). The
  "communication ≫ judgment ≫ verification" gradient is an **above-floor** phenomenon.

The pivot: **stop treating "articulate more" as the only lever.** Articulation is one *installation
channel*. There are others — fine-tuning, RL, weight/activation editing ("knowledge patching") — and
the user's hypothesis is that some kinds of tacit knowledge are *constitutively* reachable only
through the non-linguistic channels, because that is how humans acquire them (constant environmental
reward/punishment working its way into the tacit system).

So the reframe: **rescue an isomorphism not by saying more, but by installing through a channel that
never states the policy propositionally at all.**

---

## 2. The organizing structure: types × channels

Two axes. Collins gives the **rows** (kinds of tacit knowledge). The pivot gives the **columns**
(how you install a policy). The existing program is one column.

### Rows — Collins' three types (increasing resistance to explication)

| Type | Why it's tacit | Machine analog | Predicted natural channel |
|---|---|---|---|
| **Relational (RTK)** | contingent social relations — secret, must-be-shown, unrecognized, mismatched salience. *Explicable in principle.* | information the model lacks but could just be told | **in-context articulation** |
| **Somatic (STK)** | rooted in the body/brain as a physical system (bike-riding) | competence installable by *doing*, not by stating | **fine-tuning / imitation on demonstrations** |
| **Collective (CTK)** | located in society; acquired only by social embedding. Collins' hard limit for machines. Governs **polimorphic** action — the "same" act needs different behavior per social context (a joke, a greeting, "what's funny here") | competence installable only by being *rewarded/punished by the collective* | **RL against an environmental reward** |
| **Statistical (added, §3g)** | a **high-dimensional regularity absorbed from data**, tacit because too high-dimensional to verbalize (forecaster/radiologist/chicken-sexer gut). *Not* in Collins — it's the type that maps onto neural-net weights. Machines get it (ML does); fails Collins' CTK test. | this *is* what an NN's weights are | **fine-tune / RL on data — the frequencies, not the words** |

Collins' **polimorphic vs mimeomorphic** distinction is the mechanism for our own gradient.
Procedural tasks (n&c) are largely *mimeomorphic* — same rule, same output — so an explicit rule
transfers. Judgment tasks (humor, taste, CW) are *polimorphic* — what counts as good depends on a
socially-read context — so a rule can't capture them, and (per Collins) *no amount of articulation
can*. That is exactly why humor articulation is inert-to-harmful. **The gradient we measured is
Collins' polimorphic axis in disguise.**

### Columns — installation channels (ordered by how much the policy is rendered as explicit language)

| Channel | Policy stated propositionally? | Human analog |
|---|---|---|
| **0. name only** | no (latent invocation) | "just do it" — invoke what's already internalized |
| **A. in-context articulation** | **yes**, fully | reading the manual |
| **B. fine-tune on demonstrations** | no — only (x, label) pairs | apprenticeship by imitation |
| **C. RL on a reward signal** | no — only reward/punishment | learning what lands by being rewarded |
| **D. knowledge patching** (activation steering / LoRA graft / model edit) **(DROPPED 2026-07-21)** | no — direct weight/activation edit toward the target's judgment direction | (no clean human analog — surgical) |

The **theoretically load-bearing cut** is A vs {B, C, D}: whether the policy is ever expressed as
explicit propositional content. That cut *is* the tacit/explicit boundary, operationalized as an
intervention rather than a description.

### The matching hypothesis (the diagonal)

**Each tacit type has a matched channel; mismatched channels fail or hurt.**

- RTK → A works (already observed: procedural/n&c).
- STK → B works where A fails.
- CTK/polimorphic → **C works where A fails, and A can even hurt** (the humor sign-inversion is the
  fingerprint of trying to reach CTK by articulation).
- Math floor is **orthogonal** — it's absent *capability*, not a channel mismatch. No channel
  rescues it; separating "channel-mismatch failure" from "capability-floor failure" is itself a
  result the 2×2 makes visible.

If this diagonal holds, the paper's claim is not "articulation fails for judgment" (a negative) but
**"the channel that installs a policy must match the epistemic type of the policy"** (a positive,
predictive law).

---

## 3. Framing threads (organized, each tagged with where it bites)

### 3a. Composability — does composing tacit knowledge require symbolizing it?
> "Logical compositions and extensions of both? Or does knowledge need to become symbolic (explicit)
> to be composable?"

Working view: **logical** composition (operate on parts, extend by rule) is a property of *symbolic*
systems — you compose symbols, not holistic competences. Tacit knowledge is holistic and doesn't
obviously support logical composition. **But** skills clearly compose *behaviorally* (a cook fuses
knife-work and heat-intuition into a new dish without articulating either). So the resolution is
probably: tacit knowledge composes by **co-practice / joint reward**, not by logical operation.
- **Bites the experiment as a probe:** RL-install skill A, RL-install skill B; does A∘B work
  zero-shot, or does it require *joint* RL? If tacit skills don't logically compose, mismatched-order
  or held-out compositions should fail without joint training — a direct test of "symbolization is
  the price of composability."
- *(Audit 2026-07-21: the task-vector composition version of this test died when channel D was
  dropped; the joint-vs-separate-training version above is the surviving form.)*

### 3b. Productive vagueness — mottos, mission statements, incomplete contracts
> "Don't be Evil", "Move fast and break things", "Democracy Dies in Darkness"; Nyarko: a fully-clear
> contract produces winners/losers → zero-sum → no one signs.

This is a **distinct category, not a point on Collins' axis.** A motto is an *explicit* token (the
words are right there) whose *function* requires it to be **tacitly and divergently interpreted** by
each stakeholder. Vagueness is **coalition technology**: it lets everyone layer their own tacit
reading and thereby unifies a big tent. The many-person generalization of Nyarko's two-person
incomplete-contract point (Hart–Moore incomplete contracts; Star & Griesemer **boundary objects** —
"plastic enough to adapt locally, robust enough to keep identity across sites" — is the exact term
of art).
- **Where it bites:** it forces a third box beyond tacit/explicit — call it the
  **strategically-underspecified**: explicit surface, tacit-divergent function, and the vagueness is
  *load-bearing*, not a defect. This is about **coordination**, not competence — a different
  dependent variable (agreement/coalition-size) than reconstruction accuracy. Likely its own
  short study, not part of the channels experiment. Worth a paragraph in the paper's discussion as
  the case where *making it explicit would destroy the thing*.
- **(Audit addition 2026-07-21) The contractability half of the user's thought is the deep link:**
  in incomplete-contract theory (Hart–Moore), contracts are incomplete precisely over qualities that
  are **observable to the parties but not verifiable to a court**. Observable-but-unverifiable ≈
  tacitly-known-but-not-contractible — i.e. **contractability is the economist's operationalization
  of explicitness**, and it maps directly onto this program's own Outcome = V + A + Taste
  decomposition (`project_verifiability_explainability_gaps`): the contractible part is V, the
  articulable-but-unverifiable part is A, and the residual that makes contracts incomplete is the
  Taste/tacit component. This is the bridge between §3b and the core program, previously missed.

### 3c. What is the right axis for organizing tacit types?
> individual→org→macro? physical→social? past→present→future? objective→subjective?

These are different axes and each captures something, but for *this* program the productive one is
**locus × channel-affinity**: *where the knowledge lives* (body / dyad / collective) crossed with
*which installation channel reaches it*. Collins' own axis ("why is it tacit") already sorts by
increasing resistance and maps onto the channels in §2. The other axes (physical→social,
objective→subjective) largely **correlate** with Collins' — physical≈somatic, social≈collective —
so I'd adopt Collins as the primary axis and note the others as correlated readings rather than
competing taxonomies.

*(Audit 2026-07-21: partially superseded by §3g, which elevates the user's past→present→future
intuition into a genuine second axis — cognitive function — rather than a correlated reading. Two of
the user's axes remain genuinely unplaced: (i) **individual→micro-org→macro-org** — dismissed too
quickly; it is the natural home of the §3b coordination thread, and "does tacit knowledge change
character with the size of the collective that holds it?" is a real open question, not a redundant
axis; (ii) **understanding→predicting** — never addressed; plausibly it names the *transition*
between the `describe` and `predict` columns of the §3g grid, i.e. whether descriptive mastery is a
precondition for predictive gut or independent of it. Both parked, neither resolved.)*

### 3d. Predictive tacit knowledge & the Moneyball inversion
> wisdom of the crowd — *which* crowd? What about Moneyball, where the crowd's gut is wrong and the
> formalized language is right?

Crucial corrective: **tacit ≠ correct.** Moneyball is the case where explicit statistical language
*beats* the tacit gut — the gut was un-formalized bias, and articulation revealed it. In our frame:
- "crowd gut" = the base model's **name-invoked** policy (channel 0).
- "formalized language" = the **articulation** (channel A).
- A domain's position on the gradient *is* its position on the **formalizability spectrum**:
  where articulation *helps* over name-only (n&c) is the Moneyball-tractable end; where it *hurts*
  (humor) is the irreducibly-polimorphic end.

So the gradient is not "how hard is the task" but "how much of the policy is formalizable without
loss." That reframing is clean and it's already what the data show.

### 3e. Is something tacit *until there is a language for it*? — the expanding explicitness frontier
> tacit predictive judgments stay tacit until we can quantify them with probability. How much is
> explicit only with *yet-to-be-invented* languages? Is this what the scaling-law work does? Do we
> see *emergence of new languages* or are prompt/OSL bounded by *known* language?

This is the deepest thread and it ties straight to the OSL / articulability-scaling line. Restated
as a measurable claim: **the tacit/explicit boundary is not fixed — it is relative to the available
representational language**, and the boundary moves when a new formalism is invented (probability
theory made a whole class of tacit judgment explicit).
- **The language-frontier probe:** when articulation *does* rescue a cell, does it succeed by
  **recombining vocabulary already present** in the domain's lexicon, or by introducing genuinely
  **novel formalization**? If rescue is always recombination, LLMs are **bounded by known language**
  (the explicitness frontier is fixed by the training corpus). If cross-family we see rescue via
  novel constructs, the frontier **expands with capability** (training technique / synthetic data /
  data efficiency pushing it out). This is answerable inside the existing apparatus and is a strong
  standalone finding either way. Connects to `project_metric_lexicon_census`,
  `project_mention_auc_state_channel`, OSL executor-scaling.

### 3f. Internal consistency — necessary for all knowledge?
> internally consistent systems that are tacitly known — is internal consistency necessary for all
> knowledge, explicit and tacit?

Working view: **explicit systems can be *checked* for consistency — that is much of what
formalization buys you.** Tacit systems can be *locally* consistent while *globally* inconsistent,
and the inconsistency is often **invisible until articulation surfaces it** (the classic "I can't
say why, and when I try to state the rule it contradicts itself"). So internal consistency may be a
*consequence* of explication rather than a precondition of knowledge. Testable corollary: an
RL-installed (tacit) competence and its own post-hoc articulation may **disagree** on edge cases —
and the size of that disagreement is a measure of how much the competence exceeds what language can
hold (see §4).

### 3g. Predictive tacit knowledge — the axis Collins is missing (2026-07-21)
> Collins doesn't discuss the *gut instinct for a prediction* — is it a subset of Collective TK, or
> its own thing?

**Neither.** Collins has no *predictive* category and, more fundamentally, **no functional axis at
all** — his three types only answer *"why does this resist explication?"* (contingency / body /
society). Prediction is a *job the knowledge does*, so it's orthogonal to his carving. And predictive
gut is **not** a subset of Collective TK: the paradigm cases (radiologist, 30-year forecaster,
chicken-sexer) live in **one brain**, expert guts **disagree**, and a **machine with the same data
gets them** — all of which fail Collins' test for CTK (shared, socially-embedded, machine-inaccessible).

What predictive gut reveals is a **fourth source of tacitness** Collins underweights —
**statistical / inductive tacitness** (§2 new row): tacit because it's a high-dimensional regularity
absorbed from data, too high-dimensional to verbalize. This is exactly what a neural net's weights
are, and it's the **Moneyball battleground**: gut (tacit statistics in weights) vs model (explicit
statistics in a regression) — same inductive knowledge, two encodings; the explicit one wins when the
tacit sample was biased. "*Which* crowd?" (the wisdom-of-crowds question) = *which sampling
distribution generated the internalized statistics*; crowds are wise when errors are independent and
cancel, Moneyball when they share the bias the gut absorbed.

**The fix is two orthogonal axes, not four siblings** — cross Collins' *source-of-tacitness* with the
*cognitive function* (which is just the [[project-four-language-functions]] axis:
describe / predict / form-coordinate / regulate; the user's own "past→present→future" scattered-axis
is this one):

| source ↓ / function → | **describe** | **predict** | **form / coordinate** | **regulate** |
|---|---|---|---|---|
| **Relational** (told) | undocumented recipe / trade secret | privileged insider info | in-group password / shibboleth | unwritten house rules |
| **Somatic** (drilled body) | *mouthfeel words — always undershoot* | athlete anticipating trajectory | ensemble/rowing entrainment | flinch, trained restraint, surgeon's hand |
| **Collective** (socially embedded) | thick concepts ("what's *rude* here") | reading a room / market | **motto / boundary object** (the vague-symbol box) | knowing-when-to-break-the-rule |
| **Statistical** (absorbed from data) | *→ becomes the regression when made explicit (Moneyball)* | **forecaster / radiologist gut** | *(weak — coordination is social)* | internalized risk limits (trader's PnL sense) |

**What filling the grid reveals:** the **`describe` column is where tacitness goes to die** — it's
thin for Somatic and Statistical *precisely because* describing = making explicit, which is what
those types resist. And that column is exactly where parts 1–2 (articulation) live. So the grid
*explains the measured gradient*: articulation tries to drag knowledge into `describe`; types with no
natural `describe` cell (Somatic, Statistical) are where it fails. Structurally: **Statistical
concentrates in `predict`; Collective owns `form`; Somatic owns bodily `coordinate` + reflexive
`regulate`; Relational is uniform-but-shallow** (all explicable, just unstated).

**Payoff for part 3:** (1) it sharpens the matching hypothesis — the FT/RL channel installs
*statistical* tacit knowledge, so RL is "the installer for anything absorbed-from-frequencies," not
only "for Collective TK"; (2) it names a **test domain we'd missed** — a genuine **forecasting task**
is the *cleanest* probe of statistical TK, because it isolates `predict × statistical` without the
social/polimorphic tangle that humor carries (humor mixes "will this land?" with collective reading).
Caution held on myself: "statistical tacitness" earns its place only because it makes these two
predictions — not because it's tidy. Polanyi would say all tacit knowing shares one from-to
structure; the reply (Collins' and mine) is that the *reasons* and *installers* differ, and for a
model that difference is the whole experiment.

**(Audit correction 2026-07-21) The "fourth type" claim was overstated — it contradicts §2's own STK
definition.** The STK row says "rooted in the body/**brain** as a physical system"; by that
definition the radiologist's gut (statistics in neural weights) *is* somatic, and Collins would
likely file it exactly there — his STK is the explicable-in-principle, mechanizable tier, which is
why machines getting it is no embarrassment to him. Downgraded claim: statistical tacitness is a
**refinement within Collins' somatic bucket, not a refutation** — for machines, somatic splits into
*motor-performative* (no LLM analog; the vestigial row) and *data-inductive* (installs via FT), and
the split earns its keep solely through the two §3g predictions. What stands untouched is the
**functional-axis** point: Collins genuinely has no `predict` column, and that half of the argument
carries the grid.

---

## 4. The operational payoff: an in-model tacitness assay

The channels frame gives a *measurement* of tacitness that doesn't exist yet:

**Executable-but-inarticulable.** Suppose channel C (RL) rescues a humor cell — the executor now
reconstructs the target's judgments at adverse-ρ ≥ .70. Now ask that same rescued executor to
**articulate its policy**, take that articulation, and hand it to a *fresh* executor (channel A).
- If the articulation transfers → the competence was **linguistically expressible** after all (RTK
  in disguise).
- If it does **not** transfer → RL installed a competence the model can *execute* but cannot *state*:
  the operational definition of **genuinely tacit knowledge installed in a model.**

The **articulation-transfer gap** (C-rescue minus A-from-C's-own-words) is a scalar tacitness score,
per cell, with no human labels anywhere. This is the thing §3e and §3f are both circling, made into
a number.

---

## 5. Experiment design — a 2-D exchange-rate surface (reuses the isomorphism apparatus)

Same estimand as Sub-line A — reconstruct a larger model's **name-invoked** judgment vector on
**unseen** items, adverse-ρ readout, same-family, **no external ground truth**. The change: instead of
sweeping only the articulation axis, sweep a **2-D dose-response grid** (per the §-intro correction):

```
                intervention dose  →   0(none)   FT@N   RL   patch
 articulation   name only (0)          [parts1-2 left column]  ...
 dose  ↓        + definition                 .        .    .    .
                + rules                       .        .    .    .
                + dossier (full)              .        .    .    .
```

Each cell = adverse-ρ. **The headline is not a binary rescue but the exchange rate** — the iso-ρ
contours: *how many tokens of articulation does N examples of fine-tuning buy?* (Precedent: the
`z×a` exchange-rate framing in the OSL line, `project_osl_executor_scaling`.) "Rescue" is the corner
case where the left column is flat and an intervention lifts it.

- **Targets / executors:** existing 70B/72B/27B name-invoked vectors + small same-family rungs.
- **Label-free discipline preserved:** every intervention trains on the **target model's own
  name-invoked judgments** on the TRAIN split — the target is the oracle, never human ground truth;
  evaluate adverse-ρ on held-out items (`feedback_reconstruction_only_no_labels`).
- **Domains:** procedural (n&c), polimorphic-judgment (humor/CW), **a pure forecasting task**
  (`predict × statistical`, per §3g — the clean statistical-TK probe), capability-floor (math)
  (`feedback_all_three_label_types_per_task`).

### What each channel actually is (the substance)

**B — Fine-tune = distillation (yes, exactly).** Soft-label distillation: minimize KL between the
executor's judgment distribution and the **target's name-invoked score distribution** over TRAIN
items (hard-label SFT on the target's argmax is the cheap fallback). This is **behavioral cloning of
the policy**, not learning the task — the executor is supervised toward *the target's judgments*, not
toward correctness. Cleanest, lowest-risk channel; run first.
- *Saturation caution:* distilling on enough of the target's labels for the exact construct can just
  clone it (ρ→1) and make the exchange rate degenerate. So the interesting regime is **low-N** — how
  *few* demonstrations match K tokens of articulation — and the guard is **held-out items** (already
  in the estimand) plus, ideally, **held-out sub-populations / cross-construct transfer** to
  distinguish *installed a policy* from *memorized items*.

**C — RL: only justified with a reward that is NOT a per-item label.** This is the crux, and the
honest answer is: **plain pointwise "reward = match the target's label" RL is pointless — SFT on that
same label strictly dominates it** (RL sees *less* information per example than the label it's derived
from). RL earns its place only when the reward carries something a supervised target can't:
- **(c1) Reward an emitted *rubric/rationale* by its downstream reconstruction effect.** The
  executor's *action* is to write an articulation; the reward is how well that text, fed to a frozen
  executor, reconstructs the target. This is **learned articulation = GEPA-by-RL** — and it is the
  version that makes parts 1–2 and part 3 *the same pipeline*: RL **produces** the explicit text
  parts 1–2 evaluate. (We already have GEPA in the ecosystem; this is the natural, in-house form of
  "C.") **Recommended primary RL arm.**
- **(c2) Comparative / ordinal reward.** Reward pairwise-order agreement with the target rather than
  pointwise value — RLHF-style. Motivated because the **estimand itself is rank-based (adverse-ρ)**,
  so an ordinal reward is better-aligned to the objective than pointwise supervision.
- **(c3) Reward from a separate environment/judge** (not the target's stored label) — genuinely
  can't-supervise. Use only if such a signal exists for the domain.
- If none of (c1–c3) apply for a domain, **don't do RL there** — do distillation and call it
  distillation (`feedback_no_overengineering`).

**D — Knowledge patching = install the policy as a portable vector/module, then graft it.** The
user's intuition ("an M_ω-equivalent elsewhere, then patching = putting it together") is exactly
**task-vector arithmetic** (Ilharco et al.): θ_finetuned − θ_base is a portable "policy vector" you
can *add* to a fresh model. Concretely, three variants in increasing surgery:
- **Activation steering:** derive the residual-stream *direction* separating the target's high- vs
  low-judgment items (contrastive), add it to the executor at inference — no training, cheapest patch.
- **LoRA graft / task-vector add:** take the adapter from the B (distillation) run and *graft/add* it
  to a fresh model — tests whether the policy is a **portable, composable object** (this is where the
  §3a composability question becomes testable: does humor-vector + CW-vector compose without joint
  training?).
- *Not* ROME/MEMIT: those edit **factual key-value associations**, and a diffuse judgment policy is
  not a fact — wrong tool, don't reach for it.
- Most speculative arm; high theory-payoff (portability/composability), low load-bearing weight for
  the core claim. **DEPRIORITIZED (user, 2026-07-21) — not pursuing patching.** Left here for record.

### 5.1 The downstream-reward design (the behaviorist core — user, 2026-07-21)

The strongest form of C is **not** "reward = match the target's per-metric label" (a static label →
SFT dominates it, §5-C). It is **reward = a distal, aggregate outcome** that the subjective metric
only *partly* drives. Behaviorist claim: **tacit knowledge is environmentally reinforced** — you are
never told "that was elegant," you only observe "accepted / rejected," and *elegance is reconstructed
from its footprint on the outcome*. The unarticulable metric is installable precisely *because* it is
never labeled, only reinforced.

**Metric-subjectivity spectrum (user's peer-review example):**

| metric | how preference-defined | articulable? |
|---|---|---|
| "experiments support the claims" | least — near-objective | high (parts 1–2 already reach it) |
| "has enough baselines" | somewhat | medium |
| "is elegant" | most — pure preference | ~none (no `describe` cell) |

**Headline hypothesis (pre-registerable slope):** the **advantage of the reward channel over
articulation rises with metric subjectivity.** x-axis = metric articulability (already measurable via
the codability / lexicon census); y-axis = ρ(reward-installed) − ρ(best-articulation), each
reconstructing the target's name-invoked metric vector on held-out items. **Upward slope confirms the
behaviorist story: reward buys most where words buy least.**

**Identifiability — the make-or-break caveat.** A distal aggregate reward installs the *gestalt*
("what predicts acceptance"), **not** isolated elegance. To isolate the subjective component you must
**residualize the outcome against the objective/articulable metrics** — regress the outcome on
{claims-support, baselines, …} and treat the **residual** as the target signal. That residual *is*
"the tacit stuff articulation can't reach," and whether reinforcement on it reconstructs the target's
*elegance* vector is a **self-checking test** (residual reconstructs elegance → residual ≈ elegance;
if not → residual is some other unarticulated bundle — reportable either way). Omitted-variable risk
is real: any unlisted subjective driver leaks into "elegance," so claim "reward installs the
**residual tacit component**" and let the reconstruction test say how much of it is elegance.

**Is this even RL?** Honest answer: with a **static corpus** of (item, outcome) pairs the downstream
reward is a fixed (confounded, distal) *label* → the minimal correct tool is **supervised regression
on the residualized outcome, NOT policy-gradient RL** (`feedback_no_overengineering`). The
behaviorist "reinforcement" is real at the level of the *knowledge source* (a distal aggregate
signal) but does not require RL *machinery*. RL re-enters only if the model must **act/generate**
(emit a judgment and be scored by downstream match, RLVR-style) — and even then residualized
regression is the baseline it must beat.

**In-model, to stay on-discipline.** Keep both signals as LLM-judge outputs of the target: reward =
the target's *holistic* accept/reject; reconstruction goal = the target's *componential* elegance
judgment. No human labels; LLM judges do all measurement
(`feedback_llm_judges_do_all_measurement`, `feedback_reconstruction_only_no_labels`). Using **real**
ICLR acceptance as the distal reward is a tempting external-validity extension but introduces a
human-generated label — **flagged, needs sign-off** (`feedback_check_before_new_approach`), not v1.

**Domain:** this makes **peer review the natural first testbed** (supersedes forecasting as the first
deep dive) — the subjectivity spectrum lives *within one domain* (so the slope is a within-domain
readout) AND there's a real downstream outcome, and it's the user's own example. Data exists (ICLR
sub-scores + meta-decision; `project_peer_review_va`).

### Predictions (pre-register before running)
1. **Peer review, the §5.1 slope — HEADLINE:** the reward channel's advantage over articulation
   **rises with metric subjectivity** (elegance ≫ enough-baselines ≫ claims-support), **after
   per-metric reliability correction** (§6 — without noise ceilings the slope is confounded by
   differential judge reliability and is not interpretable).
2. **Humor/CW:** an intervention (§5.1-style or B) shifts the curve up where articulation is
   inert/harmful — **scoped to a fixed executor rung, on the cells A demonstrably does not reach at
   that rung** (our own Qwen-14B interim shows A rescues ~20% of humor cells; the unscoped claim "A
   fails for judgment" is falsified by our own ladder).
3. **Forecasting (`predict × statistical`):** FT/distillation (B) rescues cleanly and **articulation
   barely moves it** — the sharpest B≫A contrast, because statistical TK has no `describe` cell.
4. **n&c:** A already works; B/C match but don't exceed — articulation is *sufficient* for RTK; here
   the interesting number is the **text saved**, not a rescue.
5. **Math:** no channel rescues — capability floor, not channel mismatch (the control that proves the
   axis is real).
6. **Articulation-transfer assay (§4):** humor/forecasting intervention-rescue does **not** survive
   re-articulation; n&c A-success **does** — separating tacit-installed from explicit-installed
   competence.

---

## 6. Cautions / honest limits (so we don't over-claim)

- **Simulacrum of the collective.** RL-ing an executor on *one* target model's labels is not social
  embedding — the target is a controllable **stand-in** for "the environment that rewards/punishes."
  We can honestly claim we're testing whether the *form* of the channel (propositional vs
  reward-shaped) matters for reconstruction; we cannot claim the model acquires human CTK. State this
  plainly.
- **RL-on-one-oracle can degenerate to imitation.** Need controls that C isn't just SFT-with-extra-
  steps: e.g. reward on *rank/ordinal* agreement, held-out items, and check C reaches cells B can't.
- **(Audit) The Collins mapping must not resurrect the KILLED codification hypothesis.** Patents was
  predicted ≳15% from MPEP codification and landed 4%; "mimeomorphic → rule transfers" is the same
  claim in new vocabulary unless disciplined. As used post-hoc the polimorphic axis is unfalsifiable
  (any failure can be reclassified "actually polimorphic"). Guard: **pre-register every domain's
  polimorphic/mimeomorphic classification before the axis is used as an explanation**, and patents is
  the standing anomaly any mapping must survive. If the pre-registered classification doesn't predict
  the gradient, drop the Collins gloss and keep the operational finding.
- **(Audit) Reliability confound in the §5.1 slope.** Subjective metrics plausibly have lower judge
  test-retest reliability → ρ is mechanically more attenuated for elegance than for claims-support →
  an upward slope can be a measurement artifact. Per-metric noise ceilings / reliability-corrected ρ
  are a **precondition** for the headline, not a robustness check.
- **(Audit) Scope the premise by capacity.** "Articulation fails for judgment" is capacity-relative
  (Qwen-14B rescues ~20% of humor). Every part-3 claim is indexed to a fixed executor rung and to the
  cells A doesn't reach *at that rung*.
- **Exchange-rate commensurability.** FT is a one-time capital cost; articulation is an operating
  cost per call. Iso-ρ contours stay meaningful but "N examples ≈ K tokens" must be glossed
  amortized-vs-marginal (same structure as the construction-vs-runtime seam, §31 of the seam line).
- ~~"Knowledge patching" (D) is the riskiest arm~~ — **D dropped per user 2026-07-21**; bullet kept
  struck for record.
- **Don't conflate the floor with tacitness.** Math failing under every channel is the *good* control
  — but only if we keep it separate from the humor channel-mismatch story in every figure
  (`feedback_same_family_scaling`, never pool).
- This note **records a hypothesis**; none of §2–§5 is a result. The sign-inversion (§1) is the only
  measured fact here.

---

## 7. What to decide before building anything

1. **Channel order = §5.1 downstream-reward (residualized) → B distillation → C(c1).** D (patching)
   **dropped** per user. The §5.1 design is now the *core* — it is the only channel whose signal is
   genuinely unavailable to articulation. B is the control/baseline it must beat; C(c1) is the bridge
   back to parts 1–2. Skip pointwise-label RL entirely (SFT dominates it).
2. **First domain = peer review** (supersedes forecasting): it carries the metric-subjectivity
   spectrum *within one domain*, so the headline slope (reward-advantage vs subjectivity) is a
   within-domain readout, and it instantiates the user's own example. Forecasting stays as the clean
   `predict × statistical` probe; humor as the where-A-*hurts* case.
3. **The DV is the exchange rate, not a rescue count.** Design the sweep as the 2-D dose grid (§5) so
   the deliverable is iso-ρ contours ("N examples ≈ K tokens of articulation"), which also covers the
   cells where articulation already works.
4. **§4 articulation-transfer assay** in v1 or follow-up? Most novel measurement, one extra pass.
5. **Language-frontier probe (§3e) — do this first, it's free.** Re-analyzes which vocabulary the
   *existing* certified/high-ρ articulation rescues used (recombination vs novel formalization). No
   GPU, real finding either way, and it directly informs whether C(c1) can even reach past known
   language.

---

## 7b. TK-local vs TK-general (user, 2026-07-22 — "GTK/STK") + the differentiation-floor result

> "STK is task-specific, learnable, doesn't require generalization... GTK: if I have tacit
> knowledge in one area, can I tacitly make a decision in a different area without having to
> absorb tacit knowledge? STK is pretty easily learned from SFT. Not sure how GTK is learned or
> how to test for it."

**Naming:** user's STK collides with Collins' Somatic TK — in this program write **TK-local**
(user's STK: specific) and **TK-general** (GTK). G1 was signed off in the same message (battery
approved; G2 prereg still gates confirmatory Phase-2 runs).

### 7b.1 The distinction, made measurable: a decay curve over a scope ladder

Don't treat GTK as a binary property. Transfer of installed tacit knowledge is tested at each
rung of `item → construct → task/domain → global`; **TK-local = transfer that dies at the
construct boundary; TK-general = slow decay across rungs.** "Can GTK be domain-specific?"
dissolves: it's a curve shape, not a category. Two distinct operationalizations (different
mechanisms — keep separate):
1. **Zero-shot transfer:** train on constructs c1..ck, measure adverse-ρ toward the TARGET's
   policy on unseen construct/domain (reconstruction-only stays intact: "generalize" = move
   toward the target's policy elsewhere, never toward truth).
2. **Meta-acceleration (the deeper claim):** GTK = a SHIFT IN THE EXCHANGE-RATE CURVE on new
   tasks — fewer examples to install the next tacit policy (Harlow learning-sets). Measured with
   the §5 tally unchanged; only the x-axis becomes "prior tacit training."

### 7b.2 The GTK ceiling is measurable BEFORE any training — and it produced a discovery

Zero-shot GTK transfer toward the target is bounded by the **shared variance of the target's own
construct policies** (if the 90 policies are orthogonal, transfer is impossible by construction).
Ran the factor analysis on existing npz (2026-07-22, 90 name-policy vectors per domain, PC1 share
of the construct×construct Spearman matrix):

| job | n&c PC1 | humor PC1 | math PC1 |
|---|---|---|---|
| Llama-70B target | 39.1% | 45.9% | 54.1% |
| **Llama-8B exec** | **66.6%** | **76.2%** | **91.3%** |
| Qwen-72B target | 37.8% | 51.4% | 63.4% |
| **Qwen-3B exec** | **48.9%** | **63.4%** | **86.6%** |
| Qwen-14B exec | — | **38.9%** | — |
| Gemma-27B target | 49.1% | 51.1% | 74.2% |
| Gemma-12B exec | — | 53.3% | — |

Findings:
1. **Ceiling exists:** targets carry a real general evaluative factor (38–74%; mean
   inter-construct ρ .34–.73) — zero-shot GTK transfer has genuine headroom.
2. **DIFFERENTIATION-FLOOR DISCOVERY:** small executors are far LESS differentiated than their
   targets — Qwen-3B math is 86.6% one factor (it judges 90 different constructs with
   essentially ONE policy); Llama-8B humor 76.2%. Differentiation increases with scale, and
   **reaches target level exactly at the 12–14B class where the articulation-rescue unlock
   happens** (Qwen-14B humor 38.9% ≤ 72B's 51.4%; Gemma-12B 53.3% ≈ 27B's 51.1%). Mechanistic
   candidate for the capacity floor: **below the floor the executor cannot maintain
   construct-specific policies at all, so an articulation has no differentiated policy-slot to
   configure; rescue turns on when differentiation does.** (Also explains the below-floor sign
   inversion: rich text loads a one-policy judge with content it cannot route.) Prereg a
   confirmatory version before claiming (correlate per-rung PC1 with per-rung conditioned-rescue
   % across all rungs/domains; noise caveat: undifferentiation is not noise — noise would LOWER
   correlations, small models show HIGHER).
3. **Conceptual corrective for GTK:** a high general factor in a weak model is
   UNDIFFERENTIATION (failure to individuate constructs), not general tacit competence. GTK-as-
   competence must be measured as transfer toward the TARGET's structure (its shared factor),
   never as self-generality of the executor.
4. Domain ordering (math most collapsed 54–91%, n&c most differentiated) matches the gradient.

### 7b.3 GTK probe design (v1, within existing apparatus)

Fleet: Qwen-7B (below unlock) + 14B (at unlock). Adapters distilled on m ∈ {1, 4, 16} humor
constructs (target policies), plus a **shuffled-label control adapter** (installs nothing real).
Test battery per adapter:
- (i) sanity: name-only ρ on trained constructs (TK-local installed?);
- (ii) zero-shot Δρ on held-out humor constructs (within-domain GTK);
- (iii) zero-shot Δρ on CW / n&c constructs (cross-domain GTK);
- (iv) low-N exchange-rate on NEW constructs, trained-adapter vs fresh (meta-acceleration);
- (v) **contrast-pair flip agreement** (beyond-memorization probe, §7b.4).
Readouts: the transfer decay curve; slope of transfer vs m (breadth hypothesis); size
interaction (scale hypothesis). All threshold-free.

### 7b.4 Probing beyond memorization ("how is knowledge used")

Maps onto the storage/extraction/manipulation split (Allen-Zhu & Li, Physics of LM 3.x — the
"how is knowledge used" reference): held-out items only test extraction; GTK needs manipulation
probes:
- **Contrast-pair flip agreement:** generate minimal pairs flipping the feature the policy
  tracks; a policy-holder's judgment flips WITH the target's; an exemplar-memorizer's doesn't.
  (LLM generates pairs; target's flip = reference; executor scored on flip agreement.)
- **Composition:** judge by c1∧c2 jointly without joint training (§3a probe).
- **Inverse probe:** from a judgment pattern, identify the construct (existing reconstruction-
  MCQ machinery — `feedback_reconstruction_mcq_default`).
- **Articulation-transfer assay (§4)** — already designed; completes the battery.

### 7b.4½ WHAT MAKES A TRANSFERRED POLICY *TACIT*? — the channel-difference discipline
(user challenge, 2026-07-22: "policies can also be explicit — how are your tests
differentiating truly tacit policies?")

The challenge is correct: policy-transfer measurements are neutral between tacit and explicit.
**Tacitness in this program is never a property of one measurement — it is always a CHANNEL
DIFFERENCE**, a relation between (policy, carrier medium, agent pair). Operational definition:
a structure is tacit-relative-to-(articulator, executor, language-space) iff the language
channel fails to carry it while another channel (weights, demonstrations) succeeds. Polanyi's
"we know more than we can tell," measured as the gap between what telling transfers and what
other channels transfer. Three design rules, now binding on every GTK/part-3 test:

1. **Selection rule:** run tacit-transfer tests on the ARTICULATION-FAILURE set (gap-conditioned
   cells where the policy demonstrably exists — native gap — but the language channel
   demonstrably fails). There, any channel that succeeds is carrying something
   beyond-current-language for that executor by construction.
2. **Matched language arm everywhere:** every transfer cell (train on A → effect on B) gets a
   control arm: hand a fresh executor the BEST articulation of A's policy and measure the same
   B-effect. Define **tacit-transfer(A→B) = transfer_weights(A→B) − transfer_language(A→B)**.
   If the sentence does the same work as the gradient, what transferred was explicit; only the
   channel gap is tacit content. (Catches exactly the "verbalizable regularity" confound.)
3. **Self-articulation loop-closer (§4 assay):** after weight-transfer, the trained model
   articulates what it gained; if that text reproduces the gain in a fresh model, the gained
   structure was explicit — or, importantly, HAS BECOME explicit (installation may
   explicitate). Either way the tacit residual shrinks by exactly the measured amount.

**Scope honesty (the "truly tacit" limit) — upgraded to a THREE-TIER certification ladder
(user, 2026-07-22: "our prompt upper bounding should play a role here"):** we can never certify
tacit-in-itself, but the claim strength is graded by how much of the language channel is closed:
- **Tier 0 — bounded non-discovery:** the searched articulation bank fails (weakest; the
  Collins-test discipline, RECONSTRUCTION_V2.md).
- **Tier 1 — optimized non-discovery:** articulation search UNDER OPTIMIZATION PRESSURE fails
  (GEPA c1 with reconstruction fitness) — the language space searched is now the reachable
  neighborhood of an optimizer, not a frozen bank.
- **Tier 2 — CERTIFIED tacit residual:** the weight channel exceeds a **certified all-prompt
  upper bound on the articulation channel** — the DPI fixed-target cap from the M_ω audit line
  (`project_momega_audit_bracket`: the only certified all-prompt bound; OPT_Ω+ε/Fano/CR-ε are
  RETRACTED, do not substitute them). When ρ_weights(cell) > C_prompt-cap(cell) + margin, the
  gap is tacit relative to the ENTIRE prompt space under the cap's assumptions, not just the
  prompts searched. Disciplines carry over: fixed-target, freeze-before-eval, never mix M_i
  scorings in one figure.
Phase 0 tightened Tier 0 (rescues use only known vocabulary). Every claim states its tier. The
outcome trichotomy at every level — {explicit: telling works} / {tacit-installable: only
weights/demos work} / {floor: nothing works} — mirrors metric_seam's articulable_only /
verifiable_only / neither verdicts, with weights as the third medium.

**(Lit-review upgrade, 2026-07-22): the tier ladder now carries NAMED PHILOSOPHICAL STAKES.**
The sophisticated intellectualist null (Stanley & Williamson; Fodor strongest form) does NOT
predict text=practice — Stanley concedes testimony insufficiency. The camps are distinguished
by exactly ONE test: does the articulation gap CLOSE ASYMPTOTICALLY under text optimization
(intellectualism vindicated: the gap was propositional-content-shaped) or PERSIST at any
richness against a certified cap (the Polanyi/Ryle/Dreyfus camp vindicated)? That is precisely
Tier 1 (GEPA-optimized search) → Tier 2 (cap) — the ladder is the operational form of the
central 70-year dispute in this literature, and the catalog reviewer independently recommends
it as the headline confirmatory test. (Full menu: notes/2026-07-22__tacit-knowledge-
operationalization-catalog.md, 28 entries and counting.)

### 7b.4¾ EXP-GTK-1 — the concrete experiment (what A and B actually are)

**Executor:** Qwen2.5-7B (largest below-unlock rung; full articulation-failure set; cheap).
Secondary replication: Qwen2.5-14B (at-unlock). **Target:** Qwen-72B name-invoked policies.

**A (training set):** 16 humor constructs from the 7B **articulation-failure set** (gap > .10,
best articulation < .70), stratified over R1/R2/R3, **items = stable-hash half-1 only**.
**Item-disjointness rule (binding):** within-domain transfer is evaluated on item-half-2 ONLY.
Constructs share the 90-item pool, and the target's policies correlate across constructs (the
general factor, §7b.2) — so training on A's (item, label) pairs and testing B on the SAME items
would let pure item memorization masquerade as structure transfer. Disjoint items make whatever
transfers item-general by construction. (Cross-domain B is naturally item-disjoint.)

**Adapter arms (all N=128/construct, same hyperparams, same trainer):**
| arm | labels | what it isolates |
|---|---|---|
| `real` | target's true per-construct policies on A | the candidate tacit transfer |
| `shuffled` | item-shuffled labels | format/test-taking adaptation (installs no signal) |
| `construct-permuted` | REAL target policies, wrong construct assignment | "coherent-judgment-style" learning + generic attentiveness, without the right content |

**B (test batteries), all scored per-arm with the adapter-aware teacher-forced readout:**
- B1: 20 held-out humor failure-set constructs, item-half-2 (within-domain zero-shot)
- B2: 10 humor articulation-SUCCESS constructs (dissociation: does weight-transfer help where
  language already works?)
- B3: creative-writing failure-set constructs (near-domain)
- B4: n&c constructs (far-domain)
- B5: exchange-rate re-training (N ∈ {8,32}) on 4 fresh constructs, real-adapter vs fresh
  executor (meta-acceleration)

**The reported quantity is always the double difference:**
  GTK(B) = [Δρ(B | real) − Δρ(B | shuffled)] and separately vs construct-permuted —
one control subtracts format adaptation, the other subtracts content-free judgment-style gains.

### 7b.4⅞ The "learned to reason explicitly" thorn — what the instrument already settles and
what it can't

The worry: A-training might not transfer tacit structure — it might teach the model an explicit
strategy (statable rule, deliberate reasoning) that produces the same Δρ. Decomposition:

1. **Calibration/temperature/base-rate learning CANNOT produce Δρ at all** — the readout is
   Spearman on teacher-forced P(YES), invariant to any monotone recalibration. The threshold-
   free rank instrument closes this confound by design.
2. **Explicit deliberation at inference CANNOT be the mechanism of anything we measure** — the
   teacher-forced readout is a single forward pass with zero intermediate tokens; there is no
   token budget in which step-by-step reasoning could run. Measurable contrast anyway
   (**CoT-delta arm**): score B1 both teacher-forced and with CoT-then-answer. Transfer gain
   that appears ONLY with CoT = explicit-reasoning mechanism; gain present teacher-forced is
   deliberation-free. (Also directly interesting: does installation move knowledge from the
   token-dependent to the token-free regime?)
3. **Internalized explicit rule (statable, applied in one pass)** — the genuinely hard residual
   case. Operational resolution, not metaphysical: run the §4 self-articulation assay on the
   trained model. If its stated rule reproduces the B-gain in a fresh executor → what was
   learned is EXPLICIT by the statability criterion (and became so through training —
   explicitation, itself a reportable outcome). If not → tacit-relative-to-its-own-articulation,
   Tier 0/1/2 per the ladder. **The tacit/explicit distinction bottoms out operationally in
   exactly two axes: statability (assay) × token-dependence (CoT-delta).** A one-pass,
   unstatable, rank-preserved transfer is tacit in every sense this program can define — and we
   assert no sense beyond that.
4. **Generic attentiveness** ("learned to read items carefully") — survives 1–2, but it is
   content-free, so it appears equally under `construct-permuted`; the double difference
   removes it.
5. **Item memorization** — killed by the item-disjointness rule (see 7b.4¾).

### 7b.4⅞+ Thorn (a) RESOLVED as the PROMPT-SUBSPACE CAP (v0 ran 2026-07-22)

The DPI cap doesn't port directly (it's fixed-target, M_ω-setting); the executor-setting analog
is built from a measurable structural fact instead: every articulation induces a score vector
over the same items; the observed family (~57 arm×form variants/cell) spans the executor's
prompt-reachable behavior subspace; **if the subspace saturates, the cap =
max corr(target-rank, span) bounds ANY articulation blend.** Guards: fit blend on item-half-1,
report OUT-OF-SAMPLE cap on half-2; saturation curve per cell; assumption stated (unseen
prompts stay in the subspace — GEPA c1 doubles as the designated escape test).
`channels/eval/prompt_subspace_cap.py`. **v0 result (Qwen-7B humor, 90 cells): median cap_oos
= .59; 83/90 cells capped BELOW .70; median effective rank of the prompt-behavior space = 7**
(fifty-seven prompt variants collapse to ~7 directions — the few-policy-slots signature again,
independently of §7b.2). Saturation strict-criterion holds for 26/90 (bank arms are correlated
variants; diverse-prompt battery + GEPA escape test strengthen or falsify). Reading: at 7B the
humor failure set is not merely unsearched — it is **capped** (Tier ~1.9, "certified given
saturation"); an adapter beating a cell's cap_oos is a per-cell tacit residual over the
observed language channel. Artifacts: outputs/tacit_channels/subspace_cap_q7b_humor.jsonl.

### 7b.4⅞++ EXP-GTK-1 PREREG FROZEN + what the 2×2 implies for each learning channel

Prereg: `notes/2026-07-22__exp-gtk-1-prereg.md`, **sha256 011a7ec8bd61497cbd5dd24881393936
988539f0cdaac4645b149b6f5a1787a9**, frozen before any adapter training. P1–P8 directional;
builder `channels/gtk/build_exp_gtk1.py` (deterministic sha-ordered stratified selection,
50/50 item halves, arms real/shuffled/construct-permuted).

The statability × token-dependence 2×2 assigns each installation channel a predicted
DESTINATION QUADRANT — that is its implication for learning strategies:

| channel | deposits into | frozen prediction |
|---|---|---|
| A in-context articulation | statable (it IS a statement) × token-free (readout forbids tokens) → works only if the executor can COMPILE a stated rule into one pass | the 12–14B floor is partly a **compilation floor** (P8: articulation+CoT partially rescues below it — tokens as scratch-space for missing policy-slots) |
| B soft-label distill | token-free BY CONSTRUCTION (gradient target is the one-pass readout); statability unknown → the §4 assay measures whether SFT **explicitates** | failure-set gains land unstatable (P7) and token-free (P6) = tacit proper |
| CoT-distill (variant, later) | statable × token-dependent | same competence as B in a different encoding → the Moneyball contrast INSIDE one model |
| GEPA c1 | statable by construction | doubles as subspace-escape test (Tier 1) |
| §5.1 reward/residual | token-free, predicted unstatable | the most tacit-targeted channel |

So the axes are not commentary — they are the experiment's dependent variables: **which quadrant
the knowledge lands in, and whether training moves it between quadrants** (explicitation =
unstatable→statable; compilation = token-dependent→token-free).

### 7b.4⅞+++ GTK construct REOPENED (user, 2026-07-22) — convergent multi-operationalization

User ruling on "what should GTK mean": **all of the above and none of them alone** — "maybe we
need many operationalizations and hopefully they will converge." The capstone's measurement
strategy is therefore **construct validation by triangulation**: a TACITNESS/GENERALITY PROFILE
per (construct, model) across operationalizations, not a single score. A deep lit review
(Collins, Polanyi, Wittgenstein, Ryle/Stanley-Williamson, Dreyfus, Reber implicit learning,
Nonaka, Bourdieu, Sternberg, …) is running to harvest every operationalization the literature
offers; profile membership will be revised when it lands.

Two user-supplied operationalizations, now first-class experiments:
1. **Situational generalization (the newsworthiness case):** "learning what newsworthiness
   means to a newsroom helps me identify the newsworthiness of NEW articles." Note this is
   generalization ACROSS SITUATIONS WITHIN one construct — and our current held-out items are
   i.i.d., which under-tests it. Upgrade: **distribution-shifted item splits** (e.g., by topic
   cluster / style stratum): install "newsworthiness" on one region of item-space, evaluate on
   a shifted region. The installed policy's covariate-shift robustness IS this operationalization.
2. **Composability (newsworthiness + brand-suitability → newsworthy-AND-brand-suitable):**
   install m1 and m2 SEPARATELY (adapters or joint training control), then judge under the
   COMPOSED construct name; reference = the target's own composed-name policy (one cheap new
   target pass for ~6 construct pairs). Contrast: {fresh, m1-only, m2-only, m1+m2-separate,
   m1+m2-joint} × {tacit-installed vs articulation-carried}. **The §3a hypothesis becomes the
   headline prediction: articulable constructs compose (symbols compose); tacitly-installed
   ones require joint training (practice blends, it doesn't compose).** If tacit-installed
   constructs DO compose zero-shot, that's evidence the installed structure is more
   symbol-like than the practice view allows — informative either way.
Also user clarification recorded: articulation is often an INSUFFICIENT channel, and is a
channel AT ALL only for the not-yet-articulated; the articulation-failure set is a MIXTURE of
{not-yet-articulated, not-articulable}, and part 3 + the tier ladder is the **mixture-separator**.

### 7b.4⁴ DE-RELATIVIZING TACITNESS (user concern, 2026-07-22: "articulable-to-72B-but-not-7B
is a limited, particular subset; we need tacitness that isn't just relative — maybe OSL")

Pair-relative tacitness (one teacher × one student) is the weakest form. Three upgrades:
1. **Population-relative:** random-effects over articulator×executor pairs (3 families × 6 rungs
   already scored); the construct-level intercept = the "intrinsic" component.
2. **OSL scaling-asymptote (PRIMARY; the user's suggestion):** tacitness as a property of the
   construct's SCALING CURVE: scaling-tacit(C) ⟺ articulation-gain(z) ≈ 0 for ALL measured z
   while the native gap persists and weights install at finite z. The asymptote replaces the
   pair. **v0 RAN (humor, Qwen 3B/7B/14B/32B): 41/90 constructs scaling-tacit (gain<.05 at
   every rung, gaps up to .81; several NEGATIVE at every rung), 27/90 eventually-articulable
   (gain≥.10 at some rung), 22 intermediate.** Caveats: point estimates, ad-hoc thresholds,
   one family — confirmatory version = cross-family curve fits, prereg'd. Two EXP-GTK-1
   A-cells are in the 41.
3. **Construct-intrinsic prediction:** if membership in the scaling-tacit class is predictable
   from target-free instruments (codability census, articulability-bits/recon-channel, lexicon
   dialect), tacitness becomes a property of constructs that model pairs merely REVEAL.
These join the profile (7b.4⅞+++): the operationalizations to converge are now {channel-diff,
statability, token-dependence, subspace-cap, situational-shift, composability, scaling-
asymptote, construct-intrinsic predictability} + whatever the lit review harvests.

### 7b.4⁵ The knowing-using gap import (arXiv 2607.08393; user, 2026-07-22)

Paper's finding (explicit-knowledge setting): SFT-installed facts pass direct retrieval (90%+)
but collapse under composition (10–20% at 2-hop, ~0% at 4-hop); mechanism = SFT encodes
POINT-WISE ASSOCIATIONS, not composable operations; the effective fix = training on composite
examples. Three imports for the capstone:

1. **Evidence-backed prediction for our composability probe:** our distillation channel IS
   SFT — their result predicts our adapters will show a knowing-using gap: P1-style
   installation can succeed while composed-construct judgment (newsworthy∧brand-suitable)
   fails hard. The symbols-compose/practice-blends hypothesis now has literature behind its
   SFT half.
2. **INTERPRETATION GUARD (important):** a future transfer/composition failure is AMBIGUOUS
   between (i) "tacit knowledge is inherently local" and (ii) "SFT installs knowledge in an
   unusable, point-wise form" — a property of the INSTALLER, not the knowledge. Disambiguator
   arms: if a different installer (reward channel §5.1; joint-composite training per their
   augmentation result) produces usable knowledge where plain distillation doesn't, the gap
   was the installer's. Never conclude tacit-locality from SFT arms alone.
3. **The generative problem space we lacked:** their KG gives infinite compositional queries;
   our analog = COMPOSITIONAL OPERATORS OVER CONSTRUCTS — {AND, OR, NOT, conditional} applied
   to construct names, with the TARGET's own composed-name policy as reference (reconstruction-
   only preserved). Usage-gap vs composition depth = their hop curve, for judgment policies.
   NOT (the anti-construct) is the cheapest memorization test: a point-wise p_yes memorizer
   fails inversion.

**EXP-COMP-1 sketch (composition ladder):** ~6 humor construct pairs stratified by
{both-articulable, both-scaling-tacit, mixed} × operators {AND, NOT, OR}; ONE new target pass
(72B under composed names); executor arms: fresh, fresh+both-articulations, adapter trained on
m1+m2 data MIXED (separate; no composed examples), adapter trained WITH composed examples
(their augmentation contrast). Headline DV: **using-gap = knowing(parts) − using(composition),
per channel** — prediction: channels match on knowing, dissociate on using (articulation
composes; separate-SFT gaps; joint-SFT closes; reward-channel = open question). Sharpens §3a +
the user's newsworthiness∧brand-suitability operationalization into the paper's central figure
candidate.

### 7b.5 How is TK-general learned — three registered hypotheses
- **H-breadth (meta-learning):** GTK emerges from breadth of installed TK-local (predicts
  transfer slope ↑ with m).
- **H-scale:** GTK is what scale buys — the differentiation floor (7b.2) is its signature
  (predicts transfer tracks base size, ~flat in m).
- **H-embedding (Collins' null):** social GTK cannot be installed by task training at all
  (predicts flat transfer everywhere; the honest null).
Priors worth stating: the human expertise literature finds far transfer RARE (Thorndike
identical-elements; chess expertise doesn't transfer), and our own sibling-lattice result
(invention is sibling-local, 0/13) points the same way — the pre-registered expectation should
be steep decay, with any slow-decay finding carrying the burden of proof.

**Verified starting assets** (checked, not assumed): `methods/codability/experiments/score_fresh_name_arms.py`
is manifest-driven with a `teacher_forced_declared_labels` readout (YES/NO label-token logits → the
score vectors in the npz files); `methods/dense/train_reward_model.py` already implements peft
LoRA/QLoRA training end-to-end (the trainer below adapts it rather than starting from scratch);
`api_field_runner.py` is the hardened GLM caller; GEPA harness exists
(`methods/metric_implementer/experiments/m_omega_gepa.py` patterns; mind the `GEPA_CORPUS` trap);
packet partitions + stable-hash splits + tally scripts all exist from parts 1–2.

**Proposed new code home:** `methods/tacit_channels/` (keep it out of `methods/codability/` — that
directory is the frozen parts-1–2 apparatus; part 3 imports from it, never edits it mid-campaign).

### The one critical integration point (build this first, everything hangs off it)

**1.3 Adapter-aware scoring.** Every intervention produces a LoRA adapter; every readout must go
through the *identical* teacher-forced scoring path as parts 1–2 or no comparison is valid.
Two routes, in order of preference:
- (a) vLLM LoRA inference: extend `score_fresh_name_arms.py` (or a thin wrapper
  `tacit_channels/eval/score_with_adapter.py` that reuses its internals) with
  `--lora-adapter <path>`, passing a `LoRARequest`; adapter provenance (target job, N, item-hash,
  seed) recorded into the npz `meta`.
- (b) **Fallback if vLLM-LoRA + teacher-forcing misbehaves on the sk2/sk3 vLLM versions:**
  `merge_adapter.py` — merge LoRA into the base checkpoint, score the merged model exactly like any
  rung. Disk-heavier but zero new inference-path risk. (Verify merged checkpoints index-vs-shard —
  the broken-merged-checkpoint landmine, `reference_fp8_vllm_sk3`.)
- **Acceptance test for either route:** a *zero-trained* adapter (or merge of an untouched base) must
  reproduce the base rung's score vector bit-for-bit-ish (ρ > .999) on a 200-item smoke slice. No
  intervention result is reportable until this passes.

### Phase 0 — Language-frontier probe (§3e). CPU/API only; can start immediately.

| # | Component | What it does | Reuses |
|---|---|---|---|
| 0.1 | `frontier_probe/extract_rescue_articulations.py` | walk existing tallies + arm banks; emit one row per rescued cell: (domain, cell, best_arm, articulation_text, adverse-ρ, gain, controls-beaten) | existing npz/tally outputs, `tacit_breadth_arm_bank_v3.json` |
| 0.2 | `frontier_probe/lexicon_overlap.py` | vocabulary/construct overlap of each rescuing articulation vs the domain's metric-bank lexicon (exact-Jaccard + embedding recall) | lexicon-census machinery (`mine_clusters` patterns), metric banks (`reference_metric_banks`) |
| 0.3 | `frontier_probe/judge_novelty.py` | LLM-judge pass: each rescuing construct classified recombination vs novel-formalization; **blinded known-label anchors in every batch** (synthetic known-recombination + known-novel items) | `api_field_runner.py` (GLM/Sonnet — judges Sonnet+ only), `feedback_anchor_test_annotation_passes` |

Output: per-domain recombination-vs-novel rates + the frontier verdict. Descriptive; no prereg gate.

### Phase 1 — Shared training infrastructure (the long pole; code now, GPU when a lane frees)

| # | Component | What it does | Key decisions locked in |
|---|---|---|---|
| 1.1 | `distill/build_policy_distill_dataset.py` | target npz → JSONL `{prompt, p_yes}` where prompt is the **verbatim name-invoked judgment prompt the scorer uses** and `p_yes` is the target's score | train ONLY on the open `tacit_breadth_search` partition — frozen calibration/eval items never touch training; splits by stable hash (`feedback_stable_hash_splits`) |
| 1.2 | `distill/train_policy_lora.py` | LoRA SFT; loss = BCE/KL on the YES/NO label-token logits vs soft label `p_yes` | adapted from `methods/dense/train_reward_model.py`; N-sweep ∈ {8, 32, 128, 512} (the low-N regime, §5-B saturation guard); fixed hyperparams, one seed, full provenance manifest per adapter |
| 1.3 | adapter-aware scoring | (above — the critical integration point) | acceptance test mandatory |
| 1.4 | `eval/tally_exchange_rate.py` | 2-D tally keyed (articulation_arm × intervention, N); emits the iso-ρ contour table: for each ρ level, tokens(articulation) vs N(examples) | extends the gap-conditioned tally; `np.asarray` decompress-once |

Env note: training needs a torch+peft env (NOT the vLLM envs); `train_reward_model.py` implies one
exists or is buildable — **verify on sk2 before scheduling GPU time**, budget one debugging session
if not. One GPU per training job, never more (`feedback_gpu_usage`); LoRA on ≤14B rungs fits easily.

### Phase 2 — Peer-review §5.1 (the core). Gated on prereg G2.

| # | Component | What it does | Guards |
|---|---|---|---|
| 2.1 | `peer_review/metric_battery_v1.json` | the subjectivity-spectrum battery (~6–10 metrics: claims-support … enough-baselines … elegance) + registry entries incl. FEW_SHOT_EXAMPLES (`feedback_local_explanations_per_task_fewshots`); articulability x-value per metric from the codability census where covered, else a pre-registered anchored LLM-judge articulability rating | **battery list needs user sign-off** (it defines the x-axis) |
| 2.2 | target judgment passes | big-model target scores papers on (a) each metric name-invoked, (b) holistic accept/reject; **≥2 reps × ≥2 prompt forms → per-metric test–retest reliability = the noise ceilings (PRECONDITION, §6)** | reuse scorer + new prompt bank; ICLR-only corpus (`project_peer_review_va` — venue-confound lesson); papers are long → fixed truncation policy (e.g. 8k tokens, abstract+intro+method) declared in the bank, identical across all passes; check judge score distributions (`feedback_check_judge_score_distribution`) |
| 2.3 | `peer_review/residualize_outcome.py` | regress target-holistic on target's articulable-metric scores (train split only); emit residual soft labels + diagnostics | **STOP-RULE: if residual variance ≈ noise floor (from 2.2 ceilings), there is nothing to install — report and halt the arm** |
| 2.4 | reward-channel training | LoRA (via 1.2) on: (i) residual, (ii) raw holistic, (iii) **placebo: shuffled residual**, (iv) **fitted-values arm: the articulable part only** | (iii)+(iv) give the double dissociation — fitted-values training should improve articulable metrics but NOT elegance; residual training the reverse. That pattern, not any single ρ, is the §5.1 result |
| 2.5 | slope readout | per metric: Δρ(reward-installed − best-articulation) on held-out items, reliability-corrected; regress Δρ on articulability | prereg'd direction; report attenuation-corrected and raw side by side |

### Phase 3 — Articulation-transfer assay (§4). After any Phase-2 rescue exists.

| # | Component |
|---|---|
| 3.1 | `assay/elicit_articulation.py` — trained checkpoint generates its own policy articulation (k samples × the standard forms; vLLM offline batch) |
| 3.2 | standard parts-1–2 pipeline scores a FRESH executor with that articulation; transfer gap = ρ(trained) − ρ(fresh + self-articulation) |

### Phase 4 — C(c1) GEPA-by-RL bridge. Cheapest after 1.3 exists.

| # | Component |
|---|---|
| 4.1 | `gepa_bridge/fitness_reconstruction.py` — GEPA fitness = candidate articulation → frozen executor scores GEPA-dev items (batched as extra arms in one vLLM pass) → ρ vs target. GEPA-dev items disjoint from eval items (stable hash). Deliverable: GEPA-articulation vs mined-articulation vs distillation at matched compute |

### Phase 5 — parked (needs separate sign-off when reached)
Forecasting domain (dataset selection + registry + few-shots); composability probe (joint-vs-separate
training, reuses 1.2); real-ICLR-acceptance reward variant (human label — flagged, §5.1).

### Sequencing, cost, gates

- **Now (no GPU):** Phase 0 entirely; 1.1/1.2/1.4 code; 2.1 battery draft for sign-off; prereg drafts.
- **When an sk2 lane frees** (ladder finishing): 1.3 acceptance test → 1.2 smoke (one humor cell,
  Qwen-7B, N=32) → first exchange-rate row on the existing humor targets.
- **After sk3 canonical pipeline completes:** 2.2 target passes (70B, GPU 0) — do NOT preempt the
  frozen calibration run.
- **GPU shape:** LoRA training is minutes-to-hours on 1 GPU for ≤14B rungs; the expensive items are
  the 2.2 target passes (70B × ~2–5k papers × ~10 metrics × reps/forms — thousands of long prompts,
  offline batch per `feedback_vllm_batch_size`) and the N-sweep × rung × metric scoring grid.
- **Prereg gates (`feedback_check_before_new_approach`, metric-lexicon prereg discipline):**
  G1 = user signs off battery (2.1) + this plan. G2 = predictions §5 frozen (incl. the slope
  direction + reliability precondition) before any confirmatory 2.4/2.5 run. G3 = the 2.3 stop-rule.
  Exploratory smokes (acceptance test, single-cell distill) are allowed pre-G2 and reported as
  exploratory only.

**Total new code estimate:** ~10 scripts, of which 4 are thin adapters over existing machinery
(0.1, 1.1, 1.4, 3.2), 3 are real engineering (1.2 trainer, 1.3 adapter-scoring, 4.1 GEPA fitness),
and 3 are judgment-pass configs (0.3, 2.1, 2.2). The single riskiest item is 1.3 (vLLM-LoRA ×
teacher-forcing interaction) — hence the merge fallback and the mandatory acceptance test.

---

## 9. TACITNESS BATTERY (2026-07-22, built + W0 run)

Plan approved and implemented same day: `methods/tacit_channels/battery/` — probe registry
(~30 executable specs, catalog A1–C42 zero-orphans enforced by test), 10 encoded gates,
memoized artifact context, Hive-parquet long-format profile store (cells_v1 pattern),
convergence analysis (probe×probe rank agreement + PC1 = the general-tacitness-factor
readout). Decisions: v1 slice Qwen×humor; ALL FOUR heavy items committed (surface-swap +
scrambled →W2, LPP curriculum →W3, Lam task×channel →W3.5); GPU order v1b → W1. 21/21 tests.

**W0 FIRST RESULT (exploratory, humor × 4 Qwen rungs, run_tag w0_v2, 6.8K profile rows):
PC1 share = 0.30 — the operationalizations do NOT collapse onto one general tacitness
factor; the structure is ≥3 coherent, near-orthogonal blocks:**
- **articulation-resistance** (P-CHAN-core × P-SCAL-1 +0.67);
- **metacognitive/generalization** (P-STAT-1 × P-GEN-1 **+0.81** — Dienes zero-correlation
  and Wittgenstein/Bourdieu OOD-decay, two unrelated instruments, pick out the SAME
  constructs — the first non-trivial convergence);
- **distributional** (P-GT-3, anti-correlated with everything).
Cross-block: articulation-resistance ⊥ metacognitive opacity (+0.10/+0.21) — "tacit by
articulation-failure" and "tacit by metacognitive-dissociation" are DIFFERENT construct sets
at W0. If this survives the confirmatory battery: **tacitness is at least three-dimensional —
the concept splits** (the informative dissociation outcome of the convergence strategy).
Instrument catches from the first read: SCAL-2 primary was gain-derived (dup of CHAN-core at
+1.00) → replaced with cross-rung divergence SLOPE (now −0.24, PC1 0.35→0.30); CEIL-1×GT-1
+0.54 is partly attenuation (reliability gate must correct cap_oos before confirmatory).
Artifacts: outputs/tacit_channels/tacit_profile_v1/battery_w0_v2.parquet + summary JSON in
notebooks/data/two_faces_20260702/tacit_profile/. NEXT: battery prereg (freeze registry sha +
convergence predictions) before any W1–W3 confirmatory pass.

**2026-07-23 cross-domain W0: the multidimensionality result REPLICATES in all three domains —
PC1 = .30 (humor) / .29 (n&c) / .36 (math); no domain shows a single general tacitness
factor.** (Store fix: per-domain parquet filenames — a same-tag second domain had clobbered
the first; summaries were unaffected; all three regenerated as w0_v3. Never-delete-data
honored.) Isomorphism-status resolution recorded in the learning-approaches note (doubt 3):
estimand promoted from phenomenon to instrument; three-act arc (named → channels/battery →
beyond-names).

### Change log
- 2026-07-21 — note created from user's framing dump + hypothesis (tacit knowledge installed via
  RL/FT, not more articulation). Organized into types×channels frame with Collins' polimorphic axis
  as the mechanism for the measured gradient; recorded the six framing threads; sketched the
  channels experiment and the articulation-transfer tacitness assay. No work launched.
- 2026-07-21 (rev.) — user corrections + discussion folded in: (a) parts 1–2 and part 3 are **not
  disjoint** — reframed to a **2-D articulation×intervention exchange-rate surface** (interventions
  reduce the *amount* of text needed, even where articulation works); (b) added the **Statistical /
  inductive** tacit-knowledge type (predictive gut — the axis Collins lacks) + the **two-axis grid**
  (source × function) with the "`describe` column is where tacitness dies" insight (§3g); (c) the
  experiment substance: **FT = soft-label distillation** (low-N regime, saturation guard); **RL only
  justified with a non-label reward** — pointwise-label RL is dominated by SFT, so the primary RL arm
  is **(c1) reward an emitted rubric by downstream reconstruction = GEPA-by-RL**, which makes parts
  1–2 and 3 one pipeline; ordinal reward (c2) as alt; **patching = task-vector arithmetic /
  activation steering** (the user's "M_ω-then-put-it-together"), NOT ROME/MEMIT; (d) added a pure
  **forecasting** domain as the clean statistical-TK probe and recommended it as the first deep dive.
  Still no work launched.
- 2026-07-21 (rev.2) — **knowledge patching DROPPED** per user. **§5.1 added: the downstream-reward
  (behaviorist) design is now the core of part 3** — reward = a *distal aggregate outcome* (paper
  accepted) that the subjective metric only partly drives, so the unarticulable metric is installed
  from its reinforcement footprint rather than ever being labeled. Headline = a pre-registerable
  **slope: reward-advantage over articulation RISES with metric subjectivity** (elegance ≫ baselines
  ≫ claims-support). Two rigor points recorded: (i) **identifiability** — a distal aggregate installs
  the gestalt, so you must **residualize against the objective metrics** and claim only "the residual
  tacit component," with the elegance-reconstruction as a self-check; (ii) **it is not necessarily
  RL** — on a static corpus the outcome is a fixed distal label, so the minimal tool is *supervised
  regression on the residual*, with RL justified only if the model must act/generate. First domain
  switched to **peer review** (subjectivity spectrum within one domain). Real-ICLR-acceptance variant
  flagged as needing sign-off (human-generated label). Still no work launched.
- 2026-07-21 (rev.3, audit pass) — full-conversation audit against the original scattered-thoughts
  dump. **Coverage:** two threads had been dropped ("understanding→predicting"; the
  individual→micro-org→macro-org axis) — now parked explicitly in §3c; the **contractability** half
  of §3b was under-read — added the Hart–Moore observable-vs-verifiable ↔ V/A/Taste bridge (the
  deepest link between §3b and the core program). **Soundness fixes:** (F1) polimorphic mapping risks
  re-asserting the killed codification hypothesis — patents anomaly + pre-registration guard added to
  §6; (F2) "statistical TK as a 4th type" contradicted §2's own brain-inclusive STK definition —
  downgraded to a refinement-within-somatic in §3g; (F3) reliability confound in the headline slope —
  per-metric noise ceilings made a precondition; (F4) "A fails for judgment" is capacity-relative —
  predictions now scoped to fixed rung; (F5) the §5.1 slope was missing from the predictions block —
  now prediction 1; (F6) exchange-rate glossed amortized-vs-marginal. Staleness: D marked dropped in
  §2/§6; §3a composability probe rebased off dead channel D. Structural debt noted: intro carries
  three layered framings and §5.1 (the core) sits at line ~350 — next major revision should be a
  clean rewrite with §5.1 promoted.
- 2026-07-21 (rev.4) — **§8 implementation plan added**, verified against the repo (scorer =
  manifest-driven teacher-forced YES/NO readout; `methods/dense/train_reward_model.py` = in-repo
  LoRA precedent to adapt). New code home `methods/tacit_channels/`; 5 phases; the critical
  integration point is **adapter-aware scoring through the identical teacher-forced path** (vLLM
  LoRARequest, merge-and-score fallback, mandatory zero-adapter acceptance test). Phase 0
  (language-frontier probe) is CPU-only and can start immediately; Phase 2 carries the prereg gates
  (G1 battery sign-off, G2 frozen predictions, G3 residual stop-rule) and the placebo/fitted-values
  **double-dissociation controls**. Nothing launched; G1 pending user.
- 2026-07-22 — **STAGE 1 BUILT AND VERIFIED + Phase 0 first results.** `methods/tacit_channels/`
  created per the approved plan (`_apparatus.py` indirection; channels/{frontier_probe,distill,
  eval,peer_review,assay,gepa_bridge}; 14/14 CPU tests green). vllm_backend.py extended
  additively (adapter-keyed engine cache + `_maybe_lora()`; default path no-op, verified).
  Teacher-forced LoRA fork with upstream-drift sha guard; zero-adapter acceptance test ready
  (needs a free sk2 GPU). Trainer = causal-LM LoRA soft-CE on {YES,NO} logits, scoring-identical
  chat rendering. **Phase 0 ran on real data: extractor found 89 conditioned rescues (exactly
  matching the ladder tallies — cross-validation) across 1,260 (rung,domain,cell) rows; lexicon
  overlap: rescuing articulations carry ~0–1% novel vocabulary — near-total RECOMBINATION of the
  domain's existing lexicon (first, token-level evidence that articulation rescue is bounded by
  known language, §3e).** GLM-4.7 judge pass (1,266 prompts incl. 6 blinded anchors) running;
  anchor-gated ingest pending. Artifacts: `outputs/tacit_channels/frontier_probe/`. Frozen
  apparatus untouched (verified via git status). G1 still pending user.
- 2026-07-22 (later) — **PHASE 0 COMPLETE: the language-frontier verdict is RECOMBINATION.**
  GLM-4.7 judged 1,252 articulations (1,266 prompts, 0 errors; blinded anchors separate
  perfectly: recombination-anchors 0.0 vs novel-anchors 9.33/10 → judge calibrated, no
  score-collapse). Distribution: 93% of articulations score exactly 0/10 novelty; only 6/1,252
  reach 10. **Rescued cells (n=89): novelty means 0.24 (humor) / 0.00 (math) / 0.11 (n&c); only
  3/89 rescuers reach ≥3.** Non-rescued contrast is slightly MORE novel (0.42/0.41 humor/math) —
  directionally, novelty anti-correlates with rescue. Combined with the token-level ~0–1% novel
  vocabulary: **within this apparatus, articulation rescue operates entirely inside known
  language — rescues deploy standard domain vocabulary well; they do not push the explicitness
  frontier (§3e answer: bounded-by-known-language).** SCOPE CAVEAT (state honestly): the v3 bank's
  articulations are mined/compiled from source hierarchies, not freely optimized — whether
  *optimization* (the GEPA c1 bridge) can find genuinely novel language that rescues is exactly
  the open question this result tees up; a positive GEPA-novelty result would be frontier
  expansion, a negative one closes the claim. Artifacts: novelty_judged.jsonl + summary in
  `outputs/tacit_channels/frontier_probe/`.
