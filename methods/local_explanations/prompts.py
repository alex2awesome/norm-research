"""Prompt templates for local explanation approaches."""

from typing import Dict


# =============================================================================
# Approach A: Rationalization + Prior Calibration
# =============================================================================

PRIOR_ESTIMATION_PROMPT = """\
Task: {task_description}

Based on your general knowledge, list {n_features} specific features that would \
typically cause a {text_type} to be {label_description}. Be concrete and specific \
-- avoid vague qualities like "well-written" or "clear". Focus on substantive, \
content-level features that someone could verify by reading the text.

Output as a JSON list of strings. Example: ["feature 1", "feature 2", ...]"""


RATIONALIZATION_PROMPT = """\
Task: {task_description}

The following {text_type} was {label_word}.

Text: {text}

Identify {n_features} specific, concrete features of THIS particular text that \
explain why it was {label_word}. Focus on features you can directly observe in \
the text -- not generic qualities that would apply to any {text_type}.

Output as JSON: {{"features": ["feature 1", "feature 2", ...]}}"""


# =============================================================================
# Approach B: Amended STaR-Local
# =============================================================================

STAR_LOCAL_PROMPT = """\
Task: {task_description}

You will be given a {text_type}. Your job is to identify the features of THIS \
particular text that a careful reviewer would treat as evidence FOR \
{positive_label} or evidence AGAINST {positive_label} (i.e., for \
{negative_label}). These features will later be used as predictive variables, so \
they must be substantive enough to discriminate THIS text from others.

Hard rules:

1. OUTPUT FORMAT must EXACTLY match the template below. Use the literal section \
   headers "FEATURES_FOR", "FEATURES_AGAINST", "PREDICTION" — no markdown (no \
   "###", no "**"), no preamble, no summary. Each feature on its own line \
   starting with "- ". No text before "FEATURES_FOR" or after the PREDICTION \
   line.

2. EACH FEATURE MUST HAVE TWO PARTS, separated by " — " (em dash with spaces):
       - [SPECIFIC CONTENT from the text] — [EVALUATION: why it matters for the outcome]

   The CONTENT part names the actual method, dataset, number, comparison, \
   mechanism, claim, comparison, or omission found in the text. The EVALUATION \
   part is a 1-2 clause judgment that explicitly says WHY this content makes \
   {positive_label} more or less likely. The evaluation should reference \
   what reviewers care about (rigor, novelty, scope, validation, reproducibility, \
   significance, etc.) rather than just restating the content.

3. SPECIFICITY. The CONTENT part must be specific enough that swapping in a \
   different {text_type} for the same task would change it. Avoid generic \
   placeholders ("novel approach", "well-written", "doesn't discuss limitations", \
   "addresses an important problem"). If your content part could appear in any \
   {text_type}, you have written a generic feature — discard it.

4. EVALUATIVENESS. The EVALUATION part must explain WHY this content matters, \
   not just label it as good/bad. "— is significant" is too vague. Spell out \
   the reviewer's reasoning: "— provides verifiable quantitative validation \
   against a strong baseline", "— a methodological gap that limits the \
   reliability of the headline claim", etc.

5. NO PADDING. List FEWER features (down to zero) rather than fabricating filler. \
   If you genuinely cannot find a concrete content+evaluation pair for one side, \
   OUTPUT THE HEADER AND THEN NO BULLETS. Do NOT write filler like "There are \
   no obvious flaws", "No apparent weaknesses", "There are no features against" \
   — these are noise and will be discarded.

6. AVOID NEAR-DUPLICATES. Each feature should evaluate a DIFFERENT aspect of \
   the text. Don't list five paraphrases of the same point.

Examples below show the target format and depth.

{few_shot_block}\
Now process this text:

Text: {text}

FEATURES_FOR ({positive_label}):
FEATURES_AGAINST ({negative_label}):
PREDICTION: {positive_label} or {negative_label}
"""


# =============================================================================
# Per-task few-shot examples
# =============================================================================
# IMPORTANT: when adding a new task to TASK_METADATA_REGISTRY, add a matching
# entry to FEW_SHOT_EXAMPLES below or the prompt will fall back to no examples.

FEW_SHOT_EXAMPLES = {
    "MathAnswerQuality": """Examples (note the "[content] — [evaluation: why this matters]" structure). \
A PREFERRED math answer is correct, well-explained, appropriately rigorous, and \
addresses the specific question. A NON-PREFERRED answer may be incorrect, \
incomplete, off-topic, link-only, or lack the clarity/rigor expected by the community.

--- Example 1 (preferred) ---
Text: Question: Prove that the sum of two continuous functions is continuous.

Answer: Let f and g be continuous at x=a. For any epsilon > 0, there exist \
delta_1, delta_2 > 0 such that |f(x)-f(a)| < epsilon/2 when |x-a| < delta_1 \
and |g(x)-g(a)| < epsilon/2 when |x-a| < delta_2. Taking delta = min(delta_1, delta_2), \
we have |(f+g)(x)-(f+g)(a)| <= |f(x)-f(a)| + |g(x)-g(a)| < epsilon/2 + epsilon/2 = epsilon \
whenever |x-a| < delta.

FEATURES_FOR (preferred):
- Starts from the epsilon-delta definition and constructs delta explicitly as min(delta_1, delta_2) — demonstrates proper proof technique by reducing to the definitions, which is what the community expects for this level of question
- Splits epsilon into epsilon/2 for each function, then recombines via triangle inequality — a clean, standard decomposition trick that shows mathematical maturity
- Every step follows logically from the previous with no gaps — no "it is easy to see" or hand-waving, which reviewers penalize

FEATURES_AGAINST (non-preferred):
- Proves only continuity at a single point rather than discussing uniform continuity or continuity on an interval — may not fully address what the asker intended if they wanted a broader result
PREDICTION: preferred

--- Example 2 (non-preferred) ---
Text: Question: How do I find the eigenvalues of a 3x3 matrix?

Answer: Just use numpy.linalg.eig() in Python. It will give you the eigenvalues directly.

FEATURES_FOR (preferred):
- Provides a practical tool that will compute the answer — technically functional for someone who just needs the numerical result

FEATURES_AGAINST (non-preferred):
- Gives a code-library reference instead of a mathematical explanation — on Math.SE, the community expects mathematical methods, not computational shortcuts
- Does not explain the characteristic polynomial det(A - lambda*I) = 0 approach — fails to teach the underlying concept, which is what upvoted math answers are expected to do
- No worked example or step-by-step procedure — a one-liner with no elaboration signals low effort and provides no learning value
- Does not address the 3x3 case specifically (Cardano's formula, special structure) — generic advice that doesn't engage with the specifics of the question
PREDICTION: non-preferred

--- Example 3 (preferred) ---
Text: Question: Why is the derivative of e^x equal to itself?

Answer: Consider the limit definition: d/dx e^x = lim_{h->0} (e^{x+h} - e^x)/h \
= e^x * lim_{h->0} (e^h - 1)/h. The key is showing lim_{h->0} (e^h - 1)/h = 1. \
One approach: if we define e = lim_{n->inf} (1 + 1/n)^n, then for small h, \
e^h approx 1 + h, so (e^h - 1)/h approx 1. More rigorously, this follows from \
the fact that ln is the inverse of exp and d/dx ln(x) = 1/x, giving \
d/dx e^x = e^x / (1/e^x) * (1/e^x) ... but the cleanest path is to define \
exp as the unique solution to f' = f with f(0) = 1, in which case the result is \
definitional.

FEATURES_FOR (preferred):
- Shows multiple approaches (limit definition, approximation argument, inverse function, ODE definition) — breadth of perspective is a hallmark of highly-upvoted explanatory answers
- Starts with the concrete limit computation before moving to the abstract definitional approach — pedagogically effective ordering from intuition to rigor
- Acknowledges that the answer depends on how you DEFINE e^x — this meta-mathematical awareness is exactly what makes "why" questions interesting and what the community rewards
- Distinguishes "approx 1" intuition from "more rigorously" — honest about levels of rigor rather than pretending a heuristic is a proof

FEATURES_AGAINST (non-preferred):
- The rigorous inverse-function argument is left incomplete (trails off with "...") — a complete answer would follow through on the chain rule computation
PREDICTION: preferred

--- Example 4 (non-preferred) ---
Text: Question: Is every subgroup of an abelian group abelian?

Answer: Yes this is obvious since subgroups inherit the group operation.

FEATURES_FOR (preferred):
- States the correct answer (yes) — factually accurate

FEATURES_AGAINST (non-preferred):
- Uses "obvious" without proof — the math community consistently downvotes answers that dismiss the question rather than engaging with it
- "Inherit the group operation" is vague and does not constitute a proof — does not show that if ab=ba for all a,b in G, then ab=ba for all a,b in H subset G, which is the actual (simple) argument
- One sentence with no mathematical notation — does not meet the community standard for proof-based questions even when the proof is short
PREDICTION: non-preferred

--- Example 5 (preferred, proof-writing tag) ---
Text: Question: Prove by induction that 1 + 2 + ... + n = n(n+1)/2.

Answer: Base case: n=1. LHS = 1, RHS = 1*2/2 = 1. Check. \
Inductive step: Assume 1+2+...+k = k(k+1)/2 for some k >= 1. Then \
1+2+...+k+(k+1) = k(k+1)/2 + (k+1) = (k+1)(k/2 + 1) = (k+1)(k+2)/2. \
This is exactly the formula with n = k+1. By the principle of mathematical \
induction, the result holds for all n >= 1.

FEATURES_FOR (preferred):
- Follows the standard induction template (base case, inductive hypothesis, inductive step, conclusion) — clean structure that the community recognizes as correct form
- Shows every algebraic step in the inductive step: k(k+1)/2 + (k+1) factored to (k+1)(k+2)/2 — no steps skipped, verifiable by the reader
- Explicitly states the inductive hypothesis ("Assume ... for some k >= 1") — a common mistake in student proofs is to not state this clearly, so doing it right signals quality
- Concludes with "By the principle of mathematical induction" — proper closure that frames the argument within the formal proof method

FEATURES_AGAINST (non-preferred):
- Standard textbook induction proof with no additional insight — while correct and complete, it doesn't offer intuition (e.g., the visual triangle argument or Gauss's pairing trick) that would distinguish it from what the asker could find in any textbook
PREDICTION: preferred

""",
    "NoticeAndComment": """Examples (note the "[content] — [evaluation: why this matters]" structure). \
In notice-and-comment rulemaking, a SUBSTANTIVE comment raises an issue an agency \
must respond to in the final rule: specific factual claims, procedural/legal \
defects, stakeholder data, or actionable alternatives grounded in the proposal's \
text. A NON-SUBSTANTIVE comment expresses preference, sentiment, or generic \
support/opposition without obligating agency response.

--- Example 1 (substantive) ---
Text: The proposed rule's 30-day comment window violates OMB Circular A-4 guidance that economically \
significant rules should provide at least 60 days for public comment. The rule's own estimated \
$2.3 billion annual impact places it well within the A-4 threshold, and the compressed timeline \
prevents affected industries from collecting the cost data the agency expressly requests in \
Question 7 of the NPRM.

FEATURES_FOR (substantive):
- Cites a specific procedural authority (OMB Circular A-4) and identifies the numeric threshold allegedly crossed — invokes a framework an agency must respond to under the APA's requirement to address procedural objections
- Quantifies the rule's economic significance ($2.3B annual impact) using the agency's own figure — ties the procedural claim to a concrete, verifiable attribute of the proposal
- Explicitly links the timeline deficiency to Question 7 of the NPRM, where the agency solicited cost data — shows the comment engages with the substance of the rule, not a generic timing complaint

FEATURES_AGAINST (non-substantive):
- Focuses on process rather than the rule's policy merits — some agencies treat pure procedural objections as preliminary matters rather than substance
PREDICTION: substantive

--- Example 2 (substantive) ---
Text: As a critical-access hospital administrator, I note that the 340B reporting threshold proposed at \
§51.305(b)(2) would exclude 73% of critical-access hospitals in our state, based on CMS Hospital Cost \
Report data for FY2022. The regulatory impact analysis assumes these hospitals have compliance staff \
equivalent to larger integrated delivery networks, which is incorrect: our 2-person HIM departments \
cannot absorb the quarterly attestation burden described in §51.340.

FEATURES_FOR (substantive):
- Cites the specific regulatory section (§51.305(b)(2)) being critiqued and quantifies the impact (73% of critical-access hospitals in state) using a named data source (CMS Hospital Cost Report FY2022) — agency must address the factual claim or explain its rebuttal
- Identifies a specific flawed assumption in the regulatory impact analysis (equivalent compliance staff across hospital sizes) and provides a falsifiable counterclaim (2-person HIM departments) — constitutes disputed material fact the agency cannot ignore
- Stakeholder identity is named and relevant (critical-access hospital administrator) — agencies weight stakeholder-specific operational evidence as substantive input

FEATURES_AGAINST (non-substantive):
- Based on personal experience at one facility rather than aggregated data beyond the CMS reference — agency may discount as anecdotal unless corroborated
PREDICTION: substantive

--- Example 3 (non-substantive) ---
Text: I strongly oppose this terrible rule. The EPA is destroying American industry. We need less \
regulation, not more. Please withdraw this proposal immediately.

FEATURES_FOR (substantive):
- Clearly communicates a position (oppose) — registers a stakeholder view, which some agencies tally even for non-substantive comments

FEATURES_AGAINST (non-substantive):
- Contains no reference to any specific provision, data, legal authority, or alternative — nothing the agency is obligated to address in a response-to-comments section
- Relies on generic political framing ("destroying American industry") rather than identifying a concrete harm or cost — courts reviewing the administrative record would not treat this as raising a disputed issue of fact or law
- Demands withdrawal without explaining why any particular element of the rule is deficient — gives the agency nothing to analyze or respond to
PREDICTION: non-substantive

--- Example 4 (non-substantive) ---
Text: Wildlife are beautiful creatures that deserve our protection. The Fish and Wildlife Service \
should do everything possible to save endangered species. Future generations will thank you for your \
stewardship.

FEATURES_FOR (substantive):
- Expresses support for the agency's underlying mission — may be aggregated in comment counts

FEATURES_AGAINST (non-substantive):
- Pure sentiment statement with no reference to any specific proposal element, species, habitat, or regulatory text — no grounding the agency can engage with
- Phrased in aspirational terms ("should do everything possible") rather than proposing any concrete action or critique — lacks the specificity courts look for in judicial review of agency response to comments
- Reads as a form-letter style expression typical of mass-comment campaigns — agencies routinely categorize these separately from substantive comments
PREDICTION: non-substantive

--- Example 5 (substantive) ---
Text: The proposed restriction on vessel transits in critical habitat zones overlooks seasonal \
migration patterns documented in NMFS's own 2019 North Atlantic Right Whale Status Review (pp. 47-52), \
which shows right whale concentrations shift 15-20 nautical miles northward between April and June. \
A static year-round boundary will fail to protect whales during peak calving months in the northern \
shifted range while unnecessarily restricting transit in the southern zone during months when whales \
have migrated away.

FEATURES_FOR (substantive):
- Cites the agency's own technical document (NMFS 2019 Right Whale Status Review pp. 47-52) to contradict the proposal — agencies face heightened scrutiny when critiqued on their own evidentiary basis
- Provides a specific quantitative claim (15-20 nautical miles northward shift, April-June timing) that the agency can verify or rebut — falsifiable factual dispute
- Identifies a concrete design flaw (static boundary vs. seasonal migration) and implies a specific alternative (seasonal boundary) — gives the agency a constructive path to modify the final rule
- Shows the restriction will fail in both directions (under-protective during calving, over-restrictive otherwise) — strengthens the argument that the rule is arbitrary as drafted

FEATURES_AGAINST (non-substantive):
- Does not attach the cited report as an exhibit or quote the specific text — reduces evidentiary weight if agency cannot locate the exact passages
PREDICTION: substantive

""",
    "PeerReviewAcceptance": """Examples (note the "[content] — [evaluation: why this matters]" structure):

--- Example 1 (accepted ICLR paper) ---
Text: We propose Mamba-SSM, a selective state-space model that scales linearly \
with sequence length. On the Pile language modeling benchmark, Mamba-2.8B \
matches Llama-7B perplexity using 2.5x less compute. We release pretrained \
checkpoints up to 7B parameters and code at github.com/state-spaces/mamba.

FEATURES_FOR (accepted):
- Reports Mamba-2.8B matching Llama-7B perplexity on the Pile at 2.5x less compute — a concrete head-to-head benchmark result against a widely-cited baseline, exactly the kind of falsifiable empirical claim reviewers reward
- Architectural contribution explicitly framed as "selective state-space model with linear sequence-length scaling" — mechanistically specific contribution that addresses a real bottleneck in transformer architectures, signaling theoretical novelty
- Releases pretrained checkpoints up to 7B parameters and code at github.com/state-spaces/mamba — full reproducibility lowers reviewer skepticism and increases citation potential
- 2.5x compute efficiency is quantified as a tradeoff rather than an unspecific "more efficient" claim — verifiable and easy for reviewers to validate or contest

FEATURES_AGAINST (rejected):
- Only the Pile benchmark cited; no downstream task results (e.g. MMLU, GSM8K, HumanEval) — narrows the claim to perplexity, which is a weaker quality signal than downstream accuracy
- Abstract does not name comparisons to other linear-attention or SSM architectures (RetNet, RWKV, S4) — leaves open whether the gain comes from the SSM family in general or this paper's specific design
PREDICTION: accepted

--- Example 2 (accepted ICLR paper) ---
Text: We introduce SETR, the first pure transformer-based encoder for semantic \
segmentation. SETR achieves 50.28% mIoU on ADE20K, outperforming the prior \
ResNet-based DeepLabV3+ baseline by 4.5 mIoU. We provide ablations over patch \
size and positional encoding.

FEATURES_FOR (accepted):
- 50.28% mIoU on ADE20K with a 4.5-point gain over DeepLabV3+ — large absolute and relative improvement on a standard benchmark, the kind of magnitude reviewers consider significant rather than within-noise
- Frames itself as "first pure transformer encoder for semantic segmentation" — establishes a clear conceptual contribution beyond incremental tuning, which top venues weight heavily
- Provides ablations over patch size and positional encoding — demonstrates methodological rigor and isolates the source of the gain, which strengthens reviewers' trust in the headline result

FEATURES_AGAINST (rejected):
- "First" claim narrowly scoped to "pure transformer encoder" — leaves room for concurrent or hybrid work to weaken the priority claim, which reviewers may probe
PREDICTION: accepted

--- Example 3 (rejected ICLR paper) ---
Text: We apply BERT to predict customer churn in a telecom dataset. Our model \
outperforms a logistic regression baseline by 2 percentage points in AUC. The \
approach demonstrates the power of large language models for tabular tasks.

FEATURES_FOR (accepted):
- Reports a 2 percentage point AUC gain over logistic regression — at least the comparison is quantitative rather than narrative, giving reviewers something concrete

FEATURES_AGAINST (rejected):
- Off-the-shelf BERT applied to one telecom dataset — no methodological contribution to the model itself, which top ML venues consistently reject as "application work" rather than research
- Only baseline is logistic regression — fails to compare against specialized tabular methods (XGBoost, TabNet, FT-Transformer) that are the genuine SOTA for tabular tasks, undercutting any "demonstrates the power" claim
- "Demonstrates the power of LLMs for tabular tasks" overgeneralizes from a single telecom benchmark — exactly the kind of unsupported sweeping claim that triggers reviewer pushback
- 2 pp AUC gain is small and likely within run-to-run variance on tabular tasks — without seeds or confidence intervals reported, reviewers cannot distinguish the gain from noise
PREDICTION: rejected

--- Example 4 (rejected ICLR paper) ---
Text: We present a new theoretical framework for understanding why neural \
networks generalize. Our framework unifies several existing perspectives and \
suggests new directions for research.

FEATURES_FOR (accepted):
- Positions itself in the high-stakes generalization-theory area — if substantive, this would be an impactful contribution

FEATURES_AGAINST (rejected):
- Abstract names no concrete theorem, definition, bound, or measurable quantity — for a "new theoretical framework", the absence of any formal object signals there is no actual technical content to evaluate
- "Unifies several existing perspectives" without naming which perspectives or how the unification works — vague enough that reviewers cannot verify the claim of unification
- "Suggests new directions for research" rather than reporting results — reviewers read "suggests directions" as workshop or position-paper level, not main-conference contribution
- No experimental verification or theoretical proof sketched in the abstract — leaves reviewers with no evidence the framework is correct, useful, or new
PREDICTION: rejected

--- Example 5 (borderline / accepted) ---
Text: We propose CLIP-Adapter, a lightweight adapter module added on top of \
frozen CLIP encoders for few-shot image classification. On 11 benchmarks \
including ImageNet, CLIP-Adapter outperforms CoOp by 1.0-3.0% absolute \
accuracy. The adapter has only 0.5M trainable parameters.

FEATURES_FOR (accepted):
- 0.5M trainable parameters quoted — concrete parameter count grounds the "lightweight" claim and is a tangible advantage reviewers can compare against alternatives
- 1.0-3.0% absolute accuracy gain over CoOp across 11 benchmarks including ImageNet — coverage across many benchmarks reduces concern about cherry-picking, and gain over the natural prior-art baseline (CoOp) is the right comparison
- Diverse benchmark suite (11 datasets) including ImageNet — breadth of evaluation is what reviewers look for to trust generalization claims rather than benchmark-specific tuning

FEATURES_AGAINST (rejected):
- 1-3% gain margin is modest with no error bars or seed sensitivity reported — reviewers will probe whether the headline gain survives across runs, and a 1-2% margin can disappear with different seeds
- Comparison limited to CoOp; no Tip-Adapter, CoCoOp, or other recent adapter methods — leaves uncertainty about whether CLIP-Adapter is actually the strongest in its line of work, the kind of gap reviewers ask about in rebuttals
PREDICTION: accepted

""",
}


# =============================================================================
# Clustering prompts
# =============================================================================

_SIM_FORMAT_TAIL = """\
Now decide for this pair:

A: {feat_a}
B: {feat_b}

Reply with EXACTLY one word: Yes or No."""


SIMILARITY_LABELING_PROMPT_FOR_FOR = """\
You are deciding whether two FOR-acceptance evaluative judgments describe the \
SAME underlying reviewer concern, or DIFFERENT ones.

Both inputs are evaluative judgments that argue why a paper SHOULD be accepted. \
You'll see only the abstract evaluative reasoning (no paper-specific names or \
numbers). Decide whether they would consolidate into a single shared positive \
rubric, or whether they identify distinct kinds of positive evidence.

Two judgments are SAME if they make essentially the same kind of evaluative \
point — they would be summarized identically by a careful reader, even though \
their surface phrasing, domains, or specifics differ. Two judgments are \
DIFFERENT if they identify distinct kinds of positive evidence (e.g., one \
about empirical results, one about reproducibility; one about novelty, one \
about scope).

When in doubt, lean DIFFERENT: better to keep distinct evidence kinds separate \
than to collapse genuinely different rubrics into one.

EXAMPLES:

Pair: A: "concrete benchmark gain over a strong baseline — verifiable claim"
      B: "demonstrates the practical impact and generalizability of the proposed methods"
Answer: No (A is empirical-magnitude on a specific baseline; B is broad-applicability/generalization)

Pair: A: "demonstrates effectiveness through rigorous evaluation on standard benchmarks"
      B: "outperforms previous works across various audio-visual benchmarks"
Answer: Yes (both: empirical-magnitude × broad-evaluation on standard benchmarks)

Pair: A: "introduces a novel cost aggregation network with self-attention"
      B: "introduces a margin-based learning framework with a specific margin constraint"
Answer: Yes (both: novel-architecture / mechanism-specificity)

Pair: A: "releases code on GitHub for reproducibility and verification"
      B: "includes ablations on the gating mechanism, isolating which design choice drives gain"
Answer: No (reproducibility vs ablation-rigor)

Pair: A: "provides quantitative evidence of effectiveness, essential for assessing the contribution"
      B: "introduces a theoretically grounded method based on category theory"
Answer: No (empirical-magnitude vs theoretical-result)

Pair: A: "reduces trainable parameters by 94% while maintaining accuracy — efficiency-with-accuracy"
      B: "lighter model with fewer parameters than baseline — practical efficiency gain"
Answer: Yes (both: efficiency-gain)

Pair: A: "provides a strong theoretical foundation, increasing credibility of the claim"
      B: "rigorous theoretical guarantees on convergence rates"
Answer: Yes (both: theoretical-result)

Pair: A: "extensive experiments across many tasks demonstrate generalizability"
      B: "ablation studies isolate the contribution of each component"
Answer: No (broad-evaluation vs ablation-rigor)

""" + _SIM_FORMAT_TAIL


SIMILARITY_LABELING_PROMPT_AGAINST_AGAINST = """\
You are deciding whether two AGAINST-acceptance evaluative judgments describe \
the SAME underlying reviewer concern, or DIFFERENT ones.

Both inputs are evaluative judgments that argue why a paper SHOULD be rejected. \
You'll see only the abstract evaluative reasoning (no paper-specific names or \
numbers). Decide whether they consolidate to a single shared concern, or \
identify distinct kinds of weakness.

Two judgments are SAME if they raise essentially the same kind of concern — \
they would be summarized identically by a careful reader, even though their \
surface phrasing, domains, or specifics differ. Two judgments are DIFFERENT if \
they identify distinct kinds of weakness (e.g., one about missing comparisons \
to baselines, one about narrow evaluation scope; one about absent theoretical \
content, one about underpowered statistics).

Note: many AGAINST judgments are about ABSENCES of something. Two absences are \
SAME only when they are absences of the SAME kind of thing (e.g., both about \
missing baseline comparisons). "Missing X" and "missing Y" are DIFFERENT when \
X and Y are different things, even though both are absences.

When in doubt, lean DIFFERENT.

EXAMPLES:

Pair: A: "no comparison to recent positional encoding methods, leaves uncertainty about superiority"
      B: "no comparison to other graph-based retrieval methods, leaves the question of relative performance"
Answer: Yes (both: missing-baseline-comparison)

Pair: A: "no discussion of potential limitations or failure cases of the proposed method"
      B: "narrow scope, limited to vision transformer models in evaluation"
Answer: No (missing-limitation-discuss vs narrow-evaluation-scope)

Pair: A: "no quantitative metrics in the abstract — reviewers cannot assess the magnitude"
      B: "no theoretical analysis or proof of the proposed method's properties"
Answer: No (missing-quantification vs missing theoretical content — different absences)

Pair: A: "evaluation limited to two benchmarks, may not cover full range of scenarios"
      B: "only three datasets used for evaluation, limited generalizability"
Answer: Yes (both: narrow-evaluation-scope)

Pair: A: "claim of strong performance is qualitative and lacks concrete numbers"
      B: "no quantitative comparison to other invariant learning methods in abstract"
Answer: No (missing-quantification of own results vs missing-baseline-comparison; close but distinct)

Pair: A: "off-the-shelf application to a single domain, limited methodological novelty"
      B: "incremental contribution, mostly engineering on top of existing architecture"
Answer: Yes (both: novelty-doubt)

Pair: A: "sweeping claim of generalizability from a single benchmark"
      B: "claim of state-of-the-art without showing the specific comparison numbers"
Answer: No (overclaim vs missing-quantification — though both are unsupported claims, axis differs)

Pair: A: "sample size of 40 patients limits statistical power and external validity"
      B: "no error bars or confidence intervals reported in the experiments"
Answer: Yes (both: statistical-power / inadequate uncertainty quantification)

""" + _SIM_FORMAT_TAIL


# Backwards-compat alias (kept so existing imports don't break — points to FOR-FOR variant)
SIMILARITY_LABELING_PROMPT = SIMILARITY_LABELING_PROMPT_FOR_FOR


CLUSTER_NAMING_PROMPT = """\
The following features were all identified as describing the same underlying concept:

{feature_list}

What is the single, concise concept these features share? Provide:
1. A short name (3-8 words)
2. A one-sentence description

Output as JSON: {{"name": "...", "description": "..."}}"""


RUBRIC_GENERATION_PROMPT = """\
Feature: "{feature_name}"
Description: {feature_description}

Example member features from this category:
{member_features}

Write a binary evaluation rubric for this feature as it applies to evaluating a {text_type}.

YES: [2-3 sentences describing what constitutes YES -- when this feature is clearly present]
NO: [2-3 sentences describing what constitutes NO -- when this feature is absent or not applicable]

Output as JSON: {{"yes": "...", "no": "..."}}"""


# =============================================================================
# Render functions
# =============================================================================

def render_prior_prompt(
    task_description: str,
    text_type: str,
    label_description: str,
    n_features: int = 10,
) -> str:
    return PRIOR_ESTIMATION_PROMPT.format(
        task_description=task_description,
        text_type=text_type,
        label_description=label_description,
        n_features=n_features,
    )


def render_rationalization_prompt(
    task_description: str,
    text_type: str,
    label_word: str,
    text: str,
    n_features: int = 7,
) -> str:
    return RATIONALIZATION_PROMPT.format(
        task_description=task_description,
        text_type=text_type,
        label_word=label_word,
        text=text,
        n_features=n_features,
    )


def render_star_local_prompt(
    task_description: str,
    text: str,
    positive_label: str,
    negative_label: str,
    text_type: str = "text",
    dataset_name: str = "",
) -> str:
    few_shot_block = FEW_SHOT_EXAMPLES.get(dataset_name, "")
    return STAR_LOCAL_PROMPT.format(
        task_description=task_description,
        text=text,
        positive_label=positive_label,
        negative_label=negative_label,
        text_type=text_type,
        few_shot_block=few_shot_block,
    )


def render_similarity_prompt(pairs: list) -> str:
    """Render similarity prompt for a single pair (legacy callers may pass multi)."""
    if not pairs:
        raise ValueError("pairs must be non-empty")
    f1, f2 = pairs[0]
    return SIMILARITY_LABELING_PROMPT.format(feat_a=f1, feat_b=f2)


def render_similarity_prompt_single(feat_a: str, feat_b: str, polarity: str = "for") -> str:
    """Render single-pair similarity prompt, picking the polarity-conditioned variant.

    polarity: "for" (both features are FOR-acceptance) or "against" (both AGAINST).
    Cross-polarity pairs should be filtered upstream and not reach this function.
    """
    if polarity == "against":
        template = SIMILARITY_LABELING_PROMPT_AGAINST_AGAINST
    else:
        template = SIMILARITY_LABELING_PROMPT_FOR_FOR
    return template.format(feat_a=feat_a, feat_b=feat_b)


def render_cluster_naming_prompt(features: list) -> str:
    return CLUSTER_NAMING_PROMPT.format(
        feature_list="\n".join(f"- {f}" for f in features)
    )


def render_rubric_prompt(
    feature_name: str,
    feature_description: str,
    member_features: list,
    text_type: str,
) -> str:
    return RUBRIC_GENERATION_PROMPT.format(
        feature_name=feature_name,
        feature_description=feature_description,
        member_features="\n".join(f"- {f}" for f in member_features[:10]),
        text_type=text_type,
    )
