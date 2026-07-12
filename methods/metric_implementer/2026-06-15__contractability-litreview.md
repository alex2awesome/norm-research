# Contractability and Articulability: The Observable–Verifiable Distinction as a Theory of the Tacit Residual

*A literature review for a law-and-economics audience, mapping contract theory's "observable vs. verifiable" wedge onto a decomposition of expert judgment into Verifiable (V), Articulable (A), and Taste components, with a capability-indexed tacit residual τ(E).*

---

## 1. The Landscape

This review organizes a large law-and-economics literature around a single hinge concept that our project independently arrived at from the machine-learning side: **the gap between what a party can perceive and what a third party can be made to enforce.** In contract theory this is the **observable-vs-verifiable distinction**; in our framework it is the **anchor-vs-rubric gap.** The claim of this review is that these are the same object, and that the law-and-economics tradition has already built — with sixty years of accumulated formal and doctrinal machinery — a theory of exactly the quantity we are trying to measure.

Our framework decomposes an expert ("anchor") quality judgment into three layers. **V (Verifiable):** the part a third party can *mechanically* check (a passing test, a citation that exists, a word count). **A (Articulable):** the part that can be put in words a competent reader can apply but cannot mechanically verify ("the argument is well-motivated," "the proof is elegant"). **Taste:** the part that resists articulation altogether. The **tacit residual τ(E)** is the share of the anchor judgment that a *human-readable rubric scored by an executor of capability E* cannot reproduce. As E rises (a more capable LLM judge, or a more sophisticated court), some of A becomes reachable; what remains as E → ∞ is, in the limit, Taste.

The law professor's framing — that **contractability is "one avenue where articulability comes up in law"** — is precise and well-founded. A term is contractable only if it can be articulated to a degree that a **verifying third party (a court)** can establish whether it was satisfied. That is exactly **articulability-to-a-verifying-third-party**, with the executor E instantiated as a generalist adjudicator rather than an LLM judge. Non-contractible quality is τ(E).

The review proceeds: §2 states the hinge distinction and gets the canonical formal sources exactly right; §3 covers the incomplete-contracts/property-rights foundations and the hold-up problem; §4 covers describability/complexity/cost as the determinant of the **articulation budget L**; §5 covers what parties do when quality is non-verifiable (relational contracts, subjective evaluation); §6 states the **legal doctrine of definiteness** with exact Restatement/UCC sections and cases; §7 covers **rules vs. standards** and deliberate contract vagueness; §8 covers **mechanism-design/implementation** foundations; §9 makes the **contractability↔articulability bridge** fully explicit (V/A/τ and the role of E); §10 is honest about where the analogy is loose and where the literatures genuinely disagree (Maskin-Tirole vs. Hart-Moore); §11 lists open questions and what is novel for our project. A concept/instrument table closes the review.

---

## 2. The Hinge: Observable vs. Verifiable

The load-bearing distinction is between information that contracting **parties observe** during performance and information that an **enforcing third party can verify**. Only the latter can ground enforcement.

The canonical formal statement is **Hart & Moore, "Incomplete Contracts and Renegotiation," 56 Econometrica 755 (1988)**: parties "may be unable to describe the states of the world in enough detail that an outsider (the courts) could later verify which state had occurred, and so the contract will be incomplete." They add that to "describe [a state] in sufficient detail that an outsider (the courts) can verify whether a particular state ... has occurred, and so enforce a contract which is contingent on [it], may be prohibitively cost[ly]." This is the explicit framing of **third-party verifiability as the binding constraint.**

The cleanest modern *definition* comes from the law-and-economics handbook chapter, **Hermalin, Katz & Craswell, "Contract Law," in 1 Handbook of Law and Economics 3 (Polinsky & Shavell eds. 2007)**: a contract is "**a mapping from verifiable events to outcomes**," where verifiable means an event is observable not only to the parties but to "any third party (e.g., a judge) who might be called upon to adjudicate a dispute." If an outcome were contingent on an unverifiable event, "there would be no way for the third party to judge the extent of breach." **This is contractability = articulability-to-a-verifying-third-party, stated as a definition.** Observable-but-not-verifiable quality is precisely τ(E).

The cleanest *taxonomy* of why contracts are incomplete is **Maskin, "On Indescribable Contingencies and Incomplete Contracts," 46 European Economic Review 725 (2002)**, which gives three reasons: (1) aspects "may not be commonly observable; in particular, whoever is responsible for enforcing the contract (e.g., the court) may not be able to ascertain these aspects (in which case, we say that the aspects are 'unverifiable')"; (2) aspects "may be unforeseen or indescribable by the parties in advance"; and (3) writing foreseen aspects "may be too costly." These three map onto our decomposition with unusual cleanliness — see §9.

A four-rung **contractibility ladder** that already reads like our scorecard appears in **Andersson, Jordahl & Josephson, "Outsourcing Public Services: Contractibility, Cost, and Quality," 65 CESifo Economic Studies 349 (2019)**: quality is "perfectly contractible" when "a deviation can be identified by a court or arbitrator and an appropriate sanction can be applied"; "observable but unverifiable" when "the parties ... all know X, but this knowledge is not 'hard' enough to be the basis for an enforceable contract" (their prison-guard-force example); and at the bottom, unobservable/non-contractible. That ladder is **V → A → Taste read top-down.**

---

## 3. Incomplete-Contracts Foundations: Property Rights and Hold-Up

The reason the observable-verifiable wedge *matters* is that it forces contracts to be incomplete, which in turn drives the property-rights theory of the firm.

**Why contracts are incomplete.** The incomplete-contracts tradition (synthesized in **Hart, *Firms, Contracts, and Financial Structure* (Oxford 1995)**) attributes incompleteness to the transaction costs of (i) foreseeing contingencies (bounded rationality), (ii) negotiating over them, and (iii) **writing them down in language a court can interpret and enforce.** The third — describability/writing cost, not mere foresight — is doing most of the work: even foreseeable, foreseen states are routinely omitted because they cannot be rendered into enforceable language at acceptable cost. **Hart & Moore, "Foundations of Incomplete Contracts," 66 Review of Economic Studies 115 (1999)** make this precise: parties may **foresee** a contingency yet be unable to **describe** it ex ante in enforceable terms, even though they can describe it ex post once it occurs. Their model makes the **"describability of trades"** the central variable and identifies the fully-incomplete case with the **"null contract."** The null contract is the limit of zero articulability — τ = 1.

**Segal, "Complexity and Renegotiation," 66 Review of Economic Studies 57 (1999)** supplies a complementary foundation: defining verifiable as "observable by a court," he shows that as the environment's complexity n grows, the optimal message-contingent contract converges to the maximally incomplete benchmark — trade becomes "contractible ex post, but not ex ante." Complexity drives the articulable share toward zero.

**Property-rights theory / residual control.** Because not everything is contractible, **Grossman & Hart, "The Costs and Benefits of Ownership: A Theory of Vertical and Lateral Integration," 94 Journal of Political Economy 691 (1986)** distinguish *specific* rights (enumerated, contracted-upon) from *residual* rights of control (authority over uses not specified), and define **ownership as the purchase of residual control rights.** When listing and verifying all specific quality terms is too costly, parties allocate residual control to one party rather than articulating the quality term. **Hart & Moore, "Property Rights and the Nature of the Firm," 98 Journal of Political Economy 1119 (1990)** generalize to many agents and assets: with non-verifiable (hence non-contractible) relationship-specific human-capital investments and infeasible long-term contracts, asset ownership matters because it fixes ex-post bargaining positions, and complementary assets should be co-owned.

**Hold-up.** The non-contractibility of relationship-specific investment produces the **hold-up problem**: a party who sinks an ex-ante investment cannot contract on it, so part of its return is expropriated in ex-post renegotiation, causing **underinvestment**. The lineage runs through **Williamson, *The Economic Institutions of Capitalism* (1985)** (asset specificity, opportunism, bounded rationality) and **Klein, Crawford & Alchian, "Vertical Integration, Appropriable Rents, and the Competitive Contracting Process," 21 Journal of Law and Economics 297 (1978)** (appropriable quasi-rents), to Grossman-Hart and Hart-Moore. **Che & Hausch, "Cooperative Investments and the Value of Contracting," 89 American Economic Review 125 (1999)** sharpen the boundary: for **cooperative** investments (raising the partner's payoff) that are non-contractible, if commitment against renegotiation is impossible, contracting has *no value over pure ex-post bargaining.* In our terms, the non-contractible/tacit quality residual is then **wholly unreachable by any rubric** — τ is not merely large, it is untouchable.

---

## 4. Describability, Complexity, and Cost-of-Writing: The Articulation Budget L

A distinct strand models contractibility as bounded by the *cost of description* — directly the economic theory of our articulation budget L (how many words/clauses the rubric gets).

**Cost of describing (L itself).** **Dye, "Costly Contract Contingencies," 26 International Economic Review 233 (1985)** gives the first formal model of a per-contingency description cost, so parties optimally omit low-value contingencies. **Battigalli & Maggi, "Rigidity, Discretion, and the Costs of Writing Contracts," 92 American Economic Review 798 (2002)** model a contract as a finite collection of primitive sentences in a formal language, with writing cost proportional to length/detail, and identify two endogenous failure modes: **rigidity** (obligations insufficiently contingent on the state) and **discretion** (behavior insufficiently specified). These are precisely the two ways a *length-capped rubric* under-fits an anchor: missing partitions (rigidity) vs. missing scoring detail within a partition (discretion).

**Complexity / computability.** **Anderlini & Felli, "Incomplete Written Contracts: Undescribable States of Nature," 109 Quarterly Journal of Economics 1085 (1994)** require the state→outcome map to be **algorithmic (computable)**; combined with a computable selection process this yields endogenously incomplete optimal contracts. Their related "Incomplete Contracts and Complexity Costs" work shows that for any complexity measure in a broad axiomatic class, simple problems exist where the optimal contract collapses to the incomplete default. The computability requirement is what makes **executor capability E** precise on the legal side: a contract is a computable map, so describability is bounded by what an algorithm (our LLM judge) can compute — **capability-indexed articulability.**

**The irreducible residual.** **Al-Najjar, Anderlini & Felli, "Undescribable Events," 73 Review of Economic Studies 849 (2006)** prove the sharpest version of intrinsic indescribability: events whose consequences and probabilities are *known* to both parties, yet for which every *finite* description necessarily omits payoff-relevant features. This is a formal **τ(∞) > 0** — an irreducible non-articulable residual independent of executor capability, i.e., **Taste** as opposed to merely language-tacit A.

**Cognition.** **Tirole, "Cognition and Incomplete Contracts," 99 American Economic Review 265 (2009)** recasts the cost as cognitive — the cost of *thinking through* contingencies and their implications — so parties use heuristics and stop short. Bounded cognition is bounded E. The deeper Williamsonian root is **Williamson, *Markets and Hierarchies* (1975)**: "all complex contracts are incomplete by reason of bounded rationality."

**The AI bridge for this strand.** **Hadfield-Menell & Hadfield, "Incomplete Contracting and AI Alignment," Proc. AIES 2019 (arXiv:1804.04268, 2018)** port the entire describability-cost apparatus to AI: a specified reward/objective *is* an incomplete contract, and misspecification is "unintentional and unavoidable." This makes explicit that an **LLM-scored rubric is a contract** and its incompleteness is reward misspecification.

---

## 5. What Parties Do When Quality Is Non-Verifiable: Relational Contracts and Subjective Evaluation

When quality is observable-but-not-verifiable (our A), parties cannot enforce it in court, so they fall back on **implicit/relational contracts** sustained by repeated interaction — the economics analog of falling back on holistic dense judgment when a rubric fails.

**Existence.** **Bull, "The Existence of Self-Enforcing Implicit Contracts," 102 Quarterly Journal of Economics 147 (1987)** proves that performance-contingent incentive contracts, being non-enforceable, can still be sustained as non-trivial equilibria of a repeated bilateral game, supported by *intrafirm* reputation.

**Characterization.** **MacLeod & Malcomson, "Implicit Contracts, Incentive Compatibility, and Involuntary Unemployment," 57 Econometrica 447 (1989)** is the canonical statement: when performance cannot be verified in court, explicit piece-rate contracts are unenforceable, and self-enforcing implicit contracts are characterized as perfect equilibria of the repeated game, with a role for bonus/efficiency-wage payments.

**Objective vs. subjective = verifiable vs. articulable.** **Baker, Gibbons & Murphy, "Subjective Performance Measures in Optimal Incentive Contracts," 109 Quarterly Journal of Economics 1125 (1994)** is the cleanest formalization for our purposes: **objective** measures enter explicit (court-enforceable) contracts; **subjective** measures enter implicit contracts enforced by a reneging/self-enforcement constraint. Strikingly, the two can be **substitutes** — a sufficiently strong objective measure can *destroy* an otherwise first-best implicit contract — or **complements**, where neither alone yields positive profit but the combination does. Objective-in-explicit ≈ V; subjective-in-implicit ≈ A.

**Coarsening under unreliability.** **Levin, "Relational Incentive Contracts," 93 American Economic Review 835 (2003)** shows optimal relational contracts take a simple stationary form; under moral hazard with subjective performance, the optimum uses just **two compensation levels and terminates** after poor performance. This mirrors how a low-fidelity executor collapses a rich quality scale into coarse pass/fail.

**The sharpest articulability analog.** **MacLeod, "Optimal Contracting with Subjective Evaluation," 93 American Economic Review 216 (2003)** studies subjective evaluation where principal and agent may *disagree*: the optimal contract pays a bonus when performance is deemed acceptable but requires the principal to **burn money** (divert surplus to a third party or destroy it) whenever the agent disputes a negative evaluation. The result is that **pay is compressed** relative to the verifiable-measure case, and disagreement forces costly surplus destruction. This is the direct analog of **executor unreliability**: when the subjective signal (rubric) is noisy/contested, the incentive gradient flattens and value is lost — the efficiency price of low articulability.

---

## 6. The Legal Doctrine of Definiteness: Contract Law's Test for Articulability

Contract law has an operational, doctrinal test for articulability-to-a-verifying-third-party: the **indefiniteness/definiteness doctrine.** A court will not enforce a term it cannot apply to find breach and craft a remedy.

**Black letter.** **Restatement (Second) of Contracts § 33 (1981)** ("Certainty"): an offer cannot be accepted to form a contract "unless the terms of the contract are reasonably certain" (§ 33(1)); and terms "are reasonably certain if they provide a basis for **determining the existence of a breach and for giving an appropriate remedy**" (§ 33(2)). The comment grounds this in the policy that "contracts should be made by the parties, not by the courts." Section 33(2) is, on the law's side, **τ(E) = 0**: a term is enforceable iff a court (executor E = a generalist judge) can mechanically apply it.

**The permissive sales counterpart.** **UCC § 2-204(3)**: "Even though one or more terms are left open a contract for sale does not fail for indefiniteness if the parties have intended to make a contract and there is a reasonably certain basis for giving an appropriate remedy." The Article 2 **gap-fillers** — notably **UCC § 2-305** (open price term) — let courts supply missing terms via commercial standards. Here **the law itself acts as an external rubric** that raises articulability: the background of trade usage and reasonableness standards lets a court complete what the parties left open.

**Canonical cases.**
- ***Varney v. Ditmars*, 217 N.Y. 223 (1916)** — a promise of a "fair share of [the] profits" held too "vague, indefinite and uncertain" to enforce because the amount "cannot be computed" and is "pure conjecture." The classic holding that an *unquantifiable quality term is non-contractible*; recovery, if any, lies only in quantum meruit. This is the canonical case of quality the parties may *feel* but cannot *specify* — the tacit residual.
- ***Joseph Martin, Jr., Delicatessen, Inc. v. Schumacher*, 52 N.Y.2d 105 (1981)** — a lease-renewal clause at a rent "to be agreed upon" is an unenforceable "agreement to agree"; a material term left to future negotiation, with no objective formula, fails for indefiniteness. Courts refuse to supply terms they cannot derive from the agreement.

**Deliberately under-specified standards.**
- ***Wood v. Lucy, Lady Duff-Gordon*, 222 N.Y. 88 (1917)** (Cardozo, J.) — a court will *imply* a "reasonable efforts" obligation ("instinct with an obligation") to rescue an otherwise illusory/indefinite promise. This is the seminal source of best-efforts as a deliberately under-specified standard, later codified at **UCC § 2-306(2)** (best efforts in exclusive-dealing arrangements).
- **Restatement (Second) of Contracts § 205 (1981)** and **UCC § 1-304** impose a duty of **good faith and fair dealing** whose content is intentionally left open. **Summers, "'Good Faith' in General Contract Law and the Sales Provisions of the Uniform Commercial Code," 54 Virginia Law Review 195 (1968)** conceptualizes good faith as an **"excluder"** — a phrase with no general positive meaning that merely rules out heterogeneous forms of bad faith. Comment d to § 205, tracking Summers, states that "a complete catalogue of types of bad faith is impossible." This is **the doctrine openly conceding that the standard is non-enumerable** — a legal admission of an irreducible tacit residual.

**The law-and-econ bridge for this strand.** **Schwartz & Scott, "Contract Theory and the Limits of Contract Law," 113 Yale Law Journal 541 (2003)** port verifiability into contract law: parties leave out terms that "cannot be observed or, even if observable ... cannot be verified by enforcers at reasonable cost," and courts should privilege the written text — enforce the *articulated*, not the merely *observed*. **Scott, "A Theory of Self-Enforcing Indefinite Agreements," 103 Columbia Law Review 1641 (2003)** shows empirically that the indefiniteness doctrine "lives on" and that many contracts are *deliberately* incomplete, declining to specify even cheaply-verifiable measures and delegating proxy-selection to courts ex post.

---

## 7. Rules vs. Standards and Deliberate Vagueness: Allocating the Cost of Articulation

Where §6 is doctrine, this strand is design: the choice between a **precise term articulated ex ante** and a **vague term adjudicated ex post** is a deliberate allocation of describability/verification cost.

**The single axis.** **Kaplow, "Rules versus Standards: An Economic Analysis," 42 Duke Law Journal 557 (1992)** defines the choice on one axis: "the only distinction between rules and standards is the extent to which efforts to give content to the law are undertaken **before or after** individuals act" (p. 560). A **rule** gives content ex ante ("driving in excess of 55 mph"); a **standard** leaves content to be supplied ex post by an adjudicator ("driving at an excessive speed"). The cost structure: "Rules typically are more costly than standards to create, whereas standards tend to be more costly for individuals to interpret ... and for an adjudicator to apply to past conduct" (p. 557). Frequency of application is pivotal: when many actors face the same situation, paying once to articulate a rule beats repeated ex-post adjudication. **This is the articulability axis exactly:** a rule is a fully-articulated rubric fixed before scoring; a standard is an under-articulated rubric whose content the executor (judge of capability E) supplies at scoring time. A rule's over/under-inclusiveness is the fidelity loss from forcing tacit quality into a fixed rubric.

**Deliberate vagueness as design.** **Scott & Triantis, "Anticipating Litigation in Contract Design," 115 Yale Law Journal 814 (2006)** introduce the **front-end/back-end** frame: choosing precise (rule) vs. vague (standard) terms "implicitly allocate[s] costs between the front and back end." Vague terms like "best efforts" or "commercial reasonableness" "delegate to the back end the task of **selecting proxies**." A **proxy** is an evidentiary signal that, when noisy, "does not perfectly correlate with the true state of the world" (e.g., quarterly net income as a proxy for long-term profitability) — i.e., a *rubric metric*. **Choi & Triantis, "Strategic Vagueness in Contract Design: The Case of Corporate Acquisitions," 119 Yale Law Journal 848 (2010)** show, in the context of material-adverse-change clauses, that vague terms can **dominate** precise proxies, with litigation cost itself acting as a screen. (A companion framing, **Scott & Triantis, "Principles of Contract Design,"** circulated as an SSRN working paper, overlaps the *Anticipating Litigation* analysis; we cite the published article with higher confidence.)

For our project, Choi-Triantis's "vagueness can dominate" result is the legal analog of the claim that **for some quality dimensions an articulated rubric is strictly worse than delegating to a capable judge** — i.e., the articulable component A is small relative to what a skilled executor adds, so τ(E) shrinks faster by raising E than by enriching the rubric. **Nay, "Large Language Models as Fiduciaries," (arXiv:2301.10095 / SSRN 4335945, 2023)** makes the executor-of-capability-E reading explicit: legal **standards** (vs. rules) "facilitate the robust communication of inherently vague and underspecified goals" and let an agent infer "the spirit of a directive" in "unspecified states of the world," with meaning completed ex post by a third party (a court — or an LLM judge).

---

## 8. Mechanism Design and Implementation: The Formal Theory of Verification

Underneath the "verifiable" half of the hinge is the formal theory of *what a third party can implement using only what it can verify* — the rigorous backbone of our V layer.

**Costly state verification.** **Townsend, "Optimal Contracts and Competitive Markets with Costly State Verification," 21 Journal of Economic Theory 265 (1979)**: the state is private and unverifiable *unless a verification cost is paid*; the optimal deterministic-audit contract is **standard debt**, with verification triggered only on default. Verification is a costly, state-contingent *action* — a direct model of **paying to expand the articulable/verifiable region** (our coverage-k budget), with debt as the optimal partition.

**Partial verifiability.** **Green & Laffont, "Partially Verifiable Information and Mechanism Design," 53 Review of Economic Studies 447 (1986)**: when the agent's available messages depend on the true state (only some claims are checkable), the Revelation Principle fails in general and holds **only under the Nested Range Condition** (a transitivity/nesting structure on which types can mimic which). **Singh & Wittman, "Implementation with Partial Verification," 6 Review of Economic Design 63 (2001)** give concrete environments violating the NRC; absent a revelation principle, deciding implementability is in general intractable. This formalizes that **partial articulability is structure-dependent** — supporting our optimal-granularity and structure-dependent-τ results.

**Hard evidence / disclosure.** **Bull & Watson, "Evidence Disclosure and Verifiability," 118 Journal of Economic Theory 1 (2004)** and **Bull & Watson, "Hard Evidence and Mechanism Design," 58 Games and Economic Behavior 75 (2007)** ground abstract "verifiable messages" in actual documentary **evidence** parties can produce or withhold; **"evidentiary normality"** is the condition under which the abstract-declaration (Green-Laffont) model faithfully represents real evidence, and any implementable outcome is achievable via a two-stage send-message-then-disclose-evidence mechanism. This is the fidelity bridge **from free-text rubric to executor-checkable claims** — the A→V reduction.

**Implementation primitive.** **Maskin, "Nash Equilibrium and Welfare Optimality," 66 Review of Economic Studies 23 (1999)** (circulated 1977) supplies the third-party-verification primitive: outcomes condition only on a checkable message profile, and **Maskin monotonicity** is necessary and (with no veto power, n ≥ 3) sufficient for Nash implementation — an upper bound on which quality/social-choice functions are implementable from verifiable reports alone.

---

## 9. The Contractability ↔ Articulability Bridge (Made Explicit)

The mapping is close to one-to-one, and — gratifyingly — it is a mapping a contract theorist already drew (the Andersson-Jordahl-Josephson ladder, §2). Here is the bridge stated rigorously.

**Identity of the wedge.** The **observable-vs-verifiable distinction IS the anchor-vs-rubric gap.** Parties *observe* quality (the anchor / full expert judgment, V + A + Taste); a court can enforce only what is *verifiable* = describable in words a third party can mechanically apply (the rubric scored by executor E). The set of quality dimensions that are observable-but-not-verifiable is precisely the **non-contractible residual = τ(E)**, here with E = a court rather than an LLM judge.

**Layer-by-layer mapping.**

| Our framework | Contract-theory rung | Doctrinal / formal anchor |
|---|---|---|
| **V — Verifiable** (mechanically checkable) | "Perfectly contractible": a court/arbitrator can identify deviation and apply a sanction | Hermalin-Katz-Craswell "mapping from verifiable events"; Restatement § 33(2); objective measures in explicit contracts (Baker-Gibbons-Murphy) |
| **A — Articulable, not verifiable** (the anchor-vs-rubric gap) | "Observable but unverifiable": parties all know X but it is not "hard" enough to enforce | Andersson-Jordahl-Josephson middle rung; subjective measures in implicit contracts (MacLeod-Malcomson, Levin); standards / "best efforts" (UCC § 2-306(2), *Wood v. Lucy*) |
| **Taste / τ(E)** (non-articulable residual) | "Non-contractible": unobservable, or observable-but-unverifiable and irreducible | Che-Hausch (no contractual value); Al-Najjar-Anderlini-Felli undescribable events (τ(∞) > 0); good-faith "excluder," § 205 cmt. d (non-enumerable) |

**Maskin's three reasons map onto our decomposition.** From Maskin (2002): (1) **unverifiability** = the V/A boundary (observable to parties, not establishable by a third party = the part that drops to A/τ); (2) **indescribability/unforeseeability** = the A boundary (cannot be put into words at all); (3) **cost of specifying** = the **granularity/coverage-k budget L** that bounds rubric length (Dye; Battigalli-Maggi; Kaplow's ex-ante specification cost).

**Budget L vs. intrinsic τ — the literature splits along our axis.** Cost-of-describing/cognition theories (Dye 1985; Williamson's bounded rationality; Kaplow's ex-ante specification cost; Segal's complexity channel; Tirole 2009; Grossman-Hart's "costly to list all rights") model a **finite articulation budget L**: given more pages, cognition, or precedent, more becomes contractible (A shrinks toward V). Indescribability theories (Maskin-Tirole's indescribable contingencies; Al-Najjar-Anderlini-Felli's undescribable events) model **intrinsic τ** that no budget removes. This is exactly our distinction between **language-tacit A** (reachable as E or L rises) and **fully-tacit Taste** (the dense-model ceiling residual).

**The executor E is the court/algorithm.** Anderlini-Felli's requirement that the contract be a *computable* map makes "executor capability E" precise on the legal side; Tirole's cognitive cost makes E a bounded-rationality parameter; Kaplow's and Nay's standards make E literally the adjudicator who supplies content ex post. A more capable E (a more sophisticated court, a stronger LLM judge) reaches more of A — the contract-theory analog of our claim that τ(E) is decreasing in E.

**Why a rubric+executor is the right operationalization (not a revelation mechanism).** The Maskin-Tirole **irrelevance theorem** (§10) recovers first-best by tying payments to *revealed payoffs* via message games among the parties — i.e., reaching the anchor *without articulating the state.* That mechanism is defeated by renegotiation/collusion among the parties. An **LLM judge applying a fixed human-readable rubric is a non-strategic verifying third party**, immune to the renegotiation channel that defeats message games. So τ(E) *measured by a rubric+executor* is robust to the irrelevance critique in a way a revelation mechanism would not be — the rubric is the "hard," renegotiation-proof instrument the theory says you need.

---

## 10. Where the Analogy Is Loose, and Where the Literatures Disagree

Intellectual honesty requires three caveats.

**(1) States vs. payoffs — the Maskin-Tirole irrelevance theorem is the adversarial null.** **Maskin & Tirole, "Unforeseen Contingencies and Incomplete Contracts," 66 Review of Economic Studies 83 (1999)** and Maskin (2002) prove an **irrelevance theorem**: if parties can assign a probability distribution to their possible future *payoffs*, then the fact that they cannot *describe the physical states* in advance is irrelevant to welfare — a message-game mechanism keyed to payoffs, plus renegotiation, recovers first-best. Note carefully what survives: indescribability (reason 2) is *dissolved*, but **unverifiability (reason 1) may still bind.** For our project this is the sharpest stress test: it warns that a measured articulability gap may be an artifact of the *operationalization* (we are articulating *states*, not *payoffs*) rather than intrinsic tacitness. A τ(E) that would vanish under a payoff-revelation scheme is not "real" tacit knowledge. This is the contract-theory version of our standing worry that **τ(E) is operationalization-dependent.**

**(2) The foundations debate is unresolved.** **Hart & Moore (1999)** and **Segal (1999)** reply to Maskin-Tirole: the clever message-game solutions break down once **renegotiation cannot be ruled out**, restoring genuine incompleteness. **Tirole, "Incomplete Contracts: Where Do We Stand?," 67 Econometrica 741 (1999)** frames the open methodological tension between full rationality and assumed transaction costs/non-verifiability and asks whether complete-contract methodology can even account for authority and ownership. This debate is *exactly* our question — whether τ(E) is irreducible or merely an artifact of finite L plus a weak executor — and the economics literature has **not settled it.** We should present our empirical τ(E) as evidence in this debate, not as something the debate has resolved in our favor.

**(3) Three genuine looseness points in the analogy.**
- *Strategic vs. non-strategic.* Contract theory's third party adjudicates between *self-interested* parties who may misreport; our executor scores a fixed input with no strategic counterparty. This is mostly favorable to us (it sidesteps the revelation-principle failures of §8), but it means results keyed to incentive-compatibility (Green-Laffont, Maskin implementation) transfer only as *upper bounds on what a verifier could in principle establish*, not as descriptions of our measurement.
- *Welfare vs. fidelity.* The economics measures non-contractibility by its *welfare cost* (underinvestment, surplus destruction, adaptation cost); we measure τ(E) by *predictive fidelity* to the anchor. These coincide only when the anchor judgment is itself the welfare-relevant object. Bajari-Houghton-Tadelis's "adaptation costs = 7.5–14% of the winning bid" (below) is a *dollar* estimate of the gap; our τ(E) is a *variance-explained* estimate. They are analogous instruments, not the same number.
- *Indexing of E.* "Court capability" and "LLM-judge capability" are both real but not obviously commensurable; the law-and-econ literature rarely varies E continuously, whereas our project's central move is to trace τ(E) as a function of E. The economics gives us the *concept* of a capability-bounded verifier (computability in Anderlini-Felli, cognition in Tirole) but not a scaling curve.

---

## 11. Procurement and Empirical Contract (In)completeness: Where the Instruments Come From

The procurement literature theorizes *and measures* incompleteness, and its instruments are directly portable as proxies for τ(E).

**Theory.** **Bajari & Tadelis, "Incentives versus Transaction Costs: A Theory of Procurement Contracts," 32 RAND Journal of Economics 387 (2001)**: the buyer "incurs a cost of providing a comprehensive design" and trades ex-ante incentives against ex-post transaction costs; fixed-price contracts induce cost-saving effort but force costly renegotiation when the design is incomplete, while cost-plus eases adaptation. Prediction: **cost-plus for more complex (harder-to-describe) projects, fixed-price for simpler ones** — complexity/describability is the lever, the same lever as articulability. **Tadelis, "Complexity, Flexibility, and the Make-or-Buy Decision," 92 American Economic Review (Papers & Proceedings) 433 (2002)** extends this to firm boundaries: harder-to-describe transactions are internalized.

**Four operational families of incompleteness measurement** — directly portable as τ(E) instruments:
1. *Pricing-form ordinal scale.* **Crocker & Reynolds, "The Efficiency of Incomplete Contracts: An Empirical Analysis of Air Force Engine Procurement," 24 RAND Journal of Economics 126 (1993)** code contract pricing provisions on a completeness ladder (firm-fixed-price most complete … cost-plus least), predicted by asset specificity, uncertainty, and repeated dealing.
2. *Coded clause/safeguard counts.* **Saussier, "Transaction Costs and Contractual Incompleteness: The Case of Électricité de France," 42 Journal of Economic Behavior & Organization 189 (2000)** uses the EDF coal-transport database (1977–1997).
3. *Contract duration.* **Joskow, "Contract Duration and Relationship-Specific Investments: Empirical Evidence from Coal Markets," 77 American Economic Review 168 (1987)** (277 coal contracts); **Crocker & Masten, "Pretia ex Machina? Prices and Process in Long-Term Contracts," 34 Journal of Law and Economics 69 (1991)**; and **Goldberg & Erickson, "Quantity and Price Adjustment in Long-Term Contracts: A Case Study of Petroleum Coke," 30 Journal of Law and Economics 369 (1987)** on price-adjustment provisions.
4. *Ex-post adaptation as realized incompleteness.* **Bajari, Houghton & Tadelis, "The Economics of Incomplete Contracts: Theory and Evidence from Highway Procurement," 104 American Economic Review 1288 (2014)** measure incompleteness via **change orders** and structurally estimate adaptation costs at **7.5–14% of the winning bid** — a dollar-denominated estimate of the non-contracted (tacit) margin. **Bajari, McMillan & Tadelis, "Auctions versus Negotiations in Procurement: An Empirical Analysis," 25 Journal of Law, Economics & Organization 372 (2009)** show auctions underperform negotiation when "projects are complex and contractual design is incomplete." The foundational relational-exchange framing is **Goldberg, "Regulation and Administered Contracts," 7 Bell Journal of Economics 426 (1976)**.

The takeaway: **articulability is costly and chosen, not binary.** The "share of value resolved by ex-post adaptation rather than ex-ante specification" is an externally validated, real-world analog of the non-rubricizable fraction of an outcome — and Bajari-Houghton-Tadelis put a price on it.

---

## 12. Open Questions and What Is Novel for Our Project

**Open questions inherited from the literature.**
1. *Is τ(E) irreducible or an operationalization artifact?* The Maskin-Tirole vs. Hart-Moore foundations debate (§10) is the same question, unsettled. Our continuous τ(E)-vs-E curves could be read as *empirical evidence* in a debate that economics has only argued theoretically — but only if we can show the gap survives the payoff-revelation reframing.
2. *States vs. payoffs.* Maskin-Tirole's irrelevance result is about *payoff* describability, not *state* describability. Do our rubrics articulate states or payoffs? If our anchor is itself a payoff (a holistic quality score), the irrelevance theorem bites harder; if it is a state description, less so. This deserves an explicit experimental separation.
3. *Structure-dependence of partial articulability.* Green-Laffont's Nested Range Condition and Singh-Wittman's intractability results predict that **which subsets of A are jointly reachable is structure-dependent and can be computationally hard.** This is a precise, importable prediction for our optimal-granularity / coverage-k findings, and we have not yet tested whether our reachable-A subsets exhibit the predicted nesting structure.

**What is novel for our project (relative to this literature).**
- *A continuous, capability-indexed measurement of the gap.* Contract theory treats verifiability as essentially binary (a term is or is not enforceable) and varies it only via discrete instruments (pricing form, duration, clause counts). Our **τ(E) traced as a smooth function of executor capability E** — a scaling curve — has no analog in the law-and-economics literature, which gives us the *concept* of a bounded verifier (computability, cognition) but never a curve.
- *Fidelity rather than welfare as the metric.* We measure the gap by *reproduction fidelity* of the anchor, not by its welfare cost. This sidesteps the welfare-relevance qualifier in Maskin-Tirole but introduces its own validity burden (the anchor must be the right target). The literatures are complementary: their dollar-denominated adaptation-cost estimates (Bajari-Houghton-Tadelis) and our variance-explained τ(E) are two readings of one underlying quantity, and *triangulating* them is novel.
- *The rubric+executor as a renegotiation-proof verifier.* Our operationalization — a fixed human-readable rubric scored by a non-strategic LLM judge — is precisely the "hard," renegotiation-immune instrument that Hart-Moore's rebuttal to Maskin-Tirole says you need. We are, in effect, *building the verifier the theory presupposes*, which lets us measure τ(E) in a setting where the irrelevance theorem does not apply.
- *The A→V reduction as an engineering target.* Bull-Watson's "evidentiary normality" is a *condition* under which abstract messages faithfully represent hard evidence; our project treats the move from free-text rubric clauses to executor-checkable claims as something to *engineer and measure*, not merely characterize. That is a constructive use of a result the economics literature states only as an existence/fidelity condition.

In sum: the law professor's framing is not a loose metaphor. **Contractability is articulability-to-a-verifying-third-party**, the observable-vs-verifiable distinction is the anchor-vs-rubric gap, and the non-contractible residual is τ(E). The economics has built the conceptual scaffolding (V/A/Taste, budget L, intrinsic τ, capability-bounded verifier) and even the empirical instruments; what it has not built — and what our project supplies — is a **continuous, capability-indexed, fidelity-based measurement of where the line between articulable and tacit actually falls.**

---

## Appendix: Concept / Instrument Crosswalk

| Our concept | Law-and-economics counterpart | Primary source(s) |
|---|---|---|
| Anchor judgment (full V+A+Taste) | Quality **observable** to the parties | Grossman-Hart 1986; Hart-Moore 1988 |
| Rubric scored by executor E | A contract = **mapping from verifiable events to outcomes**; rule (ex ante) vs. standard (ex post) | Hermalin-Katz-Craswell 2007; Kaplow 1992 |
| V (Verifiable) | "Perfectly contractible"; objective measure in explicit contract | Andersson-Jordahl-Josephson 2019; Baker-Gibbons-Murphy 1994 |
| A (Articulable, not verifiable) | "Observable but unverifiable"; subjective measure in relational contract; standard/"best efforts" | MacLeod-Malcomson 1989; Levin 2003; UCC § 2-306(2); *Wood v. Lucy* |
| Taste / τ(E) | Non-contractible residual; undescribable events; "excluder" good faith | Che-Hausch 1999; Al-Najjar-Anderlini-Felli 2006; Summers 1968 |
| τ(E) = 0 (law's test) | Term "reasonably certain" → court can find breach + craft remedy | Restatement (Second) § 33(2) |
| Articulation budget L | Per-contingency description/writing cost; rigidity vs. discretion | Dye 1985; Battigalli-Maggi 2002 |
| Executor capability E | Computability of the contract map; cognitive cost of contingencies | Anderlini-Felli 1994; Tirole 2009 |
| Coverage-k / pay-to-verify | Costly state verification; audit-on-default (debt) | Townsend 1979 |
| Structure-dependence of reachable A | Partial verifiability; Nested Range Condition | Green-Laffont 1986; Singh-Wittman 2001 |
| A→V reduction (text → checkable claims) | Hard evidence; "evidentiary normality" | Bull-Watson 2004, 2007 |
| Adversarial null for τ(E) | Indescribability **irrelevance theorem** | Maskin-Tirole 1999; Maskin 2002 |
| Rebuttal (why rubric+executor is right) | Renegotiation defeats message games → genuine incompleteness | Hart-Moore 1999; Segal 1999 |
| τ(E) as dollar instrument | Ex-post adaptation cost = 7.5–14% of bid | Bajari-Houghton-Tadelis 2014 |
| Rubric = incomplete contract (AI link) | Reward function as incomplete contract; standards for LLM goals | Hadfield-Menell & Hadfield 2019; Nay 2023 |


## References (auto-verified BibTeX, 2026-06-15)

> 58 citations, web-verified + independently audited (search → fetch → resolvable id; attributed claim checked against the located source). Legal sources (Restatement/UCC/cases) verified to exist. See "needs manual review" for 0 contradicted-claim and 2 unlocatable/rejected items.

```bibtex
@article{alnajjar2006undescribable,
  title={Undescribable Events},
  author={Al-Najjar, Nabil I. and Anderlini, Luca and Felli, Leonardo},
  journal={The Review of Economic Studies},
  volume={73},
  number={4},
  pages={849--868},
  year={2006},
  doi={10.1111/j.1467-937X.2006.00399.x}
}

@article{anderlini1994incomplete,
  title={Incomplete Written Contracts: Undescribable States of Nature},
  author={Anderlini, Luca and Felli, Leonardo},
  journal={The Quarterly Journal of Economics},
  volume={109},
  number={4},
  pages={1085--1124},
  year={1994},
  doi={10.2307/2118357}
}

@article{anderlini1999complexity,
  title={Incomplete Contracts and Complexity Costs},
  author={Anderlini, Luca and Felli, Leonardo},
  journal={Theory and Decision},
  volume={46},
  number={1},
  pages={23--50},
  year={1999},
  doi={10.1023/A:1004917722235}
}

@article{andersson2019outsourcing,
  title   = {Outsourcing Public Services: Contractibility, Cost, and Quality},
  author  = {Andersson, Fredrik and Jordahl, Henrik and Josephson, Jens},
  journal = {CESifo Economic Studies},
  volume  = {65},
  number  = {4},
  pages   = {349--372},
  year    = {2019},
  doi     = {10.1093/cesifo/ifz009}
}

@article{bajari2001incentives,
  title   = {Incentives versus Transaction Costs: A Theory of Procurement Contracts},
  author  = {Bajari, Patrick and Tadelis, Steven},
  journal = {The RAND Journal of Economics},
  volume  = {32},
  number  = {3},
  pages   = {387--407},
  year    = {2001},
  doi     = {10.2307/2696361}
}

@article{bajari_houghton_tadelis_2014,
  author  = {Bajari, Patrick and Houghton, Stephanie and Tadelis, Steven},
  title   = {Bidding for Incomplete Contracts: An Empirical Analysis of Adaptation Costs},
  journal = {American Economic Review},
  year    = {2014},
  volume  = {104},
  number  = {4},
  pages   = {1288--1319},
  doi     = {10.1257/aer.104.4.1288}
}

@article{bajari_mcmillan_tadelis_2009,
  author  = {Bajari, Patrick and McMillan, Robert and Tadelis, Steven},
  title   = {Auctions Versus Negotiations in Procurement: An Empirical Analysis},
  journal = {The Journal of Law, Economics, and Organization},
  year    = {2009},
  volume  = {25},
  number  = {2},
  pages   = {372--399},
  doi     = {10.1093/jleo/ewn002}
}

@article{baker1994subjective,
  author  = {Baker, George and Gibbons, Robert and Murphy, Kevin J.},
  title   = {Subjective Performance Measures in Optimal Incentive Contracts},
  journal = {The Quarterly Journal of Economics},
  year    = {1994},
  volume  = {109},
  number  = {4},
  pages   = {1125--1156},
  doi     = {10.2307/2118358}
}

@article{battigalli2002rigidity,
  title={Rigidity, Discretion, and the Costs of Writing Contracts},
  author={Battigalli, Pierpaolo and Maggi, Giovanni},
  journal={American Economic Review},
  volume={92},
  number={4},
  pages={798--817},
  year={2002},
  doi={10.1257/00028280260344470}
}

@article{bull1987existence,
  author  = {Bull, Clive},
  title   = {The Existence of Self-Enforcing Implicit Contracts},
  journal = {The Quarterly Journal of Economics},
  year    = {1987},
  volume  = {102},
  number  = {1},
  pages   = {147--159},
  doi     = {10.2307/1884685}
}

@article{che_hausch_1999,
  title   = {Cooperative Investments and the Value of Contracting},
  author  = {Che, Yeon-Koo and Hausch, Donald B.},
  year    = {1999},
  journal = {American Economic Review},
  volume  = {89},
  number  = {1},
  pages   = {125--147},
  doi     = {10.1257/aer.89.1.125}
}

@article{choi2010strategic,
  author  = {Choi, Albert H. and Triantis, George G.},
  title   = {Strategic Vagueness in Contract Design: The Case of Corporate Acquisitions},
  journal = {Yale Law Journal},
  volume  = {119},
  number  = {5},
  pages   = {848--924},
  year    = {2010},
  url     = {https://www.jstor.org/stable/20698313}
}

@article{crocker1993efficiency,
  title   = {The Efficiency of Incomplete Contracts: An Empirical Analysis of Air Force Engine Procurement},
  author  = {Crocker, Keith J. and Reynolds, Kenneth J.},
  journal = {The RAND Journal of Economics},
  volume  = {24},
  number  = {1},
  pages   = {126--146},
  year    = {1993},
  doi     = {10.2307/2555956}
}

@article{crocker_masten_1991,
  author  = {Crocker, Keith J. and Masten, Scott E.},
  title   = {Pretia ex Machina? Prices and Process in Long-Term Contracts},
  journal = {The Journal of Law and Economics},
  year    = {1991},
  volume  = {34},
  number  = {1},
  pages   = {69--99},
  doi     = {10.1086/467219}
}

@article{dye1985costly,
  title={Costly Contract Contingencies},
  author={Dye, Ronald A.},
  journal={International Economic Review},
  volume={26},
  number={1},
  pages={233--250},
  year={1985},
  note={JSTOR stable 2526538},
  url={https://www.jstor.org/stable/2526538}
}

@article{goldberg_1976,
  author  = {Goldberg, Victor P.},
  title   = {Regulation and Administered Contracts},
  journal = {The Bell Journal of Economics},
  year    = {1976},
  volume  = {7},
  number  = {2},
  pages   = {426--448},
  doi     = {10.2307/3003265}
}

@article{goldberg_erickson_1987,
  author  = {Goldberg, Victor P. and Erickson, John R.},
  title   = {Quantity and Price Adjustment in Long-Term Contracts: A Case Study of Petroleum Coke},
  journal = {The Journal of Law and Economics},
  year    = {1987},
  volume  = {30},
  number  = {2},
  pages   = {369--398},
  doi     = {10.1086/467141}
}

@article{green1986partially,
  author  = {Green, Jerry R. and Laffont, Jean-Jacques},
  title   = {Partially Verifiable Information and Mechanism Design},
  journal = {The Review of Economic Studies},
  volume  = {53},
  number  = {3},
  pages   = {447--456},
  year    = {1986},
  doi     = {10.2307/2297639}
}

@article{grossman1986costs,
  author  = {Grossman, Sanford J. and Hart, Oliver D.},
  title   = {The Costs and Benefits of Ownership: A Theory of Vertical and Lateral Integration},
  journal = {Journal of Political Economy},
  year    = {1986},
  volume  = {94},
  number  = {4},
  pages   = {691--719},
  doi     = {10.1086/261404}
}

@inproceedings{hadfieldmenell2019incomplete,
  author    = {Hadfield-Menell, Dylan and Hadfield, Gillian K.},
  title     = {Incomplete Contracting and AI Alignment},
  booktitle = {Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society (AIES)},
  year      = {2019},
  pages     = {417--422},
  doi       = {10.1145/3306618.3314250}
}

@article{hart1988incomplete,
  author  = {Hart, Oliver and Moore, John},
  title   = {Incomplete Contracts and Renegotiation},
  journal = {Econometrica},
  year    = {1988},
  volume  = {56},
  number  = {4},
  pages   = {755--785},
  doi     = {10.2307/1912698}
}

@article{hart1990property,
  author  = {Hart, Oliver and Moore, John},
  title   = {Property Rights and the Nature of the Firm},
  journal = {Journal of Political Economy},
  year    = {1990},
  volume  = {98},
  number  = {6},
  pages   = {1119--1158},
  doi     = {10.1086/261729}
}

@book{hart1995firms,
  author    = {Hart, Oliver},
  title     = {Firms, Contracts, and Financial Structure},
  series    = {Clarendon Lectures in Economics},
  publisher = {Oxford University Press (Clarendon Press)},
  address   = {Oxford},
  year      = {1995},
  isbn      = {9780198288817}
}

@article{hart1999foundations,
  author  = {Hart, Oliver and Moore, John},
  title   = {Foundations of Incomplete Contracts},
  journal = {Review of Economic Studies},
  year    = {1999},
  volume  = {66},
  number  = {1},
  pages   = {115--138},
  doi     = {10.1111/1467-937X.00080}
}

@incollection{hermalin2007contract,
  author    = {Hermalin, Benjamin E. and Katz, Avery W. and Craswell, Richard},
  title     = {Contract Law},
  booktitle = {Handbook of Law and Economics},
  editor    = {Polinsky, A. Mitchell and Shavell, Steven},
  volume    = {1},
  chapter   = {1},
  pages     = {3--138},
  publisher = {Elsevier},
  address   = {Amsterdam},
  year      = {2007},
  url       = {https://ideas.repec.org/b/eee/lawhes/1.html}
}

@misc{josephmartin1981,
  author = {{New York Court of Appeals (Fuchsberg, J.)}},
  title = {Joseph Martin, Jr., Delicatessen, Inc. v. Schumacher, 52 N.Y.2d 105, 417 N.E.2d 541 (1981)},
  year = {1981},
  url = {https://www.courtlistener.com/opinion/2589351/joseph-martin-jr-delicatessen-inc-v-schumacher/}
}

@article{joskow1987contract,
  title   = {Contract Duration and Relationship-Specific Investments: Empirical Evidence from Coal Markets},
  author  = {Joskow, Paul L.},
  journal = {The American Economic Review},
  volume  = {77},
  number  = {1},
  pages   = {168--185},
  year    = {1987}
}

@article{kaplow1992rules,
  author  = {Kaplow, Louis},
  title   = {Rules versus Standards: An Economic Analysis},
  journal = {Duke Law Journal},
  volume  = {42},
  number  = {3},
  pages   = {557--629},
  year    = {1992},
  doi     = {10.2307/1372840}
}

@article{klein_crawford_alchian_1978,
  title   = {Vertical Integration, Appropriable Rents, and the Competitive Contracting Process},
  author  = {Klein, Benjamin and Crawford, Robert G. and Alchian, Armen A.},
  year    = {1978},
  journal = {The Journal of Law and Economics},
  volume  = {21},
  number  = {2},
  pages   = {297--326},
  doi     = {10.1086/466922}
}

@article{levin2003relational,
  author  = {Levin, Jonathan},
  title   = {Relational Incentive Contracts},
  journal = {American Economic Review},
  year    = {2003},
  volume  = {93},
  number  = {3},
  pages   = {835--857},
  doi     = {10.1257/000282803322157115}
}

@article{macleod1989implicit,
  author = {MacLeod, W. Bentley and Malcomson, James M.},
  title = {Implicit Contracts, Incentive Compatibility, and Involuntary Unemployment},
  journal = {Econometrica},
  year = {1989},
  volume = {57},
  number = {2},
  pages = {447--480},
  doi = {10.2307/1912559}
}

@article{macleod2003optimal,
  author  = {MacLeod, W. Bentley},
  title   = {Optimal Contracting with Subjective Evaluation},
  journal = {American Economic Review},
  year    = {2003},
  volume  = {93},
  number  = {1},
  pages   = {216--240},
  doi     = {10.1257/000282803321455232}
}

@article{maskin1999nash,
  author  = {Maskin, Eric S.},
  title   = {Nash Equilibrium and Welfare Optimality},
  journal = {The Review of Economic Studies},
  volume  = {66},
  number  = {1},
  pages   = {23--38},
  year    = {1999},
  doi     = {10.1111/1467-937X.00076}
}

@article{maskin_2002,
  title   = {On Indescribable Contingencies and Incomplete Contracts},
  author  = {Maskin, Eric},
  year    = {2002},
  journal = {European Economic Review},
  volume  = {46},
  number  = {4--5},
  pages   = {725--733},
  doi     = {10.1016/S0014-2921(01)00209-4}
}

@article{maskin_tirole_1999,
  title   = {Unforeseen Contingencies and Incomplete Contracts},
  author  = {Maskin, Eric and Tirole, Jean},
  year    = {1999},
  journal = {Review of Economic Studies},
  volume  = {66},
  number  = {1},
  pages   = {83--114},
  doi     = {10.1111/1467-937X.00079}
}

@article{nay_2023,
  author  = {Nay, John J.},
  title   = {Large Language Models as Fiduciaries: A Case Study Toward Robustly Communicating With Artificial Intelligence Through Legal Standards},
  year    = {2023},
  eprint  = {2301.10095},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL}
}

@misc{rest2d_contracts_205,
  title        = {Restatement (Second) of Contracts {\S} 205 --- Duty of Good Faith and Fair Dealing},
  author       = {{American Law Institute}},
  year         = {1981},
  howpublished = {\url{https://opencasebook.org/casebooks/10902-contracts-2023/resources/8.3.2-comments-to-restatement-2d-205-duty-of-good-faith-and-fair-dealing/}}
}

@misc{restatement2contracts33,
  author = {{American Law Institute}},
  title = {Restatement (Second) of Contracts {\S} 33 (Certainty)},
  year = {1981},
  howpublished = {American Law Institute}
}

@article{saussier2000transaction,
  title   = {Transaction costs and contractual incompleteness: the case of {\'E}lectricit{\'e} de France},
  author  = {Saussier, St{\'e}phane},
  journal = {Journal of Economic Behavior \& Organization},
  volume  = {42},
  number  = {2},
  pages   = {189--206},
  year    = {2000},
  doi     = {10.1016/S0167-2681(00)00085-8}
}

@article{schwartz_scott_2003,
  title   = {Contract Theory and the Limits of Contract Law},
  author  = {Schwartz, Alan and Scott, Robert E.},
  journal = {Yale Law Journal},
  volume  = {113},
  number  = {3},
  pages   = {541--619},
  year    = {2003},
  note    = {\url{https://digitalcommons.law.yale.edu/ylj/vol113/iss3/1/}}
}

@article{scott2003selfenforcing,
  author  = {Scott, Robert E.},
  title   = {A Theory of Self-Enforcing Indefinite Agreements},
  journal = {Columbia Law Review},
  volume  = {103},
  number  = {7},
  pages   = {1641--1699},
  year    = {2003},
  doi     = {10.2307/3593401}
}

@article{scott2005principles,
  author  = {Scott, Robert E. and Triantis, George G.},
  title   = {Principles of Contract Design},
  journal = {Yale Law Journal},
  volume  = {115},
  pages   = {814--879},
  year    = {2005},
  note    = {SSRN working paper},
  url     = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=722263}
}

@article{scott2006anticipating,
  author  = {Scott, Robert E. and Triantis, George G.},
  title   = {Anticipating Litigation in Contract Design},
  journal = {Yale Law Journal},
  volume  = {115},
  pages   = {814--879},
  year    = {2006},
  url     = {https://www.yalelawjournal.org/article/anticipating-litigation-in-contract-design}
}

@article{segal1999complexity,
  author  = {Segal, Ilya},
  title   = {Complexity and Renegotiation: A Foundation for Incomplete Contracts},
  journal = {Review of Economic Studies},
  year    = {1999},
  volume  = {66},
  number  = {1},
  pages   = {57--82},
  doi     = {10.1111/1467-937X.00078}
}

@article{singh2001implementation,
  author  = {Singh, Nirvikar and Wittman, Donald},
  title   = {Implementation with Partial Verification},
  journal = {Review of Economic Design},
  volume  = {6},
  number  = {1},
  pages   = {63--84},
  year    = {2001},
  doi     = {10.1007/PL00013697}
}

@article{summers_good_faith_1968,
  title   = {{"Good Faith"} in General Contract Law and the Sales Provisions of the Uniform Commercial Code},
  author  = {Summers, Robert S.},
  journal = {Virginia Law Review},
  volume  = {54},
  number  = {2},
  pages   = {195--267},
  year    = {1968},
  note    = {\url{https://www.jstor.org/stable/1071860}}
}

@article{tadelis2002complexity,
  title   = {Complexity, Flexibility, and the Make-or-Buy Decision},
  author  = {Tadelis, Steven},
  journal = {American Economic Review},
  volume  = {92},
  number  = {2},
  pages   = {433--437},
  year    = {2002},
  doi     = {10.1257/000282802320191750}
}

@article{tirole1999incomplete,
  title={Incomplete Contracts: Where Do We Stand?},
  author={Tirole, Jean},
  journal={Econometrica},
  volume={67},
  number={4},
  pages={741--781},
  year={1999},
  doi={10.1111/1468-0262.00052}
}

@article{tirole2009cognition,
  author  = {Tirole, Jean},
  title   = {Cognition and Incomplete Contracts},
  journal = {American Economic Review},
  year    = {2009},
  volume  = {99},
  number  = {1},
  pages   = {265--294},
  doi     = {10.1257/aer.99.1.265}
}

@article{townsend1979optimal,
  author  = {Townsend, Robert M.},
  title   = {Optimal Contracts and Competitive Markets with Costly State Verification},
  journal = {Journal of Economic Theory},
  volume  = {21},
  number  = {2},
  pages   = {265--293},
  year    = {1979},
  doi     = {10.1016/0022-0531(79)90031-0}
}

@misc{ucc2204,
  author = {{American Law Institute} and {National Conference of Commissioners on Uniform State Laws}},
  title = {Uniform Commercial Code {\S} 2-204 (Formation in General), subsection (3)},
  year = {1951},
  url = {https://www.law.cornell.edu/ucc/2/2-204}
}

@misc{ucc2305,
  author = {{American Law Institute} and {National Conference of Commissioners on Uniform State Laws}},
  title = {Uniform Commercial Code {\S} 2-305 (Open Price Term)},
  year = {1951},
  url = {https://www.law.cornell.edu/ucc/2/2-305}
}

@misc{ucc_1_304,
  title        = {Uniform Commercial Code {\S} 1-304 --- Obligation of Good Faith},
  author       = {{American Law Institute and National Conference of Commissioners on Uniform State Laws (Uniform Law Commission)}},
  year         = {2001},
  howpublished = {\url{https://www.law.cornell.edu/ucc/1/1-304}}
}

@misc{ucc_2_306,
  title        = {Uniform Commercial Code {\S} 2-306 --- Output, Requirements and Exclusive Dealings},
  author       = {{American Law Institute and National Conference of Commissioners on Uniform State Laws (Uniform Law Commission)}},
  year         = {1951},
  howpublished = {\url{https://www.law.cornell.edu/ucc/2/2-306}}
}

@misc{varney1916,
  author = {{New York Court of Appeals}},
  title = {Varney v. Ditmars, 217 N.Y. 223, 111 N.E. 822 (1916)},
  year = {1916},
  url = {https://law.justia.com/cases/new-york/court-of-appeals/1916/217-n-y-223-111-n-e-822-1916.html}
}

@book{williamson_1975,
  title     = {Markets and Hierarchies: Analysis and Antitrust Implications. A Study in the Economics of Internal Organization},
  author    = {Williamson, Oliver E.},
  year      = {1975},
  publisher = {Free Press},
  address   = {New York},
  isbn      = {9780029353608}
}

@book{williamson_1985,
  title     = {The Economic Institutions of Capitalism: Firms, Markets, Relational Contracting},
  author    = {Williamson, Oliver E.},
  year      = {1985},
  publisher = {Free Press},
  address   = {New York},
  isbn      = {9780029348208}
}

@misc{wood_v_lucy_1917,
  title        = {Wood v. Lucy, Lady Duff-Gordon},
  author       = {{New York Court of Appeals (Cardozo, J.)}},
  year         = {1917},
  note         = {222 N.Y. 88, 118 N.E. 214 (N.Y. 1917)},
  howpublished = {\url{https://en.wikipedia.org/wiki/Wood_v._Lucy,_Lady_Duff-Gordon}}
}

@article{bull2004evidence,
  author  = {Bull, Jesse and Watson, Joel},
  title   = {Evidence Disclosure and Verifiability},
  journal = {Journal of Economic Theory},
  volume  = {118},
  number  = {1},
  pages   = {1--31},
  year    = {2004},
  doi     = {10.1016/j.jet.2003.12.002}
}

@article{bull2007hard,
  author  = {Bull, Jesse and Watson, Joel},
  title   = {Hard Evidence and Mechanism Design},
  journal = {Games and Economic Behavior},
  volume  = {58},
  number  = {1},
  pages   = {75--93},
  year    = {2007},
  doi     = {10.1016/j.geb.2006.03.003}
}

```

### Citations needing manual review

All 60 extracted citations resolve to real works. Two had a DOI collision in the first
verification pass (the proposed DOI pointed to a neighboring article in the same volume);
the audit pass independently found the correct DOIs via Crossref, now used above:

- **Bull & Watson 2004**, *Evidence Disclosure and Verifiability*, J. Econ. Theory 118(1):1–31 — corrected DOI `10.1016/j.jet.2003.12.002`.
- **Bull & Watson 2007**, *Hard Evidence and Mechanism Design*, Games Econ. Behav. 58(1):75–93 — corrected DOI `10.1016/j.geb.2006.03.003`.

