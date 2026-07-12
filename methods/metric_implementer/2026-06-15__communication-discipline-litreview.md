# Articulability Is Social: How the Discipline of Communication Constitutes and Moves the Tacit Residual

> *Framing for this project.* In the `metric_implementer` setup, a **rubric** is a human-readable description of a quality construct; an **LLM judge of capability E** (the *executor*) runs that rubric to reproduce an **anchor** judgment Y about a datapoint X. Articulability-at-budget-L is the predictive V-information `I_{V_{L,E}}(X→Y)` decodable by the best rubric of `≤ L` words on executor E; the part no finite rubric+executor reproduces is the **tacit residual** with asymptote `τ(E) = lim_{L→∞} G_E(L)`. Prior reviews covered cross-field tacit-knowledge *measurement*, concept-*naming* psycholinguistics, and LLM-as-*annotator*. This review reads the **social-science discipline of Communication** — journalism studies, framing, agenda-setting, science/risk/health/strategic communication, rhetoric, diffusion, organizational comm, interpersonal/CMC, and Language & Social Interaction — through that lens, to test the user's two hypotheses: **(H1)** articulability is *largely socially constituted*, and **(H2)** *communication strategies move it*, so that articulability may be a function of (concept × executor E × **encoder strategy/skill**), not (concept × E) alone. The load-bearing question for our formalization: **does the encoder warrant a third axis beyond L and E?**

---

## 1. The Landscape

Communication, more than any field surveyed so far, treats "how much of a judgment can be put into shareable form, and what is lost" as its *native subject*, because that is what communication *is*. Where psychology asks whether an individual can verbalize a private skill, and information theory asks how much usable signal a channel carries, Communication asks the third thing our project actually needs: **what an encoder can do to a message to change how much of a complex judgment a decoding community recovers** — and it has spent sixty years measuring it.

Three orientations recur across the subfields and map onto our axes:

1. **Articulability as a property of the message representation, not the content (encoder-form effects).** The same fact, reworded, transfers differently: natural frequencies vs. conditional probabilities (Gigerenzer & Hoffrage 1995), equivalence framing (Tversky & Kahneman 1981), single-event vs. reference-class probability (Gigerenzer et al. 2005). This is the cleanest in-discipline proof of **H2** with content held fixed.
2. **Articulability as a co-produced social relation (encoder × decoder × shared context).** Conversational grounding (Clark & Wilkes-Gibbs 1986), the deficit-vs-dialogue debate in science communication (Wynne 1992; Bucchi 1998), constitutive views (Craig 1999; Pearce & Cronen 1980), and ethnomethodology's indexicality (Garfinkel 1967) all locate articulability in the *relation*, not the concept. This is **H1**.
3. **Articulability as historically/institutionally contingent and trainable.** News values shift to "shareworthiness" under metrics (Trilling et al. 2017); interpretive packages rise and fall over decades (Gamson & Modigliani 1989); doxic news judgment is convertible to orthodox criteria via professionalization (Schultz 2007). Articulability is a moving target, not a fixed per-concept scalar.

The discipline also supplies the **strongest in-principle ceiling argument** we have seen — Garfinkel's *etcetera* principle: the situations a rule applies to "can never be finitely enumerated," so every rubric carries an implicit "and so on" filled by tacit competence. This is a sociological statement that `τ(E) > 0` for any finite L. And it supplies the **strongest cautions**: the easiness effect (Scharrer et al. 2017) and illusory-truth/fluency effects (Dechêne et al. 2010) show that strategies move *perceived* articulability and felt comprehension *without* moving the truth — a fidelity/readability dissociation we must measure separately. Finally, O'Keefe's meta-analytic corpus is the field's own honest counterweight: across ~1,149 studies and 30 message-design choices, the median main effect is `r ≈ 0.10`, with most prediction intervals straddling zero — strategy levers are real but, on average, *small and unreliable*.

A blunt framing caveat up front: much of what follows is **convergence, not validity**. Communication theories are *isomorphic* to our construct and supply vocabulary, mechanisms, and measurement templates; few of them were built to predict an anchor label, and almost none separate "the encoder raised recoverable signal" from "the encoder raised the decoder's confidence/agreement." We flag this throughout.

---

## 2. Subfield Findings (the per-area yield)

**2.1 Journalism studies / news values / gatekeeping.** This subfield *is* a sixty-year attempt to articulate a tacit expert judgment ("the nose for news") into a scorable checklist, so its arc is unusually diagnostic. Galtung & Ruge (1965) posited ~12 news factors plus an **additivity hypothesis** (more factors → higher selection probability) — a literal additive articulable scoring function. Harcup & O'Neill (2001; 2017) empirically revised the list while conceding "no taxonomy can ever explain everything" — an explicit tacit residual. The deepest articulability lesson is **Staab's (1990) functional critique**: news factors are not inherent event properties but are *deliberately added by sources/journalists* to make stories newsworthy — newsworthiness is a manufacturable achievement, the strongest in-field statement of H2. **Bednarek & Caple's Discursive News Values Analysis (2017)** operationalizes this: newsworthiness is *constructed* via specific devices (intensifiers, comparatives/superlatives, quoted emotion, eliteness/attribution markers, proximity deixis, visual framing), so the same event scores higher under a better encoder. On the tacit side, **Schultz (2007, Bourdieu)** splits judgment into articulable *orthodox* values and silent *doxic* values lodged in habitus — a clean per-field operationalization of articulable-vs-residual that locates the residual in *socialization* (a social source of tacitness). **De Maeyer (2020)** argues the "nose for news" is Deweyan — a continuum from felt intuition to systematized appraisal, *relational* (value emerges by comparison among the day's competing stories). On measurability, Boukes et al. (2022) quantitatively link factor-count to prominence (length, front-page placement), and Spangher et al. (the project's own line, ~AUC 0.88–0.93 for front-page/layout newsworthiness with cross-corpus transfer) show a large articulable fraction with a measurable residual.

**2.2 Framing theory (Entman / Goffman / Druckman).** The Communication subfield most aligned with H1+H2. To frame is to **select aspects and make them salient** (Entman 1993) — an encoder-supplied compression that makes latent considerations *applicable*, not more numerous (the **applicability, not accessibility, mechanism**: Nelson, Oxley & Clawson 1997). A single conceptual-metaphor word (crime-as-beast vs. virus) reroutes policy reasoning more than the partisan gap (Thibodeau & Boroditsky 2011) — minimal cost, large shift. The subfield is explicit that **not every encoder can frame**: source credibility bounds effects (Druckman 2001), and frame strength/repetition and the *competitive* environment govern uptake (Chong & Druckman 2007). The honest ceiling: framing effects are "real but limited," shrinking under competition (Amsalem & Zoizner 2022). Measurement is mature: frame-element decomposition + cluster recovery (Matthes & Kohring 2008), generic-frame inventories (Semetko & Valkenburg 2000), the annotated Media Frames Corpus / Policy Frames Codebook of 15 frames (Card et al. 2015), and unsupervised semantic-axis scoring (FrameAxis; Kwak, An, Jing & Ahn 2021).

**2.3 Agenda-setting / priming / attribute and network agenda-setting.** Agenda-setting is a theory of how *repeated communication makes elements salient and shared* — exactly the social process that moves an attribute from "in the encoder's head" to "nameable and agreeable in the receiver's." Its three levels are a ladder of articulability units: first-level object salience (McCombs & Shaw 1972), second-level **attribute agenda-setting** (Ghanem 1997; McCombs & Evatt 1995) — *an attribute agenda IS a rubric* — and third-level **Network Agenda-Setting** (Guo & McCombs 2011; Vu, Guo & McCombs 2014), which transfers *bundles* of co-associated attributes and measures network-to-network correspondence by **QAP** (r ≈ 0.6–0.85). A rubric is exactly such a bundle, so NAS hands us both a model and a metric for "how much of the encoder's attribute-network the executor reproduces." Empirically, media→public correspondence is `r ≈ 0.49` overall and `~0.65` at the attribute level (Luo, Burley, Moe & Sui 2019). Two H1/H2-bearing refinements: the **compelling-arguments hypothesis** (some attributes transfer regardless of frequency, especially affective/valenced ones; Kim & Kiousis 2012) and **Need for Orientation** (transfer strongest when receiver relevance is high and certainty low; Weaver 1977; Matthes 2008) — an executor-side moderator.

**2.4 Science communication / public understanding / popularization.** The closest disciplinary analogue to our whole program: it literally studies how much expert knowledge re-encodes into lay-usable form and what is lost. The **deficit vs. dialogic** debate (Wynne 1992; Bucchi & Trench) is the question of whether articulability is a fixable property of the message or a co-produced relation — H1 in the field's own terms. **Bucchi's continuity model** recasts popularization not as dilution but as a stepwise descent of *registers* (intra-specialist → interdisciplinary → pedagogic → popular) where uncertainty is stripped and content *gains* lay articulability as it crosses levels — articulability is register-dependent, a communicative choice. Bucchi imports **boundary objects** (Star & Griesemer 1989): a representation "plastic enough to adapt to local needs yet robust enough to maintain a common identity" lets a partial-coverage articulation carry usable signal to a weaker decoder while the expert keeps the fuller referent — the optimal rubric as boundary object, not lossy reduction. **Collins' three-fold tacit knowledge** (relational / somatic / collective) types the residual by reducibility: relational is articulable in principle (a strategy lever), collective is the hard floor; and **interactional expertise** (Collins & Evans 2007) names precisely our *language-tacit* layer — fluency recoverable by a language-only decoder immersed in the discourse. Cautions are sharp: the **easiness effect** (Scharrer, Rupieper, Stadtler & Bromme 2017) shows simplification inflates self-rated understanding and lowers reliance on experts without raising true competence, and metaphor can raise *perceived* but not *actual* comprehension (Reijnierse et al.) — measure recovery against the anchor, never decoder-reported comprehensibility.

**2.5 Strategic / public relations communication.** The strongest disciplinary *home* for H2, because it studies clarity as a deliberate **design choice the encoder controls**. **Strategic ambiguity** (Eisenberg 1984) inverts the default: clarity is "non-normative," and skilled communicators *deliberately lower* articulability to preserve unified diversity, deniability, and flexibility — articulability is a *signed, bidirectional* lever, not a thing to maximize. **Message design logics** (O'Keefe 1988) rank encoders on a sophistication ladder (expressive < conventional < rhetorical); higher-logic encoders convey the same intention more effectively — an encoder-skill axis. **Equivocation theory** (Bavelas, Black, Chovil & Mullett 1990) is the key counter-hypothesis: ambiguity is produced by *situational* avoidance–avoidance binds, "more dependent on situations than on individuals" — and it ships a validated 4-dimension judge-scaling instrument (sender / content / receiver / context). NLP now quantifies the dial: large-scale extraction of CEO promise specificity-vs-flexibility from transcripts (the Majzoubi et al. line) and greenwashing-vagueness scoring validated against real performance.

**2.6 Rhetoric / rhetoric of science.** Rhetoric is a theory of articulability-as-social-relation. The **enthymeme** makes this explicit: an argument states only what is *not* already shared and lets the audience supply the elided premise from **endoxa** (reputable opinions, graded by who accepts them) — so what must be made explicit at budget L is exactly the gap between concept and the audience's shared knowledge, and the unstated premise *is* the tacit residual for that audience. **Topoi** are a community's articulated machinery for making implicit value-standards explicit (heuristics, not algorithms — the residual remains). **Stasis theory** is a decomposition tool that localizes *where* a judgment becomes contestable (fact / definition / quality / procedure) — letting us test whether the residual concentrates in the *quality/evaluation* stasis. **Burke's terministic screens** warrant terminology as a first-order lever. Empirically, **Fahnestock (1986)** tracks how popularization shifts the stasis and *inflates certainty* as a claim crosses audiences; **Swales' CARS** model (1990) converts tacit "how to write an intro" into codeable rhetorical moves (with computational successors); and the clean causal experiment that a terminology choice (jargon vs. plain) moves comprehension/engagement (Bullock, Colón Amill, Shulman & Dixon 2019).

**2.7 Diffusion of innovations / two-step flow / opinion leadership.** Reframes articulability as a *process-and-network* property co-produced as a concept travels. Rogers' perceived-attribute theory makes **complexity** a measured negative predictor of adoption (the five attributes explain ~49–87% of adoption-rate variance) — complexity as inverse articulability — and **reinvention** (Rice & Rogers 1980) documents *simplification of an overly complex innovation* as a driver of sustained adoption: a communicative act that raises articulability is causal to spread. **Two-step flow** (Katz & Lazarsfeld 1955) posits an intermediary encoder layer that interprets and simplifies; **Banerjee et al. (2019)** show experimentally that seeding central "gossip" nodes raises information spread, evidence that uptake is partly an encoder-*placement* variable. The same fluency cautions recur (illusory truth; easiness effect), so simplified diffusion can inflate apparent articulability.

**2.8 Organizational communication & knowledge management.** A decades-long debate *about* articulability. **Nonaka's SECI / externalization** (1994) is the optimistic pole: tacit→explicit conversion via *metaphor, analogy, concept, model* is real and amplifiable — a skilled encoder using the right figurative device raises articulability (the "twisting stretch" coinage that partially codified a kneading skill into machine specs). **Tsoukas (2003)**, reading Polanyi, is the loyal opposition: subsidiary awareness is destroyed the moment it becomes focal, so *full* conversion is impossible — an irreducible `τ` — but his positive proposal ("re-punctuating distinctions," "instructive talk," "drawing attention to how we draw attention") is itself an encoder-side strategy account *with a ceiling*. **Weick's sensemaking** treats articulation as a social, retrospective, language-mediated accomplishment (plausibility over accuracy). For measurement, **Zander & Kogut (1995)** give the canonical validated **codifiability** scale (with teachability and complexity subscales) predicting transfer speed — codifiability *is* articulability-at-budget made into Likert items — and **Boisot's I-Space** (1995) makes codification a continuous axis explicitly flagged to trade against context/fidelity. **Bechky (2003)** and **Hargadon & Sutton (1997)** show empirically that analogy and common-ground repair are encoder strategies that raise cross-group articulability, and that it is intersubjective (H1).

**2.9 Interpersonal / relational / constitutive communication.** The strongest endorsement of H1. **Berger & Luckmann (1966)** (externalization → objectivation → internalization) and symbolic interactionism (Mead; Blumer 1969) establish the base mechanism: a quality becomes a nameable *social object* only after repeated interaction *typifies* it — pre-typification it is private and inarticulable; reification is the endpoint of maximal (and taken-for-granted) articulability. **CMM** (Pearce & Cronen 1980) and the **constitutive metamodel** (Craig 1999) push to: communication doesn't transmit meaning, it makes the social world — so articulability is encoder-dependent *by theory*. The measurable wing is **conversational grounding**: under *least collaborative effort* (Clark & Brennan 1991), repeated reference compresses a long hedged description to a 1–2-word convention in 3–6 rounds (Clark & Wilkes-Gibbs 1986) — articulability rising as a function of communicative work — and **conceptual pacts** are partner-specific (term reuse drops from ~48% to ~18% on a partner swap; Brennan & Clark 1996), so the residual is *relational*: tacit for one dyad, fully articulable for another. **Schmid's (2020)** Entrenchment-and-Conventionalization model formalizes the individual↔social feedback loop; **Gentner's (1983)** structure-mapping gives the cognitive mechanism by which analogy externalizes previously-implicit relational structure. The boundary condition (Bormann's Symbolic Convergence Theory; convergence *can* fail and fragment by structural position) shows the lever is conditional, not monotone.

**2.10 Language & Social Interaction (ethnomethodology / CA / Goffman / Sacks).** The most theoretically pointed source for our `τ`. **Indexicality + the etcetera principle** (Garfinkel 1967; Garfinkel & Sacks 1970): meaning depends on unstated circumstances, the triggering conditions of any rule can never be finitely enumerated, and trying to replace indexical with context-free expressions yields an *endless regress* — a sociological proof that `I_{V_{L,E}}(X→Y)` saturates below the anchor for any finite L. Three empirical instruments operationalize the residual: **breaching experiments** (size the unwritten background by the repair effort to restore order), **"good organizational reasons for bad records"** (documents are indices readers reconstruct from tacit knowledge), and **instructed-action analysis** (Amerine & Bilmes 1988; Lynch & Lindwall 2022 — the rubric→execution gap on real tasks). On H2, the field reframes rather than rejects: **formulation** (Heritage & Watson 1979) is the explicit interactional act of glossing gist/upshot via *deletion, selection, transformation* — the interactional prototype of writing a rubric, with a built-in lossiness taxonomy; **keying/framing** (Goffman 1974) changes what a strip is taken to be evidence of; **recipient design** and **membership categorization** (Sacks 1992) make sayability audience-relative and compress complex judgments via shared category inferences. **Suchman (1987)** is the canonical EM→AI bridge: plans (read: rubrics) are *resources* for situated action, never complete specifications.

---

## 3. The Social Constitution of Articulability (H1)

The discipline's verdict on H1 is strong and near-unanimous at the level of *theory*: articulability is co-produced and community-relative, not an intrinsic property of (concept × E). Four convergent claims, with their best evidence:

- **A quality must be socially *typified* before it can be reliably named.** Symbolic interactionism and social-construction-of-reality (Mead; Blumer 1969; Berger & Luckmann 1966) make the un-named, un-objectified quality the *paradigm case* of the tacit residual — not because executors are weak, but because the *community* has not yet built shared language. This predicts that articulability of a construct should rise as a research community matures around it, independent of executor capability — a longitudinal, not cross-sectional, prediction.
- **The residual is partly *relational*, attaching to the encoder–decoder pair, not the concept.** Conceptual-pact partner-specificity (Brennan & Clark 1996, 48%→18%) is the cleanest number: the *same* articulation is tacit across one pairing and explicit across another. Collins' relational tacit knowledge names exactly this layer as in-principle articulable (strategy-reducible), distinct from the collective floor. *This is the single most important H1 result for us, because it predicts a measurable, non-intrinsic, executor-pair-specific component of `τ`.*
- **Articulability is institutionally and historically contingent.** Doxic→orthodox conversion via professionalization (Schultz 2007), shareworthiness displacing newsworthiness under metrics (Trilling et al. 2017), and interpretive-package lifecycles (Gamson & Modigliani 1989) all show the *articulable target itself moves*. This complicates any claim that `τ(E)` is a fixed concept-and-executor constant.
- **The ineffability is, at root, indexical/social.** Garfinkel's etcetera principle locates the irreducible residual not in the individual head but in the unbounded, society-held background expectancies that no finite text enumerates. This is a *social* (collective-tacit) reading of the floor, consonant with Collins' "collective tacit knowledge = property of society."

**Honest qualification.** The discipline establishes that articulability is *socially constituted* far more convincingly than it establishes that it is *largely* so versus cognitively/intrinsically so. Two countercurrents matter: (i) Bavelas's equivocation theory explicitly argues ambiguity is *situational, not individual* — which is "social" but in a way that *denies* encoder-skill, cutting against H2 even as it supports H1; (ii) the compelling-arguments hypothesis (some attributes transfer regardless of communicative effort) and Collins' collective-tacit floor both assert a concept-intrinsic component that no social work removes. The defensible synthesis: **`τ` decomposes into a social/relational part (movable, the field's home turf) and a collective/intrinsic part (the floor) — and the discipline's contribution is to show the first part is large, real, and manipulable, not that the floor is zero.**

---

## 4. Communication Strategies That Move Articulability (H2) — evidence-graded

We grade each lever on *how strong and how clean the evidence is that it moves recoverable articulability* (not merely persuasion, felt comprehension, or spread). **Grade A** = controlled, content-held-fixed evidence of moved *comprehension/recovery*; **B** = robust effect on uptake/spread with plausible but contested mapping to recovery; **C** = real effect but mostly on *perceived*/persuasion outcomes, or small/fragile/signed; **D** = theorized lever, thin direct measurement. The "signed/bidirectional" column flags levers that can *lower* articulability on purpose.

| Strategy | Mechanism (claim) | Best evidence | Grade | Signed? | Caveat for us |
|---|---|---|---|---|---|
| **Equivalence reframing** (gain/loss; "10% unemployed" vs "90% employed") | Pure encoding-form change of the *representation*, content identical | Tversky & Kahneman 1981; framing-effect meta (Kühberger) | A (for representation effect); C (for outcome size) | no | The cleanest H2 instrument: holds X fixed, varies only form |
| **Reference-class / natural-frequency reframing** | Making the implicit reference class explicit removes systematic mis-decoding | Gigerenzer & Hoffrage 1995; Gigerenzer et al. 2005 | A | no | Strongest "representation, not content, sets articulability" result in the corpus |
| **Visualization / icon arrays / fact boxes** | Offloads denominator neglect; lifts low-capability decoders | Gigerenzer et al. 2005; Franconeri et al. 2021 (graph literacy gates it) | A (gated by graph-literacy) | no | Conditional on decoder; an E×encoder interaction, not a universal gain |
| **Conceptual metaphor / single-source analogy** | Externalizes implicit relational structure; induces a transferable schema | Thibodeau & Boroditsky 2011 (reasoning shift); Gentner 1983 (mechanism) | B (reasoning); C (comprehension — perceived ≠ actual) | no | Reijnierse et al.: raises *perceived* comprehension, maybe not actual — validate vs anchor |
| **Narrative / storytelling / transportation** | Story form raises comprehension, recall, engagement for nonexperts | Dahlstrom 2014; Green & Brock 2000 (transportation scale) | B | no | Reduces counterarguing → can inflate apparent recovery; accuracy/persuasion tradeoff |
| **De-jargonization / register lowering** | Lowers processing cost; defining-in-text is *insufficient*, substitution works | Bullock et al. 2019; Rakedzon et al. 2017 (De-Jargonizer) | A | no | Readability ≠ comprehension; background knowledge still gates |
| **Emphasis framing (selection + salience)** | Raises *applicability* of latent considerations, not new info | Entman 1993; Nelson, Oxley & Clawson 1997 | B | no | Effect medium and shrinks under competition (Amsalem & Zoizner 2022) |
| **Repetition / entrenchment** | Raises accessibility/fluency; conventionalizes an articulation community-wide | Iyengar & Kinder 1987 (priming); Schmid 2020; Dechêne et al. 2010 | B (accessibility); C (truth/fluency confound) | no | **Illusory-truth caution**: repetition raises felt truth without content fidelity |
| **Conceptual-pact formation / lexical entrainment** | Iterated joint use compresses + locks a shared term | Clark & Wilkes-Gibbs 1986; Brennan & Clark 1996 | A | no | Gains are *partner-specific* — rebound on executor swap measures the relational residual |
| **Terminology coinage / naming a phenomenon** | Turns a quality into a shareable *social object*; later re-recognizable | Burke (terministic screens); Berger & Luckmann 1966 (typification) | C/D | yes (poor naming can deflect) | Mostly theoretical for *evaluative* constructs; cross-task naming-RT literature is the proxy |
| **Externalization via metaphor/model (SECI)** | Figurative device partially codifies an inarticulable skill | Nonaka 1994; Hargadon & Sutton 1997 (analogy as cross-domain transfer) | C | no | Tsoukas: bounded — never reaches 100%; SECI measures *activity*, not content recovered |
| **Frame strength / stronger (not just repeated) framing** | A high-strength frame extracts more movement than a repeated weak one | Chong & Druckman 2007 | B | no | Quality of the framing, not just budget L, governs uptake |
| **Boundary-object construction** | Plastic-yet-robust representation enables partial coverage by design | Star & Griesemer 1989; Bucchi 1998 | D | no | Compelling theory; no clean recovery-against-anchor measurement |
| **Audience-tailored register descent (Bucchi continuity)** | Moving toward popular register strips uncertainty to gain decodability | Bucchi 1998; Fahnestock 1986 (certainty inflation) | C | yes (fidelity↓ as coverage↑) | Explicit articulability-vs-fidelity tradeoff: gains decodability at the cost of hedging/scope |
| **Strategic ambiguity / deliberate vagueness** | Encoder *lowers* articulability for flexibility/deniability | Eisenberg 1984; Bavelas et al. 1990 | C | **yes (the canonical down-lever)** | Proof that the encoder axis is bidirectional, not a maximizer |
| **Formulation (gist/upshot glossing)** | Explicit interactional articulation of the tacit upshot | Heritage & Watson 1979 | D | no | Direct interactional analogue of writing a rubric; lossiness via deletion/selection/transformation |
| **Intermediary relay / seeding skilled encoders** | Routing through opinion leaders / central nodes raises uptake | Katz & Lazarsfeld 1955; Banerjee et al. 2019 | B (spread); D (recovery) | no | Encoder *placement* effect; spread ≠ recovery of the underlying judgment |

**Reading of the table.** The levers with the cleanest, content-held-fixed evidence of moved *recovery* (Grade A) are exactly the ones that operate on the **representation of fixed content**: equivalence/reference-class reframing, visualization, de-jargonization, and conceptual-pact entrainment. The levers that are most celebrated in theory (metaphor, narrative, naming, externalization) are mostly Grade B/C/D because their measured effect is on *persuasion, spread, or perceived comprehension*, with the perceived-vs-actual gap repeatedly documented. **H2 is well supported for representation-level encoding and for entrainment, and is supported-but-confounded for the richer semantic strategies.** The single most important methodological inheritance is therefore the field's own **perceived-vs-actual dissociation design** (easiness effect; illusory truth): any encoder lever we test must be scored on recovery against the anchor, never on agreement/confidence, or we will measure a phantom.

---

## 5. The Discipline's Own Ineffability / Residual Analogs

Communication has independently re-derived our `τ(E)` several times, each time as **anchor minus best codified channel**:

| Subfield | The `τ(E)`-analog | Status of the floor | Source |
|---|---|---|---|
| Journalism studies | "no taxonomy explains everything"; doxic (silent) vs orthodox (articulable) | non-zero, located in habitus/socialization | Harcup & O'Neill 2017; Schultz 2007 |
| Org comm / KM | Tsoukas's irreducible "from-to" residual; Boisot codification context-loss | in-principle non-zero (Tsoukas) | Tsoukas 2003; Boisot 1995 |
| Sci comm | collective tacit knowledge (the hard floor) vs relational (movable) | typed: collective floor > 0 | Collins & Evans 2007 |
| Persuasion | perspective-dependent residual in "argument quality" (annotator background shifts AQ) | non-zero, annotator-relative | O'Keefe & Jackson; Wachsmuth et al. 2017 |
| Ethnomethodology / LSI | the etcetera principle / endless regress — finite enumeration impossible | **provably non-zero for any finite L** | Garfinkel 1967; Garfinkel & Sacks 1970 |
| Interpersonal | conceptual-pact partner-specific residual; CMM context-stripping | non-zero, *relational* (pair-specific) | Brennan & Clark 1996; Pearce & Cronen 1980 |

The LSI/ethnomethodology entry is the field's distinctive contribution: it is the only place that gives an **in-principle, not merely empirical, argument that `τ > 0` for finite L** — the indexical remainder cannot be enumerated. There is one tension with our framing: this tradition treats the residual as *irreducible in principle*, whereas we treat it as *budget- and executor-relative* (`τ(E) = lim_{L→∞}`). Their "in principle" is best reread as **our `L→∞` asymptote**: the etcetera principle says the limit is approached but never reached at any finite L, which is exactly the shape of a saturating-concave `fidelity(L)` with a non-zero floor — fully compatible.

---

## 6. Borrowable Measures

These are the instruments worth porting into the scorecard or scaling experiments, ranked by directness.

**Most directly portable**

1. **QAP / network-correlation of attribute bundles (NAS).** Build an attribute co-occurrence network from the rubric and another from the executor's reproduced verdicts; correlate adjacency matrices via Quadratic Assignment Procedure with a permutation null. Gives a **network-level articulability score** beyond per-clause fidelity — how much of the *relational structure* of a multi-clause rubric the executor reproduces (NAS benchmark r ≈ 0.6–0.85; Guo & McCombs 2011; Vu et al. 2014). Direct answer to "did the executor get the bundle, not just the items."
2. **Perceived-vs-actual comprehension dissociation (easiness / illusory-truth design).** Always measure executor *recovery against the anchor* separately from executor *self-rated confidence / inter-judge agreement*; the **felt-minus-genuine gap** is itself a metric of *illusory articulability* — the exact failure mode where a better-worded rubric raises agreement without raising fidelity (Scharrer et al. 2017; Dechêne et al. 2010). This is a mandatory guardrail, not an optional add.
3. **Conceptual-pact partner-swap rebound.** Converge a minimal rubric on executor A; deploy on executor B; the fidelity drop estimates the *relational/executor-specific* component of `τ` (Brennan & Clark 1996). Sweeping B across model tiers turns this into a τ(E)-vs-E curve with a built-in "how much is pair-specific vs concept-intrinsic" split.
4. **Frame-element decomposition + cluster recovery (Matthes-Kohring).** Code a rubric's atomic clauses independently, score each, recover the operative rubric as a co-occurring cluster — a reliability protocol *and* a clause-attribution method for which clauses carry the V-information (mirrors our existing leaf re-clustering).
5. **De-Jargonizer accessibility score (Rakedzon et al. 2017).** A 0–100 lexical-decodability score (common / mid-frequency / jargon proportions) computable on any candidate rubric *before* running an executor — a cheap encoder-side covariate for the scaling experiment.
6. **Equivalence-framing minimal pairs.** Hold the underlying criterion fixed, vary only rubric wording; the executor-agreement delta measures the *pure encoder-strategy contribution* to articulability (ΔV-information at fixed content) — the field's cleanest instrument for isolating the encoder axis, directly portable as an ablation.

**Useful as covariates / nulls / calibration**

7. **News-factor count as an additive articulability score** (Galtung-Ruge additivity; Boukes et al. 2022 validated count→prominence) — template for "each rubric clause is an additive predictor, measure per-clause lift."
8. **Bavelas 4-dimension equivocation scaling** — a multi-rater rubric for "how committal/articulable is this verdict," with a built-in inter-rater-reliability protocol for an LLM-judge executor.
9. **Generic-frame inventory / Policy Frames Codebook (Card et al. 2015, 15 frames)** — a capped-vocabulary, transferable frame set; measure how much of Y is recoverable from frame-presence features alone (a V-information lower bound at fixed frame budget).
10. **FrameAxis bias+intensity on antonym semantic axes (Kwak et al. 2021)** — unsupervised, annotation-free continuous proxy for "how articulated is X along named axes" at scale.
11. **Readability-grade delta with a fidelity constraint** (PLABA/BioLaySumm metric stacks: SARI, Flesch-Kincaid, SMOG paired with BERTScore/factuality) — the two-axis (readability gain ∧ fidelity retention) evaluation our scorecard needs for any rewrite lever.
12. **Need-for-Orientation index (relevance × uncertainty; Weaver 1977; Matthes 2008)** — a receiver-side moderator covariate, letting articulability-at-budget be modeled as a function of executor *state*, not just capability.
13. **Zander-Kogut codifiability scale (1995)** — Likert articulability-at-budget for human-anchor elicitation, validated to predict transfer.
14. **Inter-coder reliability (Krippendorff/Holsti) on the rubric scheme** — the *noise floor*: rubric articulability is bounded above by how reliably humans apply the explicit criteria; separates genuine `τ` from coding noise (the same "subtract the anchor-reliability floor before claiming tacit" discipline our prior reviews demand).

---

## 7. Where It Complements vs. Contradicts Our Cognitive Framing

**Complements.**
- The field *adds the encoder* to a picture our prior reviews built mostly from (concept × executor). Where psycholinguistics gave codability as a property of (domain × namer) and tacit-knowledge measurement gave τ(E) as (concept × E), Communication insists the *encoder's choices* are a first-class determinant — and supplies content-held-fixed evidence (Tversky-Kahneman, Gigerenzer) that this is real, not a confound.
- It *re-derives our floor independently* (etcetera principle, Tsoukas, collective tacit knowledge), strengthening the claim that `τ > 0` is structural, not an estimator artifact.
- It *converges on a saturating-concave budget curve with a floor*: conversational grounding's reduction curves, Bucchi's register ladder, diffusion adoption curves, and grounding-cost accounting all trace `fidelity(work)` with diminishing returns toward a ceiling — the same shape as our `fidelity(L)`.
- Its **perceived-vs-actual dissociation** is the missing guardrail for our scorecard: it names precisely the way an LLM-judge rubric can post a high agreement number while the recovered fidelity is flat.

**Contradicts / complicates.**
- **The encoder may not be a free maximizer.** Bavelas's situational equivocation and O'Keefe's small/unreliable average effects warn that "skill" may be (a) frequently *situationally determined* rather than freely chosen, and (b) on average a *small* lever. Our framework, if it adds an encoder axis, should expect *modest and conditional* gains, not large free ones.
- **The articulable target moves.** Shareworthiness, package lifecycles, and doxic→orthodox conversion say `τ(concept, E)` is not time-invariant — a complication our static formalization elides.
- **Strategy is signed.** Strategic ambiguity is a *deliberate down-lever*; our framing implicitly treats more/better articulation as monotone-good. The encoder axis must be allowed to *lower* articulability.
- **Convergence ≠ validity (again).** Symbolic Convergence and naming-game-style results show communities can converge on a *shared but wrong* articulation; agreement-driven encoder gains can be herding. This re-flags the "reliability ≠ validity" caution from the naming review at the *community* level.

---

## 8. Implications for the τ(E) / fidelity(L) Formalization — does the encoder need a third axis?

This is the load-bearing question. Three defensible positions, in increasing cost to the formalism.

**Position 1 — No third axis; the encoder is absorbed into the `sup_L` (recommended default).** Our object is already `I_{V_{L,E}}(X→Y) = sup over rubrics s of length ≤ L of [usable info E decodes]`. The encoder's strategy *is* the choice of rubric `s` — so the *best* encoder strategy at budget L is, by definition, the `sup`. On this reading, **Communication's "encoder strategy" is not a new axis; it is the optimization the `sup` already performs**, and the field's contribution is *empirical guidance on which `s` achieves the sup* (representation-level reframing, entrainment, the right metaphor/frame) plus a *catalog of strategies* to search over. The equivalence-framing result — two rubrics with identical content but different `I_{V_{L,E}}` — is then not evidence of a missing axis but a demonstration that the `sup` over forms is non-trivial and that naive rubrics sit *below* it (an `ε`-style efficiency gap, exactly the IB "distance to optimum" the naming review imported). **This keeps the formalism two-dimensional (L, E) and reframes encoder skill as the gap between an *achieved* rubric and the budget-L frontier.**

**Position 2 — A bounded-rationality encoder axis is needed if the proposer cannot reach the `sup`.** The `sup` is an idealization. In practice the rubric is produced by a *proposer* (a model, a human, GEPA) of finite skill, and the field's whole point is that **encoders differ systematically in how close they get to the frontier** (message design logics: expressive < conventional < rhetorical; "who can frame"). If we want to model the *achieved* (not optimal) articulability, we need `fidelity(L, E, P)` where `P` indexes proposer/encoder capability, and the **encoder gap** `sup_s I_{V_{L,E}} − I_{V_{L,E}}[s_P]` is a real, measurable quantity (it is the `ε` efficiency loss, and it is what equivalence-framing pairs and the De-Jargonizer covariate measure). This is the *minimal* honest extension: keep `τ(E)` as the L→∞, optimal-encoder floor, but report an **achieved-encoder shortfall** as a separate term — analogous to how the naming review separated `ε` (waste below the best rubric) from `gNID`/`τ` (wrong granularity / irreducible residual). We recommend **adopting Position 2's *reporting* (an achieved-vs-frontier `ε` term) while keeping Position 1's *formalism***.

**Position 3 — A genuinely irreducible third axis (relational/decoder-pair).** The strongest case for a *non-absorbable* third axis is **partner-specificity** (Brennan & Clark 48%→18%) and CMM's context-indexing: articulability that exists for the *encoder-decoder pair* but not for the concept or the executor alone. If a non-trivial slice of `τ` is pair-specific — recoverable by *some* (E, encoder) pairing but by no rubric-on-E in isolation — then `I_{V_{L,E}}` is genuinely under-specified and needs a relational term. **This is the one place the discipline suggests our two-axis model may be structurally incomplete**, and it is *testable*: the partner-swap rebound instrument measures exactly the pair-specific component. Our recommendation: **treat the relational residual as an open empirical question — run the swap experiment; if the pair-specific component is small, Position 1/2 suffices; if large, the formalism needs a relational axis.**

**Net recommendation.** Keep `(L, E)` as the formal skeleton with the encoder folded into the `sup_L`. *Add reporting* of an achieved-encoder efficiency gap `ε` (Position 2) — this is the cleanest, most-supported contribution of the discipline and slots directly into the existing IB/ε machinery. *Hold open* the relational axis (Position 3) pending the partner-swap measurement. And *bake in* the perceived-vs-actual guardrail at every step, since it is the field's loudest warning and the failure mode our scorecard is most exposed to.

---

## 9. Mapping to Our Task Domains

| Task domain | Best-fit Communication subfield | Encoder-strategy lever most likely to move articulability | Caveat |
|---|---|---|---|
| **news_homepages** (spatial layout / placement) | Journalism: news values, DNVA, prominence-as-anchor | DNVA device inventory as rubric features; factor-count additivity; placement as revealed-preference anchor Y (matches the project's existing front-page work) | Shareworthiness drift; outlet/position/topic confounds already documented |
| **press_releases** | Strategic comm / PR; agenda-building; greenwashing-vagueness NLP | Specificity-vs-flexibility (Majzoubi line); information-subsidy pre-articulation; equivocation scaling for "how committal" | Strategic ambiguity is a *deliberate* down-lever here — articulability may be intentionally low |
| **peer_review** | Rhetoric of science (stasis, topoi, CARS); persuasion/argument-quality | Stasis classification (residual concentrates in *quality* stasis); special-topoi value inventory; CARS-move coding | "Argument quality" has a documented annotator-relative residual (perspective-dependence) |
| **creative_writing** | Interpersonal/constitutive; codability; verbal overshadowing | Conceptual-pact entrainment for evaluative vocabulary; metaphor/analogy for tacit style qualities | The taste-residual case where forcing a rubric most risks overshadowing (lowering fidelity) |
| **humor** | Symbolic convergence; conversational grounding; fluency | Entrainment / shared in-group reference; narrative framing | Convergence-vs-validity acute; community-specific, fragments by sub-group |
| **math** (elegance/clarity) | Sci-comm popularization; Bucchi register ladder; boundary objects | Register descent + analogy (Nonaka externalization); de-jargonization | Articulability-vs-fidelity tradeoff sharp (stripping rigor for decodability) |
| **code_review** | Org comm / KM (SECI, codifiability); instructed-action | Zander-Kogut codifiability scale; SECI externalization of review heuristics | Tsoukas ceiling: review "feel" partially irreducible; SECI measures activity not content |
| **notice_and_comment / legal** | LSI / ethnomethodology (indexicality, instructed-action); definiteness | Formulation/gist-glossing; instructed-action gap analysis | The etcetera principle bites hardest here — strongest in-principle `τ>0` |
| **patents** | Rhetoric (genre/move conventions); definiteness | CARS-style move coding; genre-convention scaffolds | Genre conventions are constructed/historical — articulability is time-varying |

---

## 10. Open Slots / What Is Novel for Us

1. **The encoder-efficiency gap `ε` for *evaluative* rubrics is unmeasured.** The IB `ε` (distance below the budget-L frontier) and the equivalence-framing delta both exist as instruments, but no one has measured, for an *anchor-predicting evaluative* rubric, how far a proposer's achieved rubric sits below the best same-length rubric. This is a clean, novel contribution and slots directly into the existing IB/ε column.
2. **Partner-specificity of articulability for LLM-judge pairs is untested.** Brennan-Clark's 48%→18% is for human dyads on reference games. Whether a rubric's articulability is *executor-pair-specific* (and thus whether Position 3's relational axis is needed) is a free, decisive experiment via the partner-swap rebound — and it is the one result that could force a structural change to our formalism.
3. **No one has separated "the encoder raised recovery" from "the encoder raised agreement" for rubric design.** The perceived-vs-actual dissociation is standard in health/sci-comm but has *not* been applied to LLM-judge rubric optimization, where the failure mode (GEPA optimizing a rubric to a higher-agreement-but-not-higher-fidelity local optimum) is live and dangerous. Building this guardrail into the scorecard is novel-for-us and protective.
4. **Stasis-localization of the residual is a genuinely new decomposition.** The hypothesis that `τ` concentrates in the *quality/evaluation* stasis (with fact/definition stases highly articulable) is testable on our tasks and not, to our knowledge, attempted anywhere — it would type *which kind of judgment* carries the residual, complementing Collins' relational/somatic/collective typing.
5. **Articulability-over-time (institutional contingency) is a prediction nobody has tested at LLM scale.** Symbolic interactionism predicts a construct's articulability rises as its community typifies it, *independent of executor capability*. With versioned corpora we could test whether a construct's `fidelity(L)` ceiling rises over the years a research community matures around it — a longitudinal articulability law, orthogonal to the executor scaling law.
6. **Strategic-ambiguity as a deliberate down-lever has no analog in our pipeline.** Our optimizer only pushes articulability *up*. Domains where the anchor's producers *deliberately* lower articulability (press releases, hedged legal language) may have a structurally *lower* ceiling for reasons of encoder intent, not concept difficulty — a confound worth modeling.
7. **Network-level (bundle) articulability is unmeasured.** NAS/QAP gives a ready metric for whether the executor reproduces the *relational structure* of a multi-clause rubric, not just the marginal clauses — a strictly richer fidelity measure than our current per-clause/aggregate scores, and a direct test of whether decomposed rubrics lose the inter-clause structure (relevant to the optimal-granularity U-shape).

**A final honesty note.** This review's central caution is its most important deliverable: most of the discipline's support for H2 is **convergence, not validity** — the theories are isomorphic to our construct and supply mechanism and measurement, but the *measured outcomes* are usually persuasion, spread, or felt comprehension, not anchor-recovery. The two places the field gives us genuinely load-bearing, content-held-fixed evidence (equivalence/reference-class reframing; conceptual-pact partner-specificity) are exactly the two we should build experiments around; the richer semantic levers (metaphor, narrative, externalization) should be treated as *candidate* levers to be validated against the anchor, with the perceived-vs-actual guardrail on, before we believe them.


## References (auto-verified BibTeX, 2026-06-15)

> 117 citations, web-verified + independently audited (search → fetch → resolvable id; attributed claim checked against the located source). See "needs manual review" for 0 contradicted-claim, 4 partial-match, and 1 unlocatable/rejected items.

```bibtex
@article{ambrosini2001tacit,
  title   = {Tacit Knowledge: Some Suggestions for Operationalization},
  author  = {Ambrosini, V{\'e}ronique and Bowman, Cliff},
  journal = {Journal of Management Studies},
  year    = {2001},
  volume  = {38},
  number  = {6},
  pages   = {811--829},
  doi     = {10.1111/1467-6486.00260}
}

@article{amerine1988following,
  title={Following instructions},
  author={Amerine, Ronald and Bilmes, Jack},
  journal={Human Studies},
  volume={11},
  number={2-3},
  pages={327--339},
  year={1988},
  doi={10.1007/BF00177308}
}

@article{amsalem2022real,
  title={Real, but Limited: A Meta-Analytic Assessment of Framing Effects in the Political Domain},
  author={Amsalem, Eran and Zoizner, Alon},
  journal={British Journal of Political Science},
  volume={52},
  number={1},
  pages={221--237},
  year={2022},
  doi={10.1017/S0007123420000253}
}

@article{aragones2000ambiguity,
  title={Strategic Ambiguity in Electoral Competition},
  author={Aragon{\`e}s, Enriqueta and Neeman, Zvika},
  journal={Journal of Theoretical Politics},
  volume={12},
  number={2},
  pages={183--204},
  year={2000},
  doi={10.1177/0951692800012002003}
}

@book{aristotle1991rhetoric,
  author      = {Aristotle},
  title       = {On Rhetoric: A Theory of Civic Discourse},
  translator  = {Kennedy, George A.},
  publisher   = {Oxford University Press},
  address     = {New York},
  year        = {1991},
  isbn        = {0195064860}
}

@inproceedings{bakshy2011everyones,
  title={Everyone's an Influencer: Quantifying Influence on Twitter},
  author={Bakshy, Eytan and Hofman, Jake M. and Mason, Winter A. and Watts, Duncan J.},
  booktitle={Proceedings of the Fourth ACM International Conference on Web Search and Data Mining (WSDM '11)},
  pages={65--74},
  year={2011},
  doi={10.1145/1935826.1935845}
}

@article{banerjee2019gossips,
  title={Using Gossips to Spread Information: Theory and Evidence from Two Randomized Controlled Trials},
  author={Banerjee, Abhijit and Chandrasekhar, Arun G. and Duflo, Esther and Jackson, Matthew O.},
  journal={The Review of Economic Studies},
  volume={86},
  number={6},
  pages={2453--2490},
  year={2019},
  doi={10.1093/restud/rdz008}
}

@article{baramtsabari2013instrument,
  title={An Instrument for Assessing Scientists' Written Skills in Public Communication of Science},
  author={Baram-Tsabari, Ayelet and Lewenstein, Bruce V.},
  journal={Science Communication},
  volume={35},
  number={1},
  pages={56--85},
  year={2013},
  doi={10.1177/1075547012440634}
}

@book{bavelas1990equivocal,
  author    = {Bavelas, Janet Beavin and Black, Alex and Chovil, Nicole and Mullett, Jennifer},
  title     = {Equivocal Communication},
  series    = {Sage Series in Interpersonal Communication},
  publisher = {Sage Publications},
  address   = {Newbury Park, CA},
  year      = {1990},
  isbn      = {0803929420}
}

@book{bazerman1988shaping,
  author    = {Bazerman, Charles},
  title     = {Shaping Written Knowledge: The Genre and Activity of the Experimental Article in Science},
  publisher = {University of Wisconsin Press},
  address   = {Madison},
  year      = {1988},
  isbn      = {9780299116941}
}

@article{bechky2003sharing,
  author  = {Bechky, Beth A.},
  title   = {Sharing Meaning Across Occupational Communities: The Transformation of Understanding on a Production Floor},
  journal = {Organization Science},
  year    = {2003},
  volume  = {14},
  number  = {3},
  pages   = {312--330},
  doi     = {10.1287/orsc.14.3.312.15162}
}

@book{bednarek2017discourse,
  title={The Discourse of News Values: How News Organizations Create Newsworthiness},
  author={Bednarek, Monika and Caple, Helen},
  publisher={Oxford University Press},
  address={New York},
  year={2017},
  isbn={9780190653941}
}

@book{berger1966social,
  author    = {Berger, Peter L. and Luckmann, Thomas},
  title     = {The Social Construction of Reality: A Treatise in the Sociology of Knowledge},
  year      = {1966},
  publisher = {Doubleday/Anchor Books},
  address   = {Garden City, NY},
  isbn      = {978-0385058988}
}

@article{berkowitz1990information,
  title={Information Subsidy and Agenda-Building in Local Television News},
  author={Berkowitz, Dan and Adams, Douglas B.},
  journal={Journalism Quarterly},
  volume={67},
  number={4},
  pages={723--731},
  year={1990},
  doi={10.1177/107769909006700426}
}

@book{blumer1969symbolic,
  author    = {Blumer, Herbert},
  title     = {Symbolic Interactionism: Perspective and Method},
  year      = {1969},
  publisher = {Prentice-Hall},
  address   = {Englewood Cliffs, NJ},
  isbn      = {978-0138799243}
}

@book{boisot1995information,
  title     = {Information Space: A Framework for Learning in Organizations, Institutions and Culture},
  author    = {Boisot, Max H.},
  year      = {1995},
  publisher = {Routledge},
  address   = {London and New York},
  isbn      = {9780415114905}
}

@article{bormann1972fantasy,
  author  = {Bormann, Ernest G.},
  title   = {Fantasy and Rhetorical Vision: The Rhetorical Criticism of Social Reality},
  journal = {Quarterly Journal of Speech},
  year    = {1972},
  volume  = {58},
  number  = {4},
  pages   = {396--407},
  doi     = {10.1080/00335637209383138}
}

@article{boukes2022newsworthiness,
  author  = {Boukes, Mark and Jones, Natalie P. and Vliegenthart, Rens},
  title   = {Newsworthiness and Story Prominence: How the Presence of News Factors Relates to Upfront Position and Length of News Stories},
  journal = {Journalism},
  year    = {2022},
  volume  = {23},
  number  = {1},
  pages   = {98--116},
  doi     = {10.1177/1464884919899313}
}

@article{brennan1996conceptual,
  author  = {Brennan, Susan E. and Clark, Herbert H.},
  title   = {Conceptual pacts and lexical choice in conversation},
  journal = {Journal of Experimental Psychology: Learning, Memory, and Cognition},
  year    = {1996},
  volume  = {22},
  number  = {6},
  pages   = {1482--1493},
  doi     = {10.1037/0278-7393.22.6.1482}
}

@book{bucchi1998science,
  title={Science and the Media: Alternative Routes to Scientific Communications},
  author={Bucchi, Massimiano},
  year={1998},
  publisher={Routledge},
  isbn={9780415189521}
}

@article{bullock2019jargon,
  author  = {Bullock, Olivia M. and Col{\'o}n Amill, Daniel and Shulman, Hillary C. and Dixon, Graham N.},
  title   = {Jargon as a barrier to effective science communication: Evidence from metacognition},
  journal = {Public Understanding of Science},
  year    = {2019},
  volume  = {28},
  number  = {7},
  pages   = {845--853},
  doi     = {10.1177/0963662519865687}
}

@incollection{burke1966terministic,
  author    = {Burke, Kenneth},
  title     = {Terministic Screens},
  booktitle = {Language as Symbolic Action: Essays on Life, Literature, and Method},
  publisher = {University of California Press},
  address   = {Berkeley},
  year      = {1966},
  pages     = {44--62},
  isbn      = {9780520001916}
}

@inproceedings{card2015media,
  title={The Media Frames Corpus: Annotations of Frames Across Issues},
  author={Card, Dallas and Boydstun, Amber E. and Gross, Justin H. and Resnik, Philip and Smith, Noah A.},
  booktitle={Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)},
  pages={438--444},
  year={2015},
  doi={10.3115/v1/P15-2072}
}

@article{carlson2018automating,
  title   = {Automating judgment? Algorithmic judgment, news knowledge, and journalistic professionalism},
  author  = {Carlson, Matt},
  journal = {New Media \& Society},
  year    = {2018},
  volume  = {20},
  number  = {5},
  pages   = {1755--1772},
  doi     = {10.1177/1461444817706684}
}

@article{centola2007complex,
  author  = {Centola, Damon and Macy, Michael},
  title   = {Complex Contagions and the Weakness of Long Ties},
  journal = {American Journal of Sociology},
  year    = {2007},
  volume  = {113},
  number  = {3},
  pages   = {702--734},
  doi     = {10.1086/521848}
}

@article{chong2007framing,
  author  = {Chong, Dennis and Druckman, James N.},
  title   = {Framing Theory},
  journal = {Annual Review of Political Science},
  year    = {2007},
  volume  = {10},
  pages   = {103--126},
  doi     = {10.1146/annurev.polisci.10.072805.103054}
}

@article{clark1986referring,
  author  = {Clark, Herbert H. and Wilkes-Gibbs, Deanna},
  title   = {Referring as a collaborative process},
  journal = {Cognition},
  year    = {1986},
  volume  = {22},
  number  = {1},
  pages   = {1--39},
  doi     = {10.1016/0010-0277(86)90010-7}
}

@incollection{clark1991grounding,
  author    = {Clark, Herbert H. and Brennan, Susan E.},
  title     = {Grounding in communication},
  booktitle = {Perspectives on Socially Shared Cognition},
  editor    = {Resnick, Lauren B. and Levine, John M. and Teasley, Stephanie D.},
  publisher = {American Psychological Association},
  address   = {Washington, DC},
  year      = {1991},
  pages     = {127--149},
  doi       = {10.1037/10096-006}
}

@book{collins2007rethinking,
  title={Rethinking Expertise},
  author={Collins, Harry and Evans, Robert},
  year={2007},
  publisher={University of Chicago Press},
  isbn={9780226113609}
}

@article{cook1999bridging,
  title   = {Bridging Epistemologies: The Generative Dance Between Organizational Knowledge and Organizational Knowing},
  author  = {Cook, Scott D. N. and Brown, John Seely},
  journal = {Organization Science},
  year    = {1999},
  volume  = {10},
  number  = {4},
  pages   = {381--400},
  doi     = {10.1287/orsc.10.4.381}
}

@article{craig1999communication,
  author  = {Craig, Robert T.},
  title   = {Communication Theory as a Field},
  journal = {Communication Theory},
  year    = {1999},
  volume  = {9},
  number  = {2},
  pages   = {119--161},
  doi     = {10.1111/j.1468-2885.1999.tb00355.x}
}

@article{dahlstrom2014using,
  title={Using narratives and storytelling to communicate science with nonexpert audiences},
  author={Dahlstrom, Michael F.},
  journal={Proceedings of the National Academy of Sciences},
  volume={111},
  number={Supplement 4},
  pages={13614--13620},
  year={2014},
  doi={10.1073/pnas.1320645111}
}

@article{dechene2010truth,
  title={The Truth About the Truth: A Meta-Analytic Review of the Truth Effect},
  author={Dech{\^e}ne, Alice and Stahl, Christoph and Hansen, Jochim and W{\"a}nke, Michaela},
  journal={Personality and Social Psychology Review},
  volume={14},
  number={2},
  pages={238--257},
  year={2010},
  doi={10.1177/1088868309352251}
}

@article{demaeyer2020nose,
  author  = {De Maeyer, Juliette},
  title   = {{``A Nose for News'': From (News) Values to Valuation}},
  journal = {Sociologica},
  year    = {2020},
  volume  = {14},
  number  = {2},
  pages   = {109--132},
  doi     = {10.6092/issn.1971-8853/11176}
}

@article{druckman2001limits,
  author  = {Druckman, James N.},
  title   = {On the Limits of Framing Effects: Who Can Frame?},
  journal = {Journal of Politics},
  year    = {2001},
  volume  = {63},
  number  = {4},
  pages   = {1041--1066},
  doi     = {10.1111/0022-3816.00100}
}

@article{eisenberg1984ambiguity,
  author  = {Eisenberg, Eric M.},
  title   = {Ambiguity as strategy in organizational communication},
  journal = {Communication Monographs},
  year    = {1984},
  volume  = {51},
  number  = {3},
  pages   = {227--242},
  doi     = {10.1080/03637758409390197}
}

@article{entman1993framing,
  author  = {Entman, Robert M.},
  title   = {Framing: Toward Clarification of a Fractured Paradigm},
  journal = {Journal of Communication},
  year    = {1993},
  volume  = {43},
  number  = {4},
  pages   = {51--58},
  doi     = {10.1111/j.1460-2466.1993.tb01304.x}
}

@article{fahnestock1986accommodating,
  author  = {Fahnestock, Jeanne},
  title   = {Accommodating Science: The Rhetorical Life of Scientific Facts},
  journal = {Written Communication},
  year    = {1986},
  volume  = {3},
  number  = {3},
  pages   = {275--296},
  doi     = {10.1177/0741088386003003001}
}

@article{farnese2019managing,
  title   = {Managing Knowledge in Organizations: A Nonaka's SECI Model Operationalization},
  author  = {Farnese, Maria Luisa and Barbieri, Barbara and Chirumbolo, Antonio and Patriotta, Gerardo},
  journal = {Frontiers in Psychology},
  year    = {2019},
  volume  = {10},
  pages   = {2730},
  doi     = {10.3389/fpsyg.2019.02730}
}

@article{franconeri2021science,
  title   = {The Science of Visual Data Communication: What Works},
  author  = {Franconeri, Steven L. and Padilla, Lace M. and Shah, Priti and Zacks, Jeffrey M. and Hullman, Jessica},
  journal = {Psychological Science in the Public Interest},
  year    = {2021},
  volume  = {22},
  number  = {3},
  pages   = {110--161},
  doi     = {10.1177/15291006211051956}
}

@article{galesic2011graph,
  title   = {Graph Literacy: A Cross-Cultural Comparison},
  author  = {Galesic, Mirta and Garcia-Retamero, Rocio},
  journal = {Medical Decision Making},
  year    = {2011},
  volume  = {31},
  number  = {3},
  pages   = {444--457},
  doi     = {10.1177/0272989X10373805}
}

@article{galtung1965structure,
  title={The Structure of Foreign News: The Presentation of the Congo, Cuba and Cyprus Crises in Four Norwegian Newspapers},
  author={Galtung, Johan and Ruge, Mari Holmboe},
  journal={Journal of Peace Research},
  volume={2},
  number={1},
  pages={64--91},
  year={1965},
  doi={10.1177/002234336500200104}
}

@article{gamson1989media,
  author  = {Gamson, William A. and Modigliani, Andre},
  title   = {Media Discourse and Public Opinion on Nuclear Power: A Constructionist Approach},
  journal = {American Journal of Sociology},
  year    = {1989},
  volume  = {95},
  number  = {1},
  pages   = {1--37},
  doi     = {10.1086/229213}
}

@book{garfinkel1967studies,
  author    = {Garfinkel, Harold},
  title     = {Studies in Ethnomethodology},
  year      = {1967},
  publisher = {Prentice-Hall},
  address   = {Englewood Cliffs, NJ},
  isbn      = {9780138583811}
}

@incollection{garfinkel1970formal,
  author    = {Garfinkel, Harold and Sacks, Harvey},
  title     = {On Formal Structures of Practical Actions},
  booktitle = {Theoretical Sociology: Perspectives and Developments},
  editor    = {McKinney, John C. and Tiryakian, Edward A.},
  year      = {1970},
  pages     = {338--366},
  publisher = {Appleton-Century-Crofts},
  address   = {New York},
  isbn      = {9780390623706}
}

@article{gentner1983structure,
  author  = {Gentner, Dedre},
  title   = {Structure-Mapping: A Theoretical Framework for Analogy},
  journal = {Cognitive Science},
  year    = {1983},
  volume  = {7},
  number  = {2},
  pages   = {155--170},
  doi     = {10.1207/s15516709cog0702_3}
}

@incollection{ghanem1997tapestry,
  title={Filling in the Tapestry: The Second Level of Agenda Setting},
  author={Ghanem, Salma},
  booktitle={Communication and Democracy: Exploring the Intellectual Frontiers in Agenda-Setting Theory},
  editor={McCombs, Maxwell and Shaw, Donald L. and Weaver, David},
  publisher={Lawrence Erlbaum Associates},
  address={Mahwah, NJ},
  pages={3--14},
  year={1997},
  doi={10.4324/9780203810880-2}
}

@article{gigerenzer1995bayesian,
  title   = {How to Improve Bayesian Reasoning Without Instruction: Frequency Formats},
  author  = {Gigerenzer, Gerd and Hoffrage, Ulrich},
  year    = {1995},
  journal = {Psychological Review},
  volume  = {102},
  number  = {4},
  pages   = {684--704},
  doi     = {10.1037/0033-295X.102.4.684}
}

@article{gigerenzer2005rain,
  title   = {``A 30\% Chance of Rain Tomorrow'': How Does the Public Understand Probabilistic Weather Forecasts?},
  author  = {Gigerenzer, Gerd and Hertwig, Ralph and van den Broek, Eva and Fasolo, Barbara and Katsikopoulos, Konstantinos V.},
  year    = {2005},
  journal = {Risk Analysis},
  volume  = {25},
  number  = {3},
  pages   = {623--629},
  doi     = {10.1111/j.1539-6924.2005.00608.x}
}

@book{goffman1974frame,
  author    = {Goffman, Erving},
  title     = {Frame Analysis: An Essay on the Organization of Experience},
  year      = {1974},
  publisher = {Harvard University Press},
  address   = {Cambridge, MA},
  isbn      = {9780674316560}
}

@article{green2000transportation,
  author  = {Green, Melanie C. and Brock, Timothy C.},
  title   = {The Role of Transportation in the Persuasiveness of Public Narratives},
  journal = {Journal of Personality and Social Psychology},
  year    = {2000},
  volume  = {79},
  number  = {5},
  pages   = {701--721},
  doi     = {10.1037/0022-3514.79.5.701}
}

@book{grunig1992excellence,
  editor    = {Grunig, James E.},
  title     = {Excellence in Public Relations and Communication Management},
  publisher = {Lawrence Erlbaum Associates},
  address   = {Hillsdale, NJ},
  year      = {1992},
  isbn      = {0805802266}
}

@inproceedings{guo2011network,
  title={Network Agenda Setting: A Third Level of Media Effects},
  author={Guo, Lei and McCombs, Maxwell},
  booktitle={Annual Conference of the International Communication Association (ICA), Political Communication Division},
  address={Boston, MA},
  year={2011},
  url={https://www.leiguo.net/publications/guo_nas_2011_ica.pdf}
}

@article{harcup2001whatisnews,
  title={What Is News? Galtung and Ruge revisited},
  author={Harcup, Tony and O'Neill, Deirdre},
  journal={Journalism Studies},
  volume={2},
  number={2},
  pages={261--280},
  year={2001},
  doi={10.1080/14616700118449}
}

@article{harcup2017whatisnews,
  title={What is News? News values revisited (again)},
  author={Harcup, Tony and O'Neill, Deirdre},
  journal={Journalism Studies},
  volume={18},
  number={12},
  pages={1470--1488},
  year={2017},
  doi={10.1080/1461670X.2016.1150193}
}

@article{hargadon1997technology,
  author  = {Hargadon, Andrew and Sutton, Robert I.},
  title   = {Technology Brokering and Innovation in a Product Development Firm},
  journal = {Administrative Science Quarterly},
  year    = {1997},
  volume  = {42},
  number  = {4},
  pages   = {716--749},
  doi     = {10.2307/2393655}
}

@incollection{heritage1979formulations,
  author    = {Heritage, John and Watson, Rod},
  title     = {Formulations as Conversational Objects},
  booktitle = {Everyday Language: Studies in Ethnomethodology},
  editor    = {Psathas, George},
  year      = {1979},
  pages     = {123--162},
  publisher = {Irvington},
  address   = {New York}
}

@book{hovland1953communication,
  title     = {Communication and Persuasion: Psychological Studies of Opinion Change},
  author    = {Hovland, Carl I. and Janis, Irving L. and Kelley, Harold H.},
  year      = {1953},
  publisher = {Yale University Press},
  address   = {New Haven, CT},
  isbn      = {978-0300005738}
}

@book{iyengar1987news,
  title={News That Matters: Television and American Opinion},
  author={Iyengar, Shanto and Kinder, Donald R.},
  publisher={University of Chicago Press},
  address={Chicago},
  year={1987},
  isbn={9780226388571}
}

@book{iyengar1991responsible,
  author    = {Iyengar, Shanto},
  title     = {Is Anyone Responsible? How Television Frames Political Issues},
  year      = {1991},
  publisher = {University of Chicago Press},
  address   = {Chicago},
  isbn      = {9780226388557}
}

@book{katz1955personal,
  author    = {Katz, Elihu and Lazarsfeld, Paul F.},
  title     = {Personal Influence: The Part Played by People in the Flow of Mass Communications},
  year      = {1955},
  publisher = {The Free Press},
  address   = {Glencoe, IL},
  isbn      = {9780029171509}
}

@article{kim2012affect,
  title={The Role of Affect in Agenda Building for Public Relations: Implications for Public Relations Outcomes},
  author={Kim, Ji Young and Kiousis, Spiro},
  journal={Journalism \& Mass Communication Quarterly},
  volume={89},
  number={4},
  pages={657--676},
  year={2012},
  doi={10.1177/1077699012455387}
}

@article{kwak2021frameaxis,
  title={FrameAxis: characterizing microframe bias and intensity with word embedding},
  author={Kwak, Haewoon and An, Jisun and Jing, Elise and Ahn, Yong-Yeol},
  journal={PeerJ Computer Science},
  volume={7},
  pages={e644},
  year={2021},
  doi={10.7717/peerj-cs.644}
}

@article{lipkus2001general,
  title   = {General Performance on a Numeracy Scale among Highly Educated Samples},
  author  = {Lipkus, Isaac M. and Samsa, Greg and Rimer, Barbara K.},
  journal = {Medical Decision Making},
  year    = {2001},
  volume  = {21},
  number  = {1},
  pages   = {37--44},
  doi     = {10.1177/0272989X0102100105}
}

@article{luo2019metaanalysis,
  title={A Meta-Analysis of News Media's Public Agenda-Setting Effects, 1972-2015},
  author={Luo, Yunjuan and Burley, Hansel and Moe, Alexander and Sui, Mingxiao},
  journal={Journalism \& Mass Communication Quarterly},
  volume={96},
  number={1},
  pages={150--172},
  year={2019},
  doi={10.1177/1077699018804500}
}

@book{lynch2023instructed,
  title={Instructed and Instructive Actions: The Situated Production, Reproduction, and Subversion of Social Order},
  editor={Lynch, Michael and Lindwall, Oskar},
  year={2023},
  publisher={Routledge},
  series={Directions in Ethnomethodology and Conversation Analysis},
  isbn={9781032230719},
  doi={10.4324/9781003279235}
}

@article{majzoubi2026ceo,
  title={Shaping expectations, losing flexibility: A study of CEO promises as strategic communication tools},
  author={Majzoubi, Majid and Murray, Alex and Mayew, William J.},
  journal={Strategic Management Journal},
  volume={47},
  number={7},
  pages={1980--2061},
  year={2026},
  doi={10.1002/smj.70068}
}

@article{matthes2008content,
  title={The Content Analysis of Media Frames: Toward Improving Reliability and Validity},
  author={Matthes, J{\"o}rg and Kohring, Matthias},
  journal={Journal of Communication},
  volume={58},
  number={2},
  pages={258--279},
  year={2008},
  doi={10.1111/j.1460-2466.2008.00384.x}
}

@article{matthes2008orientation,
  title={Need for Orientation as a Predictor of Agenda-Setting Effects: Causal Evidence from a Two-Wave Panel Study},
  author={Matthes, J{\"o}rg},
  journal={International Journal of Public Opinion Research},
  volume={20},
  number={4},
  pages={440--453},
  year={2008},
  doi={10.1093/ijpor/edn042}
}

@article{mccombs1972agenda,
  title={The Agenda-Setting Function of Mass Media},
  author={McCombs, Maxwell E. and Shaw, Donald L.},
  journal={Public Opinion Quarterly},
  volume={36},
  number={2},
  pages={176--187},
  year={1972},
  doi={10.1086/267990}
}

@article{mccombs1995temas,
  title={Los temas y los aspectos: explorando una nueva dimensi{\'o}n de la agenda setting},
  author={McCombs, Maxwell and Evatt, Dixie},
  journal={Comunicaci{\'o}n y Sociedad},
  volume={8},
  number={1},
  pages={7--32},
  year={1995},
  url={https://doaj.org/article/62af2a073da04e39b3a1b127b400cc77}
}

@article{miller1984genre,
  author  = {Miller, Carolyn R.},
  title   = {Genre as Social Action},
  journal = {Quarterly Journal of Speech},
  year    = {1984},
  volume  = {70},
  number  = {2},
  pages   = {151--167},
  doi     = {10.1080/00335638409383686}
}

@article{nelson1997toward,
  author  = {Nelson, Thomas E. and Oxley, Zoe M. and Clawson, Rosalee A.},
  title   = {Toward a Psychology of Framing Effects},
  journal = {Political Behavior},
  year    = {1997},
  volume  = {19},
  number  = {3},
  pages   = {221--246},
  doi     = {10.1023/A:1024834831093}
}

@article{nisbet2009whats,
  author  = {Nisbet, Matthew C. and Scheufele, Dietram A.},
  title   = {What's next for science communication? Promising directions and lingering distractions},
  journal = {American Journal of Botany},
  year    = {2009},
  volume  = {96},
  number  = {10},
  pages   = {1767--1778},
  doi     = {10.3732/ajb.0900041}
}

@article{noar2007tailoring,
  title   = {Does tailoring matter? Meta-analytic review of tailored print health behavior change interventions},
  author  = {Noar, Seth M. and Benac, Christina N. and Harris, Melissa S.},
  journal = {Psychological Bulletin},
  year    = {2007},
  volume  = {133},
  number  = {4},
  pages   = {673--693},
  doi     = {10.1037/0033-2909.133.4.673}
}

@article{nonaka1994dynamic,
  title={A Dynamic Theory of Organizational Knowledge Creation},
  author={Nonaka, Ikujiro},
  journal={Organization Science},
  volume={5},
  number={1},
  pages={14--37},
  year={1994},
  doi={10.1287/orsc.5.1.14}
}

@article{okeefe1988logic,
  author  = {O'Keefe, Barbara J.},
  title   = {The logic of message design: Individual differences in reasoning about communication},
  journal = {Communication Monographs},
  year    = {1988},
  volume  = {55},
  number  = {1},
  pages   = {80--103},
  doi     = {10.1080/03637758809376159}
}

@incollection{okeefe1995argument,
  title     = {Argument quality and persuasive effects: A review of current approaches},
  author    = {O'Keefe, Daniel J. and Jackson, Sally},
  year      = {1995},
  booktitle = {Argumentation and Values: Proceedings of the Ninth Alta Conference on Argumentation},
  editor    = {Jackson, Sally},
  pages     = {88--92},
  publisher = {Speech Communication Association},
  url       = {https://dokeefe.net/publicationlist.html}
}

@book{pearce1980communication,
  author    = {Pearce, W. Barnett and Cronen, Vernon E.},
  title     = {Communication, Action, and Meaning: The Creation of Social Realities},
  year      = {1980},
  publisher = {Praeger},
  address   = {New York},
  isbn      = {978-0030576119}
}

@book{perelman1969newrhetoric,
  author    = {Perelman, Cha{\"i}m and Olbrechts-Tyteca, Lucie},
  title     = {The New Rhetoric: A Treatise on Argumentation},
  year      = {1969},
  origyear  = {1958},
  translator = {Wilkinson, John and Weaver, Purcell},
  publisher = {University of Notre Dame Press},
  address   = {Notre Dame, IN},
  isbn      = {9780268004460}
}

@article{peters2006numeracy,
  title   = {Numeracy and Decision Making},
  author  = {Peters, Ellen and V{\"a}stfj{\"a}ll, Daniel and Slovic, Paul and Mertz, C. K. and Mazzocco, Ketti and Dickert, Stephan},
  journal = {Psychological Science},
  year    = {2006},
  volume  = {17},
  number  = {5},
  pages   = {407--413},
  doi     = {10.1111/j.1467-9280.2006.01720.x}
}

@incollection{petty1986elaboration,
  title={The Elaboration Likelihood Model of Persuasion},
  author={Petty, Richard E. and Cacioppo, John T.},
  booktitle={Advances in Experimental Social Psychology},
  volume={19},
  pages={123--205},
  year={1986},
  publisher={Academic Press},
  doi={10.1016/S0065-2601(08)60214-2}
}

@article{pinch1984social,
  title={The Social Construction of Facts and Artefacts: or How the Sociology of Science and the Sociology of Technology might Benefit Each Other},
  author={Pinch, Trevor J. and Bijker, Wiebe E.},
  journal={Social Studies of Science},
  volume={14},
  number={3},
  pages={399--441},
  year={1984},
  doi={10.1177/030631284014003004}
}

@article{rakedzon2017jargon,
  title={Automatic jargon identifier for scientists engaging with the public and science communication educators},
  author={Rakedzon, Tzipora and Segev, Elad and Chapnik, Noam and Yosef, Roy and Baram-Tsabari, Ayelet},
  journal={PLOS ONE},
  volume={12},
  number={8},
  pages={e0181742},
  year={2017},
  doi={10.1371/journal.pone.0181742}
}

@article{reijnierse2025metaphor,
  title={The differential effects of metaphor on comprehensibility and comprehension of environmental concepts},
  author={Reijnierse, W. Gudrun and Brugman, Britta C. and Droog, Ellen},
  journal={JCOM (Journal of Science Communication)},
  volume={24},
  number={4},
  year={2025},
  doi={10.22323/150520250702095506}
}

@article{reyna2008numeracy,
  title   = {Numeracy, ratio bias, and denominator neglect in judgments of risk and probability},
  author  = {Reyna, Valerie F. and Brainerd, Charles J.},
  year    = {2008},
  journal = {Learning and Individual Differences},
  volume  = {18},
  number  = {1},
  pages   = {89--107},
  doi     = {10.1016/j.lindif.2007.03.011}
}

@article{rice1980reinvention,
  author  = {Rice, Ronald E. and Rogers, Everett M.},
  title   = {Reinvention in the Innovation Process},
  journal = {Knowledge: Creation, Diffusion, Utilization},
  year    = {1980},
  volume  = {1},
  number  = {4},
  pages   = {499--514},
  doi     = {10.1177/107554708000100402}
}

@book{rogers2003diffusion,
  author    = {Rogers, Everett M.},
  title     = {Diffusion of Innovations},
  year      = {2003},
  edition   = {5th},
  publisher = {Free Press},
  address   = {New York},
  isbn      = {9780743222099}
}

@article{rothman1997shaping,
  title   = {Shaping Perceptions to Motivate Healthy Behavior: The Role of Message Framing},
  author  = {Rothman, Alexander J. and Salovey, Peter},
  journal = {Psychological Bulletin},
  year    = {1997},
  volume  = {121},
  number  = {1},
  pages   = {3--19},
  doi     = {10.1037/0033-2909.121.1.3}
}

@article{sacks1974simplest,
  author  = {Sacks, Harvey and Schegloff, Emanuel A. and Jefferson, Gail},
  title   = {A Simplest Systematics for the Organization of Turn-Taking for Conversation},
  journal = {Language},
  year    = {1974},
  volume  = {50},
  number  = {4},
  pages   = {696--735},
  doi     = {10.2307/412243}
}

@book{sacks1992lectures,
  author    = {Sacks, Harvey},
  title     = {Lectures on Conversation, Volumes I and II},
  editor    = {Jefferson, Gail},
  note      = {Introductions by Emanuel A. Schegloff},
  year      = {1992},
  publisher = {Blackwell},
  address   = {Oxford},
  isbn      = {9781557867056}
}

@article{scharrer2017when,
  title={When science becomes too easy: Science popularization inclines laypeople to underrate their dependence on experts},
  author={Scharrer, Lisa and Rupieper, Yvonne and Stadtler, Marc and Bromme, Rainer},
  journal={Public Understanding of Science},
  volume={26},
  number={8},
  pages={1003--1018},
  year={2017},
  doi={10.1177/0963662516680311}
}

@article{schegloff1977preference,
  author  = {Schegloff, Emanuel A. and Jefferson, Gail and Sacks, Harvey},
  title   = {The Preference for Self-Correction in the Organization of Repair in Conversation},
  journal = {Language},
  year    = {1977},
  volume  = {53},
  number  = {2},
  pages   = {361--382},
  doi     = {10.2307/413107}
}

@article{scheufele1999framing,
  author  = {Scheufele, Dietram A.},
  title   = {Framing as a Theory of Media Effects},
  journal = {Journal of Communication},
  year    = {1999},
  volume  = {49},
  number  = {1},
  pages   = {103--122},
  doi     = {10.1111/j.1460-2466.1999.tb02784.x}
}

@article{scheufele2007framing,
  title={Framing, Agenda Setting, and Priming: The Evolution of Three Media Effects Models},
  author={Scheufele, Dietram A. and Tewksbury, David},
  journal={Journal of Communication},
  volume={57},
  number={1},
  pages={9--20},
  year={2007},
  doi={10.1111/j.0021-9916.2007.00326.x}
}

@book{schmid2020dynamics,
  author    = {Schmid, Hans-J{\"o}rg},
  title     = {The Dynamics of the Linguistic System: Usage, Conventionalization, and Entrenchment},
  publisher = {Oxford University Press},
  address   = {Oxford},
  year      = {2020},
  isbn      = {9780198814771}
}

@article{schultz2007journalistic,
  title={The Journalistic Gut Feeling: Journalistic doxa, news habitus and orthodox news values},
  author={Schultz, Ida},
  journal={Journalism Practice},
  volume={1},
  number={2},
  pages={190--207},
  year={2007},
  doi={10.1080/17512780701275507}
}

@article{semetko2000framing,
  title={Framing European Politics: A Content Analysis of Press and Television News},
  author={Semetko, Holli A. and Valkenburg, Patti M.},
  journal={Journal of Communication},
  volume={50},
  number={2},
  pages={93--109},
  year={2000},
  doi={10.1111/j.1460-2466.2000.tb02843.x}
}

@book{shoemaker2009gatekeeping,
  title     = {Gatekeeping Theory},
  author    = {Shoemaker, Pamela J. and Vos, Tim P.},
  publisher = {Routledge},
  year      = {2009},
  isbn      = {9780415981392}
}

@article{staab1990role,
  title={The Role of News Factors in News Selection: A Theoretical Reconsideration},
  author={Staab, Joachim Friedrich},
  journal={European Journal of Communication},
  volume={5},
  number={4},
  pages={423--443},
  year={1990},
  doi={10.1177/0267323190005004003}
}

@article{star1989institutional,
  title={Institutional Ecology, `Translations' and Boundary Objects: Amateurs and Professionals in Berkeley's Museum of Vertebrate Zoology, 1907-39},
  author={Star, Susan Leigh and Griesemer, James R.},
  journal={Social Studies of Science},
  volume={19},
  number={3},
  pages={387--420},
  year={1989},
  doi={10.1177/030631289019003001}
}

@article{stivers2009universals,
  title={Universals and cultural variation in turn-taking in conversation},
  author={Stivers, Tanya and Enfield, N. J. and Brown, Penelope and Englert, Christina and Hayashi, Makoto and Heinemann, Trine and Hoymann, Gertie and Rossano, Federico and de Ruiter, Jan Peter and Yoon, Kyung-Eun and Levinson, Stephen C.},
  journal={Proceedings of the National Academy of Sciences},
  volume={106},
  number={26},
  pages={10587--10592},
  year={2009},
  doi={10.1073/pnas.0903616106}
}

@article{stivers2015coding,
  title={Coding Social Interaction: A Heretical Approach in Conversation Analysis?},
  author={Stivers, Tanya},
  journal={Research on Language and Social Interaction},
  volume={48},
  number={1},
  pages={1--19},
  year={2015},
  doi={10.1080/08351813.2015.993837}
}

@book{suchman1987plans,
  title={Plans and Situated Actions: The Problem of Human-Machine Communication},
  author={Suchman, Lucy A.},
  year={1987},
  publisher={Cambridge University Press},
  series={Learning in Doing: Social, Cognitive and Computational Perspectives},
  isbn={9780521337397}
}

@book{swales1990genre,
  author    = {Swales, John M.},
  title     = {Genre Analysis: English in Academic and Research Settings},
  publisher = {Cambridge University Press},
  address   = {Cambridge},
  series    = {Cambridge Applied Linguistics},
  year      = {1990},
  isbn      = {9780521338134}
}

@article{thibodeau2011metaphors,
  title={Metaphors We Think With: The Role of Metaphor in Reasoning},
  author={Thibodeau, Paul H. and Boroditsky, Lera},
  journal={PLoS ONE},
  volume={6},
  number={2},
  pages={e16782},
  year={2011},
  doi={10.1371/journal.pone.0016782}
}

@article{trilling2017newsworthiness,
  author  = {Trilling, Damian and Tolochko, Petro and Burscher, Bj{\"o}rn},
  title   = {From Newsworthiness to Shareworthiness: How to Predict News Sharing Based on Article Characteristics},
  journal = {Journalism \& Mass Communication Quarterly},
  year    = {2017},
  volume  = {94},
  number  = {1},
  pages   = {38--60},
  doi     = {10.1177/1077699016654682}
}

@incollection{tsoukas2003tacit,
  title={Do We Really Understand Tacit Knowledge?},
  author={Tsoukas, Haridimos},
  booktitle={The Blackwell Handbook of Organizational Learning and Knowledge Management},
  editor={Easterby-Smith, Mark and Lyles, Marjorie A.},
  pages={410--427},
  publisher={Blackwell},
  year={2003},
  isbn={9780631226727}
}

@article{tversky1981framing,
  author  = {Tversky, Amos and Kahneman, Daniel},
  title   = {The Framing of Decisions and the Psychology of Choice},
  journal = {Science},
  year    = {1981},
  volume  = {211},
  number  = {4481},
  pages   = {453--458},
  doi     = {10.1126/science.7455683}
}

@article{vu2014exploring,
  title={Exploring ``the World Outside and the Pictures in Our Heads'': A Network Agenda-Setting Study},
  author={Vu, Hong Tien and Guo, Lei and McCombs, Maxwell E.},
  journal={Journalism \& Mass Communication Quarterly},
  volume={91},
  number={4},
  pages={669--686},
  year={2014},
  doi={10.1177/1077699014550090}
}

@inproceedings{wachsmuth2017computational,
  title     = {Computational Argumentation Quality Assessment in Natural Language},
  author    = {Wachsmuth, Henning and Naderi, Nona and Hou, Yufang and Bilu, Yonatan and Prabhakaran, Vinodkumar and Thijm, Tim Alberdingk and Hirst, Graeme and Stein, Benno},
  year      = {2017},
  booktitle = {Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers},
  pages     = {176--187},
  address   = {Valencia, Spain},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/E17-1017/}
}

@incollection{weaver1977orientation,
  title={Political Issues and Voter Need for Orientation},
  author={Weaver, David H.},
  booktitle={The Emergence of American Political Issues: The Agenda-Setting Function of the Press},
  editor={Shaw, Donald L. and McCombs, Maxwell E.},
  publisher={West Publishing Co.},
  address={St. Paul, MN},
  pages={107--119},
  year={1977},
  isbn={0829901426}
}

@book{weick1995sensemaking,
  title     = {Sensemaking in Organizations},
  author    = {Weick, Karl E.},
  year      = {1995},
  publisher = {Sage Publications},
  address   = {Thousand Oaks, CA},
  series    = {Foundations for Organizational Science},
  isbn      = {9780803971776}
}

@article{weiss2005quick,
  title   = {Quick Assessment of Literacy in Primary Care: The Newest Vital Sign},
  author  = {Weiss, Barry D. and Mays, Mary Z. and Martz, William and Castro, Kelley Merriam and DeWalt, Darren A. and Pignone, Michael P. and Mockbee, Joy and Hale, Frank A.},
  journal = {The Annals of Family Medicine},
  year    = {2005},
  volume  = {3},
  number  = {6},
  pages   = {514--522},
  doi     = {10.1370/afm.405}
}

@article{white1950gatekeeper,
  title   = {The ``Gate Keeper'': A Case Study in the Selection of News},
  author  = {White, David Manning},
  journal = {Journalism Quarterly},
  year    = {1950},
  volume  = {27},
  number  = {4},
  pages   = {383--390},
  doi     = {10.1177/107769905002700403}
}

@article{wynne1992misunderstood,
  title={Misunderstood misunderstanding: social identities and public uptake of science},
  author={Wynne, Brian},
  journal={Public Understanding of Science},
  volume={1},
  number={3},
  pages={281--304},
  year={1992},
  doi={10.1088/0963-6625/1/3/004}
}

@article{zander1995knowledge,
  title   = {Knowledge and the Speed of the Transfer and Imitation of Organizational Capabilities: An Empirical Test},
  author  = {Zander, Udo and Kogut, Bruce},
  journal = {Organization Science},
  year    = {1995},
  volume  = {6},
  number  = {1},
  pages   = {76--92},
  doi     = {10.1287/orsc.6.1.76}
}

```

### Citations needing manual review

**Could not be located / rejected (1)**:

- Collins, Evans & Weinel 2021 (Modified Imitation Game) — verify: MISATTRIBUTION. No 2021 'Modified Imitation Game' paper authored by Harry Collins, Robert Evans, and Martin Wein

**Partial claim-match (4)** — spot-check exact wording/numbers:

- `bullock2019jargon`; `luo2019metaanalysis`; `rice1980reinvention`; `trilling2017newsworthiness`

