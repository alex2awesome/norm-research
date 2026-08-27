# Literature sweep: articulation gaps in scholarly peer review / academic manuscript judgment

Domain: what referees value in a paper — "significance," "novelty," "rigor" — and evidence that reviewers
converge on verdicts without being able to state (or without their stated criteria matching) the reasons.

## 1. Already in our bib (NOT new finds — confirmed via grep)

Checked: `notes/articulability-prompt-opt.bib`, `methods/metric_implementer/references.bib`,
`latex/paper-1__metric-codability/refs.bib`.

- `coupe2013peer` — Coupé, "Peer review versus citations — An analysis of best paper prizes" (Research Policy, 2013)
- `wainer2015peer` — Wainer, Eckmann & Rocha, "Peer-Selected 'Best Papers' — Are They Really That 'Good'?" (PLOS ONE, 2015)
- `pier2018low` — Pier et al., "Low agreement among reviewers evaluating the same NIH grant applications" (PNAS, 2018)
- `cortes2021inconsistency` — Cortes & Lawrence, "Inconsistency in Conference Peer Review: Revisiting the 2014 NeurIPS Experiment" (2021)
- `kennard2022disapere` — Kennard et al., DISAPERE dataset (NAACL 2022)
- `cole1981chance` — Cole, Cole & Simon, "Chance and Consensus in Peer Review" (Science, 1981)
- `merton1968matthew` — Merton, "The Matthew Effect in Science" (Science, 1968) — adjacent, not the Zuckerman & Merton 1971 referee piece
- (also present but out of this domain: `li2015` grant panels "Big names or big ideas" — another agent's grants territory)

Only found in `methods/metric_implementer/references.bib` (the fullest bib); the other two files had no additional peer-review entries beyond DISAPERE/rubric citations already listed above.

---

## 2. Top new finds, ranked

### #1 — Kang, Ammar, Dalvi, van Zuylen, Kohlmeier, Hovy & Schwartz, "A Dataset of Peer Reviews (PeerRead): Collection, Insights and NLP Applications" (NAACL 2018)
**ACADEMIC STUDY (NLP/computational, quantitative).** Directly on-point: this is a *measured* dissociation between the criteria reviewers are explicitly asked to score and the overall recommendation they give.

Why it's a gap: reviewers score papers on named sub-criteria (Substance, Clarity, Appropriateness, Impact, Meaningful comparison, Originality, Soundness/Correctness) *and* give an overall Recommendation. If the stated criteria actually drove the verdict, the two most substantive/rigor-related criteria should correlate most with the final call. They don't.

> **[VERIFIED]** Table 2, ACL 2017 subset — Pearson correlation of each aspect score with overall RECOMMENDATION: Substance 0.59, Clarity 0.42, Appropriateness 0.30, Impact 0.16, Meaningful_comparison 0.15, **Originality 0.08**, **Soundness/Correctness 0.01**. Paper's own gloss: "The aspects which correlate most strongly with the final recommendation are substance … and clarity. In contrast, soundness/correctness and originality are least correlated with the final recommendation."
> — https://ar5iv.labs.arxiv.org/abs/1804.09635 (Table 2, §3 "Data-Driven Analysis of Peer Reviews"); paper: NAACL 2018, https://aclanthology.org/N18-1149/

Use: the two criteria reviewers are told to evaluate for rigor and novelty are essentially uncorrelated with what they actually decide — a clean quantitative articulation-gap result in a domain (NLP peer review) with real ground-truth scores, not self-report.

---

### #2 — Christian Greiffenhagen, "Checking correctness in mathematical peer review" (*Social Studies of Science*, 2024, 54(2):184–209)
**ACADEMIC STUDY (ethnomethodological/STS, interviews + document analysis: 95 interviews with math journal editors, 100+ referee reports).** Shows correctness-checking in math — the most "objective," checklist-like of all review criteria — is actually done via unformalizable pattern recognition, and that editors struggle to even *state* what they expect referees to check.

> **[VERIFIED]** "In fields that people are familiar with, [referees] *know* what can go wrong. … So it's not a question of formally checking line by line so much as understanding where the problems are likely to appear." (section on checking correctness)
> **[VERIFIED]** "The way you frequently find mistakes is you see some little place in some little lemma somewhere in the paper, it doesn't quite fit with what you thought about how the world worked in this area."
> **[VERIFIED]** "When a question is really well known, the experts understand the bottleneck. … Even if the paper is 70 pages, I open to page 37, because I know that *this* is where the action has to be."
> **[VERIFIED]** "It was more challenging for my interviewees to express what they expected from referees with respect to checking correctness." (author's own observation about interview difficulty — a direct articulation-gap statement)
> — https://pmc.ncbi.nlm.nih.gov/articles/PMC10981185/ (page numbers unknown — PMC HTML, sections named "Checking Correctness in Practice," "Responsibility for Correctness")

Use: even "rigor"/correctness — the criterion assumed to be most checklist-amenable — is executed as tacit expert pattern-matching ("I know where page 37 is the action"), and the editors themselves have trouble putting their expectations for referees into words.

---

### #3 — Christian Greiffenhagen, "Judging Importance before Checking Correctness: Quick Opinions in Mathematical Peer Review" (*Science, Technology, & Human Values*, 2024) — companion paper to #2, same dataset
**ACADEMIC STUDY.** Directly names the phenomenon we care about: prestigious math journals get "quick opinions" on *importance/significance* — a holistic, fast, pre-verification judgment — before anyone even checks whether the proof is right.

> **[SNIPPET]** Editors "solicit several 'quick opinions' about the importance of results, and only after a positive evaluation do they ask a referee to check their correctness" — paraphrase-close synthesis from search of the paper (I could not get the PDF to render text for exact quotes; DOI: 10.1177/01622439231203445).
> **[LEAD]** — could not verify verbatim quotes; PDF at https://ira.lib.polyu.edu.hk/bitstream/10397/109688/1/Greiffenhagen_Judging_Importance_Before.pdf did not render as text.

Use: significance-judgment is structurally *prior to and separate from* correctness-checking in the actual editorial workflow — i.e., the "hard," articulable criterion (correctness) is gated by the soft, holistic one (importance), inverting the naive model where rigor comes first.

---

### #4 — Guetzkow, Lamont & Mallard, "What Is Originality in the Humanities and the Social Sciences?" (*American Sociological Review*, 2004, 69(2):190–212)
**ACADEMIC STUDY (81 interviews with panelists on 5 multidisciplinary fellowship competitions).** The single most direct paper on "a criterion everyone invokes decisively but no one can define the same way."

> **[SNIPPET]** Panelists in the social sciences and humanities define originality "much more broadly: as using a new approach, theory, method, or data; studying a new topic; doing research in an understudied area; or producing new findings" — i.e., the word covers a large, non-overlapping cluster of things, yet panelists deploy it as if it names one property.
> **[SNIPPET]** The paper shows panelists treat originality "as an indication of the researcher's moral character, especially of his/her authenticity and integrity" — the criterion slides from a property of the text to a trait of the person, which is itself evidence the stated criterion is not what's actually doing the evaluative work.
> **[LEAD]** — attempted direct verbatim panelist quotes ("I know it when I see it"-type statements); PDF fetches (HAL, proseminarcrossnationalstudies.wordpress.com mirror) returned access-denied or undecodable binary. Free copy exists at https://scholar.harvard.edu/files/lamont/files/guetzkow._lamont._mallard._2004.pdf (fetch blocked, 403) and https://hal.science/hal-00871416/ (access denied).

Flag: covers fellowship/grant panels (adjacent to the other agent's domain) but the *object being judged* is scholarly writing/proposals for originality, which is squarely a manuscript-judgment criterion — recommend keeping for the manuscript-judgment paper regardless.

---

### #5 — Mallard, Lamont & Guetzkow, "Fairness as Appropriateness: Negotiating Epistemological Differences in Peer Review" (*Science, Technology & Human Values*, 2009, 34(5):573–606)
**ACADEMIC STUDY (same 81-interview corpus as #4).** Shows that when panelists from different epistemological "styles" (constructivist, comprehensive, positivist, utilitarian) judge the same manuscript/proposal, they don't converge on a shared *definition* of quality — they converge on a *procedural* fairness norm ("cognitive contextualization": judge each work by the standards of its own style) instead. This is an articulation gap one level up: the panel can agree the process was fair without ever agreeing what "good" meant.

> **[SNIPPET]** Reviewers "define a fair decision-making process as one in which panelists engage in 'cognitive contextualization,' that is, use epistemological styles most appropriate to the field or discipline of the proposal under review" — synthesis from abstract/secondary description; direct text not fetched (paywalled at Sage; no open PDF located within budget).
> **[LEAD]**

---

### #6 — Michèle Lamont, *How Professors Think: Inside the Curious World of Academic Judgment* (Harvard UP, 2009)
**SCHOLARLY BOOK.** The strongest known vein per the brief, and clearly on-topic (81 interviews + observation across 5 national fellowship panels), but I was **unable to obtain a single verbatim quote** despite ~10 targeted fetch attempts (archive.org catalog page only has metadata, not full text; academia.edu PDF 403s; Google Books snippet view did not render through WebFetch; ResearchGate/HUP pages are all publisher blurbs).

> **[SNIPPET]** Publisher/reviewer paraphrase (Internet Archive catalog blurb, consistent across multiple secondary sources): "Judging quality isn't robotically rational; it's emotional, cognitive, and social, too." — https://archive.org/details/howprofessorsthi0000lamo
> **[SNIPPET]** "The exact meaning of these criteria is unclear and contested — not all panelists and officials agree on what terms like clarity, originality, significance, or quality mean" — recurring paraphrase across multiple book-review secondary sources (Goodreads/HUP copy family); not a verbatim book quote.
> **[LEAD]** Known content (from prior knowledge / secondary description, NOT verified by fetch) that panelists rely on "gut feelings," flair, and connoisseurship-like judgment that resists full verbalization, and that "customary rules" substitute for a shared explicit definition of excellence. **Do not quote this as verbatim text — I could not verify it.**

Recommend: someone with direct book access (library copy, Google Books logged-in preview, or a JSTOR/Project MUSE review essay behind institutional login) pull 2-3 verbatim interview quotes — panelists in the book do describe excellence in terms like "you know it," and the book is explicitly framed around unverbalized "customary rules," but I cannot certify exact wording from open web access.

---

### #7 — Sven E. Hug, "How do referees integrate evaluation criteria into their overall judgment? Evidence from grant peer review" (arXiv:2312.04569 / Scientometrics 2024)
**ACADEMIC STUDY (quantitative, fast-and-frugal-tree vs. logistic-regression modeling of real referee score sheets).** Tests exactly our question empirically: can a small number of named criteria, combined by a simple rule, reproduce the referee's overall judgment? Answer: only partially.

> **[VERIFIED]** Abstract (fetched in full): "…referees use many criteria and integrate the criteria using complex rules. However, and most importantly, the revised style could describe most — but not all — of the referees' judgments. Future studies should therefore examine how referees' judgments can be characterized in those cases where the uniform style failed."
> — https://arxiv.org/abs/2312.04569

Use: even the best-fitting formal model of "how criteria combine into a verdict" fails on a residual of cases — direct evidence that overall judgment is not fully reducible to stated criteria, from the reviewer's own scoring behavior (not self-report). Grant-context, but Hug is exactly the author flagged in the brief ("Hug & Aeschbach").

---

### #8 — Stefan Hirschauer, "Editorial Judgments: A Praxeology of 'Voting' in Peer Review" (*Social Studies of Science*, 2010, 40(1):71–103)
**ACADEMIC STUDY (ethnographic, 10 years participant observation + 1,800 reviews + taped editors' meetings, sociology journal).** Exactly the vein flagged in the brief. I could not get past the SSS paywall or find an open mirror within the tool/search budget, so no verbatim quote — but the paper's argument, per multiple independent secondary descriptions, is squarely on-thesis.

> **[SNIPPET]** Secondary characterization (via search synthesis, phrase reproduced across sources describing the paper — not independently verified against Hirschauer's own text): editorial judgments are analyzed as a "spontaneous impression" / "spontaneous expression of taste" rather than a reasoned application of criteria.
> **[SNIPPET]** The paper "shows a hidden interactivity in peer review, which is overlooked both by authors who impute social causes to unwelcome decisions, and by the preoccupation with 'reliability' prevalent in peer review research" (search-engine synthesis of abstract).
> **[LEAD]** for verbatim text — recommend institutional-login retrieval.

Companion piece, same author, same case study, easier topic-abstract access:

**Hirschauer, "How Editors Decide: Oral Communication in Journal Peer Review"** (*Human Studies*, 2015, 38(1):37–55). Corpus: 1,800 external reviews, 850 letters to authors, 4,000 written editorial votes, 24 hours of taped editors' meetings.
> **[SNIPPET]** "The operative nucleus of peer review processes has largely remained a 'black box' to analytical empirical research" — framing language reproduced across secondary sources describing the paper's motivation.
> **[LEAD]** for verbatim in-meeting talk quotes (the actual payload — editors' spoken reasoning) — paywalled, not retrieved.

---

## 3. Leads (exist, on-topic, not quote-verified — worth a follow-up pass with institutional access)

- **Travis & Collins, "New Light on Old Boys: Cognitive and Institutional Particularism in the Peer Review System"** (*Science, Technology & Human Values*, 1991, 16(3):322–341). Observation of British SERC grant committees; identifies "cognitive particularism" — reviewers favor work resembling their own cognitive/theoretical commitments while believing they are applying neutral criteria. Grant-adjacent but conceptually a good fit (stated criteria vs. actual mechanism).
- **Zuckerman & Merton, "Patterns of Evaluation in Science: Institutionalisation, Structure and Functions of the Referee System"** (*Minerva*, 1971, 9:66–100). Foundational sociology-of-science referee-system paper; establishes the referee system's structure but is less specifically about articulation failure than about institutional function — lower priority than #1-#8 but a citable anchor for "referee system" framing.
- **Chubin & Hackett, *Peerless Science: Peer Review and U.S. Science Policy*** (SUNY Press, 1990). Extracts named formal criteria (effectiveness, accountability, responsiveness, rationality, fairness, validity, reliability) for judging peer review *as a system*, not for judging papers — searches did not surface tacit-criteria content specifically; likely a weaker fit than initially expected from the brief.
- **Roumbanis, "Peer Review or Lottery? A Critical Analysis…"** (*Science, Technology, & Human Values*, 2019) and companion "Two Modes of Reasoning in Panel Review" — observed 10 Swedish Research Council panels; documents chance, intuition, power struggles, and emotion embedded in panel merging of individual scores into a group decision. Grant-panel domain (flagged in the brief as belonging partly to the grants agent) — report per instructions since it bears directly on "articulated criteria vs. actual mechanism," but treat as secondary to the manuscript-specific finds above.
- **Bloxham, Boyd & Orr, "Mark my words: the role of assessment criteria in UK higher education grading practices"** (*Studies in Higher Education*, 2011) and **Bloxham, den Outer, Hudson & Price, "Let's stop the pretence of consistent marking"** (*Assessment & Evaluation in Higher Education*, 2016). Not peer review of manuscripts, but same structural claim (explicit criteria used post hoc to justify a holistic/tacit judgment; think-aloud protocols with 12-24 experienced graders; Kelly's repertory grid showing wide variation in which implicit criteria different assessors even attend to). Strongly analogous mechanism; flagged as adjacent-domain evidence rather than a direct manuscript-peer-review finding.
  > **[SNIPPET]** "Markers often rely on holistic judgments informed by tacit knowledge and norm referencing, with explicit criteria primarily used post hoc to justify grades" (Bloxham, Boyd & Orr, synthesized from search of the paper — PDF at https://scispace.com/pdf/mark-my-words-the-role-of-assessment-criteria-in-uk-higher-3u6dzrswii.pdf did not render machine-readable text for verbatim extraction).
- **Kaltenbrunner, Birch & Amuchastegui, "Editorial Work and the Peer Review Economy of STS Journals"** (*Science, Technology, & Human Values*, 2022; open access PMC9260483). Fetched in full — frames peer review as a "gift economy" sustained by editorial curatorial labor; contains one usable line ("Trying to stitch together the fabric of our field, and not let each year be something completely different" — an editor describing curatorial judgment) but overall argues editors describe their judgment in fairly *articulable* terms (epistemic fit, field standards) — **weaker fit than expected**, borderline dead end for the specific articulation-gap thesis.
- **Teplitskiy, Acuna, Elamrani-Raoult, Kording & Evans, "The Social Structure of Consensus in Scientific Review"** (arXiv:1802.01270; ~8,000 PLOS ONE neuroscience manuscripts). About network-proximity bias in review scores, not directly about criteria-articulation failure — tangential.
- **Teplitskiy, Peng, Blasco & Lakhani, "Is Novel Research Worth Doing? Evidence from Peer Review at 49 Journals"** (PNAS, 2022). Shows novelty predicts acceptance even conditional on reviewer recommendation — interesting for a "criteria that move the needle in ways not fully captured by stated recommendations" angle, but not fetched in full (403); treat as lead only.

## 4. Dead ends

- Direct verbatim text of Lamont's *How Professors Think* — inaccessible via all open-web routes tried (Internet Archive metadata-only, academia.edu 403, Google Books snippet view not renderable by WebFetch, no HathiTrust/library-genesis attempted as out of scope).
- Hirschauer 2010 and 2015 full text — Sage/Springer paywalled, no open mirror found within the search/fetch budget.
- Guetzkow/Lamont/Mallard 2004 and Mallard/Lamont/Guetzkow 2009 full PDFs — HAL "access denied," Harvard scholar.harvard.edu 403, wordpress mirror returned an undecodable binary to the fetch tool.
- Chubin & Hackett *Peerless Science* — no tacit-criteria content surfaced despite multiple query angles; likely a weaker source for this specific thesis than the brief anticipated.
- Bornmann's peer-review-criteria body of work — search only surfaced generic multi-criteria descriptions (originality/rigor/impact), nothing on the specific articulation-gap mechanism; would need a more targeted title-by-title pass (e.g., Bornmann & Daniel 2008 "What do citation counts measure?" is citation-metrics, not criteria-articulation — didn't pursue).
- Langfeldt — search only returned generic panel-decision-process claims, no specific "cannot be reduced to criteria" quote located.
- Sandström (& Hällsten) — search surfaced only "productivism"/network-bias findings, nothing on undefined-criteria articulation.
- Web search budget was exhausted (200/200) partway through the Hug & Aeschbach follow-up and Bloxham-quote-verification passes — a next session with a fresh search budget could close several of the LEAD items above (especially: verbatim Lamont quotes, verbatim Hirschauer quotes, Hug & Aeschbach 2020 criteria paper).
