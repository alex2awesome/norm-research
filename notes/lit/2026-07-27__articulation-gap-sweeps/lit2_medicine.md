# Literature sweep: articulation gaps in medicine (clinical judgment, diagnostic intuition, assessment of competence)

Domain: MEDICINE. Task: upgrade 4 existing leads (Benner 1984, Montgomery 2006, Braude 2012, Klein 1998) to real quotes,
and mine the assessment-of-competence literature (entrustment/EPA, checklists vs. global ratings, OSATS, Hodges/Norman/Eva).

Tooling note: WebSearch quota was exhausted for the session almost immediately; all research below was done via WebFetch,
routed through (a) `https://r.jina.ai/<url>` as a scraping proxy for Google/Startpage/Bing pages, (b) the NCBI E-utilities
JSON/text API (`eutils.ncbi.nlm.nih.gov`) for PubMed/PMC — this was the most reliable channel by far and is where almost
every [VERIFIED] tag below comes from — and (c) direct PMC article fetches. DuckDuckGo and Google web search pages
captcha'd/rate-limited repeatedly; archive.org's scanned copies of the Benner and Klein books are access-restricted
(403 on the `_djvu.txt` and the search-inside endpoint) so those two could only be reached via Google Books' snippet view
(worked for Klein, not for Benner — see below).

---

## 1. Already have (per instructions — not re-verified here)

- `dawes1989clinical` — Dawes, Faust & Meehl, "Clinical versus actuarial judgment," *Science* 1989.
- `grove2000clinical` — Grove et al., "Clinical versus mechanical prediction: a meta-analysis," 2000.
- `benner1984novice`, `montgomery2006how`, `braude2012intuition`, `klein1998sources` — LEAD-only entries, addressed in §2.
- Also present in `refs-shared.bib`: `highhouse2008stubborn` (adjacent, hiring not medicine) and an Argyris & Schön entry — noted for context only, not medicine.

---

## 2. Upgraded leads

### 2a. Klein 1998, *Sources of Power* — now [VERIFIED] with page numbers

Archive.org's scan (`sourcesofpowerho0000klei`) is access-restricted (403 on both the direct `_djvu.txt` file and the
`fulltext/inside.php` search-inside endpoint — confirmed by `curl`). However, Google Books' snippet view for the
**20th-anniversary edition** (id `F201DwAAQBAJ`) is searchable and returned the exact passage with page numbers:

> **[VERIFIED]** (Google Books snippet view, `books.google.com/books?id=F201DwAAQBAJ`, search term `"I don't make decisions"`, p. 11): *"'I don't make decisions,' he announced to his startled listeners. 'I don't remember when I've ever made a decision.'"*

> **[VERIFIED]** (same source, p. 12, search term `"there were options"`): *"...there were options, yet it was usually obvious what to do in any given situation. We soon realized that he was defining the making of a decision in the same way as Soelberg's students — generating a set of options and evaluating them to..."*

> **[VERIFIED]** (same source, p. 24, search term `"recognition-primed"`): the RPD (recognition-primed decision) model is described as fusing "two processes: the way decision makers size up the situation to recognize which course of action makes sense, and the way they evaluate that" action.

Caveat: page numbers are from the 20th-anniversary MIT Press reprint (2017), not the 1998 first edition cited in
`klein1998sources`; the chapter-1 text is unchanged between editions but pagination may shift by a page or two. If exact
1998-edition pagination is required for the paper, flag for a physical-copy check; the quotes themselves are unaltered
Klein prose either way.

### 2b. Montgomery 2006, *How Doctors Think* — now [SNIPPET] via two independent book reviews (not the primary text itself — Google Books has no snippet view for this title, and it isn't on archive.org)

> **[SNIPPET]** (Jeffrey D. "Books: How Doctors Think... The Importance of Anecdote." *Br J Gen Pract* 2018;68(667):88. PMCID PMC5774955 — fetched full text): Montgomery "argues that clinical medicine is not a science but an interpretative practice," describing medicine as "a practice or, paradoxically, as a 'science of individuals'" that values clinical experience and takes account of context; "the anecdote, a patient's story, upon which the process of clinical medicine is based" is paradoxically ranked as the lowest form of evidence.

> **[SNIPPET]** (Barraclough K. "How Doctors Think..." *BMJ* 2006;332(7547):979. PMCID PMC1444827 — fetched full text): reviewer quotes Montgomery's central claim that "attempts to mimic the judgments of experienced clinicians algorithmically mostly fail," and via her, Feigenbaum's diagnosis of why: "At this point knowledge threatens to break down into ten thousand special cases." Also: "the clinician's generalised truths are more modest beasts, hedged in with uncertainty."

Both reviews independently confirm Montgomery's thesis (medicine as Aristotelian *phronesis*/practical reasoning,
irreducible to an "invariant, replicable science") and both explicitly cite the algorithmic-capture-fails claim, which is
exactly the "explicit criteria fail to capture what experts do" pattern the paper needs. I could not get inside the book
itself (no Google Books preview, not on archive.org) so these remain reviewer paraphrase-plus-quoted-fragments rather
than a page-cited primary quote — an actual physical/library-access pass on the book would upgrade this to [VERIFIED].

### 2c. Benner 1984, *From Novice to Expert* — still substantially a LEAD; adjacent successor quotes found instead

Archive.org has the exact item (`fromnovicetoexpe0000benn`) but it is access-restricted (403 on `_djvu.txt`, confirmed by
`curl`). Google Books' snippet view for this title (ids `LEVtAAAAMAAJ`, `tMsQAQAAMAAJ`) reports that pages matching
"intuitive grasp" exist but would not render the snippet text through any proxy tried (Google rate-limited/blocked the
session after ~10 requests). I could not obtain a direct 1984-book quote. **Do not fabricate the well-known
patient-deterioration vignette without a physical-copy check.**

What I did verify is a secondary paper that block-quotes Benner's *successor* volume (Benner, Tanner & Chesla,
*Expertise in Nursing Practice: Caring, Clinical Judgment, and Ethics*, 2nd ed., 2009 — same author, same research
program, not the 1984 book):

> **[SNIPPET]** (via Ozdemir, "Working Chance: Peirce's Semiotic Contrasted with Benner's Intuition," PMCID PMC11624903 — fetched full text, quotes carry page numbers from the secondary source): intuition defined as "understanding without rationale" (Benner & Tanner 1987, p. 23); expert intuitive judgment in *familiar* situations is contrasted with the "detached deliberation of an expert facing a *novel* situation in which he has no intuition and so… must resort to abstract principles" (Benner, Tanner & Chesla 2009, p. 320–1); "detached analytic reasoning is needed in cases of breakdown, where direct apprehension [intuition] does not occur" (ibid., p. 387); expert recognition is characterized as "recognizing the unexpected — that is, when tacit global expectations of patient's recovery are not met" (ibid., p. xvii); "knowing the patient" is called "a vital aspect of interpreting early warnings" (ibid., p. 347).

Recommend keeping `benner1984novice` as LEAD but adding a new `benner2009expertise` entry (BibTeX below) carrying these
verified-with-page-number quotes, since they are the direct intellectual continuation of the 1984 claim and are properly
sourced. A follow-up pass with real book access (library/interlibrary loan) is the only way to get the 1984 text itself.

### 2d. Braude 2012, *Intuition in Medicine* — largely still a LEAD, but now scoped and contextualized

Not on archive.org, no Google Books preview snippet available (confirmed via Startpage-site-search — zero
`books.google.com` hits for this title). Recovered:

- Table of contents (via UChicago Press catalog page, `press.uchicago.edu`, id `bo12839065`), which shows the book is
  directly on-topic for the paper: **Ch. 3** "The Place of Aristotelian *Phronesis* in Clinical Reasoning," **Ch. 6**
  "Clinical Intuition versus Statistical Reasoning," **Ch. 8** "Abduction: The Intuitive Support of Clinical Induction."
- **[VERIFIED]** (UChicago Press description page, same URL): Braude argues "ethical responsibility for the other lies at
  the heart of clinical judgment" — but this is an ethics claim, not directly an articulation-gap claim, so it is weak
  evidence for this paper's thesis even though it is a genuine quote from the publisher's page.
- A 2014 *American Journal of Bioethics* review exists — Schwab A. "The Limits of Intuition in Medicine: A Review of
  Hillel Braude's *Intuition in Medicine*." *Am J Bioethics* 2014;14(6):54–55 (DOI 10.1080/15265161.2014.903645) — but I
  could not retrieve its text (Taylor & Francis paywalled; author's own site listed ~13 unlabeled PDF links I did not
  have a reliable way to match to this specific piece in the time available).

Net: Braude remains the least-cashed-in of the four leads. Chapter 6 title ("Clinical Intuition versus Statistical
Reasoning") strongly suggests it directly engages Meehl/Dawes-style clinical-vs-actuarial material from inside medicine —
worth a dedicated library-access pass; likely rich, as originally suspected, but unconfirmed.

---

## 3. New finds, ranked (the assessment-of-competence vein)

This is the strongest material in the sweep — nearly all of it reached via NCBI PubMed/PMC (E-utilities), which was
reliable where general web search was not.

**1. [VERIFIED] Hodges B, Regehr G, McNaughton N, Tiberius R, Hanson M. "OSCE checklists do not capture increasing levels
of expertise." *Acad Med.* 1999;74(10):1129–34.** (PMID 10536636, PubMed abstract fetched directly, confirmed twice with
identical wording.) Fourteen clerks, 14 family-practice residents, 14 experienced physicians each did two 15-minute
standardized-patient interviews, scored both by binary checklist and by global process rating. The finding is the exact
inversion the brief asked for: *"on global scales, the experienced clinicians scored significantly better than did the
residents and clerks, but on checklists, the experienced clinicians scored significantly worse than did the residents
and clerks."* Conclusion: *"binary checklists may not be valid measures of increasing clinical competence."* This is the
single best empirical anchor for "the itemized explicit instrument is worse than the unexplained holistic one, measured."

**2. [VERIFIED] Regehr G, MacRae H, Reznick RK, Szalay D. "Comparing the psychometric properties of checklists and
global rating scales for assessing performance on an OSCE-format examination." *Acad Med.* 1998;73(9):993–7.** (PMID
9759104, PubMed abstract fetched directly, confirmed twice with identical wording.) 53 surgery residents, 8-station
technical-skills exam. *"Global rating scales scored by experts showed higher inter-station reliability, better
construct validity, and better concurrent validity than did checklists."* Adding checklist scores did not improve the
global scale's psychometrics. This is the companion/founding paper to #1 and the direct empirical basis for the
"checklists vs. global ratings" literature the brief named.

**3. [VERIFIED] ten Cate O. "Trust, competence, and the supervisor's role in postgraduate training." *BMJ.*
2006;333(7571):748–51.** (PMID 17023469; full text at PMC1592396, fetched directly.) This is the best single quote in
the whole sweep for the paper's thesis. Abstract: *"The decision to trust a trainee to manage a critically ill patient is
based on much more than tests of competence."* Body text: *"Supervisors often know who to pick, even if they can't tell
exactly why. This gut feeling does not always match with formally assessed knowledge or skill, but it may be more valid
for its purpose."* And: *"No external body or procedure can replace this type of expert judgment."* And on why
atomization fails: *"Attempting to assess them separately may result in a trivialised set of attained
abilities...To further develop educational technology and sophistication of assessment methods does not seem the right
direction."* This is ten Cate's founding argument for entrustment decisions over itemized competency assessment,
explicitly on the grounds that the holistic gut judgment outperforms the formal one.

**4. [VERIFIED] ten Cate O. "Nuts and bolts of entrustable professional activities." *J Grad Med Educ.* 2013;5(1):157–8.**
(PMID 24404246; full text at PMC3613304, fetched directly.) Names the core entrustment question directly: *"The key
question is: Can we trust this trainee to execute this EPA?"* — this is the literal "would you let this trainee do it
unsupervised?" reframing the brief asked for, in ten Cate's own words. Also: EPAs were adopted because standard
"competency frameworks would otherwise be too theoretical to be useful for training and assessment in daily practice."

**5. [VERIFIED] Martin JA, Regehr G, Reznick R, MacRae H, Murnaghan J, Hutchison C, Brown M. "Objective structured
assessment of technical skill (OSATS) for surgical residents." *Br J Surg.* 1997;84(2):273–8.** (PMID 9052454, PubMed
abstract fetched directly.) Founding OSATS paper. 20 residents, bench-model and live-animal formats, three scoring
methods. *"Global ratings discriminated between resident levels"* more effectively than checklists — the surgical-skill
version of finding #2, from the instrument's own inventors.

**6. [VERIFIED] Reznick R, Regehr G, MacRae H, Martin J, McCulloch W. "Testing technical skill via an innovative 'bench
station' examination." *Am J Surg.* 1997;173(3):226–30.** (PMID 9124632, PubMed abstract fetched directly.) 48 residents,
8 stations, task-specific checklists vs. global ratings scored side by side; *"high reliability and construct
validity"* with global ratings showing the strongest discrimination by training level. Companion validation paper to #5.

**7. [VERIFIED] Faulkner H, Regehr G, Martin J, Reznick R. "Validation of an objective structured assessment of
technical skill for surgical residents." *Acad Med.* 1996;71(12):1363–5.** (PMID 9114900, PubMed abstract fetched
directly.) Earliest of the three OSATS papers; OSATS rankings agreed well with faculty's own (unstructured, holistic)
rankings of senior residents specifically — i.e., the instrument was validated *against* faculty gestalt, not the
reverse.

**8. [VERIFIED] Govaerts M, van der Vleuten CP. "Validity in work-based assessment: expanding our horizons." *Med
Educ.* 2013;47(12):1164–74.** (PMID 24206150, PubMed abstract fetched directly.) Argues traditional psychometric validity
models are the wrong frame for workplace assessment because "learning, competence (as inferred from performance) as well
as performance interpretations are to be seen as inherently contextualised, and can only be understood 'in situ.'"
Proposes replacing psychometric models with interpretivist/socio-cultural ones — i.e., an explicit argument that the
formal validity apparatus cannot capture what assessors are actually doing.

**9. [VERIFIED] Hodges B. "Assessment in the post-psychometric era: learning to love the subjective and collective."
*Med Teach.* 2013;35(7):564–8.** (PMID 23631408, PubMed abstract fetched directly.) States assessment has been
"dominated by a discourse of psychometrics" since the 1970s and argues the field needs to move toward embracing
"subjective judgment and collective evaluation methods" as clinical practice becomes more team-based and workplace-based
evaluation outgrows the high-stakes exam model. Good general-argument citation for "the field is moving away from
itemized/psychometric toward holistic."

**10. [VERIFIED] Hodges B, McNaughton N, Regehr G, Tiberius R, Hanson M. "The challenge of creating new OSCE measures to
capture the characteristics of expertise." *Med Educ.* 2002;36(8):742–8.** (PMID 12191057, PubMed abstract fetched
directly.) Follow-up to #1: coded every utterance in the videotaped interviews looking for a checklist-style signature of
expertise; found only subtle differences, concluding that new measures "sensitive to the nature of expertise" are needed
because simple item-counting checklists cannot supply them.

**11. [VERIFIED] Moulton CA, Regehr G, Lingard L, Merritt C, MacRae H. "'Slowing down when you should': initiators and
influences of the transition from the routine to the effortful." *J Gastrointest Surg.* 2010;14(6):1019–26.** (PMID
20309647, PubMed abstract fetched directly.) 28 surgeons interviewed about intraoperative judgment; surgeons decelerate
from automatic to effortful processing at moments they can name post hoc but do not consciously monitor in the moment.
The paper's stated purpose is to give surgeons "language" for a judgment that otherwise stays tacit — a good example of
research literally trying to reverse-engineer an inarticulate expert skill.

**12. [SNIPPET — secondary citation within study] Duijn CCMA, Welink LS, Bok HGJ, ten Cate OTJ. "When to trust our
learners? Clinical teachers' perceptions of decision variables in the entrustment process." *Perspect Med Educ.*
2018;7(3):192–9.** (PMCID PMC6002285, fetched directly.) Opens by noting that "assessors frequently compare learner
performance with what they would do (self as standard)... some rely on a gut feeling" (citing prior literature) before
proposing 21 decision variables to structure entrustment decisions. Useful as evidence the field *knows* the judgment is
currently gut-based and is trying (with limited success, per #3's argument against over-formalizing) to formalize it —
i.e., documents the tension directly, though the paper's own agenda pushes toward structuring rather than confirming the
gap.

**13. [VERIFIED, background/theory not gap-evidence] Eva KW. "What every teacher needs to know about clinical
reasoning." *Med Educ.* 2005;39(1):98–106.** (PMID 15612906, PubMed abstract fetched directly.) Reviews "analytic (i.e.
conscious/controlled) versus non-analytic (i.e. unconscious/automatic) reasoning strategies" — standard framing citation
for the analytic/non-analytic dual-process split in clinical reasoning that underlies why expert judgment resists
verbal reconstruction. Background citation, not itself a gap-finding.

**14. [VERIFIED, background/theory not gap-evidence] Norman G. "Research in clinical reasoning: past history and
current trends." *Med Educ.* 2005;39(4):418–27.** (PMID 15813765, PubMed abstract fetched directly.) Reviews 30 years of
clinical-reasoning research; concludes expertise involves "multiple coordinated knowledge representations" (schemas,
exemplars) rather than one general reasoning skill, and that deliberate practice with varied exemplars — not explicit
rule-learning — drives expertise. Good citation for "pattern recognition," weaker as direct gap-evidence than #1–#11.

---

## 4. Leads (not chased to a quote, worth a follow-up pass)

- Braude 2012, Ch. 6 "Clinical Intuition versus Statistical Reasoning" — likely directly engages Meehl/Dawes from
  inside medicine; needs physical/library access.
- Schwab AL, "The Limits of Intuition in Medicine" (review of Braude), *Am J Bioethics* 2014;14(6):54–55 — paywalled;
  author posted ~13 unlabeled manuscript PDFs on `abeschwab.com` that I could not cheaply disambiguate.
- Klein, Calderwood & Clinton-Cirocco (1986), "Rapid decision making on the fireground" (Proc. Human Factors Society) —
  the original conference paper behind the Klein 1998 book anecdote; not attempted this pass (book quote in §2a already
  supplies verified primary text with page numbers, so this is now lower priority).
- Groopman J. *How Doctors Think* (2007) — confirmed via archive.org search to be a **different book** from Montgomery's
  (same title, different subtitle/author/thesis: Groopman is journalistic/narrative on diagnostic error, Montgomery is
  philosophical/phronesis). Not pursued for quotes since the brief's target is Montgomery, but flagging clearly so no
  future pass conflates the two — this is an easy citation error to make.
- Hodges B, "Was Hippocrates a scientist?" or similar Hodges theoretical pieces on OSCE-as-social-construction — I could
  not find a PubMed-indexed piece with exactly that framing in the time available; the closest hit was Hodges 2013
  (#9 above), which is thematically adjacent but not the "socially constructed instrument" argument by name.
- Benner 1984 direct text — see §2c; archive.org restricted, Google Books non-rendering. A physical-copy or
  interlibrary-loan pass would likely surface the canonical patient-deterioration vignette quickly (Google Books
  confirmed 2 pages match "intuitive grasp" but would not render the snippet).

---

## 5. Dead ends

- WebSearch tool: quota exhausted essentially immediately (before this task's own queries could run) — had to route
  everything through WebFetch proxies instead.
- DuckDuckGo (`html.duckduckgo.com`, `lite.duckduckgo.com`): worked for the first ~4 queries, then captcha'd
  ("select all squares containing a duck") for the rest of the session.
- Plain `google.com/search` and `books.google.com` direct fetch: worked briefly, then 429/CAPTCHA'd after ~10 requests;
  routing through `r.jina.ai` bought a few more before that too got blocked.
- `books.googleapis.com` (official Books API): quota is 0/day for unauthenticated callers — dead on arrival without an
  API key.
- Bing (`bing.com/search`): returned generic/location-based junk results (Redmond, WA hotels; dictionary definitions)
  unrelated to the query — the query terms did not appear to reach Bing's ranking properly through this fetch path.
- Mojeek, PhilPapers: 403 on every attempt.
- archive.org: both `fromnovicetoexpe0000benn` (Benner) and `sourcesofpowerho0000klei` (Klein) exist as scanned items but
  are in the access-restricted/printdisabled collection — `_djvu.txt` and the `fulltext/inside.php` search-inside
  endpoint both 403. Montgomery's and Braude's books are not on archive.org at all (searched by title/creator, zero
  hits — the archive.org "How Doctors Think" hits are all Groopman, not Montgomery).
- HAL (`hal.science`) PDF for a Philosophy-of-Medicine chapter citing Braude: served an Anubis anti-bot challenge page
  instead of the PDF, both direct and via `r.jina.ai`.
- Dove Press "Intuitive Medicine: A New Vision for Medical Education" article: 403/404 on every fetch route tried.

---

## 6. Ready-to-paste BibTeX

```bibtex
% ---------- upgraded: existing LEAD entries, replace in place ----------

@book{klein1998sources,
  author    = {Klein, Gary},
  title     = {Sources of Power: How People Make Decisions},
  publisher = {MIT Press},
  year      = {1998},
  keywords  = {domain=medicine-adjacent-ndm; gap=felt-not-stated; type=interview-study},
  annote    = {VERIFIED (Google Books snippet view, 20th-anniversary MIT Press edition, id F201DwAAQBAJ; archive.org
               scan sourcesofpowerho0000klei is access-restricted, both _djvu.txt and fulltext/inside.php 403). p. 11:
               "'I don't make decisions,' he announced to his startled listeners. 'I don't remember when I've ever made
               a decision.'" p. 12: "...there were options, yet it was usually obvious what to do in any given
               situation. We soon realized that he was defining the making of a decision in the same way as Soelberg's
               students -- generating a set of options and evaluating them..." p. 24 defines the recognition-primed
               decision (RPD) model as fusing "the way decision makers size up the situation to recognize which course
               of action makes sense, and the way they evaluate that" action. Fireground commanders explicitly deny
               that their high-stakes judgments are "decisions" in the deliberative sense the interview protocol
               presupposed -- felt expertise the actor cannot recognize as a process, let alone articulate. Page
               numbers are from the 2017 20th-anniversary reprint; chapter-1 text is unchanged from 1998 but exact
               1998-edition pagination unverified.}
}

@book{montgomery2006how,
  author    = {Montgomery, Kathryn},
  title     = {How Doctors Think: Clinical Judgment and the Practice of Medicine},
  publisher = {Oxford University Press},
  year      = {2006},
  keywords  = {domain=medicine; gap=felt-not-stated; type=theory},
  annote    = {SNIPPET (via two independent book reviews; book itself has no Google Books preview and is not on
               archive.org). Jeffrey D., Br J Gen Pract 2018;68(667):88 (PMCID PMC5774955): Montgomery "argues that
               clinical medicine is not a science but an interpretative practice," describing it as a "'science of
               individuals'" grounded in "the anecdote, a patient's story" despite anecdote ranking lowest in
               evidence hierarchies. Barraclough K., BMJ 2006;332(7547):979 (PMCID PMC1444827): quotes Montgomery's
               claim that "attempts to mimic the judgments of experienced clinicians algorithmically mostly fail,"
               and via her, Feigenbaum's diagnosis that expert-system formalization of medical knowledge "threatens
               to break down into ten thousand special cases." Thesis is medicine as Aristotelian phronesis, resistant
               to algorithmic/rule-based capture. Not yet verified against the primary text directly -- a
               library-access pass would upgrade this to VERIFIED with page numbers.}
}

@book{benner1984novice,
  author    = {Benner, Patricia},
  title     = {From Novice to Expert: Excellence and Power in Clinical Nursing Practice},
  publisher = {Addison-Wesley},
  year      = {1984},
  keywords  = {domain=medicine; gap=felt-not-stated; type=interview-study},
  annote    = {LEAD (primary text still not retrieved -- archive.org scan fromnovicetoexpe0000benn is
               access-restricted, 403 on _djvu.txt; Google Books confirms 2 pages match "intuitive grasp" but would
               not render snippet text through any proxy before rate-limiting). Do not invent the
               patient-deterioration vignette without a physical-copy check. See benner2009expertise for verified
               page-cited quotes from the same author's direct successor volume.}
}

@book{braude2012intuition,
  author    = {Braude, Hillel D.},
  title     = {Intuition in Medicine: A Philosophical Defense of Clinical Reasoning},
  publisher = {University of Chicago Press},
  year      = {2012},
  keywords  = {domain=medicine; gap=felt-not-stated; type=philosophy},
  annote    = {LEAD, now scoped (not on archive.org, no Google Books preview). Table of contents (UChicago Press,
               bo12839065) confirms direct relevance: Ch. 3 "The Place of Aristotelian Phronesis in Clinical
               Reasoning," Ch. 6 "Clinical Intuition versus Statistical Reasoning," Ch. 8 "Abduction: The Intuitive
               Support of Clinical Induction" -- Ch. 6 likely engages Meehl/Dawes (dawes1989clinical/grove2000clinical)
               directly from inside medicine. One VERIFIED but off-thesis quote from the publisher description page:
               "ethical responsibility for the other lies at the heart of clinical judgment" (an ethics claim, not
               an articulation-gap claim). A 2014 Am J Bioethics review exists (Schwab, DOI
               10.1080/15265161.2014.903645) but was paywalled. Needs a library-access pass; still the least-cashed-in
               of the four original leads.}
}

% ---------- new: adjacent successor to Benner 1984 ----------

@book{benner2009expertise,
  author    = {Benner, Patricia and Tanner, Christine A. and Chesla, Catherine A.},
  title     = {Expertise in Nursing Practice: Caring, Clinical Judgment, and Ethics},
  edition   = {2nd},
  publisher = {Springer},
  year      = {2009},
  keywords  = {domain=medicine; gap=felt-not-stated; type=interview-study},
  annote    = {SNIPPET (via Ozdemir, "Working Chance: Peirce's Semiotic Contrasted with Benner's Intuition," PMCID
               PMC11624903, which block-quotes this book with page numbers). Intuition defined as "understanding
               without rationale" (Benner \& Tanner 1987, p. 23). Expert intuitive judgment in familiar situations is
               contrasted with the "detached deliberation of an expert facing a novel situation in which he has no
               intuition and so... must resort to abstract principles" (p. 320-1). "Detached analytic reasoning is
               needed in cases of breakdown, where direct apprehension [intuition] does not occur" (p. 387). Expert
               recognition is "recognizing the unexpected -- that is, when tacit global expectations of patient's
               recovery are not met" (p. xvii); "knowing the patient" is "a vital aspect of interpreting early
               warnings" (p. 347). Same author/research program as benner1984novice (2nd ed. of the 1984 book's
               successor line), not a substitute for the 1984 primary text itself but a directly verified adjacent
               source with the same claim.}
}

% ---------- new: assessment-of-competence literature ----------

@article{hodges1999osce,
  author  = {Hodges, Brian and Regehr, Glenn and McNaughton, Nancy and Tiberius, Richard and Hanson, Mark},
  title   = {OSCE checklists do not capture increasing levels of expertise},
  journal = {Academic Medicine},
  volume  = {74},
  number  = {10},
  pages   = {1129--1134},
  year    = {1999},
  doi     = {10.1097/00001888-199910000-00017},
  keywords = {domain=medicine; gap=explicit-instrument-worse-than-holistic; type=empirical},
  annote  = {VERIFIED (PubMed abstract, PMID 10536636, fetched directly, confirmed twice with identical wording).
             14 clerks, 14 family-practice residents, 14 experienced physicians each did two 15-minute
             standardized-patient interviews, scored by both binary checklist and global process rating. The
             beautiful inversion: "on global scales, the experienced clinicians scored significantly better than
             did the residents and clerks, but on checklists, the experienced clinicians scored significantly worse
             than did the residents and clerks." Conclusion: "binary checklists may not be valid measures of
             increasing clinical competence." The single best empirical anchor in this sweep for "the itemized
             explicit instrument is measurably worse than the unexplained holistic one."}
}

@article{regehr1998comparing,
  author  = {Regehr, Glenn and MacRae, Helen and Reznick, Richard K. and Szalay, David},
  title   = {Comparing the psychometric properties of checklists and global rating scales for assessing performance on an OSCE-format examination},
  journal = {Academic Medicine},
  volume  = {73},
  number  = {9},
  pages   = {993--997},
  year    = {1998},
  doi     = {10.1097/00001888-199809000-00020},
  keywords = {domain=medicine; gap=explicit-instrument-worse-than-holistic; type=empirical},
  annote  = {VERIFIED (PubMed abstract, PMID 9759104, fetched directly, confirmed twice with identical wording).
             53 surgery residents, 8-station OSCE-format technical-skills exam. "Global rating scales scored by
             experts showed higher inter-station reliability, better construct validity, and better concurrent
             validity than did checklists." Adding checklist scores did not improve the global scale's psychometrics.
             Founding paper of the checklist-vs-global-rating literature; companion to hodges1999osce.}
}

@article{tencate2006trust,
  author  = {ten Cate, Olle},
  title   = {Trust, competence, and the supervisor's role in postgraduate training},
  journal = {BMJ},
  volume  = {333},
  number  = {7571},
  pages   = {748--751},
  year    = {2006},
  doi     = {10.1136/bmj.38938.407569.94},
  keywords = {domain=medicine; gap=stated-ne-used; type=theory},
  annote  = {VERIFIED (full text, PMCID PMC1592396 -- fetched directly; abstract also confirmed via PubMed, PMID
             17023469). Best single quote in the sweep. Abstract: "The decision to trust a trainee to manage a
             critically ill patient is based on much more than tests of competence." Body: "Supervisors often know
             who to pick, even if they can't tell exactly why. This gut feeling does not always match with formally
             assessed knowledge or skill, but it may be more valid for its purpose." And: "No external body or
             procedure can replace this type of expert judgment." On why atomized checklists fail: "Attempting to
             assess them separately may result in a trivialised set of attained abilities... To further develop
             educational technology and sophistication of assessment methods does not seem the right direction."
             ten Cate's founding argument for entrustment over itemized competency assessment, explicitly because
             the holistic gut judgment outperforms the formal one.}
}

@article{tencate2013nuts,
  author  = {ten Cate, Olle},
  title   = {Nuts and bolts of entrustable professional activities},
  journal = {Journal of Graduate Medical Education},
  volume  = {5},
  number  = {1},
  pages   = {157--158},
  year    = {2013},
  doi     = {10.4300/JGME-D-12-00380.1},
  keywords = {domain=medicine; gap=stated-ne-used; type=theory},
  annote  = {VERIFIED (full text, PMCID PMC3613304 -- fetched directly; abstract also confirmed via PubMed, PMID
             24404246). Names the core entrustment question directly: "The key question is: Can we trust this
             trainee to execute this EPA?" -- the literal "would you let this trainee do it unsupervised?" reframing
             in ten Cate's own words. Also: EPAs were adopted because standard "competency frameworks would
             otherwise be too theoretical to be useful for training and assessment in daily practice."}
}

@article{martin1997osats,
  author  = {Martin, J. A. and Regehr, G. and Reznick, R. and MacRae, H. and Murnaghan, J. and Hutchison, C. and Brown, M.},
  title   = {Objective structured assessment of technical skill (OSATS) for surgical residents},
  journal = {British Journal of Surgery},
  volume  = {84},
  number  = {2},
  pages   = {273--278},
  year    = {1997},
  doi     = {10.1046/j.1365-2168.1997.02502.x},
  keywords = {domain=medicine; gap=explicit-instrument-worse-than-holistic; type=empirical},
  annote  = {VERIFIED (PubMed abstract, PMID 9052454, fetched directly). Founding OSATS paper. 20 residents,
             bench-model and live-animal formats, three scoring methods compared. "Global ratings discriminated
             between resident levels" more effectively than checklists -- the surgical-skill analogue of
             hodges1999osce/regehr1998comparing, from the instrument's own inventors.}
}

@article{reznick1997bench,
  author  = {Reznick, R. and Regehr, G. and MacRae, H. and Martin, J. and McCulloch, W.},
  title   = {Testing technical skill via an innovative "bench station" examination},
  journal = {American Journal of Surgery},
  volume  = {173},
  number  = {3},
  pages   = {226--230},
  year    = {1997},
  doi     = {10.1016/s0002-9610(97)89597-9},
  keywords = {domain=medicine; gap=explicit-instrument-worse-than-holistic; type=empirical},
  annote  = {VERIFIED (PubMed abstract, PMID 9124632, fetched directly). 48 residents, 8 stations, task-specific
             checklists vs. global ratings scored side by side; "high reliability and construct validity" with
             global ratings showing the strongest discrimination by training level. Companion validation paper to
             martin1997osats.}
}

@article{faulkner1996validation,
  author  = {Faulkner, H. and Regehr, G. and Martin, J. and Reznick, R.},
  title   = {Validation of an objective structured assessment of technical skill for surgical residents},
  journal = {Academic Medicine},
  volume  = {71},
  number  = {12},
  pages   = {1363--1365},
  year    = {1996},
  doi     = {10.1097/00001888-199612000-00023},
  keywords = {domain=medicine; gap=validated-against-holistic; type=empirical},
  annote  = {VERIFIED (PubMed abstract, PMID 9114900, fetched directly). Earliest of the three OSATS papers; OSATS
             rankings agreed well with faculty's own unstructured, holistic rankings for senior residents
             specifically -- the new instrument was validated AGAINST faculty gestalt, not the reverse, and agreement
             was weaker for junior trainees.}
}

@article{moulton2010slowing,
  author  = {Moulton, Carol-Anne and Regehr, Glenn and Lingard, Lorelei and Merritt, Christopher and MacRae, Helen},
  title   = {"Slowing down when you should": initiators and influences of the transition from the routine to the effortful},
  journal = {Journal of Gastrointestinal Surgery},
  volume  = {14},
  number  = {6},
  pages   = {1019--1026},
  year    = {2010},
  doi     = {10.1007/s11605-010-1178-y},
  keywords = {domain=medicine; gap=felt-not-stated; type=interview-study},
  annote  = {VERIFIED (PubMed abstract, PMID 20309647, fetched directly). 28 surgeons interviewed about
             intraoperative judgment; surgeons decelerate from automatic to effortful processing at moments they can
             name post hoc but do not consciously monitor as they happen. Explicit stated purpose: give surgeons
             "language" for a judgment that otherwise stays tacit -- a study literally trying to reverse-engineer an
             inarticulate expert skill for teaching purposes.}
}

@article{duijn2018trust,
  author  = {Duijn, Charlotte C. M. A. and Welink, Lisanne S. and Bok, Harold G. J. and ten Cate, Olle Th. J.},
  title   = {When to trust our learners? Clinical teachers' perceptions of decision variables in the entrustment process},
  journal = {Perspectives on Medical Education},
  volume  = {7},
  number  = {3},
  pages   = {192--199},
  year    = {2018},
  doi     = {10.1007/s40037-018-0430-0},
  keywords = {domain=medicine; gap=stated-ne-used; type=interview-study},
  annote  = {SNIPPET (full text, PMCID PMC6002285, fetched directly). Opens by noting assessors "frequently compare
             learner performance with what they would do (self as standard)... some rely on a gut feeling" before
             proposing 21 decision variables to structure entrustment decisions. Documents the field's own awareness
             that entrustment judgment is currently gut-based, while itself pushing toward formalizing it -- read
             together with tencate2006trust (which argues AGAINST over-formalizing), this pair captures the live
             tension in the field.}
}

@article{govaerts2013validity,
  author  = {Govaerts, Marjan and van der Vleuten, Cees P. M.},
  title   = {Validity in work-based assessment: expanding our horizons},
  journal = {Medical Education},
  volume  = {47},
  number  = {12},
  pages   = {1164--1174},
  year    = {2013},
  doi     = {10.1111/medu.12289},
  keywords = {domain=medicine; gap=stated-ne-used; type=theory},
  annote  = {VERIFIED (PubMed abstract, PMID 24206150, fetched directly). Argues traditional psychometric validity
             models are the wrong frame for workplace assessment: "learning, competence (as inferred from
             performance) as well as performance interpretations are to be seen as inherently contextualised, and
             can only be understood 'in situ.'" Proposes interpretivist/socio-cultural models in place of
             psychometric ones -- an explicit argument that the formal validity apparatus cannot capture what
             assessors are actually doing.}
}

@article{hodges2013postpsychometric,
  author  = {Hodges, Brian},
  title   = {Assessment in the post-psychometric era: learning to love the subjective and collective},
  journal = {Medical Teacher},
  volume  = {35},
  number  = {7},
  pages   = {564--568},
  year    = {2013},
  doi     = {10.3109/0142159X.2013.789134},
  keywords = {domain=medicine; gap=stated-ne-used; type=theory},
  annote  = {VERIFIED (PubMed abstract, PMID 23631408, fetched directly). Assessment has been "dominated by a
             discourse of psychometrics" since the 1970s; argues the field must move toward embracing "subjective
             judgment and collective evaluation methods" as clinical practice becomes team-based and workplace
             evaluation outgrows the high-stakes exam model. General-argument citation for the field's own
             move away from itemized/psychometric toward holistic/collective assessment.}
}

@article{hodges2002challenge,
  author  = {Hodges, Brian and McNaughton, Nancy and Regehr, Glenn and Tiberius, Richard and Hanson, Mark},
  title   = {The challenge of creating new OSCE measures to capture the characteristics of expertise},
  journal = {Medical Education},
  volume  = {36},
  number  = {8},
  pages   = {742--748},
  year    = {2002},
  doi     = {10.1046/j.1365-2923.2002.01270.x},
  keywords = {domain=medicine; gap=explicit-instrument-worse-than-holistic; type=empirical},
  annote  = {VERIFIED (PubMed abstract, PMID 12191057, fetched directly). Follow-up to hodges1999osce: coded every
             utterance in the same videotaped interviews looking for a checklist-style behavioral signature of
             expertise; found only subtle differences between experience levels, concluding new measures "sensitive
             to the nature of expertise" are needed because item-counting checklists structurally cannot supply
             them.}
}

@article{eva2005teacher,
  author  = {Eva, Kevin W.},
  title   = {What every teacher needs to know about clinical reasoning},
  journal = {Medical Education},
  volume  = {39},
  number  = {1},
  pages   = {98--106},
  year    = {2005},
  doi     = {10.1111/j.1365-2929.2004.01972.x},
  keywords = {domain=medicine; gap=background-theory; type=review},
  annote  = {VERIFIED (PubMed abstract, PMID 15612906, fetched directly). Reviews "analytic (i.e.
             conscious/controlled) versus non-analytic (i.e. unconscious/automatic) reasoning strategies" -- the
             standard dual-process framing for why expert clinical judgment resists verbal reconstruction.
             Background/theory citation, not itself gap-evidence.}
}

@article{norman2005research,
  author  = {Norman, Geoffrey},
  title   = {Research in clinical reasoning: past history and current trends},
  journal = {Medical Education},
  volume  = {39},
  number  = {4},
  pages   = {418--427},
  year    = {2005},
  doi     = {10.1111/j.1365-2929.2005.02127.x},
  keywords = {domain=medicine; gap=background-theory; type=review},
  annote  = {VERIFIED (PubMed abstract, PMID 15813765, fetched directly). Thirty-year review of clinical-reasoning
             research; concludes expertise involves "multiple coordinated knowledge representations" (schemas,
             exemplars) rather than one general reasoning skill, and deliberate practice with varied exemplars --
             not explicit rule-learning -- drives expertise. Background citation for pattern-recognition-based
             expertise; weaker as direct gap-evidence than the checklist/global-rating empirical papers above.}
}
```
