# Articulation-gap quote bank (2026-07-27)

Sweep of the expert-preference literature for **articulation gaps**: experts converge on
quality judgments but cannot state the criteria, or explicit criteria demonstrably fail to
capture what experts do.

- **Bibliography:** `latex/refs-shared.bib` (106 entries, new to the repo; cite with
  `\bibliography{../refs-shared}` from any `latex/paper-*/main.tex`).
- **Raw per-domain sweeps** (dead ends, access notes, unpursued leads):
  `notes/lit/2026-07-27__articulation-gap-sweeps/`.
- **Method:** ten parallel domain agents, each grepping the three pre-existing bibs before
  searching. Every quote carries a verification tag.

**Verification tags.** `VERIFIED` = the string was confirmed in text actually fetched or
OCR'd. `SNIPPET` = exact string from a search snippet or a named secondary source quoting
the primary. `LEAD` = real and on-topic, no quote — never quote these.

The user's standard, restated: Hall (1973) is the target *shape* of quote but is not an
academic study; Galtung & Ruge (1965) is the anti-pattern (derives explicit factors, no
usable quote). Prefer empirical studies; flag trade testimony as trade testimony.

---

## The five best quotes in the sweep

| # | Quote | Source | Tag |
|---|---|---|---|
| 1 | "human beings can be understood to act according to rules that they cannot state" | Argyris, Putnam & Smith, *Action Science* (1985), p. 82 | VERIFIED |
| 2 | Originality .08, Soundness/Correctness .01 — correlation of reviewers' own criterion scores with their own overall recommendation (vs. Substance .59) | Kang et al., PeerRead, NAACL 2018, Table 2 | VERIFIED |
| 3 | "the rules which must guide this choice are extremely fine and delicate. It is almost impossible to state them precisely; **they are felt rather than formulated**" | Poincaré, "Mathematical Creation" (1908/1910) | VERIFIED |
| 4 | "the similarities in question are not, and cannot be, expressed in terms of criteria, no more than the similarities of many other kinds of objects, such as **human faces, tunes, or tastes of wine**, can be thus expressed" | Naur, "Programming as Theory Building" (1985) | VERIFIED |
| 5 | "the judge really decides by feeling and not by judgment, by hunching and not by ratiocination, such ratiocination appearing only in the opinion" | Hutcheson, 14 Cornell L. Rev. 274 (1929) | SNIPPET |

#2 is the one to build a figure around — it is a *measured* dissociation between stated and
operative criteria, in behaviour rather than self-report, in a domain we already have data for.

---

## By domain

### Journalism
Your Hall quote now has academic company. **`gans1979deciding`** is the key acquisition:
news judgment is "partly a matter of feel" (p. 171); journalists "act on the basis of quick,
virtually intuitive judgments, which some ascribe to 'feel'" (p. 82) [SNIPPET via De Maeyer
2020, whose full text was fetched]. Gans is normally read as *the* explicit-considerations
taxonomy, which makes the concession load-bearing.

**`epstein1973news`** [VERIFIED, full OCR] is the strongest empirical case: NBC's selection
system was "unwritten but generally known" (pp. 168–169); correspondents were blacklisted on
"offhanded descriptions" — "too old," "lack of sex appeal," "not punchy enough" (p. 189); and
the CBS News president "would refuse to answer any questions about news judgments" (p. 10).
Criteria that exist, produce consistent outcomes, are explicitly unwritten, and are actively
shielded from articulation.

Also: `cotter2010newstalk` [VERIFIED] on apprenticeship transmission and "nose for news" as
"somewhat magical"; `ellingsberg2026cultural` [SNIPPET] — editorial hiring "contingent on
tacit judgments that cannot easily be articulated or externally validated."

**Already held, under-mined:** `schultz2007journalistic` quotes journalists calling the gut
feeling "part of your spinal cord," "in the back of your head," "something like a feeling"
(p. 199). This is the closest academic analogue to Hall and we already own it.

**Highest-value unresolved:** `breed1955social` (the canonical osmosis-not-instruction study,
JSTOR-walled) and `clayman1998gatekeeping` (conversation analysis of real editorial
conferences — structurally the ideal design).

### Mathematics
`poincare1910mathematical` is quote #3 above. `thurston1994proof` [VERIFIED] gives the social
-standard argument: "it is preposterous to claim that mathematics as we practice it is
anywhere near formally correct," and "One-on-one, people use wide channels of communication
that go far beyond formal mathematical language."

Empirically, `burton1999intuition` [VERIFIED, n=70 interviews] is the closest match to your
anchor's method: "There was no agreement about whether these terms represented different,
distinctive states and, if so, how these states might be recognised" (p. 28).
`geist2010peer` [VERIFIED] shows referees have no stated uniform standard for what "checked"
means, in the discipline where you'd most expect one.

**Read `wells1990beautiful` and `sa2024agree` as a pair.** Wells concludes "the idea that
mathematicians largely agree in their aesthetic judgements is at best grossly oversimplified";
Sa et al. recover agreement under comparative judgement, arguing mathematicians "might have
different absolute standards for beauty ... but might nevertheless agree about which objects
are more or less beautiful." Reliable discrimination without portable absolute criteria —
directly relevant to our estimand, and honest about the field's instability.

**Complication to engage, not bury:** `inglis2020aesthetic` — the same proof was rated more
beautiful when attributed to "Proofs from THE BOOK." The stated appraisal isn't a stable
readout of a private standard either.

**Closest unfetched analogue to your anchor:** `mejiaramos2021explanatory` — comparative
judgement on explanatory value, high agreement, results inconsistent with the philosophical
literature's stated account. Same template, second criterion.

### Coding
`naur1985programming` is quote #4 and the strongest in-principle claim in any domain:
knowledge of design harmony "could not, in principle, be expressed in terms of rules."
`fowler1999smells` gives the practitioner refusal, verified against the real Pearson text
(the quote-aggregator sites don't have it): "In our experience no set of metrics rivals
informed human intuition" (p. 75).

`buse2010learning` [VERIFIED] is agreement-without-definition *measured*: consensus that
readability determines quality "but not about which factors contribute"; readability "was
intentionally left formally undefined in order to capture the unguided and intuitive notions
of participants"; humans "agree significantly ... but not to an overwhelming extent."

`soloway1984empirical` [VERIFIED via OCR] gives the mechanism: unstated "rules of programming
discourse" that experts detect only through violation. `latoza2010hard` [VERIFIED] gives the
self-directed version — rationale ("why wasn't it done this other way?") is the single
largest hard-to-answer category for the code's own authors.

Institutional evidence: `sadowski2018modern` — Google's readability certification is granted
when reviewers "are confident the developer understands the code style," i.e. an
apprenticeship standing in for a codified standard at scale.

### Academic articles / peer review
`kang2018peerread` is quote #2 and the headline. `greiffenhagen2024checking` [VERIFIED] is
the best qualitative companion, and its author's own methodological aside is the quote:
"It was more challenging for my interviewees to express what they expected from referees with
respect to checking correctness." The practice itself: "it's not a question of formally
checking line by line so much as understanding where the problems are likely to appear" —
"Even if the paper is 70 pages, I open to page 37, because I know that this is where the
action has to be." Even *correctness* runs on tacit pattern-matching.

Its companion `greiffenhagen2024judging` inverts the naive model: editors solicit "quick
opinions" on importance *before* anyone checks correctness — the soft judgment gates the hard one.

`hug2024referees` [VERIFIED] is the formal-modelling result: the best-fitting criteria-plus-
combination-rule model describes "most — but not all — of the referees' judgments." A residual
that resists formalization, from scoring behaviour.

### Grant proposals
**`guetzkow2004originality` is the best verified source in the entire sweep** (read directly
from the JSTOR PDF). Three quotes worth having:
- p. 194 — "the guidelines are very general, and panelists are given no indication of the
  specific meaning of the suggested criteria (such as feasibility or significance) or the
  weight to be given to each one."
- p. 195 — "Only one of our fellowship programs specifically mentioned the 'originality' of
  the proposal in their guidelines; yet it was of major concern to almost all the reviewers
  interviewed ... few of the reviewers expressed concern for or even knowledge of the
  institutional guidelines."
- p. 203 — what "originality" decodes to: "adventurous, ambitious, bold, courageous,
  independent, intellectually honest, curious, and risk-taking... this lexicon describes
  qualities that indicate whether or not one possesses intellectual authenticity." A
  moral-aesthetic judgment of the *person*, in no rubric anywhere.

`brunet2022making` [VERIFIED] was found by nobody's prediction and is fully sourced: ERC
reviewers "attempted to get a 'feel for the sample'" (p. 492); they "did not assess any
specific aspect of each proposal but rather looked to establish a preliminary, almost
embodied, impression of the whole set" (p. 493); "There was no consensus between panel
reviewers about how strong or loose the articulation between different elements should be"
(p. 494). *Represent their framing honestly* — the authors call this "evaluative pragmatism"
and would resist a strong inarticulacy reading.

**`lamont2009how` remains the gap in the evidence.** Two agents made ~20 fetch attempts;
archive.org is metadata-only, Google Books renders as images, academia.edu 403s. We have one
page-cited SNIPPET via a Bryn Mawr review (p. 227). Do not quote the widely-paraphrased
"gut feelings" lines — they are unverified. This needs a library copy.

### Legal argument
`hutcheson1929judgment` is quote #5 and the domain's best find — a sitting federal appellate
judge stating the written reasons are constructed afterward. Citation VERIFIED at the Cornell
repository; the quotes are SNIPPET (all primary fetches 403'd) and need one HeinOnline pass.

`cardozo1921nature` [VERIFIED, full text] gives a line that nearly defines our estimand:
"It is often through these subconscious forces that judges are kept consistent with
themselves, and inconsistent with one another."

`neumann2000schon` [VERIFIED, OCR'd pp. 401–413] is the richest single source: "we actually
know very little about how effective lawyers go about solving problems" (p. 405); quoting
Schön, "Tacit knowing or knowing-in-action has this property: we exhibit it by the competent
behavior we carry out but we are unable to describe what it is that we do" (p. 407); and on
transmission failure, students "literally did not know what they were doing, and their
teachers could not tell them" (p. 408).

`jacobellis1964` [VERIFIED] supplies the "I know it when I see it" primary, correctly pinned
at 378 U.S. at 197.

**Unresolved and worth it:** `guthrie2007blinking` — controlled experiments with sitting
judges, the best empirical evidence in the domain, all three fetch routes failed. And nobody
got to moot-court/brief-grading inter-rater reliability at all.

### Creative writing
The academic anchor is the Consensual Assessment Technique, whose premise *is* our thesis:
`amabile2012perspectives` [VERIFIED, p. 8] — "the CAT overcomes the difficulty of defining
ultimate 'objective' criteria for creativity," with agreement substituted for definition.
`baer2004extension` extends it to writing specifically with "an extremely high level of
interrater agreement."

The three pseudonym experiments (`ross1975steps`, `lessing1984somers`, `lassman2007rejecting`)
are vivid and all VERIFIED against their accounts — **but each carries a documented
methodological objection**, recorded in the bib entries. Use them as contested trade lore, not
as clean evidence. The genuinely useful artifact is Houghton Mifflin's rejection of *Steps*,
which praises what it rejects: "admiration for writing and style... [but] it doesn't add up to
a satisfactory whole." Stated assessment not predicting the verdict.

`makkai_prizes` [VERIFIED] is precise trade testimony on juries that never agree what
dimension they are judging, where outcomes turn on "Who can sway other judges?"

**Your Csikszentmihalyi question, answered:** *Creativity* (1996) contains the systems model —
creativity validated by a "field" of expert gatekeepers via social evaluation — but **no
evidence that gatekeepers cannot articulate their criteria**. That is a claim about where
validation authority sits, not about inarticulacy. Recorded as a negative result in the bib
(`csikszentmihalyi1996creativity`) so it isn't re-searched. Becker and Bourdieu, which we
already hold, cover the gatekeeping-structure point better.

### Humor
Thin domain, honestly. `amir2016frog` [VERIFIED] is the one real academic find: it proposes an
indirect behavioural test *because* direct articulation is unreliable — "Proponents of the
different theoretical accounts often show a high degree of conviction, suggesting introspection
might not be the best tool for judging the validity of humor theories."

`hessel2023androids` [VERIFIED] gives the optimization-pressure version: best models fall 30
points behind humans on matching, and human explanations beat machine ones in >2/3 of
head-to-heads even given ground-truth scene descriptions. Frame it as "resists explicit
decomposition," not as an editor admission — it isn't one.

The E. B. White epigram (`white1941humor`) needed three corrections: it is **jointly authored**
with Katharine S. White; the primary reads "**purely** scientific mind"; and the punchier
"Analyzing humor is like dissecting a frog..." is a separate paraphrase. Twain attribution is
unsupported.

**Warning on `friedman2014comedy`:** the most-recommended source may cut *against* us — a
review indicates respondents give explicit, class-patterned criteria (cleverness/difficulty vs.
pleasure/relaxation), i.e. stratified explicit taste rather than inarticulacy. Pull the book
before investing in it.

**Already held and relevant:** `bielby1994allhits` ("All Hits Are Flukes") — TV executives
explicitly disavowing the ability to articulate what makes a show succeed.

### Press releases, notice-and-comment
Both thin, and the sweep says so plainly. PR has no clean tacit-knowledge literature;
`pieczka2002public` is the best available and only its abstract was reachable: "Professional
expertise emerges from the analysis as a body not of abstract, but of practical knowledge."
For notice-and-comment, `farina2013rulemaking`'s "situated knowledge" is the best quote
candidate and sits at SNIPPET — worth one direct download to upgrade. `balla2022lost`
[VERIFIED] gives the structural version: agencies apply a substantive/non-substantive filter
whose effects are measurable and whose decision procedure is nowhere stated.

### Cross-cutting
`argyris1985action` is quote #1 and belongs in the introduction. `highhouse2008stubborn`
[VERIFIED] covers hiring: expert predictors "lack insight into how they arrive at predictions."
`johansson2005failure` supplies the causal mechanism that licenses treating stated criteria as
unreliable evidence — choice blindness, with confabulated justifications for choices never made.

`biederman1987sexing` (chicken sexing) is the **positive control** the program has wanted:
experts accurate, their own accounts vague, the real cue reverse-engineered by analysts and
then teachable in a minute. It is a LEAD — paywalled everywhere. Get it.

---

---

# Wave 2 (same day) — six further fields

Bib now holds **174 entries**: 71 VERIFIED, 29 SNIPPET, 47 LEAD, 1 negative result, rest
mixed-tag. Wave-2 sweeps archived alongside wave 1 as `lit2_*.md`.

## The finding that makes the corpus more than a quote list

**Rubric sub-scores collapse to one factor, in two unrelated fields that each spent years
building the dimensions.**

- `cheng2022frozen` — figure skating: Program Component (artistic) scores are linearly
  predictable from Technical Element scores across all four disciplines. The 2004 ISU system
  was built after the 2002 scandal *specifically* to make the two axes independent and
  auditable.
- `kelly2020globalobservation` — teaching observation: one eigenvalue >1 explaining **80–90%**
  of variance across 6–8 supposedly distinct sub-domains, in all three major protocols
  (FFT, PLATO, MQI); kappa .05–.28. The authors name it: "a tendency for overall perceptions
  to create a halo-effect."

Two fields, no contact between them, same result: raters form one impression and distribute it
across the boxes. This is the strongest available evidence that itemization does not
decompose the judgment — it just re-encodes a holistic one. It should probably be a figure.

## Best new quotes

| Quote | Source | Tag |
|---|---|---|
| "on global scales, the experienced clinicians scored significantly better than did the residents and clerks, but **on checklists, the experienced clinicians scored significantly worse**" | `hodges1999osce` | VERIFIED |
| "These are essentially **mute forms of knowledge** in the sense that their precepts do not lend themselves to being either formalized or spoken. No one learns to be a connoisseur or diagnostician by restricting himself to practicing only preexistent rules." | `ginzburg1989clues` | VERIFIED |
| "Supervisors often know who to pick, **even if they can't tell exactly why**. This gut feeling does not always match with formally assessed knowledge or skill, but it may be more valid for its purpose." | `tencate2006trust` | VERIFIED |
| 82.3% of professional musicians say sound matters most; from sound alone they pick winners at 25.7% (**below** 33% chance, p=.002), from silent video at 47.0% | `tsay2013sight` | VERIFIED |
| "I admit that, although I have a very unpleasant feeling about this kouros, I am unable to discover this kind of evidence in its partial forms." | `gettykouroscolloquium1993` (Lambrinoudakis) | VERIFIED |
| "decision ultimately rests with something which cannot be discussed... we resign as scholars and even as writers" | `friedlander1942onart` | VERIFIED |
| Evaluation criteria in design juries "were often defensible only on the grounds of **'Good Taste and Intuition'**" | `anthony1991juries` via `salamaelattar2010jury` | VERIFIED |
| "fit" as decisive-but-undefined: "He did well on the case and was very articulate... but I don't think he'd be a good fit." | `rivera2012hiring` | VERIFIED |

## Chicken sexing, resolved

`biederman1987sexing` upgraded LEAD → VERIFIED with full pagination. Naive subjects
**60.5% → 84%** after a ~1-minute instruction sheet (no-instruction control 59.0% → **54.1%**,
i.e. declined); item-wise correlation with experts **r = .21 → .82**; variance in expert
judgments predictable from naive subjects **4% → 67%**. Cue = cloacal bead, convex (male) vs
concave/flat (female).

Three corrections to the folklore, all of which matter for how we use it:
1. The sexers were **imprecise, not mute** — they had gist descriptors ("round"/"pointy") but
   no transferable rule. The cue was recovered by having the master sexer **circle regions on
   images**, not from verbal report. Pointing worked where telling failed — which maps onto
   our decompression rungs directly.
2. Stimuli were **static photographs of deliberately hard cases, untimed**. Experts scored 72%
   on them; the paper says this undersells their ~99% field accuracy. So the *cue* is statable
   and fast to learn — not that a minute makes a professional sexer.
3. Transmission was pure apprenticeship anyway: "Not a single sexer recalled being shown simple
   diagrams... training was accomplished by sexing live birds, checked by experts."

`horsey2002chicken` (retrieved via a Wayback snapshot of the dead UCL host) is the careful
academic treatment and explicitly corrects the popular retelling; it adds a second elicitation
case, WWII aircraft spotters who "had no idea how they had acquired their skills."

**The honest counterweight**, and it belongs in the paper: `camererjohnson1991paradox`
reproduces a table across five studies where a linear model of a judge's *own* cues beats that
judge — the stated/used gap quantified — but ALSO supplies two failure cases. Expert-system
rules faithfully encoding clinicians' stated configural rules do not reliably beat a simple
linear model of the same cues; and Chapman & Chapman's illusory correlation is a confidently
articulated rule that is simply false. Elicitation sometimes recovers a real cue and sometimes
recovers a confabulation that performs worse than a dumb model. Both outcomes are live for us.

---

## What to do next

| Priority | Item | Why |
|---|---|---|
| 1 | Library pull: `lamont2009how`, `hutcheson1929judgment`, `breed1955social`, `schon1983reflective`/`schon1987educating`, `anthony1991juries`, `braude2012intuition` | All load-bearing, all stuck below VERIFIED. Schön especially — we cite him only secondhand, which is a real weakness given how central knowing-in-action is to the framing. Most are behind controlled-digital-lending, so a library card fixes them, not more agents |
| 2 | Verify before camera copy | `singh2017evaluating` 16% figure; `lewis2008compromised` 80/12 figures; `frank1930law` once-removed quote; Berenson "my stomach" anecdote is flagged DO NOT USE |
| 3 | Decide bib wiring | Papers 2, 3, 4 have no `\bibliography` command; paper-1's `refs.bib` is untracked in git |
| 4 | Patents | The wave-1 patents sweep never returned. Relaunch scoped down: KSR's "common sense," Learned Hand on invention as a "fugitive, impalpable, wayward, and vague phantom," and the examiner-variation literature (Cockburn/Kortum/Stern, Lemley/Sampat) |
| 5 | Consider the halo result as a designed experiment | If itemized sub-scores collapse to one factor in skating and teaching, our own multi-criterion rubrics are a natural third test — and we already have the grids to check it |

## Infrastructure notes for future sweeps

- WebSearch has a **session-wide 200-call cap** shared across concurrent agents. Wave 2 ran
  almost entirely without it; agents routed through `r.jina.ai`-proxied search pages, OpenAlex,
  Crossref, Unpaywall, and NCBI E-utilities instead. E-utilities was the single most reliable
  channel and produced most of the medicine VERIFIEDs.
- Raw-PDF WebFetch nearly always fails. What works: `curl` + `pdftotext -layout`; `pdftoppm` +
  `tesseract` for scans; `r.jina.ai/<url>` as a reader proxy; or save the binary and open it
  with the Read tool.
- Reliably blocked, do not retry: ResearchGate, academia.edu, Harvard DASH, Google Books,
  web.archive.org (in this environment), Paris Review, HathiTrust, JSTOR.
- **Crashed agents leave their downloads behind.** The wave-2 retrieval agent stalled without
  writing its file, but its `curl`'d PDFs survived in the scratchpad and one of them
  (`mallard2009fairness`) was upgraded SNIPPET → VERIFIED by reading it locally. Check the
  scratchpad before rerunning anything.

**Patents** was the tenth sweep and had not returned when this note was written — append its
entries to `latex/refs-shared.bib` under a PATENTS heading when it lands.
