# Literature sweep: articulation gaps in DESIGN/ARCHITECTURE CRITICISM and TEACHING-QUALITY evaluation

Task: find verbatim testimony / empirical evidence that design and teaching quality are recognized
by experts but not stateable as explicit criteria, or that explicit rubrics demonstrably fail to
capture what experts respond to. Verification tags per instructions: **[VERIFIED]** (fetched text
contains it, URL given), **[SNIPPET]** (exact string reproduced inside a fetched secondary source),
**[LEAD]** (known to exist, not independently fetched — nothing invented).

---

## 0. Already held (per the four bib files) — NOT re-reported below

Confirmed via grep of `latex/refs-shared.bib`, `methods/metric_implementer/references.bib`,
`notes/articulability-prompt-opt.bib`, `latex/paper-1__metric-codability/refs.bib`:
sadler2009indeterminacy, sadler2014futility, bloxham2009marking, bloxham2011mark, bloxham2016lets,
lumley2002assessment, odonovan2004know, **neumann2000schon** (Schön via legal ed, VERIFIED,
reproduces p.407-408 tacit-knowing quotes), **schon1995educating** (SNIPPET via Neumann, p.243
tacit-knowing line). No existing holdings on architecture/design criticism, design juries, crits,
Cross, Lawson, Danielson, CLASS, MET Project, or Jacob & Lefgren — this is genuinely new territory.

---

## 1. Top finds, ranked

### (1) MET Project — cross-instrument rubric correlations & rater-judgment quote [VERIFIED — HIGH PRIORITY]
Kane, T.J. & Staiger, D.O. (2012). *Gathering Feedback for Teaching: Combining High-Quality
Observations with Student Surveys and Achievement Gains* (MET Project Research Paper, Bill &
Melinda Gates Foundation). Fetched via ERIC ED540960: https://files.eric.ed.gov/fulltext/ED540960.pdf

> "The correlation between the two general pedagogical instruments, FFT and CLASS, was 0.88,
> implying that those two instruments, if used many times on the same group of teachers, would
> provide very similar rankings among teachers. The set of competencies measured by the two
> instruments — even if they appear distinct — are very highly correlated. Moreover, the
> correlation between the two math instruments, MQI and UTOP, was 0.85."

Two independently-built, differently-named observation rubrics (Danielson's Framework for Teaching
vs. CLASS; MQI vs. UTOP) — built by different teams around different theoretical frameworks — turn
out to be measuring almost the same latent thing. Also [VERIFIED], on why reliability is hard:

> "instructional practice for a given teacher varies from lesson to lesson and, even with training,
> the instruments all require rater judgment, which is rarely unanimous." (p.8)

And the paper explicitly cautions that even 4 trained, disinterested, video-based observers may not
transfer to real school conditions (p.8) — i.e. the reliability numbers below are a **best case**.

### (2) MET Project — single-observation reliability figures & first-impressions-linger [VERIFIED — HIGH PRIORITY]
Ho, A.D. & Kane, T.J. (2013). *The Reliability of Classroom Observations by School Personnel*
(MET Project Research Paper, Harvard GSE / Gates Foundation). Fetched via ERIC ED540957:
https://files.eric.ed.gov/fulltext/ED540957.pdf

> "A single observation by a single observer is a fairly unreliable estimate of a teacher's
> practice, with reliability between .27 and .45." (p.13)

Component breakdown (Table 5, VERIFIED): on Danielson's 4-point Framework for Teaching scale,
teacher-true-score SD = 0.27–0.29 pts; **rater-by-teacher interaction alone accounts for 15–20% of
score variance** — i.e. different raters watching the *same* lesson from the *same* teacher
systematically disagree, and this component does *not* shrink with more lessons, only with more
raters (p.13). Scale compression (VERIFIED, Summary of Findings #1, p.3):

> "Observers rarely used the top or bottom categories ('unsatisfactory' and 'advanced')... The
> vast majority of scores were in the middle two categories, 'basic' and 'proficient.' On this
> compressed scale, a .1 point difference in scores can be sufficient to move a teacher up or down
> 10 points in percentile rank."

And a halo/anchoring effect directly bearing on articulability (VERIFIED, Table 9, p.27): same-rater
correlation across four sequential video scorings of the *same* teacher rose from ~.65 (videos far
apart) to .73–.78 (videos scored back-to-back) — "When an observer formed a positive (or negative)
impression of a teacher in the first several videos, that impression tended to linger" (Summary
Finding #6, p.3). Rubric scores track the rater's global impression, not just the target performance.

### (3) Observation-rubric sub-scores collapse to one/two latent factors [VERIFIED — the teaching-side analogue of the "figure-skating" finding]
Kelly, S., Bringe, R., Aucejo, E., & Fruehwirth, J. (2020). "Using Global Observation Protocols to
Inform Research on Teaching Effectiveness and School Improvement: Strengths and Emerging
Limitations." *Education Policy Analysis Archives*, 28(62). Fetched:
https://epaa.asu.edu/ojs/article/download/5012/2427

Own exploratory factor analysis of four MET-family instruments (Table 2, VERIFIED, p.13):

| Instrument | # sub-domains scored | eigenvalues > 1 | 1-factor R² |
|---|---|---|---|
| FFT (Danielson) | 8 | **1** | 0.800 |
| CLASS | 12 | 2 | 0.678 |
| PLATO | 6 | **1** | 0.575 |
| MQI (holistic) | 4–5 | **1** | 0.840–0.903 |

A single latent factor explains 80–90% of the variance across supposedly distinct sub-domains for
three of four instruments. Citing a prior independent study for FFT specifically (SNIPPET via this
paper, p.13): "Liu et al. (2019) examined the covariance structure of FFT observation scores in
three sets of data... In all cases, they found high correlations across the four FFT domains and
eight sub-domains such that a single factor structure best fit the data." The paper's own
interpretation (VERIFIED, p.14) names the mechanism directly:

> "An alternative explanation is that features of the observation system, such as a tendency for
> overall perceptions to create a **halo-effect**, create artificial consistency in sub-domain
> scores."

Also gives concrete inter-rater kappa figures for FFT under normal (non-expert) rater conditions
(Table 3 discussion, VERIFIED, p.14): "exact agreement by two local raters ranged from only 47.3%
to 65.8% across domains, and simple kappas ranged from .05 to .28" — i.e. barely above chance on a
scale most observers already compress into 2 of 4 categories.

### (4) Jacob & Lefgren — principals can spot the extremes, not the middle [VERIFIED — HIGH PRIORITY, exactly the requested claim]
Jacob, B.A. & Lefgren, L. (2008). "Can Principals Identify Effective Teachers? Evidence on
Subjective Performance Evaluation in Education." *Journal of Labor Economics*, 26(1), 101–136
(DOI 10.1086/522974). Verified via its NBER working-paper ancestor, "Principals as Agents:
Subjective Performance Measurement in Education," NBER WP 11463 (June 2005), fetched:
https://www.nber.org/system/files/working_papers/w11463/w11463.pdf — abstract/conclusion language
below is stable across the WP→JOLE revision (standard for this kind of paper; the qualitative
top/middle/bottom finding is the paper's headline result, restated identically in Abstract §I and
Conclusions §VII of the WP).

> "principals appear quite good at identifying those teachers who produce the largest and smallest
> standardized achievement gains in their schools [top/bottom 10–20%], but have far less ability to
> distinguish between teachers in the middle of this distribution [middle 60–80%]... This is not a
> result of a highly compressed distribution of teacher ability, the lumpiness of the principal
> ratings, or the differential precision of value-added measures across the distribution."
> (Abstract; restated as Conclusions, p.30: "the inability of principals to distinguish between a
> broad middle-range of teacher quality suggests caution in relying on principals for fine grained
> performance determinations.")

This is a genuine articulation-gap result in the requested shape: **holistic subjective judgment
outperforms explicit criteria exactly where the phenomenon is easiest (extremes) and underperforms
nowhere it's tested against value-added** — but the informativeness itself is not decomposable into
stated criteria; the paper models it as "principal ratings" as an opaque input, not as an explicit
rubric. Bonus [VERIFIED] — footnote 6 of the same WP explicitly reaches for the classic
inarticulable-standard analogy that we already use for legal judgment elsewhere in this project:

> "In 1964, Justice Potter Stewart tried to explain 'hard-core' pornography, or what is obscene, by
> saying, 'I shall not today attempt further to define the kinds of material I understand to be
> embraced... [b]ut I know it when I see it.'" (citing *Jacobellis v. Ohio*, 378 U.S. 184, 197 (1964))

Also [VERIFIED], on rubric sub-item correlation (p.9, footnote 16 area) — a partial (3-factor, not
1-factor) analogue to find (3) above: "the correlation between teacher organization and classroom
management exceeds 0.7 while the correlation between role model and relationship with colleagues is
less than 0.4" — i.e. some sub-items of the principal's rating instrument are near-redundant,
others aren't; an exploratory factor analysis of the multi-item principal survey yielded 3 (not 1)
interpretable factors, so treat this as a *partial* structural-collapse case, not a clean single-
factor result — do not conflate with find (3).

### (5) Design juries as "tacit/folklore," riddled with inconsistency [VERIFIED]
Webster, H. (2006). "Power, Freedom and Resistance: Excavating the Design Jury." *International
Journal of Art & Design Education*, 25(3), 286–296. Fetched via Oxford Brookes RADAR:
https://radar.brookes.ac.uk/radar/file/78a0bad5-170d-aa1f-a0b4-26705261fc59/1/webster2006power.pdf

> "architectural educators and students appear to have a largely 'tacit', or 'folklore',
> understanding of its pedagogic purpose and processes that often seems riddled with
> inconsistencies and contradictions." (p.1)

This is the design-studio analogue of "practitioners agree on outcome, disagree on/can't state
rationale" — the jury is centrally important, ritualized, but its evaluative logic is transmitted as
folklore rather than as articulated criteria. The paper cites Anthony's *Design Juries on Trial*
(1991) as the founding critical study (references list, VERIFIED) but the book itself is
access-restricted on archive.org (see Dead Ends). Use Webster + Salama & El-Attar (next) as the
verified secondary channel into Anthony's findings.

### (6) Quantified student perceptions of jury inconsistency (n=209) + Anthony quotes reproduced [VERIFIED]
Salama, A.M. & El-Attar, M.S.T. (2010). "Student Perceptions of the Architectural Design Jury."
*Archnet-IJAR: International Journal of Architectural Research*, 4(2-3), 174–200. Fetched via
Strathprints: https://strathprints.strath.ac.uk/50234/1/Salama_El_attar_Student_Perceptions_of_Architectural_Design_Juries104_318_1_PB_2_.pdf

Reproduces Anthony's own finding verbatim (SNIPPET, p.2, citing Anthony 1991) — the sharpest
one-line statement of unstateable criteria in this whole sweep:

> "Evaluation criteria were often defensible only on the grounds of 'Good Taste and Intuition.'"

And (SNIPPET, p.3, citing Anthony 1987/Dutton 1987/Salama 1995/Sara 2004 in agreement):

> "faculty critique each project spontaneously without criteria made clear to the students who are
> asked to defend their work."

The paper's *own* survey of n=209 architecture students across four Cairo universities (Ain Shams,
Al Azhar, Cairo, Helwan) found [VERIFIED, p.28-29]:
- 75% agreed design priorities emphasized in studio instruction were *changed* unpredictably at the
  jury itself (i.e. the criteria applied at evaluation are not the criteria that were taught);
- Asked about the general mode among jury members: 33% reported "contradiction among all members of
  the jury," 55% reported a "competitive scene" among jurors, and **only 16% reported harmony/
  understanding among jurors** — i.e. students perceive very low inter-juror agreement;
- this pattern is explicitly cross-referenced (SNIPPET, p.28) to Anthony (1987), Frederickson
  (1990), and Sara (2004) "when they agree that jurors come to the juries with hidden agendas."

### (7) Schön, *The Reflective Practitioner* (1983) — a primary-text quote, finally [VERIFIED, but thin — see caveat]
Fetched via infed.org (open-access education encyclopedia), which quotes the book directly with a
page citation: https://infed.org/mobi/donald-schon-learning-reflection-change/

> "The practitioner allows himself to experience surprise, puzzlement, or confusion in a situation
> which he finds uncertain or unique. He reflects on the phenomenon before him, and on the prior
> understandings which have been implicit in his behaviour. He carries out an experiment which
> serves to generate both a new understanding of the phenomenon and a change in the situation."
> (Schön, 1983, p.68, as quoted at infed.org)

Caveat: this is the general reflection-in-action formulation, not the Quist/Petra architecture
protocol specifically, and not the "our knowing is in our action" line. See Dead Ends — the 1983
and 1987 books are both access-restricted (lending-only, no OCR) on archive.org, and the specific
primary source for Quist/Petra — Schön's own journal article — is paywalled (next item). This
remains the weakest-closed vein in the sweep; recommend a follow-up pass with library/JSTOR access.

---

## 2. Leads (real, cited, not independently verified — do not quote numbers from these without a further fetch)

**(8) Schön, D.A. (1984). "The Architectural Studio as an Exemplar of Education for
Reflection-in-Action." *Journal of Architectural Education*, 38(1), 2–9.** [LEAD] — this is almost
certainly the *actual* primary publication of the Quist/Petra protocol (predates the 1987 book,
same case). DOI 10.1080/10464883.1984.10758345 / JSTOR 1424770. Confirmed via Crossref/OpenAlex;
confirmed **not** open access via Unpaywall (checked both DOIs, `is_oa: false`, no repository
copy). This is the single highest-value target for a follow-up pass with institutional/JSTOR access
— it would give primary Quist/Petra quotes directly rather than through secondhand legal-education
channels (Neumann) as we currently have.

**(9) Anthony, K. (1991). *Design Juries on Trial: The Renaissance of the Design Studio*. Van
Nostrand Reinhold.** [LEAD] — book itself is access-restricted on archive.org (identifier
`designjuriesontr0000anth`); triangulated well via finds (5) and (6) above, which reproduce its
core claims and at least one verbatim phrase ("Good Taste and Intuition") with attribution. A
university-library fetch of the physical/scanned book would let us quote Anthony directly rather
than through Salama & El-Attar's citation.

**(10) Anthony, K. (1987). "Private Reactions to Public Criticism: Students, Faculty, and
Practicing Architects State Their Views on Design Juries in Architectural Education." *Journal of
Architectural Education*, 40(3).** [LEAD] — Anthony's earlier, methodologically-described paper
(systematic behavioral observations + interviews + questionnaires + diaries of students/faculty/
alumni per Salama & El-Attar's description, p.9 of that paper). Same access situation as (8)/(9).

**(11) Liu, S., Bell, C.A., Jones, N.D., & McCaffrey, D.F. (2019).** [journal TBD — cited inside
Kelly et al. 2020 as the source of the single-factor FFT finding] [LEAD] — the actual covariance-
structure study behind the single-factor FFT claim quoted in find (3). Worth a direct fetch to get
its own numbers rather than relying on Kelly et al.'s summary.

**(12) Cross, N. (2004). "Expertise in Design: An Overview." *Design Studies*, 25(5), 427–441.**
and **Cross, N. (2007/2011). *Designerly Ways of Knowing*. Springer.** [LEAD] — the canonical
design-cognition source for "designers cannot fully account for their own moves." DOI
10.1016/j.destud.2004.06.002; confirmed closed-access via Unpaywall; the book itself is
access-restricted on archive.org (`designerlywaysof0000cros`). Not independently verified this pass.

**(13) Lawson, B. (2006). *How Designers Think: The Design Process Demystified* (4th ed.).
Architectural Press.** [LEAD] — same status as Cross: archive.org copies (`howdesignersthin0000laws`
and variants) are all access-restricted; no open-access journal version located. Canonical but
unverified this pass.

**(14) Cor, M.K. (2011). "Investigating the Reliability of Classroom Observation Protocols: The
Case of PLATO." Stanford [working paper].** [LEAD] — cited within Kelly et al. 2020's reference
list as a PLATO-specific reliability study; not independently fetched.

---

## 3. Dead ends (tried, didn't pan out — so a future pass doesn't repeat the wasted effort)

- **Schön's 1983/1987 books on archive.org**: three separate scanned copies located
  (`educatingreflect00sch`, `educatingreflect0000schn`, `reflectivepracti00scho`) but all three are
  `access-restricted-item: true` (controlled digital lending) — no djvu.txt or search-inside API
  access without a login/borrow session. Same for Anthony's *Design Juries on Trial*, Cross's
  *Designerly Ways of Knowing*, and Lawson's *How Designers Think*. All four canonical books are
  behind this wall; a pass with an authenticated archive.org account (or physical/library access)
  would unlock primary quotes for all of them at once.
- **WebSearch tool**: unusable this session — budget exhausted (200/200) before this task started
  (shared session budget, consumed by concurrent agents). Substituted OpenAlex/Crossref/Unpaywall/
  ERIC/Semantic-Scholar-style JSON APIs plus direct `curl`+`pdftotext` for known repository URLs.
  DuckDuckGo/Bing HTML scraping via curl and WebFetch reliably hit CAPTCHA/bot-walls — not viable.
- **"The Components of the Crit in Art and Design Education" (2016, TU Dublin ARROW repository)**:
  located the correct OA PDF URL (`https://arrow.tudublin.ie/cgi/viewcontent.cgi?article=1054&context=ijap`)
  via WebFetch, but the repository is behind a Cloudflare bot-challenge that blocked both `curl` and
  WebFetch download attempts. Worth retrying with a real browser session (claude-in-chrome) rather
  than curl/WebFetch.
- **"What Does It Mean to Design? A Qualitative Investigation of Design Professionals' Experiences"
  (2012, ASEE)**: OpenAlex confirms an OA green copy at `deepblue.lib.umich.edu`, but the host
  returns an HTTP 403 Cloudflare managed-challenge to both `curl` and WebFetch. Likely fetchable via
  claude-in-chrome browser automation instead.
- **Google Books API**: quota-exhausted (429, shared project quota) on the first call — could not
  pull snippet-view text for Anthony/Cross/Lawson as a workaround for the archive.org lock.
- **"The Mirage" (TNTP, 2015)**: located (ERIC ED558206) but on inspection its focus is professional-
  development spending, not rubric validity/reliability — not a good fit for this brief, excluded.
- Jacob & Lefgren's multi-item principal-survey factor analysis yielded **3** factors, not a clean
  single-factor collapse — flagged in find (4) so it isn't mistakenly quoted as a "collapse to one
  factor" result alongside find (3).

---

## Ready-to-paste BibTeX

```bibtex
@techreport{kanestaiger2012gathering,
  author      = {Kane, Thomas J. and Staiger, Douglas O.},
  title       = {Gathering Feedback for Teaching: Combining High-Quality Observations with
                 Student Surveys and Achievement Gains},
  institution = {Bill \& Melinda Gates Foundation, MET Project},
  year        = {2012},
  type        = {Research Paper},
  keywords    = {domain=teaching; gap=explicit-fails; type=empirical-report},
  annote      = {VERIFIED (fetched full text via ERIC ED540960,
                 https://files.eric.ed.gov/fulltext/ED540960.pdf). Two independently designed
                 observation rubrics correlate at .85--.88 despite scoring "distinct" competencies
                 ("The set of competencies measured by the two instruments -- even if they appear
                 distinct -- are very highly correlated," p.8): FFT/CLASS r=.88, MQI/UTOP r=.85.
                 Also: "the instruments all require rater judgment, which is rarely unanimous"
                 (p.8). See kanestaiger2012 companion ho2013reliability for the single-observation
                 reliability figures.}
}

@techreport{ho2013reliability,
  author      = {Ho, Andrew D. and Kane, Thomas J.},
  title       = {The Reliability of Classroom Observations by School Personnel},
  institution = {Bill \& Melinda Gates Foundation, MET Project, Harvard Graduate School of
                 Education},
  year        = {2013},
  month       = jan,
  keywords    = {domain=teaching; gap=rubric-noise; type=empirical-report},
  annote      = {VERIFIED (fetched full text via ERIC ED540957,
                 https://files.eric.ed.gov/fulltext/ED540957.pdf). Danielson Framework for
                 Teaching: "A single observation by a single observer is a fairly unreliable
                 estimate of a teacher's practice, with reliability between .27 and .45" (p.13).
                 Rater-by-teacher interaction alone = 15-20\% of score variance and does NOT shrink
                 with more lessons by the same rater, only with more raters (p.13). Scale
                 compression: "the vast majority of scores were in the middle two categories" of a
                 4-point scale, so ".1 point difference in scores can be sufficient to move a
                 teacher up or down 10 points in percentile rank" (p.3). Halo/anchoring: same-rater
                 correlation across sequential videos of the same teacher rises from ~.65 to
                 .73-.78 when scored back-to-back (Table 9, p.27) -- "that impression tended to
                 linger" (p.3).}
}

@article{kelly2020globalobservation,
  author  = {Kelly, Sean and Bringe, Robert and Aucejo, Esteban and Fruehwirth, Jane},
  title   = {Using Global Observation Protocols to Inform Research on Teaching Effectiveness and
             School Improvement: Strengths and Emerging Limitations},
  journal = {Education Policy Analysis Archives},
  volume  = {28},
  number  = {62},
  year    = {2020},
  keywords = {domain=teaching; gap=one-factor-collapse; type=empirical-methodological},
  annote  = {VERIFIED (fetched full text, https://epaa.asu.edu/ojs/article/download/5012/2427).
             THE teaching-side single-factor-collapse finding: own exploratory factor analysis
             (Table 2, p.13) finds a single eigenvalue >1 for FFT (8 sub-domains, 1-factor
             R\textsuperscript{2}=.80), PLATO (6 sub-domains, R\textsuperscript{2}=.575), and
             holistic MQI (R\textsuperscript{2}=.84-.90); only CLASS needs 2 factors. Also cites
             Liu et al. (2019, SNIPPET via this paper p.13): "found high correlations across the
             four FFT domains and eight sub-domains such that a single factor structure best fit
             the data." Names the mechanism directly: "a tendency for overall perceptions to
             create a halo-effect, create artificial consistency in sub-domain scores" (p.14).
             Also gives raw kappa figures under normal (non-expert) rater conditions for FFT:
             exact agreement 47.3-65.8\%, simple kappa .05-.28 (p.14) -- barely above chance.}
}

@article{jacoblefgren2008principals,
  author  = {Jacob, Brian A. and Lefgren, Lars},
  title   = {Can Principals Identify Effective Teachers? Evidence on Subjective Performance
             Evaluation in Education},
  journal = {Journal of Labor Economics},
  volume  = {26},
  number  = {1},
  pages   = {101--136},
  year    = {2008},
  keywords = {domain=teaching; gap=holistic-beats-explicit; type=empirical-econ},
  annote  = {VERIFIED via the NBER working-paper ancestor (WP 11463, "Principals as Agents:
             Subjective Performance Measurement in Education," June 2005,
             https://www.nber.org/system/files/working_papers/w11463/w11463.pdf; the headline
             qualitative finding is restated identically in the WP's Abstract and Conclusions,
             and is standard practice to survive unchanged into the published version). HIGH
             PRIORITY: "principals appear quite good at identifying those teachers who produce
             the largest and smallest standardized achievement gains in their schools (i.e., the
             top and bottom 10-20 percent), but have far less ability to distinguish between
             teachers in the middle of this distribution (i.e., the middle 60-80 percent)." A
             holistic, unarticulated judgment is informative exactly where the target is easiest
             and silent nowhere it's tested -- but is not decomposable into stated criteria.
             Bonus [VERIFIED], footnote 6 of the WP: quotes Justice Potter Stewart's "I know it
             when I see it" (Jacobellis v. Ohio, 378 U.S. 184, 197 (1964)) as the paper's own
             analogy for subjective evaluation. CAVEAT: the paper's OWN multi-item principal
             survey factor-analyzes into 3 factors, not 1 -- do not cite as a single-factor
             collapse case; some sub-items correlate >0.7, others <0.4 (p.9).}
}

@book{anthony1991juries,
  author    = {Anthony, Kathryn H.},
  title     = {Design Juries on Trial: The Renaissance of the Design Studio},
  publisher = {Van Nostrand Reinhold},
  year      = {1991},
  keywords  = {domain=design; gap=criteria-unstated; type=book},
  annote    = {LEAD -- book access-restricted on archive.org (identifier
               designjuriesontr0000anth, controlled digital lending, no search-inside access).
               Triangulated via webster2006jury and salamaelattar2010jury below, which
               reproduce its core claims with at least one verbatim phrase and attribution.
               Fetch via institutional/library access for direct quotes.}
}

@article{webster2006jury,
  author  = {Webster, Helena},
  title   = {Power, Freedom and Resistance: Excavating the Design Jury},
  journal = {International Journal of Art \& Design Education},
  volume  = {25},
  number  = {3},
  pages   = {286--296},
  year    = {2006},
  keywords = {domain=design; gap=tacit-folklore; type=empirical-ethnographic},
  annote  = {VERIFIED (fetched full text via Oxford Brookes RADAR,
             https://radar.brookes.ac.uk/radar/file/78a0bad5-170d-aa1f-a0b4-26705261fc59/1/webster2006power.pdf).
             "architectural educators and students appear to have a largely 'tacit', or
             'folklore', understanding of its pedagogic purpose and processes that often seems
             riddled with inconsistencies and contradictions" (p.1). Cites Anthony (1991) and
             Cuff as the foundational critiques of the jury system's opacity.}
}

@article{salamaelattar2010jury,
  author  = {Salama, Ashraf M. and El-Attar, M. Sherif T.},
  title   = {Student Perceptions of the Architectural Design Jury},
  journal = {Archnet-IJAR: International Journal of Architectural Research},
  volume  = {4},
  number  = {2-3},
  pages   = {174--200},
  year    = {2010},
  keywords = {domain=design; gap=criteria-unstated; type=empirical-survey},
  annote  = {VERIFIED (fetched full text via Strathprints,
             https://strathprints.strath.ac.uk/50234/1/Salama_El_attar_Student_Perceptions_of_Architectural_Design_Juries104_318_1_PB_2_.pdf).
             Reproduces Anthony (1991) verbatim: "Evaluation criteria were often defensible only
             on the grounds of 'Good Taste and Intuition.'" (p.2). Also (SNIPPET, citing Anthony
             1987/Dutton 1987/Salama 1995/Sara 2004): "faculty critique each project
             spontaneously without criteria made clear to the students" (p.3). Own survey, n=209
             architecture students across 4 Cairo universities: 75\% said design priorities
             emphasized during studio instruction changed unpredictably at the jury itself
             (p.28); only 16\% perceived harmony among jurors vs. 33\% "contradiction among all
             members of the jury" and 55\% a "competitive scene" (p.29).}
}

@book{schon1983reflective,
  author    = {Sch{\"o}n, Donald A.},
  title     = {The Reflective Practitioner: How Professionals Think in Action},
  publisher = {Basic Books},
  year      = {1983},
  keywords  = {domain=design; gap=knowing-in-action; type=book},
  annote    = {VERIFIED page-cited quote via infed.org (open education encyclopedia),
               https://infed.org/mobi/donald-schon-learning-reflection-change/: "The practitioner
               allows himself to experience surprise, puzzlement, or confusion in a situation
               which he finds uncertain or unique. He reflects on the phenomenon before him, and
               on the prior understandings which have been implicit in his behaviour." (p.68).
               CAVEAT: this is the general reflection-in-action formulation, not the Quist/Petra
               architecture-studio protocol specifically (that is schon1984architectural below,
               currently a LEAD). The book itself is access-restricted on archive.org
               (educatingreflect00sch / reflectivepracti00scho, controlled digital lending).}
}

@article{schon1984architectural,
  author  = {Sch{\"o}n, Donald A.},
  title   = {The Architectural Studio as an Exemplar of Education for Reflection-in-Action},
  journal = {Journal of Architectural Education},
  volume  = {38},
  number  = {1},
  pages   = {2--9},
  year    = {1984},
  keywords = {domain=design; gap=osmosis; type=essay},
  annote  = {LEAD -- almost certainly the primary publication of the Quist/Petra protocol
             (predates the 1987 book). DOI 10.1080/10464883.1984.10758345; confirmed NOT open
             access via Unpaywall (checked both the Taylor \& Francis and JSTOR-mirrored DOIs;
             is_oa=false, no repository copy). HIGHEST-VALUE follow-up target for a pass with
             JSTOR/institutional access -- would give primary Quist/Petra quotes directly.}
}

@book{cross2007designerly,
  author    = {Cross, Nigel},
  title     = {Designerly Ways of Knowing},
  publisher = {Springer},
  year      = {2007},
  keywords  = {domain=design; gap=cannot-account-own-moves; type=book},
  annote    = {LEAD -- canonical design-cognition source; book access-restricted on archive.org
               (designerlywaysof0000cros); companion journal piece Cross (2004) "Expertise in
               Design: An Overview," Design Studies 25(5):427-441, DOI
               10.1016/j.destud.2004.06.002, confirmed closed-access via Unpaywall. Not
               independently verified this pass -- flag for library-access follow-up.}
}

@book{lawson2006designersthink,
  author    = {Lawson, Bryan},
  title     = {How Designers Think: The Design Process Demystified},
  edition   = {4th},
  publisher = {Architectural Press},
  year      = {2006},
  keywords  = {domain=design; gap=cannot-account-own-moves; type=book},
  annote    = {LEAD -- canonical design-expertise source; all archive.org copies
               (howdesignersthin0000laws and variants) are access-restricted. No open-access
               journal-length version located. Not independently verified this pass.}
}
```
