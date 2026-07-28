# Literature sweep: articulation gaps in high-stakes selection judgment (VC/investing, hiring, admissions)

Scope per assignment: VC/angel investing, hiring, university/fellowship admissions. Verification
tags per instruction: **[VERIFIED]** = fetched full text contains the quote/stat (URL given),
**[SNIPPET]** = exact string found in a search result / secondary citing source, **[LEAD]** =
nothing fetched, name only.

---

## 0. Already-have (confirmed in repo bibs — NOT new finds)

Checked `latex/refs-shared.bib`, `methods/metric_implementer/references.bib`,
`notes/articulability-prompt-opt.bib` (empty — no matches), `latex/paper-1__metric-codability/refs.bib`
(empty — no matches).

| key | where | note |
|---|---|---|
| `highhouse2008stubborn` | refs-shared.bib | Highhouse 2008, IOP — hiring intuition; already has the "10% of variance" interview quote in its annote |
| `dawes1989clinical` | metric_implementer/references.bib | Dawes, Faust & Meehl 1989, *Science* |
| `grove2000clinical` | metric_implementer/references.bib | Grove et al. 2000, *Psychological Assessment* |
| `tetlock2005expert` | metric_implementer/references.bib | Tetlock 2005, *Expert Political Judgment* |
| `guetzkow2004originality` | refs-shared.bib | Lamont fellowship-panel cluster — 81-interview originality study, ASR 2004 |
| `mallard2009fairness` | refs-shared.bib | Lamont cluster — "cognitive contextualization" procedural-fairness paper |
| `lamont2009how` | refs-shared.bib | Lamont, *How Professors Think* (2009) |
| `lamont2011comparing` | refs-shared.bib | Lamont & Huutoniemi 2011 — "customary rules" panel calibration |

None of these are re-reported below as new.

---

## 1. Top finds, ranked

### 1. Gompers, Gornall, Kaplan & Strebulaev (2020), "How Do Venture Capitalists Make Decisions?" — [VERIFIED]
*Journal of Financial Economics* 135(1): 169-190 (NBER WP 22587 text fetched: https://www.nber.org/system/files/working_papers/w22587/w22587.pdf).

885 institutional VCs at 681 firms, survey across 8 decision areas. Direct hit on the "stated
process vs. stated determinant" gap the assignment asks for:

> "In selecting investments, VCs place the greatest importance on the management/founding team.
> The management team was mentioned most frequently both as an important factor (by 95% of VC
> firms) and as the most important factor (by 47% of VC firms)." (abstract / p.5 of the WP)

> "Few VCs use discounted cash flow or net present value techniques to evaluate their
> investments... Only 22% of the VC investors use NPV methods... 9% of the VCs claim that they
> do not use any financial metrics... **almost half of the VCs, particularly the early-stage, IT,
> and smaller VCs, admit to often making gut investment decisions.** We also asked respondents
> whether they quantitatively analyze their past investment decisions and performance. This is
> very uncommon, with only one out of ten VCs doing so." (p.~15 of WP)

This is about as clean as it gets: nearly half self-report "gut" decisions, one in ten ever
checks their own track record quantitatively, and formal DCF/NPV is marginal — yet team/founder
assessment (the least formalizable input) is what they name as decisive. Directly usable for the
"stated criterion vs. absence of a checkable process" half of the articulation-gap argument.

### 2. Rivera (2012), "Hiring as Cultural Matching: The Case of Elite Professional Service Firms" — [VERIFIED]
*American Sociological Review* 77(6): 999-1022. Full text fetched: https://www.asanet.org/wp-content/uploads/savvy/journals/ASR/Dec12ASRFeature.pdf

120 interviews + hiring-committee observation at elite law/consulting/banking firms. "Fit" is a
**formally mandated** evaluation criterion with no formal definition — evaluators are required to
score it but cannot specify what it is beyond ad hoc heuristics ("use yourself as a proxy,"
résumé-scanning for shared hobbies, "chemistry").

> "Evaluators described fit as being one of the three most important criteria they used to assess
> candidates in job interviews; **more than half reported it was the most important criterion** at
> the job interview stage, rating fit over analytical thinking and communication." (p.1007)

> Banker Nicholae: "A lot of this job is attitude, not aptitude... You can be the smartest guy
> ever, but I don't care... You need chemistry. Not only that the person is smart, but that you
> like him." (p.1008)

> Manager Hans, arguing against a candidate: "He did well on the case and was very articulate. He's
> a very interesting guy with a good story. But I think he's too intellectual for [FIRM]... I
> don't think he'd be a good fit." The candidate was not invited back. (p.1010)

This is the single best "explicit criteria fail to capture what experts actually select on" case
in the hiring vein — technical performance and articulateness are explicitly overridden by an
unspecifiable "fit" judgment.

### 3. Huang & Pearce (2015), "Managing the Unknowable: The Effectiveness of Early-Stage Investor Gut Feel in Entrepreneurial Investment Decisions" — [VERIFIED]
*Administrative Science Quarterly* 60(4): 634-670. Full text fetched:
https://cpb-us-e2.wpmucdn.com/sites.uci.edu/dist/9/311/files/2015/11/HuangPearceASQAngels2015.pdf

Inductive interview study + experiment + 4-year longitudinal field test of angel investors.
Investors name their own process "gut feel":

> "... that's how I make my decisions... I just use my gut feel. I trust my gut." (p.644)

> "I don't care about the financials... the business plan... as much as I care about the
> entrepreneur. My most successful deals have come when I trust my gut feelings... when I trust
> only what my gut tells me about the entrepreneur, and filter everything else out." (p.645)

**Important nuance for the paper** (this is a partial counter-example to a pure
"can't-articulate-at-all" claim, and should be used carefully): in Study 2, investors *could*
name the two ingredients of their gut feel (business-viability data vs. perceptions of the
entrepreneur) and accurately predicted how peers would rate a case — "angel investors' criteria
are at least partially conscious" (p.652). But the deeper mechanism — *how* the two are blended
into a holistic "leap" decision — remains the unarticulated "gut feel," and business-viability
data itself turned out **not to predict outcomes at all**, while the entrepreneur-assessment
component did predict extraordinary profitability four years later (p.657, "Discussion" of Study
3) — i.e., the analytically legible half of their stated criteria is the part that doesn't work.
Also surfaces the citation **Hisrich & Jankowicz (1990)**, who call this the VC "mystery factor"
[LEAD — not independently fetched].

### 4. Dana, Dawes & Peterson (2013), "Belief in the unstructured interview: The persistence of an illusion" — [VERIFIED]
*Judgment and Decision Making* 8(5): 512-520. Full text fetched (Cambridge Core mirror):
https://www.cambridge.org/core/services/aop-cambridge-core/content/view/5BBA77932EF22EBEAA1E8020126A1925/S1930297500003612a.pdf/belief_in_the_unstructured_interview_the_persistence_of_an_illusion.pdf

Three studies. Study 1: participants interviewed a "candidate" who was actually giving
**random, meaningless answers** via a scripted response system; interviewers formed confident
impressions anyway ("sensemaking") and these impressions actively hurt prediction accuracy
("dilution"):

> "Consistent with sensemaking, participants formed interview impressions just as confidently
> after getting random responses as they did after real responses... interviews actually led
> participants to make worse predictions." (Abstract, p.512)

Opens with the well-known University of Texas Medical School (Houston) 1979 natural experiment:
50 late-admitted applicants, originally rejected on unstructured-interview impressions, showed
**no meaningful difference** in attrition, academic performance, clinical performance, or honors
vs. the initially-accepted group (citing Devaul et al. 1987) (p.512). Study 3: people prefer a
random (i.e., content-free) interview to no interview at all — confidence in the judgment
persists even when subjects are told the interview is uninformative by design.

This is the strongest "explicit process is demonstrably decoupled from what's actually being
measured" evidence in the packet — it's not just that interviewers can't articulate their
criteria, it's that the criteria are shown to attach to noise.

### 5. Kaplan, Sensoy & Strömberg (2009), "Should Investors Bet on the Jockey or the Horse?" — [VERIFIED]
*Journal of Finance* 64(1): 75-115. Full text fetched:
http://home.cerge-ei.cz/ortmann/corp_finance/Kaplan_Sensoy_Stromberg_Jockey_or_Horse.pdf

Archival study of 50 VC-backed firms from business plan through IPO (+ all 2004 IPOs as a
robustness sample). Business line/market is remarkably stable over a firm's life; management
turnover is substantial. The paper explicitly targets a VC folk-belief and finds it doesn't hold:

> "The results call into question the claim of Arthur Rock that a great management team can find
> a good opportunity even if they have to make a huge leap from the market they currently occupy
> (in Quindlen (2000))... firms that go public rarely change or make a huge leap from their
> initial business idea or line of business... At the same time, firms commonly replace their
> founders and initial managers and still be [successful]." (pp.4, 28)

Arthur Rock is one of the most quoted legendary VCs for the "bet on the jockey" doctrine; this
paper is archival, not self-report, and it cuts the other way from what VCs *say* drives them
(cf. finding #1: 95%/47% name team as most/most-important factor). Pairs cleanly with Gompers et
al. as a stated-belief-vs-revealed-pattern contrast, though note it is a *normative/predictive*
claim about what *should* matter, not a direct measurement of what the individual VC's own choices
actually tracked (that's Zacharakis & Meyer, #7 below).

### 6. McDaniel, Whetzel, Schmidt & Maurer (1994), "The Validity of Employment Interviews: A Comprehensive Review and Meta-Analysis" — [VERIFIED]
*Journal of Applied Psychology* 79(4): 599-616. Full text fetched:
https://home.ubalt.edu/tmitch/645/articles/McDanieletal1994CriterionValidityInterviewsMeta.pdf

245 coefficients / 86,311 individuals. Corrected validity: structured interviews r=.44 (raw .24,
N=12,847, k=106) vs. unstructured r=.33 (raw .18, N=9,330, k=39) against job-performance criteria
(Table, line ~515-516 of extracted text). Background/context citation (this is the "structured
beats unstructured" meta-analytic backbone Highhouse and Dana-Dawes-Peterson both build on), not
itself an articulation-gap claim — include as supporting evidence, not a headline quote.

### 7. Zacharakis & Meyer (1998), "A lack of insight: do venture capitalists really understand their own decision process?" — [SNIPPET]
*Journal of Business Venturing* 13(1): 57-76. Could not obtain full text (JSTOR/ScienceDirect/
ResearchGate all CAPTCHA'd or paywalled after ~6 attempts via direct fetch, jina.ai mirror, and
curl with browser UA). Citation-context strings recovered via scite.ai's citing-paper excerpts
(https://scite.ai/reports/a-lack-of-insight-do-L1e60l), i.e. these are *other papers'*
characterizations of Zacharakis & Meyer's finding, not the original authors' own words:

> "VCs exhibit limited introspection about their own decisions and actions (Zacharakis and Meyer,
> 1998)"

> "differences between investors' espoused criteria and their actual in-use criteria as perceived
> by entrepreneurs (Zacharakis and Meyer, 1998)"

This is exactly the title and topic the assignment names as high priority (stated vs. captured
decision weights) — worth chasing via institutional access before drafting, since right now we
only have secondhand paraphrase, not the paper's own numbers (which reportedly compare VCs'
self-reported importance rankings to policy-captured weights from a conjoint-style task).

### 8. Bastedo, Bowman, Glasener & Kelly (2018), "What Are We Talking About When We Talk About Holistic Review? Selective College Admissions and Its Effects on Low-SES Students" — [SNIPPET]
*Journal of Higher Education* 89(5): 782-805. Could not get past JSTOR/T&F paywall; citation
recovered via University of Michigan project page summary
(https://sites.marsal.umich.edu/mac/research-initiatives/holistic-admissions-practices/) and its
quoted abstract fragment:

> "Inconsistent definitions of a core admissions concept make it more difficult for the public to
> comprehend the 'black box' of college admissions." [abstract, as quoted by UMich research page]

Mixed-methods study of 311 admissions officers (open-response survey + focus groups + experimental
simulation) finding **three incompatible operational definitions** of "holistic review" in active
use — "whole file," "whole person," "whole context" — and that which definition an officer holds
changes how they treat identical low-SES applicant files in a simulation. This is exactly the
"deliberately unspecified construct" vein the assignment calls out; worth getting the actual PDF
(T&F/JSTOR) before citing definitional percentages, since only the three-way typology and the
"black box" framing line are independently confirmed here.

### 9. Mitchell Stevens (2007), *Creating a Class: College Admissions and the Education of Elites* — [SNIPPET] / mostly [LEAD]
Harvard University Press. Front matter only was retrievable (archive.org catalog page has no
in-line text; vdoc.pub served an early excerpt). One page cite recovered, and it is worth flagging
as a **complication** rather than confirming evidence:

> "officers do not evaluate applicants; they evaluate applications" (p.20, per vdoc.pub excerpt)

That line cuts *against* a strong "unstateable holistic gut judgment" reading — Stevens's early
framing emphasizes that officers work from paper files, not personal impressions. The
book's payoff chapters on committee "Decisions" and class-shaping trade-offs (where the
unarticulated-standards material is expected to live, per the assignment's brief) were not in the
retrievable excerpt. Needs library/institutional PDF or a legally purchased ebook to confirm or
disconfirm the expected quotes about class-shaping and committee agreement-without-shared-criteria.

---

## 2. Leads (named, not fetched — do not cite without verification)

- **Zacharakis & Meyer (2000)**, "The potential of actuarial decision models: Can they improve the
  venture capital investment decision?" *Journal of Business Venturing* 15(4): 323-346. Blocked by
  Cloudflare/paywall on ScienceDirect and ResearchGate. This is the assignment's other named
  high-priority VC paper (stated-vs-captured weights); still unverified.
- **Riquelme & Rickards (1992)**, "Hybrid conjoint analysis: An estimation probe in new venture
  decisions," *Journal of Business Venturing* 7(6): 505-518 — classic conjoint/policy-capturing
  study of VC decision criteria; located citation and PDF URL candidates
  (ResearchGate-hosted PDF link found) but did not fetch/verify content.
- **Hisrich & Jankowicz (1990)** — coins the VC "mystery factor" language quoted secondhand inside
  Huang & Pearce (2015); full citation and content not independently confirmed.
- **MacMillan, Siegel & Narasimha (1985)**, "Criteria used by venture capitalists to evaluate new
  venture proposals," *Journal of Business Venturing* 1(1) — the foundational VC-criteria survey
  cited inside both Zacharakis/Meyer and Huang/Pearce lineages; not directly searched this pass.
- **Kaplan, Sensoy & Strömberg's** own citations of **Quindlen (2000)** (source of the Arthur Rock
  "jockey" quote) and **Gompers & Lerner (2001)** textbook anecdotes about Tom Perkins (Kleiner
  Perkins, technology-first) vs. Don Valentine (Sequoia, market-first) investment styles — these
  are practitioner/textbook sources, not peer-reviewed, but could supply more "VCs describe an
  unstateable style" color if the paper wants practitioner voice.

## 3. Dead ends

- JSTOR PDF fetches for `stable/pdf/*` and `stable/26772134` consistently return a CAPTCHA/access
  wall to both direct WebFetch and the `r.jina.ai` proxy — no way found around this in-session.
- ResearchGate "publication" pages return HTTP 403 to WebFetch; direct-hosted author/university PDF
  mirrors (e.g., `cpb-us-e2.wpmucdn.com`, `home.cerge-ei.cz`, `home.ubalt.edu`, `asanet.org`) work
  reliably instead — worth trying that pattern first for any future fetch in this space.
- Semantic Scholar API returned HTTP 429 (rate-limited) on every attempt this session; not usable
  as a fallback here.
- WebSearch tool was already at its per-session budget cap (200/200) inherited from the parent
  session before this task began, so all discovery had to run through WebFetch on `r.jina.ai`-proxied
  DuckDuckGo HTML result pages instead of the native search tool — slower per query but functional.
- `journal.sjdm.org` (canonical JDM host) 301-redirects to `jbaron.org/journal/...`; the older
  `sjdm.org/12/...` guessed URL pattern 404s — use the `jbaron.org` mirror going forward for JDM
  papers.

---

## 4. BibTeX (ready to paste)

```bibtex
@article{gompers2020howdo,
  author  = {Gompers, Paul and Gornall, Will and Kaplan, Steven N. and Strebulaev, Ilya A.},
  title   = {How Do Venture Capitalists Make Decisions?},
  journal = {Journal of Financial Economics},
  volume  = {135},
  number  = {1},
  pages   = {169--190},
  year    = {2020},
  keywords = {domain=venture-capital; gap=stated-ne-used; type=survey},
  annote  = {VERIFIED (NBER WP22587 full text, https://www.nber.org/system/files/working_papers/w22587/w22587.pdf).
             885 VCs/681 firms. "The management team was mentioned most frequently both as an
             important factor (by 95% of VC firms) and as the most important factor (by 47% of
             VC firms)." Meanwhile only 22% use NPV, 9% use no financial metrics at all, and
             "almost half of the VCs...admit to often making gut investment decisions," while
             only 1 in 10 ever quantitatively checks their own track record. Stated determinant
             (team/gut) vs. near-absent formal-verification process, in the VCs' own words.}
}

@article{rivera2012hiring,
  author  = {Rivera, Lauren A.},
  title   = {Hiring as Cultural Matching: The Case of Elite Professional Service Firms},
  journal = {American Sociological Review},
  volume  = {77},
  number  = {6},
  pages   = {999--1022},
  year    = {2012},
  keywords = {domain=hiring; gap=defn-contested; type=interview-study},
  annote  = {VERIFIED (full PDF, https://www.asanet.org/wp-content/uploads/savvy/journals/ASR/Dec12ASRFeature.pdf).
             120 interviews + hiring-committee observation. "Fit" is a formally mandated criterion
             with no formal definition; "more than half reported it was the most important
             criterion" at interview stage, over analytical thinking and communication (p.1007).
             Manager rejecting a technically strong candidate: "He did well on the case and was
             very articulate...But I think he's too intellectual for [FIRM]...I don't think he'd
             be a good fit" (p.1010). Best single "explicit performance overridden by unstateable
             criterion" case in the hiring vein.}
}

@article{huang2015managing,
  author  = {Huang, Laura and Pearce, Jone L.},
  title   = {Managing the Unknowable: The Effectiveness of Early-Stage Investor Gut Feel in Entrepreneurial Investment Decisions},
  journal = {Administrative Science Quarterly},
  volume  = {60},
  number  = {4},
  pages   = {634--670},
  year    = {2015},
  keywords = {domain=venture-capital; gap=felt-not-stated; type=interview-study},
  annote  = {VERIFIED (full PDF, https://cpb-us-e2.wpmucdn.com/sites.uci.edu/dist/9/311/files/2015/11/HuangPearceASQAngels2015.pdf).
             Angel investors name their process "gut feel": "I just use my gut feel. I trust my
             gut" (p.644). NUANCE: investors could name the two ingredients (business-viability
             data vs. entrepreneur perception) and predict peer ratings (p.652, "at least
             partially conscious"), but business-viability data did NOT predict 4-year outcomes
             while entrepreneur-assessment did (p.657) -- the analytically legible half of their
             stated criteria is the part that fails. Cites Hisrich & Jankowicz (1990) "mystery
             factor" for VC decisions [lead, unverified].}
}

@article{dana2013belief,
  author  = {Dana, Jason and Dawes, Robyn M. and Peterson, Nathanial},
  title   = {Belief in the Unstructured Interview: The Persistence of an Illusion},
  journal = {Judgment and Decision Making},
  volume  = {8},
  number  = {5},
  pages   = {512--520},
  year    = {2013},
  keywords = {domain=hiring; gap=stated-ne-used; type=experiment},
  annote  = {VERIFIED (full PDF via Cambridge Core mirror,
             https://www.cambridge.org/core/services/aop-cambridge-core/content/view/5BBA77932EF22EBEAA1E8020126A1925/S1930297500003612a.pdf/belief_in_the_unstructured_interview_the_persistence_of_an_illusion.pdf).
             Interviewees gave RANDOM, meaningless answers via a scripted response system;
             interviewers "formed interview impressions just as confidently after getting random
             responses as they did after real responses," and the interview made predictions
             WORSE (abstract, p.512) ("sensemaking" + "dilution"). Opens with the 1979 UT Houston
             Medical School natural experiment: 50 late-admitted, initially-rejected-on-interview
             applicants showed no meaningful outcome difference from accepted peers (p.512,
             citing Devaul et al. 1987). Strongest "criteria demonstrably attach to noise" case
             in the packet.}
}

@article{kaplan2009jockey,
  author  = {Kaplan, Steven N. and Sensoy, Berk A. and Str{\"o}mberg, Per},
  title   = {Should Investors Bet on the Jockey or the Horse? Evidence from the Evolution of Firms from Early Business Plans to Public Companies},
  journal = {Journal of Finance},
  volume  = {64},
  number  = {1},
  pages   = {75--115},
  year    = {2009},
  keywords = {domain=venture-capital; gap=espoused-ne-archival; type=archival},
  annote  = {VERIFIED (full PDF, http://home.cerge-ei.cz/ortmann/corp_finance/Kaplan_Sensoy_Stromberg_Jockey_or_Horse.pdf).
             50 VC-backed firms, business-plan-to-IPO archival panel. "The results call into
             question the claim of Arthur Rock that a great management team can find a good
             opportunity even if they have to make a huge leap from the market they currently
             occupy...firms that go public rarely change or make a huge leap from their initial
             business idea" (pp.4, 28) -- business line stable, management turnover common. Cuts
             against the "bet on the jockey/team" folk doctrine that VCs themselves report holding
             (cf. gompers2020howdo: 95%/47% name team as [most] important factor). Pair these two
             for a stated-belief-vs-archival-pattern contrast; note this paper is normative/
             predictive, not a within-subject stated-vs-actual measurement (that would be
             zacharakis1998lackofinsight, still unverified).}
}

@article{mcdaniel1994validity,
  author  = {McDaniel, Michael A. and Whetzel, Deborah L. and Schmidt, Frank L. and Maurer, Steven D.},
  title   = {The Validity of Employment Interviews: A Comprehensive Review and Meta-Analysis},
  journal = {Journal of Applied Psychology},
  volume  = {79},
  number  = {4},
  pages   = {599--616},
  year    = {1994},
  keywords = {domain=hiring; gap=context; type=meta-analysis},
  annote  = {VERIFIED (full PDF, https://home.ubalt.edu/tmitch/645/articles/McDanieletal1994CriterionValidityInterviewsMeta.pdf).
             245 coefficients / 86,311 individuals. Corrected validity: structured r=.44 (raw
             .24, k=106, N=12,847) vs. unstructured r=.33 (raw .18, k=39, N=9,330) against
             job-performance criteria. Background/support citation underlying Highhouse (already
             held) and dana2013belief's motivating claims -- not itself an articulation-gap
             finding, cite as meta-analytic backbone only.}
}

@article{zacharakis1998lackofinsight,
  author  = {Zacharakis, Andrew L. and Meyer, G. Dale},
  title   = {A Lack of Insight: Do Venture Capitalists Really Understand Their Own Decision Process?},
  journal = {Journal of Business Venturing},
  volume  = {13},
  number  = {1},
  pages   = {57--76},
  year    = {1998},
  keywords = {domain=venture-capital; gap=stated-ne-captured; type=policy-capturing},
  annote  = {SNIPPET ONLY -- full text blocked (JSTOR/ScienceDirect/ResearchGate all
             CAPTCHA'd/paywalled). Citation-context strings from citing papers, via scite.ai
             (https://scite.ai/reports/a-lack-of-insight-do-L1e60l): "VCs exhibit limited
             introspection about their own decisions and actions"; "differences between
             investors' espoused criteria and their actual in-use criteria as perceived by
             entrepreneurs." This is the assignment's named high-priority stated-vs-captured
             paper -- GET INSTITUTIONAL ACCESS before quoting any number from it; nothing here
             is the authors' own words.}
}

@article{bastedo2018whatarewe,
  author  = {Bastedo, Michael N. and Bowman, Nicholas A. and Glasener, Kristen M. and Kelly, Jandi L.},
  title   = {What Are We Talking About When We Talk About Holistic Review? Selective College Admissions and Its Effects on Low-SES Students},
  journal = {Journal of Higher Education},
  volume  = {89},
  number  = {5},
  pages   = {782--805},
  year    = {2018},
  keywords = {domain=admissions; gap=defn-contested; type=mixed-methods},
  annote  = {SNIPPET ONLY -- full text paywalled (JSTOR/Taylor \& Francis). Citation and abstract
             fragment recovered via University of Michigan project page
             (https://sites.marsal.umich.edu/mac/research-initiatives/holistic-admissions-practices/):
             311 admissions officers hold three incompatible operational definitions of
             "holistic review" ("whole file," "whole person," "whole context"); "inconsistent
             definitions of a core admissions concept make it more difficult for the public to
             comprehend the 'black box' of college admissions" [quoted abstract fragment].
             GET FULL PDF before citing the disagreement rate or simulation numbers.}
}

@book{stevens2007creating,
  author    = {Stevens, Mitchell L.},
  title     = {Creating a Class: College Admissions and the Education of Elites},
  publisher = {Harvard University Press},
  year      = {2007},
  keywords  = {domain=admissions; gap=osmosis; type=ethnography},
  annote    = {SNIPPET / mostly LEAD -- only front matter retrievable this pass (archive.org
             catalog page and a partial vdoc.pub excerpt; no access to the "Decisions" chapter
             the assignment specifically flags). One confirmed page cite, offered as a
             COMPLICATION not confirmation: "officers do not evaluate applicants; they evaluate
             applications" (p.20) -- cuts against a naive gut-judgment reading. Needs
             library/institutional PDF for the class-shaping and committee-consensus-without-
             shared-criteria material the assignment expects this book to contain.}
}
```
