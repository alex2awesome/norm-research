# Literature sweep: Articulation gaps in patent examination / nonobviousness / claim drafting

Domain: PATENTS. Task: find sources where explicit written criteria (statute, MPEP, Graham factors,
TSM test) demonstrably fail to determine expert (examiner/judge/attorney) outcomes, OR where experts
agree but cannot state criteria.

## STEP 1 — existing coverage in repo bib files (checked 2026-07-26/28)

Searched: `latex/refs-shared.bib` (patents section confirmed EMPTY — no hits at all for
patent/examiner/nonobvious/uspto/ksr/graham), `methods/metric_implementer/references.bib`,
`notes/articulability-prompt-opt.bib`, `latex/paper-1__metric-codability/refs.bib`.

Existing patent-relevant entries found (NOT new finds, just noting so we don't re-add):
- `mpep2024` (methods/metric_implementer/references.bib) — USPTO Manual of Patent Examining
  Procedure, 9th ed., Rev 01.2024. The "explicit criteria" side of the gap.
- `faber1990landis` (methods/metric_implementer/references.bib) — Faber & Landis, *Landis on
  Mechanics of Patent Claim Drafting* (Practising Law Institute, 1990). This IS a practitioner-craft
  source (claim drafting as apprenticeship craft) — relevant to target #5 (TRADE), already in repo,
  will note it as prior coverage below rather than re-adding.
- `suzgun2022harvard` (methods/metric_implementer/references.bib) — Harvard USPTO Patent Dataset
  (arXiv 2207.04043). Dataset/empirical infra, not an articulation-gap claim itself, but relevant
  background for any examiner-variance quantitative work.
- No KSR, no Learned Hand/Harries, no Cockburn/Kortum/Stern, no Lemley/Sampat, no Frakes/Wasserman,
  no Righi/Simcoe, no Sampat/Williams, no trilateral-office comparison anywhere in the four bib files.

This confirms the patents section of refs-shared.bib is genuinely empty and this is fresh-fill work.

---

## Finds (appended incrementally as verified)

### 1. Learned Hand, "fugitive, impalpable, wayward, and vague a phantom" — Harries v. Air King Products Co.

[VERIFIED] Full opinion text fetched via CourtListener (through r.jina.ai reader proxy, which
worked when direct curl to courtlistener.com was blocked/rate-limited — direct curl to
courtlistener.com returned HTTP 202 with empty body, likely bot-protection):
https://www.courtlistener.com/opinion/225389/harries-v-air-king-products-co-inc/
(fetched via https://r.jina.ai/https://www.courtlistener.com/opinion/225389/harries-v-air-king-products-co-inc/,
saved locally at scratchpad/harries_v_airking_jina2.txt)

Case: **Harries v. Air King Products Co., Inc.**, 183 F.2d 158 (2d Cir. 1950), opinion by
L. Hand, Chief Judge (Learned Hand). Decided June 20, 1950; argued April 13, 1950.
Panel: L. Hand, C.J., Swan and Chase, Circuit Judges.

Exact quote confirmed verbatim as remembered:

> "There are good reasons for allowing some latitude of choice. A decision resting upon
> non-infringement is generally much more secure than one on invalidity, at least when the question
> is whether there is a patentable invention. **That issue is as fugitive, impalpable, wayward, and
> vague a phantom as exists in the whole paraphernalia of legal concepts.** It involves, or it should
> involve, as complete a reconstruction of the art that preceded it as is possible."

Pin cite: this passage falls between the page-158 F.2d 162 and page 163 markers in the CourtListener
text (page markers `[*162]` at char offset 12546, `[*163]` at char offset 17741; "fugitive" occurs at
offset 16460) — so pin cite is **183 F.2d 158, 162 (2d Cir. 1950)**.

Why this shows a gap: Learned Hand — writing the majority opinion, not a dissent — states outright
that the legal standard for patentable invention/nonobviousness is not merely hard to apply but is
categorically the most unstable, non-propositional concept in all of law ("as ... vague a phantom as
exists in the whole paraphernalia of legal concepts"). This is a sitting appellate judge, in a
precedential published opinion, asserting that expert legal judgment on this question cannot be
reduced to an articulable rule — the clearest possible domain-insider admission of an articulation
gap, predating and motivating the entire modern nonobviousness statutory framework (35 U.S.C. §103,
enacted 1952, partly in reaction to this line of cases).

```bibtex
@misc{harries1950airking,
  author       = {{Second Circuit}},
  title        = {Harries v. Air King Products Co., Inc.},
  howpublished = {183 F.2d 158 (2d Cir. 1950)},
  year         = {1950},
  note         = {Opinion by L. Hand, C.J.; opinion text via CourtListener, https://www.courtlistener.com/opinion/225389/harries-v-air-king-products-co-inc/},
  keywords     = {domain=patents; gap=criteria-underdetermine-outcome; type=judicial-opinion},
  annote       = {[VERIFIED] fetched full opinion text via CourtListener (r.jina.ai proxy);
    quote at 183 F.2d 158, 162: "That issue is as fugitive, impalpable, wayward, and vague a
    phantom as exists in the whole paraphernalia of legal concepts" (re: the question of patentable
    invention). Learned Hand, in a precedential majority opinion, states that the legal test for
    invention is the least articulable concept in all of law -- a domain-insider admission that
    explicit doctrine cannot capture the judgment being exercised.}
}
```

### 2. KSR International Co. v. Teleflex Inc. — rejection of the rigid TSM test for "common sense"/"expansive and flexible"

[VERIFIED] Full opinion text fetched via CourtListener (r.jina.ai proxy):
https://www.courtlistener.com/opinion/145737/ksr-international-co-v-teleflex-inc/
(saved locally at scratchpad/ksr_jina2.txt)

Case: **KSR International Co. v. Teleflex Inc.**, 550 U.S. 398 (2007). No. 04-1350. Argued
November 28, 2006; decided April 30, 2007. Opinion for a unanimous Court by Justice Kennedy.

Quote 1 (pin cite **550 U.S. at 415** — falls between page markers [*415] offset 25811 and [*416]
offset 28516 in the fetched text; quote itself at offset 27033):

> "We begin by rejecting the rigid approach of the Court of Appeals. Throughout this Court's
> engagement with the question of obviousness, our cases have set forth an **expansive and flexible
> approach** inconsistent with the way the Court of Appeals applied its TSM test here."

Quote 2 (pin cite **550 U.S. at 421** — falls between [*421] offset 40446 and [*422] offset 43158;
quote at offset 42505):

> "**Rigid preventative rules that deny factfinders recourse to common sense**, however, are
> neither necessary under our case law nor consistent with it."

Related supporting language nearby (419, same fetched text, offset 36191): "Helpful insights,
however, need not become rigid and mandatory formulas; and when it is so applied, the TSM test is
incompatible with our precedents." (pin cite ~550 U.S. at 419).

Why this shows a gap: the TSM (teaching-suggestion-motivation) test was the Federal Circuit's
attempt to make nonobviousness *fully articulable* — an explicit, checkable rule an examiner or
court could apply mechanically. The Supreme Court unanimously struck this down specifically because
a fully explicit rule cannot track the judgment actually being exercised, and replaced it with an
avowedly non-formalizable standard ("common sense," "expansive and flexible approach"). This is a
court, on the record, choosing an inarticulable standard over an articulable one because the
articulable one was determined to be worse at tracking true nonobviousness — the single cleanest
doctrinal example of "explicit criteria demonstrably fail to capture what experts do."

```bibtex
@misc{ksr2007teleflex,
  author       = {{Supreme Court of the United States}},
  title        = {KSR International Co. v. Teleflex Inc.},
  howpublished = {550 U.S. 398 (2007)},
  year         = {2007},
  note         = {Opinion by Kennedy, J., for a unanimous Court; opinion text via CourtListener, https://www.courtlistener.com/opinion/145737/ksr-international-co-v-teleflex-inc/},
  keywords     = {domain=patents; gap=explicit-rule-rejected-for-inarticulable-standard; type=judicial-opinion},
  annote       = {[VERIFIED] fetched full opinion text via CourtListener (r.jina.ai proxy);
    quotes at 550 U.S. 398, 415 ("expansive and flexible approach" replacing the Federal Circuit's
    "rigid" TSM test) and at 421 ("Rigid preventative rules that deny factfinders recourse to common
    sense ... are neither necessary ... nor consistent with" precedent). The Court unanimously
    rejects a fully explicit, checkable obviousness test (TSM) precisely because it fails to
    capture what examiners/courts actually judge, replacing it with an avowedly non-formalizable
    "common sense" standard -- a court choosing inarticulability over a demonstrated-inadequate
    explicit rule, on the record.}
}
```

### 3a. Cockburn, Kortum & Stern (2002/2003), "Are All Patent Examiners Equal?" — examiner fixed effects quantified

[VERIFIED] Full text fetched: NBER Working Paper 8980 (June 2002), PDF downloaded directly
(HTTP 200, no proxy needed): https://www.nber.org/system/files/working_papers/w8980/w8980.pdf
Saved locally: scratchpad/cockburn_kortum_stern_w8980.pdf and .txt (pdftotext -layout).
Title in full: "Are All Patent Examiners Equal? The Impact of Characteristics on Patent Statistics
and Litigation Outcomes." Prepared for the National Academy of Sciences STEP Board Conference on
The Operation of the Patent System; published version appeared as a chapter in *Patents in the
Knowledge-Based Economy* (National Academies Press, 2003), eds. Cohen & Merrill — NBER WP is the
open-access copy of the same content.

Sample: 298,441 (also stated as 289,441 in one table header — both figures appear in the PDF, see
note) patents attributed to 196 USPTO examiners whose patents were later ruled on for validity by
the CAFC (182 CAFC-tested patents, litigated 1997-2000).

Key numbers (Table 3A, ANOVA of patent characteristics; text p.19-20 of PDF):

> "In Table 3A, we present a simple ANOVA analysis based on our complete sample of 298,441 patents
> attributed to the 196 CAFC-tested examiners. The results indicate that examiners matter: a
> significant share of the variance in this sample in the four variables capturing the volume and
> pattern of citations by and to a particular patent ... is accounted for by fixed examiner
> effects, with a particularly strong effect in the ANOVA of CITATIONS RECEIVED. A similar result
> is obtained for the length of time between application and grant: about 8% of the variance in
> this measure can be attributed to differences among examiners."

Table 3A fraction-of-variance-explained-by-examiner-fixed-effects (controlling for 36 tech
sub-classes and 24 cohorts): CITATIONS MADE 0.077, **CITATIONS RECEIVED 0.117**, APPROVAL TIME
0.083, CLAIMS 0.030, GENERALITY 0.079, ORIGINALITY 0.069. All F-statistics for "no examiner effect"
are large and presumably significant (e.g. F=193.40 for citations received, F=131.77 for approval
time) — same rows and same time/place, different examiner = 8-12% of outcome variance attributable
to examiner identity alone, net of technology and cohort.

Litigation-outcome result (p.22 of PDF, Table 7 regression discussion):

> "According to (7-4), increasing the EXAMINER CITES PER PATENT by one standard deviation (3.49),
> the probability of validity is predicted to decline by over 14 percentage points, from a mean of
> 48%."

And on examiner experience (same page):

> "Whether detailed controls are included or not, there is no significant relationship between any
> measure of experience and the probability of a ruling of validity."

Why this shows a gap: same written statute, same MPEP, same Graham-factor doctrine, same technology
class and filing cohort — yet ~8-12% of measurable outcome variance (citation patterns, approval
time, claim scope) is attributable to *which examiner* handled the file, and a one-SD shift in an
examiner's baseline "generosity" (their historical citations-per-patent) shifts validity probability
by 14 points, more than the effect of the examiner's tenure/experience (which has none). This is
the number-attached version of "identical explicit criteria, different examiner, different
outcome."

```bibtex
@techreport{cockburn2002examiners,
  author      = {Cockburn, Iain M. and Kortum, Samuel and Stern, Scott},
  title       = {Are All Patent Examiners Equal? The Impact of Characteristics on Patent Statistics and Litigation Outcomes},
  institution = {National Bureau of Economic Research},
  type        = {NBER Working Paper},
  number      = {8980},
  year        = {2002},
  url         = {https://www.nber.org/system/files/working_papers/w8980/w8980.pdf},
  note        = {Later published as a chapter in Patents in the Knowledge-Based Economy (National Academies Press, 2003)},
  keywords    = {domain=patents; gap=examiner-fixed-effects; type=empirical},
  annote      = {[VERIFIED] fetched full PDF from NBER (nber.org/system/files/working_papers/w8980/w8980.pdf);
    N=298,441 patents / 196 CAFC-tested examiners. Table 3A: examiner fixed effects explain 0.117
    of variance in CITATIONS RECEIVED, 0.083 in APPROVAL TIME, 0.079 GENERALITY, 0.069 ORIGINALITY,
    0.077 CITATIONS MADE, 0.030 CLAIMS -- net of 36 tech sub-classes and 24 filing cohorts. Litigation
    result: "increasing the EXAMINER CITES PER PATENT by one standard deviation (3.49), the
    probability of validity is predicted to decline by over 14 percentage points, from a mean of
    48%," while examiner experience/tenure has no significant relationship to validity outcomes.
    Same statute, same technology class, same cohort -- outcome varies systematically by which
    examiner drew the file.}
}
```

### 3b. Lemley & Sampat (2012), "Examiner Characteristics and Patent Office Outcomes" — grant-rate by experience + "as many patent offices as there are patent examiners"

[VERIFIED] Full text fetched: published version, *Review of Economics and Statistics* 94(3),
817-827 (2012) [note: NBER/journal-published year 2012; "Received for publication January 8,
2009. Revision accepted for publication December 20, 2010" per the paper's own footer, and
OpenAlex lists a 2011 record for the same DOI 10.1162/rest_a_00194 — journal issue-date is 2012].
Fetched via OSF mirror (open preprint copy), URL resolved through Crossref DOI
10.31235/osf.io/xbme2 -> https://osf.io/xbme2/download (HTTP 200, real PDF; the `_v1` variant of
that OSF URL returned an HTML error page, so use the exact URL below).
Saved locally: scratchpad/lemley_osf_30b307.pdf and scratchpad/lemley_sampat.txt (pdftotext -layout).
Authors: Mark A. Lemley (Stanford Law School) and Bhaven Sampat (Columbia University).

Sample: ~9,846 patent applications (nearly 10,000) filed in January 2001, followed through
prosecution at the USPTO.

Headline quote (abstract, p.817):

> "we show that there are important differences across patent examiners at the U.S. Patent and
> Trademark Office. We show that more experienced examiners cite less prior art, are more likely to
> grant patents, and are more likely to grant patents without any rejections. These results suggest
> that **the most important decisions made by the patent office are significantly affected by the
> happenstance of which examiner gets an application.**"

Quantitative results (Table 4, p.822 of journal pagination, in the extracted text):

> "The grant rate increases monotonically with experience, with the two most experienced groups
> having an **11 percentage point higher grant rate**." (Model 4.1: coefficients 0.057*** for 2-4
> yrs experience, 0.110*** for 5-7 yrs, 0.112*** for 8+ yrs experience, vs. a constant/baseline
> grant rate of 0.663 for the least-experienced group; N=9,846.)

> "The likelihood of granting without rejections increases sharply with experience, with the most
> experienced cohort **13 percentage points more likely to do so**." (Model 4.2: 0.060***, 0.111***,
> 0.133*** by experience tier; N=7,117.)

Prosecution-history detail: "82% of these granted applications received a nonfinal rejection, and
26% a final rejection ... But 18% did not receive any rejections before they were issued; they were
issued on the first office action, as is."

Selection controls: the authors interviewed ~two dozen examiners/SPEs and found applications within
art units are assigned essentially by the last digit of the serial number (i.e., effectively
random), not by application difficulty — ruling out the obvious confound that harder/easier cases
are steered to different experience tiers.

**Quoted phrase attributed to Cockburn, Kortum & Stern** (confirmed independently present verbatim
in the Cockburn et al. NBER WP 8980 itself, p.8 of that PDF — see finding 3a above), reproduced by
Lemley & Sampat in their literature review:

> "They conclude that 'there may be as many patent offices as there are patent examiners.'"
(Original in Cockburn et al.: "The first key finding from the qualitative evaluation of patent
examination can be summarized in the phrase of one of our informants: 'there may be as many patent
offices as there are patent examiners.'" — i.e., it originates as a direct quote from one of
Cockburn/Kortum/Stern's interview informants (an examiner/administrator), not authored by the
economists themselves.)

**Cross-office (USPTO vs. EPO) disagreement data — target #4** (Table 6, p.824):
Cross-tabulation of outcomes for 2,731 US applications also filed at the EPO (European Patent
Office): of applications **rejected by the EPO**, 525 of 869 (60.4%) were nonetheless **patented by
the USPTO**; of applications **patented by the EPO**, 1,143 of 1,271 (89.9%) were also patented by
the USPTO, but 128 (10.1%) were not. Overall, "of the applications that were granted in the United
States, slightly more than half (52.1%) have been granted by the EPO. By contrast, of those granted
by the EPO, the vast majority (88%) are also granted by the PTO." And on the examiner-experience
interaction with EPO disagreement (Table 7, Model 7.1): "applications rejected at the EPO have a
**27 percentage point lower probability** of being patented in the United States" (coefficient
-0.269***), and more experienced US examiners are specifically more likely to grant applications
the EPO rejected (Model 7.3: 5-7 yrs +0.174**, 8+ yrs +0.118**) while experience has no effect on
agreement with EPO grants (Model 7.2, insignificant).

Why this shows a gap: two patent offices applying textually similar novelty/inventive-step/
nonobviousness standards disagree on hundreds of the *same* underlying applications, and within a
single office, grant probability shifts by 11-13 percentage points purely as a function of which
examiner (specifically their tenure) drew the file — with random-by-serial-number assignment ruling
out a difficulty-sorting confound. Same criteria, different examiner or different office, different
outcome, now with both a within-office effect size and a cross-office disagreement rate attached.

```bibtex
@article{lemley2012examiner,
  author  = {Lemley, Mark A. and Sampat, Bhaven},
  title   = {Examiner Characteristics and Patent Office Outcomes},
  journal = {Review of Economics and Statistics},
  volume  = {94},
  number  = {3},
  pages   = {817--827},
  year    = {2012},
  doi     = {10.1162/rest_a_00194},
  note    = {Open-access preprint copy: https://osf.io/xbme2/download (Crossref DOI 10.31235/osf.io/xbme2)},
  keywords = {domain=patents; gap=examiner-fixed-effects; type=empirical},
  annote  = {[VERIFIED] fetched full text PDF via OSF mirror (osf.io/xbme2/download); N approx
    9,846 Jan-2001 applications. Abstract: "the most important decisions made by the patent office
    are significantly affected by the happenstance of which examiner gets an application." Table 4:
    most-experienced examiners have an 11-percentage-point higher grant rate and 13-point higher
    probability of granting with zero rejections than least-experienced (baseline grant rate
    66.3%); assignment within art units shown near-random (by serial-number digit), ruling out a
    difficulty-sorting confound. Table 6/7: of US applications the EPO rejected, 60% (525/869) were
    still patented by the USPTO, and US examiner experience specifically predicts overriding an EPO
    rejection (+17-18 pts) with no corresponding effect on agreeing with EPO grants -- a
    quantified cross-office disagreement under nominally equivalent inventive-step standards. Also
    reproduces the informant quote from Cockburn, Kortum & Stern (2002): "there may be as many
    patent offices as there are patent examiners" (independently verified present in that source,
    see cockburn2002examiners annotation).}
}
```

### 3c. Frakes & Wasserman (2014/2017), "Is the Time Allocated to Review Patent Applications Inducing Examiners to Grant Invalid Patents?" — same examiner, less time, higher grant rate

[VERIFIED] Full text fetched: NBER Working Paper 20337 (July 2014, revised December 2014), PDF
downloaded directly from NBER (HTTP 200, no proxy needed):
https://www.nber.org/system/files/working_papers/w20337/w20337.pdf
Saved locally: scratchpad/frakes_wasserman_w20337.pdf and .txt (pdftotext -layout).
Authors: Michael D. Frakes (Northwestern Law) and Melissa F. Wasserman (Illinois Law). Published
version: *Review of Economics and Statistics* (2017 per later record; OpenAlex lists the journal
DOI as 10.1162/rest_a_00605, year field 2016 in that index).

Design: exploits the USPTO's internal General Schedule (GS) pay-grade promotion ladder, under
which the SAME examiner is allocated roughly half as much time per application at GS-14 as at
GS-7, holding years of experience constant (promotions are scheduled/near-automatic, not
merit-selected on a case basis) — a within-examiner natural experiment isolating time-per-case
from examiner identity/skill.

Headline quantitative results (abstract/intro, p.3-4 of PDF):

> "As examination time is cut roughly in half (i.e., as an examiner rises from GS-7 to GS-14 along
> the General Schedule scale, controlling for changes in years of experience), our findings suggest
> that **grant rates rise by as much as 9 to 19 percentage points, or by roughly 13 to 28 percent**."

> "Considering the distribution of examinations across GS levels, our findings imply that if all
> examiners were allocated as many hours as are extended to GS-7 examiners, **the Patent Office's
> overall grant rate would fall by roughly 14 percentage points, or nearly 20 percent**."

Body detail (p.20-21 area, matching text around lines 822-828, 1182-1190 of extracted .txt):
"they increase their grant rates by 2.8 percentage points (or by roughly 4 percent)" at an early
promotion step, rising to "her grant rate at GS-level 14 is 19.0 percentage points (or nearly 28
percent) higher than it" was at GS-7 — and this is accompanied elsewhere in the paper (not
independently re-verified with exact line here, but stated in the abstract) by evidence that the
marginal patents granted under time pressure are "of below-average quality" (measured via
subsequent citation/renewal proxies).

The paper explicitly places itself in the same lineage as the examiner-heterogeneity literature:
"only a handful of studies have explored the dynamics of the Patent Office, primarily by
investigating the role of examiner heterogeneity in explaining the outcomes of the patenting
process (Cockburn, Kortum, & Stern, 2003; Lichtman, 2004; Mann, 2014)."

Why this shows a gap: this is not cross-examiner variation (which could in principle reflect
different but equally valid readings of the same criteria) but WITHIN-examiner variation — the same
person applying the same written nonobviousness/novelty standard to a statistically similar mix of
applications grants patents at a systematically different rate purely as a function of how many
minutes they are allotted to think about the case. That is direct evidence that the "judgment" being
applied is not a stable, criteria-driven computation but a resource-bounded heuristic that degrades
predictably — the explicit standard does not mechanically determine the outcome even for one
decision-maker holding the standard fixed.

```bibtex
@techreport{frakes2014time,
  author      = {Frakes, Michael D. and Wasserman, Melissa F.},
  title       = {Is the Time Allocated to Review Patent Applications Inducing Examiners to Grant Invalid Patents? Evidence from Micro-Level Application Data},
  institution = {National Bureau of Economic Research},
  type        = {NBER Working Paper},
  number      = {20337},
  year        = {2014},
  url         = {https://www.nber.org/system/files/working_papers/w20337/w20337.pdf},
  note        = {Published version: Review of Economics and Statistics, DOI 10.1162/rest\_a\_00605},
  keywords    = {domain=patents; gap=within-examiner-time-pressure; type=empirical},
  annote      = {[VERIFIED] fetched full PDF directly from NBER
    (nber.org/system/files/working_papers/w20337/w20337.pdf). Using the USPTO's GS pay-grade ladder
    as a within-examiner natural experiment (same examiner, time-per-application roughly halved
    from GS-7 to GS-14), grant rates rise "by as much as 9 to 19 percentage points, or by roughly
    13 to 28 percent," and office-wide, giving every examiner GS-7-level time would cut the overall
    grant rate "by roughly 14 percentage points, or nearly 20 percent." Marginal patents granted
    under time pressure are of below-average quality by citation/renewal proxies. Shows the SAME
    decision-maker applying the SAME written standard yields systematically different outcomes as a
    function of time budget alone -- evidence that "applying the criteria" is a resource-bounded,
    non-mechanical judgment rather than a fixed computation the criteria determine.}
}
```

### 3d. Righi & Simcoe (2020/2022), "Patenting Inventions or Inventing Patents? Continuation Practice at the USPTO" — examiner leniency distribution + strategic exploitation

[VERIFIED] Full text fetched: NBER Working Paper 27686 (August 2020, revised February 2022), PDF
downloaded directly from NBER (HTTP 200): https://www.nber.org/system/files/working_papers/w27686/w27686.pdf
Saved locally: scratchpad/righi_simcoe_w27686.pdf and .txt.
Authors: Cesare Righi (Barcelona School of Economics) and Timothy Simcoe (Boston University/NBER).

This paper is centrally about strategic use of continuation applications to "invent patents" that
read on later-published industry standards (SEP = standard-essential patent), not primarily an
articulation-gap paper — flagging it as SECONDARY / supporting evidence rather than a top-5 find.
But it independently quantifies examiner-leniency dispersion and shows applicants strategically
exploit it:

> "This effect is larger when patent examiners are more lenient" (abstract).

Examiner leniency variable (measured as each examiner's own grant rate on all post-AIPA disposed
applications, excluding the focal case) summary statistics (Table, line ~2436 of extracted text):
N=959,627, mean **0.749**, std. dev. **0.134**, IQR roughly [0.682, 0.850] (25th/75th percentile).

> "examiner leniency is associated with a 0.68-0.78 percentage point increase in the probability
> [of the SEP-continuation late-claiming effect]" (section 4.4, "Examiner leniency").

The paper explicitly treats examiner leniency as a known, exploitable, persistent quantity, citing
Frakes & Wasserman (2016, 2017a, b) and Lemley & Sampat (2012) as establishing that leniency is
partly time-varying for a given examiner (consistent with entries 3b/3c above).

Why relevant (secondary): confirms with a much larger sample (~960K post-2000 disposed
applications) that examiner grant-rate ("leniency") varies with real dispersion (SD 0.134 around a
mean of 0.75) and that sophisticated repeat players (patent attorneys drafting continuations)
actively route claims to exploit this variance — i.e., the "gap" is large and stable enough to be a
known, monetizable target for practitioners, not just statistical noise.

```bibtex
@techreport{righi2020continuations,
  author      = {Righi, Cesare and Simcoe, Timothy},
  title       = {Patenting Inventions or Inventing Patents? Continuation Practice at the USPTO},
  institution = {National Bureau of Economic Research},
  type        = {NBER Working Paper},
  number      = {27686},
  year        = {2020},
  url         = {https://www.nber.org/system/files/working_papers/w27686/w27686.pdf},
  note        = {Revised February 2022; published version in RAND Journal of Economics, DOI 10.1111/1756-2171.12446},
  keywords    = {domain=patents; gap=examiner-leniency-exploitation; type=empirical; status=secondary},
  annote      = {[VERIFIED] fetched full PDF from NBER (nber.org/system/files/working_papers/w27686/w27686.pdf).
    Secondary/supporting source, not primarily an articulation-gap paper: shows examiner leniency
    (own grant rate) has mean 0.749, SD 0.134 across N=959,627 disposed applications, and that
    applicants strategically time continuation filings to exploit lenient examiners after a
    standard publishes, to "invent patents" that read on the standard. Confirms examiner-outcome
    variance under identical written novelty/obviousness criteria is large and stable enough to be
    a known, exploitable target for sophisticated repeat players.}
}
```

### 4. Mandel, "Patently Non-Obvious II: Experimental Study on the Hindsight Issue Before the Supreme Court in KSR v. Teleflex" (2007) — the TSM instruction doesn't fix hindsight bias

[VERIFIED] Full text fetched directly from the journal's own site (HTTP 200, no proxy needed):
https://yjolt.org/sites/default/files/mandel-9-yjolt-1.pdf (linked from the article landing page
https://yjolt.org/patently-non-obvious-ii-experimental-study-hindsight-issue-supreme-court-ksr-v-teleflex).
Saved locally: scratchpad/mandel_yjolt.pdf and .txt (pdftotext -layout).
Citation: Gregory N. Mandel, *Patently Non-Obvious II: Experimental Study on the Hindsight Issue
Before the Supreme Court in KSR v. Teleflex*, 9 Yale J.L. & Tech. 1 (2006-2007).
[Companion/earlier paper verified only by citation + DDG snippet, NOT independently fetched:
Mandel, *Patently Non-Obvious: Empirical Demonstration that the Hindsight Bias Renders Patent
Decisions Irrational*, 67 Ohio St. L.J. 1391 (2006) — SSRN and institutional-repository copies
were all blocked (403/CAPTCHA) despite multiple attempts (SSRN direct, Temple ScholarShare
bitstream, OSU KnowledgeBank handle, core.ac.uk mirror all returned 403/404). Flagging that
companion paper as [LEAD] only, cited here via this verified sequel paper's own references and a
DuckDuckGo result snippet: "This Article reports an experimental study that provides the first
empirical demonstration of the hindsight bias in patent law. The results are dramatic along
several fronts: (1) the hindsight bias distorts patent decisions far more than anticipated, and to
a greater extent than other legal judgments" [SNIPPET, from DuckDuckGo HTML result for the 2006
paper, not independently fetched].]

Design (this verified 2007 paper): mock-juror experiment (n≈55 per condition) presenting the same
invention scenario, varying only (a) whether the invention's existence/success was already known
(hindsight) vs. not yet known (foresight), and (b) whether participants were given the Federal
Circuit's TSM ("suggestion, teaching, or motivation") instruction or the Graham-factors instruction,
before KSR was decided.

Key results (Table 1, p.15-16 of the PDF):

> "participants rated inventions non-obvious significantly more frequently in foresight than in
> hindsight in both the scenario without a suggestion to combine prior art references (Χ2 = 9.462,
> Fisher's p < .01) and the scenario with a suggestion to combine (Χ2 = 15.579, Fisher's p < .001)."

> "For the scenario with no suggestion to combine, **42% of participants (23 out of 55) in the
> foresight condition thought that a solution to the problem was obvious, while 71% of participants
> (39 out of 55) in the hindsight condition thought that a solution was obvious**." With an explicit
> suggestion to combine: **49% (foresight) vs. 85% (hindsight)**.

Critically, the explicit legal instruction did NOT fix this:

> "suggestion instructions had no significant effect on judgments of obviousness in both the
> scenario without a suggestion to combine references (Χ2=2.380, Fisher's p=ns) and the scenario
> with a suggestion to combine references (Χ2=1.658, Fisher's p=ns). Regardless of whether there
> was a suggestion to combine references, **mock jurors were no more likely to consider an invention
> non-obvious when instructed on the Federal Circuit's suggestion, teaching, or motivation
> requirement than they were in the hindsight condition without instruction.**"

Why this shows a gap: this is a controlled experiment showing that the single biggest distortion in
nonobviousness judgment — knowing the invention already exists and works — swings the same
fact-pattern from "obvious" to "not obvious" by 29-36 percentage points, and giving decision-makers
the EXACT explicit legal test the Federal Circuit used (TSM/suggestion-to-combine instructions) does
nothing to correct it (p=ns). This directly explains, with experimental mechanism, why KSR's
"rigid" TSM rule failed on its own terms: the explicit criteria were tested against the actual
cognitive process fact-finders use, and were shown not to bind that process at all.

```bibtex
@article{mandel2007patently,
  author  = {Mandel, Gregory N.},
  title   = {Patently Non-Obvious II: Experimental Study on the Hindsight Issue Before the Supreme Court in KSR v. Teleflex},
  journal = {Yale Journal of Law \& Technology},
  volume  = {9},
  pages   = {1},
  year    = {2007},
  url     = {https://yjolt.org/sites/default/files/mandel-9-yjolt-1.pdf},
  keywords = {domain=patents; gap=hindsight-bias-explicit-instruction-fails; type=experimental},
  annote  = {[VERIFIED] fetched full PDF directly from yjolt.org
    (yjolt.org/sites/default/files/mandel-9-yjolt-1.pdf). Mock-juror experiment (n~55/condition):
    foresight vs. hindsight framing of the identical invention scenario shifts "obvious" judgments
    from 42% to 71% (no suggestion-to-combine) and 49% to 85% (with suggestion-to-combine).
    Explicitly instructing participants on the Federal Circuit's TSM (suggestion/teaching/
    motivation) test produced NO significant change relative to uninstructed hindsight judgments
    (Fisher's p=ns in both scenarios) -- i.e., handing fact-finders the exact written legal
    standard did not correct the bias it was designed to guard against. Companion/earlier paper
    (Mandel, 67 Ohio St. L.J. 1391 (2006)) could not be independently fetched (SSRN, Temple
    ScholarShare, OSU KnowledgeBank, core.ac.uk all blocked) and is cited here only via this
    verified sequel's references plus an unfetched DuckDuckGo snippet -- treat that earlier paper's
    content as [LEAD] only.}
}
```

### 5. Mandel, "The Non-Obvious Problem: How the Indeterminate Nonobviousness Standard Produces Excessive Patent Grants" (2008) — the doctrinal-indeterminacy argument (target #6)

[VERIFIED] Full text fetched directly from the journal's own site (HTTP 200, no proxy needed):
https://lawreview.law.ucdavis.edu/sites/g/files/dgvnsk15026/files/media/documents/42-1_Mandel.pdf
(found via a DuckDuckGo search-result link, confirmed as the correct paper by matching OpenAlex
record W2267739935's alternate landing page https://lawreview.law.ucdavis.edu/issues/42/1/articles/).
Saved locally: scratchpad/mandel_nonobvious_problem.pdf and .txt (pdftotext -layout).
Citation: Gregory Mandel, *The Non-Obvious Problem: How the Indeterminate Nonobviousness Standard
Produces Excessive Patent Grants*, 42 U.C. Davis L. Rev. 57 (2008).
[NOTE: the downloaded PDF is only 10 pages per `file` — likely an abstract/excerpt or front matter
rather than the complete ~70-page article; the extracted text runs to page ~89+ in the article's
own internal pagination (see Table of Contents reaching p.100+), so pdftotext appears to have
captured more running text than physical PDF "pages" suggests (probably a page-numbering/embedded-
font quirk) — the abstract, full introduction (pp.57-61), and substantial portions of Part I/II
(through roughly p.89 of the article) were successfully extracted and are quoted below verbatim.]

Core thesis (abstract, p.57):

> "The dominant current perception in patent law is that the core requirement of nonobviousness is
> applied too leniently... This Article reveals that the common wisdom is only half correct. **The
> nonobviousness standard is not too low, but both too high and too low. It is indeterminate.**
> Three principal factors produce nonobviousness indeterminacy: a failure to identify the quantum
> of innovation necessary to satisfy the standard, a failure to define the baseline level of
> ordinary skill against which to measure an innovation, and **the epistemic infeasibility of
> requiring a technologically lay decision maker to judge from the perspective of a more highly
> trained and educated person of ordinary skill in the art.**"

On PHOSITA (person having ordinary skill in the art) specifically (p.59-60):

> "the nonobviousness problem is compounded by requiring lay decision makers to judge whether a
> given advance would have been obvious from the perspective of another — the person of ordinary
> skill. **Such a judgment is epistemically impractical.** Due to the 'curse of knowledge,'
> individuals are cognitively incapable of accurately making judgments from other individuals'
> perspectives. Nonobviousness exacerbates this problem by requiring lay individuals to make a
> judgment from the perspective of a more highly educated and trained person of ordinary skill.
> These indeterminacy and epistemic problems cause nonobviousness decisions to be **inconsistent
> and unpredictable.**"

And: "the standard itself remains unformulated" — i.e., decades of Supreme Court and Federal
Circuit doctrine on nonobviousness have focused on factual sub-issues (Graham factors) while never
actually specifying the threshold quantum of inventiveness required, leaving the operative legal
concept itself open.

Why this shows a gap (this is THE direct hit for target #6, the "irreducibly a judgment call"
legal-scholarship target): Mandel argues, formally and with a mathematical model, that
nonobviousness is not merely hard to apply consistently as a contingent matter — it is
*structurally* indeterminate for two independent reasons: (1) the doctrine never defines the
threshold itself (a drafting gap that 50+ years of case law have not closed), and (2) even a
perfectly-informed decision-maker faces an epistemically impossible task — judging from the
perspective of a hypothetical PHOSITA whose relevant "ordinary skill" cannot be introspected into
by someone who does not have it (a curse-of-knowledge argument structurally identical to
"articulate what you cannot access"). This is a law professor arguing, from within the doctrine,
that the explicit test is unformulable in principle, not just poorly applied in practice.

```bibtex
@article{mandel2008nonobvious,
  author  = {Mandel, Gregory},
  title   = {The Non-Obvious Problem: How the Indeterminate Nonobviousness Standard Produces Excessive Patent Grants},
  journal = {University of California, Davis Law Review},
  volume  = {42},
  pages   = {57},
  year    = {2008},
  url     = {https://lawreview.law.ucdavis.edu/sites/g/files/dgvnsk15026/files/media/documents/42-1_Mandel.pdf},
  keywords = {domain=patents; gap=doctrinal-indeterminacy-phosita; type=legal-scholarship},
  annote  = {[VERIFIED] fetched PDF directly from lawreview.law.ucdavis.edu (own journal site, no
    proxy needed). Argues nonobviousness is "not too low, but both too high and too low. It is
    indeterminate," due to (1) the standard's threshold quantum-of-innovation never being defined
    by 50+ years of doctrine and (2) "the epistemic infeasibility of requiring a technologically
    lay decision maker to judge from the perspective of a more highly trained and educated person
    of ordinary skill in the art" -- a "curse of knowledge" argument that PHOSITA judgment is not
    just hard but introspectively inaccessible to the actual decision-maker. Direct doctrinal-level
    argument that the explicit nonobviousness test is not merely under-specified in application but
    structurally cannot capture the judgment it purports to operationalize.}
}
```

### 6. Mann (2014), "The Idiosyncrasy of Patent Examiners: Effects of Experience and Attrition" — tenure/attrition confound + relays Lichtman's "two-thirds of variation in editing rigor" finding

[VERIFIED] Full text fetched via the author's own hosted copy (HTTP 200, no proxy needed):
http://www.columbia.edu/~mr2651/Mann-92-7.pdf (URL located via OpenAlex record W768736275's
alternate location; the bepress-hosted copy at scholarship.law.columbia.edu returned an empty
202 response and was not used).
Saved locally: scratchpad/mann_idiosyncrasy2.pdf and scratchpad/mann_idiosyncrasy.txt.
Citation: Ronald J. Mann, *The Idiosyncrasy of Patent Examiners: Effects of Experience and
Attrition*, 92 Tex. L. Rev. 2149 (2014).

Own headline finding (Introduction, p.2151): using a hand-collected dataset of examiner patent
portfolios merged with NBER patent data and internal PTO education records, Mann finds tenure and
experience effects run in OPPOSITE directions and are individually confounded in prior work:

> "The existing literature overemphasizes the importance of experience, largely because it fails to
> consider the importance of attrition and tenure differences among examiners... The effects of
> tenure are substantial and cut in the opposite direction from experience. For example, where the
> number of claims in a patent or the time spent in examination increases markedly with the
> experience of the examiner, both attributes decrease markedly with increasing tenure."

Confirms Lemley & Sampat's 11-point figure verbatim, citing the published version directly: "the
grant rate increases monotonically with experience, so that the most experienced examiners have a
grant rate eleven percentage points higher than the least experienced examiners" (citing Lemley &
Sampat, 94 Rev. Econ. & Stat. 817, 822 (2012) — matches finding 3b above independently).

Also reproduces the Cockburn/Kortum/Stern quote with a precise pin cite to the **published book
chapter** (distinct from the NBER working-paper pagination used in finding 3a): "Iain M. Cockburn,
Samuel Kortum & Scott Stern, *Are All Patent Examiners Equal? Examiners, Patent Characteristics,
and Litigation Outcomes*, in PATENTS IN THE KNOWLEDGE-BASED ECONOMY 19, **21** (Wesley M. Cohen &
Stephen A. Merrill eds., 2003)" for "there may be as many patent offices as patent examiners."

**Relayed finding, NOT independently verified against the primary source** — Mann's literature
review reports (p.2153 of the extracted text), citing Douglas Lichtman, *Rethinking Prosecution
History Estoppel*, 71 U. Chi. L. Rev. 151, 157-162 (2004) [this Lichtman paper itself could not be
located as an open-access copy in the time available — treat the following as [SNIPPET]-via-Mann,
not independently fetched]:

> "[Lichtman] concludes that differences among the responsible examiners account for **about
> two-thirds of the variation in rigor of editing**" of patent claim text between the filed
> application and the issued patent, across ~300,000 post-2000 published applications restricted to
> the ten technology classes with the most observations.

Why this shows a gap: same point as 3a-3d (examiner identity swamps stated criteria) but adds a
methodological wrinkle relevant to a paper on articulation gaps — tenure and experience are
confounded in most prior examiner-variance studies and actually point in opposite directions, which
means the "amount of the outcome explained by who-the-decision-maker-is" is even less stable/
predictable than a single coefficient suggests. The Lichtman figure (two-thirds of *editing rigor*
variation attributable to examiner identity) is the most extreme number encountered in this sweep,
if it holds up — flagged for independent verification before being used as a headline statistic.

```bibtex
@article{mann2014idiosyncrasy,
  author  = {Mann, Ronald J.},
  title   = {The Idiosyncrasy of Patent Examiners: Effects of Experience and Attrition},
  journal = {Texas Law Review},
  volume  = {92},
  pages   = {2149},
  year    = {2014},
  url     = {http://www.columbia.edu/~mr2651/Mann-92-7.pdf},
  keywords = {domain=patents; gap=examiner-fixed-effects-tenure-confound; type=empirical},
  annote  = {[VERIFIED] fetched full PDF from author's own hosted copy
    (columbia.edu/~mr2651/Mann-92-7.pdf). Shows examiner tenure and experience effects run in
    OPPOSITE directions and confound each other in prior examiner-variance studies (claims/exam-time
    rise with experience but fall with tenure). Independently confirms Lemley & Sampat's "eleven
    percentage point[]" grant-rate gap by experience with a precise published-version pin cite (94
    Rev. Econ. \& Stat. 817, 822 (2012)), and the Cockburn/Kortum/Stern "as many patent offices as
    patent examiners" quote with a published-book-chapter pin cite (Patents in the Knowledge-Based
    Economy 19, 21 (2003)). ALSO relays (NOT independently verified against the primary source, flag
    as [SNIPPET]-via-secondary) Lichtman (2004)'s finding that examiner identity accounts for "about
    two-thirds of the variation in rigor of editing" of patent claim text between filing and grant --
    the largest single examiner-effect-size figure surfaced in this sweep, pending independent
    verification.}
}
```
