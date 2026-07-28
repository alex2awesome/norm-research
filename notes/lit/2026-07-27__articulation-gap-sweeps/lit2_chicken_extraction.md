# Literature retrieval: chicken sexing + articulation-gap extraction cases

Bibs checked for duplicates first: `latex/refs-shared.bib` (found existing `biederman1987sexing` stub, LEAD-only, no quote), `methods/metric_implementer/references.bib` (found `hoffman1998use`, `militello1998applied`, `tofelgrehl2013cognitive`, `karelaia2008determinants`, `dawes1989clinical`, `cowan2001expert` — all already present, none duplicated below), `notes/articulability-prompt-opt.bib`, `latex/paper-1__metric-codability/refs.bib`. No existing entry for Horsey 2002, Camerer & Johnson 1991, Hoffman 1960, Slovic 1969, Goldberg 1970, Kundel & LaFollette 1972, or Chapman & Chapman 1967/1969 — all new below.

---

## PART 1 — Biederman & Shiffrar (1987), "Sexing Day-Old Chicks"

**RETRIEVED IN FULL.** A prior agent's routes (APA/JSTOR/CiteSeerX/sci-hub) failed. What worked: a Google-indexed direct PDF mirror at a course-materials site, found via a DuckDuckGo HTML search proxied through WebFetch (`https://chrissnijders.com/materials/Biederman_Shiffrar_1987.pdf`, a Tilburg University professor's course-materials page), downloaded with `curl` (Chrome UA), extracted with `pdftotext -layout`. Full 6-page OCR-clean text obtained, journal pagination intact (pp. 640–645).

Citation: Biederman, I., & Shiffrar, M. M. (1987). Sexing day-old chicks: A case study and expert systems analysis of a difficult perceptual-learning task. *Journal of Experimental Psychology: Learning, Memory, and Cognition*, 13(4), 640–645.

### Stimulus conditions (this matters — the popular retelling overstates the result)
**[VERIFIED]** — the experiment was run on **static photographs**, not live chicks, and was **not timed** for the naive subjects:
> "Naive subjects were shown 18 pictures of cloacal regions of male and female chicks (in random appearing arrangement) and asked to judge the sex of each chick. The pictures included a number of rare and difficult configurations." (p. 640, abstract)

Crucially, the pictures were **deliberately the hard cases**: "the pictures shown in Figure 1 ... were initially created by Canfield (1940, 1941) to depict rare and difficult types. Their high level of difficulty provided us with an opportunity to obtain sufficient errors..." (p. 643). This means the professionals' 72% accuracy figure below is *not* comparable to their real 98%+ on-the-job rate — the paper says so explicitly (see below).

Real-world sexer conditions (self-report, not the experiment itself): "The task requires that the cloaca be everted... Gentle but firm pressure from the two thumbs and right forefinger are exerted to spread the ventral surface of the cloaca upwards to expose the eminence, called the 'bead.' ... Mr. Carlson worked quickly and steadily, spending approximately .5 s actually looking at each eminence under magnification and a bright 200-W bulb." (pp. 640–641)

### The diagnostic cue (convex vs. concave)
**[VERIFIED]**
> "Our examination of the beads revealed a simple difference in the contours between males and females. In males, the eminence was convex; in females, flat or concave. This differentiation corresponded to the descriptions offered by some of the sexers who described males as 'round' and females as 'pointy.'" (p. 641)

> "...a brief instruction that merely described where a nonaccidental contrast in shape (concavity-convexity) could serve as the basis for classification." (p. 644, Discussion)

The instructional sheet given to naive subjects (Figure 3, verbatim, p. 643) told them: locate "the two large cylindrical side lobes near the bottom of each picture," find the genitals "either between the ends of these two lobes... or slightly above the ends," and that "Male chicken genitals tend to look round and full ... like a ball or watermelon" vs. female "pointed, like an upside down pine tree, or flatish." **~1 minute** was required for this instruction (abstract, p. 640; confirmed again p. 641 "Approximately 1 min was required for this instruction").

### How the authors characterize the articulation gap — nuanced, not absolute
This is the part where the popular ("Gladwell") retelling overstates the paper. The authors do **not** claim total inarticulacy; they report a **partial, imprecise, untransferable** verbal account that had to be reverse-engineered and sharpened by the researchers:

**[VERIFIED]**
- Sexers had a rough verbal descriptor ("round"/"pointy") that *correlated* with the true cue but was not precise or actionable enough to teach: "This differentiation corresponded to the descriptions offered by some of the sexers..." (p. 641) — i.e., partial correspondence, discovered only after the researchers had already independently found the contour cue by asking the master sexer to *circle* the critical regions on the picture, not by asking him to describe them in words: "We asked Mr. Carlson to circle the critical areas in each of the pictures on a copy of Figure 1." (p. 641)
- Sexers named the target structure but not the diagnostic feature of it: "Although the sexers said that they were looking for the bead, they also said that they were matching 'types.'" (p. 643) — they could name *where* to look but described their decision process as instance-matching against a large, largely tacit catalog of learned "types," not as a simple shape rule.
- Training was not explicit-rule-based and diagrams like the researchers' Figure 3 were never used: "Not a single sexer recalled being shown simple diagrams such as those in Figure 3 as part of their training. Training was accomplished by sexing live birds, which were then checked by experts." (p. 643)
- Even the professionals' account of *what was hard* was vague, not mechanistic: "In response to a question as to what was the most difficult part of the task, most sexers noted an aspect of image interpretation such as 'reading the bead' or 'different types.'" (p. 643)
- One explicit cognitive strategy (avoiding the gambler's fallacy when a tray ran unbalanced) was practiced by every sexer but *none had a name for it*: "Every sexer interviewed noted the necessity of avoiding the 'gambler's fallacy' (although none used that expression)." (p. 643)
- Estimates of the size of their own "type" vocabulary varied wildly and were themselves folkloric, not measured: one sexer said "45 of each sex," another said "they say there are over a 1,000 types for each sex" (p. 643, footnote 1: "Indeed, the purpose of the Canfield (1940, 1941) articles was to illustrate the various types.")

So the correct, page-cited characterization for the paper: **experts' spontaneous verbal account was partially correct in gist (they knew to attend to "the bead" and had a loose shape-word for it) but was not precise, complete, or teachable as stated — the researchers had to extract the actual geometric invariant (convex/concave, non-accidental under viewpoint change) by direct behavioral elicitation (having the expert circle regions on images) rather than by taking his verbal report at face value.**

### The accuracy numbers — reported exactly as stated in the paper, all page-cited
**[VERIFIED]**, all from Results (pp. 643–644) unless noted:

- Naive subjects (n=36, UCSC + SUNY Buffalo, no chick-sexing experience), unweighted:
  - Pretest: **60.5% correct** (10.9/18 pictures)
  - Posttest, after ~1 min instruction: **84% correct**
  - (Abstract, p. 640, gives the same 60.5%→84% figures; body text p. 644 repeats "The naive subjects averaged 60.5% correct (10.9 correct choices out of 18 pictures) in pretest and 84% correct in posttest, after instruction.")
- Control check — the gain was not just from a second look: the 16 naive subjects who got a *second* trial with **no instruction** actually got *worse*: 59.0% (trial 1) → 54.1% (trial 2), "a decline of 4.9%" (p. 644).
- Professional sexers (n=5; four retired ex-Kimberly-Farms sexers + one currently employed; mean 24.4 years full-time experience, range 18–36 years): **72% mean accuracy** (unweighted, per-picture average) on the same 18 hard/rare Canfield pictures (p. 644).
  - Paper explicitly flags this 72% as an underestimate of real-world performance because the stimulus set was deliberately the rare/hard cases and because photographs deprive the sexer of the *dynamic* tactile cue (pressing/releasing the eminence): "The 72% mean accuracy per picture in this study for the professional sexers was markedly lower than the high 90% typically reported for on-the-job accuracy levels. About half of this difference could be accounted for by the higher miss rates for the rare and difficult types." (p. 644) "...it was perhaps not surprising that most of the professional sexers voiced reservations about having to perform this task from pictures." (p. 644)
  - When accuracy is *reweighted* by the true population frequency of each picture-type (i.e., correcting for the fact that the stimulus set over-sampled rare/hard cases), expected accuracy rises: **experts 84.1%**, **instructed naive subjects 89.8%**, **naive (uninstructed) 65.3%** (p. 643, "expected performance levels for the experts averaged 84.1% accuracy, the instructed subjects 89.8% accuracy, and the naive subjects, 65.3%").
- Self-reported on-the-job performance (from structured interviews of the 5 professionals, not from the picture experiment): mean maximum accuracy **99.4%** (range 99.3–99.5%), at a mean rate of **960 birds/hour** (range 900–1,000); mean **2.4 months** (range 1.5–3.5) to reach 95% accuracy; **2 to 6 years** (M = 3.3 years) to reach personal maximum performance, at roughly 1 million birds/year (p. 643, "Interviews" subsection).
- Item-level correlation with the experts (the paper's headline transfer statistic): pretest naive vs. experts, **r = .21**; posttest (instructed) naive vs. experts, **r = .82** (abstract p. 640 and body p. 644, "The Pearson product-moment correlation between the pretest naive subjects and the professionals was .21. The r between the posttest scores for the instructed subjects and the professionals was .82."). For calibration, the paper also reports: pre-to-post correlation for the *instructed* group was r = .63; a random split-half correlation *among the professionals themselves* was r = .57; and trial-to-trial correlation for the *uninstructed* naive group was r = .87 (p. 644) — i.e., after instruction, naive subjects' item-by-item error pattern resembled the experts' (.82) *more* than it resembled their own uninstructed selves a minute earlier (.63).
- Variance-explained framing (their strongest single claim): "Before instruction, 4% of the variance of the professional sexers could be predicted by the naive subjects; after training this value increased to 67%." (p. 644, Discussion)
- Above-chance baseline explained: naive subjects were already at 60.5% (not 50%) before any instruction because "the presence of a prominent bead ... was interpreted as being male" — a real but weak and imperfect correlate the naive subjects had guessed on their own (p. 644–645).

**Bottom line for the paper's argument**: this is a real, quantified, page-cited case of (a) a genuine, high stakes, high-accuracy perceptual expertise; (b) a verbal self-report from experts that named the right general locus of attention ("the bead") but not the transferable rule; (c) a rule that outside investigators extracted by non-verbal elicitation (asking the expert to mark the image, not to describe it) and formalized as a precise geometric invariant; (d) a large, measured transfer effect from a ~1-minute instruction (item-correlation with experts jumping from .21 to .82, variance explained from 4% to 67%); but (e) an explicit caveat, in the authors' own words, that the professionals' 72%-picture-task accuracy undersells their true ~99% field accuracy, because photographs strip out a real, non-verbal, dynamic tactile cue (repeated compression of the eminence) that sexers rely on and that the researchers could not reduce to a static rule.

---

## Horsey (2002), "The Art of Chicken Sexing"

**RETRIEVED IN FULL.** UCL Working Papers in Linguistics did not resolve at the modern UCL URL (403/redirect dead), but the Wayback Machine had a clean 2023 snapshot of the old `phon.ucl.ac.uk` mirror: `http://web.archive.org/web/20230314211825/https://www.phon.ucl.ac.uk/home/PUB/WPL/02papers/horsey.pdf`. Downloaded and `pdftotext`'d, full 8-page/10-page-journal (pp. 107–117 in the volume) text obtained.

Citation: Horsey, R. (2002). The art of chicken sexing. *UCL Working Papers in Linguistics*, 14, 107–117.

This is exactly the "careful academic treatment of the folklore" the task asked for. Key content, **[VERIFIED]**:

- Horsey's actual thesis is a *debunking/normalizing* one: he argues chicken sexing is not evidence of a special "ineffable" faculty, but an instance of the same automatic, introspection-inaccessible categorization that underlies everyday object recognition (faces, chairs, dogs); the "mystery" is only about *why it's hard to learn*, not that it's uniquely inarticulable (Abstract, p. 107).
- He explicitly **corrects the popular version of Biederman & Shiffrar's numbers**, and flags that the naive-subject above-chance baseline is weak, not strong: "In fact, Biederman & Shiffrar (1987) report that untrained subjects perform slightly better than chance at chicken sexing, probably because they interpreted a prominent bead as an indication that the chick was male. Although this is not an accurate diagnostic, there is a weak statistical correlation." (footnote 4, pp. 109–110)
- He gives the folklore quote that the popular retellings run with, sourced to a chick-sexing memoirist, not to a peer-reviewed study — useful for showing the paper the difference between folklore and data: "'To be close to 100 per cent accurate at 800 to 1200 chickens per hour for a long day, intuition comes in to play in many of your decisions, even if you are not consciously aware of it. As one of my former colleagues said to me... "There was nothing there but I knew it was a cockerel". This was intuition at work.'" — quoted from R. D. Martin, *The Specialist Chick Sexer* (Martin 1994), via Bernal Publishing's website (p. 109–110, footnote 5).
- He draws the parallel case of **WWII aircraft-spotter training** as a *documented knowledge-elicitation problem*, citing Allan (1958): "expert 'spotters' ... did exist, but there were too few of them. Training centres were therefore set up, but the problem was that the experts had no idea how they had acquired their skills in the first place, or how to transmit those skills to others. Training regimens therefore had to be developed somewhat by trial-and-error." (p. 108–109) — this is a genuine primary-adjacent case of elicitation being *hard and ad hoc*, not a clean success; worth citing as a partial-failure/friction case (extraction eventually worked via trial-and-error training design, not via getting a clean verbal rule from the experts).
- Bird-watchers' "jizz" (gestalt species-ID by overall impression) is presented as another same-class case: "birdwatchers perceive the jizz as a gestalt, but cannot say what the features are that make up the whole." (p. 110)
- Horsey's own theoretical resolution (Gigerenzer & Goldstein's "fast and frugal heuristics" / cue-order framework, and implicit-learning research by Jiang & Chun 2001 and Lewicki et al. 1992) is that expert cue-use is real, feature-based, and can be taught by directing attention to the cues — even though the resulting skill becomes non-introspectable once automatized (pp. 111–116). This is a strong theoretical citation for "articulable-in-principle, non-introspectable-in-practice."

---

## PART 2 — Documented extraction successes (and one important internal failure) in the lens-model / bootstrapping literature

**Primary sources targeted (Hoffman 1960, Slovic 1969, Goldberg 1970) are all APA/Elsevier-paywalled with no open-access copy** — confirmed via OpenAlex + Unpaywall + Crossref DOI lookups (`10.1037/h0047807` Hoffman 1960; `10.1037/h0027773` Slovic 1969 *Journal of Applied Psychology*; `10.1037/h0029230` Goldberg 1970) — all return `is_oa: false`, `best_oa_location: null`. No course-page or repository mirror found for any of the three within the search budget available (WebSearch tool was already at its session cap of 200/200 calls before I started; DuckDuckGo/Bing scraping via WebFetch/curl hit CAPTCHA/anti-abuse blocks). I did **not** fabricate numbers for these — see below for exactly what is and isn't recoverable.

**However**, I found and fully retrieved an open-access, high-quality secondary source that quotes all three with real page numbers and reproduces an actual data table:

> Camerer, C. F., & Johnson, E. J. (1991). The process-performance paradox in expert judgment: How can experts know so much and predict so badly? In K. A. Ericsson & J. Smith (Eds.), *Toward a General Theory of Expertise: Prospects and Limits* (pp. 195–217). Cambridge University Press.

**[VERIFIED]** — full text retrieved open-access via Caltech's institutional repository (`https://authors.library.caltech.edu/records/qrp4n-9az74/files/334945.pdf`, found via the OpenAlex API's `primary_location.pdf_url` field), `pdftotext`'d cleanly with page headers preserved (pp. 195–217 confirmed via in-text running heads).

### The quantified "stated/used" gap: bootstrapping models beat the judges they're modeled on
This is the "gold" quantified case the task asked for — a table (Table 8.1, p. 201 in-text) of five separate published studies, each comparing a judge's own accuracy (`r_a`) against the accuracy of a simple linear regression model built to *mimic that same judge's own inputs* (`r_m`, the "bootstrapping model"), an equal-weight model (`r_ew`), and the best possible actuarial model on the same cues (`R_e`):

**[VERIFIED]**, reproduced exactly as printed (p. 201):

| Study | Task | Model fit R_s | Judge r_a | Bootstrap model r_m | Bootstrap residual r_z | Equal-weight r_ew | Actuarial R_e |
|---|---|---|---|---|---|---|---|
| Goldberg (1970) | Psychosis vs. neurosis | .77 | .28 | .31 | .07 | .34 | .45 |
| Dawes (1971) | PhD admissions | .78 | .19 | .25 | .01 | .48 | .38 |
| Einhorn (1972) | Disease severity | .41 | .01 | .13 | .06 | n.a. | .35 |
| Libby (1976)* | Bankruptcy | .79 | .50 | .53 | .13 | n.a. | .67 |
| Wiggins & Kohen (1971) | Grades | .85 | .33 | .50 | .01 | .60 | .57 |

(*Libby figures are Goldberg's 1976 recalculation. Source line as printed: "Adapted from Camerer (1981a) and Dawes & Corrigan (1974).")

In every single row, the model built from the judge's own cues (r_m) beats the judge's own actual judgment (r_a) — this is the "captured weights beat stated/used weights" pattern in quantified form, across five independent domains. Camerer & Johnson's gloss on why, with the explicit citation to Hoffman (1960) for the term "paramorphic":

**[VERIFIED]**
> "Of course, such an explanation is 'paramorphic' (Hoffman, 1960): It describes judgments in a purely statistical way, as if experts were weighing and combining cues in their heads; the process they use might be quite different." (p. 199, footnote 6)

And the mechanism (lens-model equation, Einhorn 1974 version) — bootstrapping wins whenever the judge's own residual variance is closer to noise than to signal, which the authors say is nearly always true empirically: "For R_s = .6 (a reasonable value; see Table 8.1), residual validity r_z must be about half as large as model accuracy for experts to outperform their own bootstrapping models. **This rarely occurs.**" (p. 200)

### Two positive extraction cases beyond the regression table

**Radiology** — **[VERIFIED]** (Camerer & Johnson quoting/paraphrasing Kundel & LaFollette 1972; I did not retrieve the 1972 primary itself, so treat the specific numbers as SNIPPET-via-secondary, not independently verified):
> "Kundel and LaFollette (1972) reported that novices and first-year medical students were unable to detect lesions from radiographs of abnormal lungs, but fourth-year students (who had had some training in radiography) were as good as full-time radiologists." (p. 201)
This is structurally the closest published analogue to the chicken-sexing result: modest explicit training collapses a gap that "years of experience" folklore says should take much longer. Camerer & Johnson's editorializing gloss is worth quoting because it's exactly the paper's thesis: "If a small amount of training can make a person as accurate as an experienced clinical psychologist or doctor, as the data imply, then lightly trained paraprofessionals could replace heavily trained experts for many routine kinds of diagnoses." (p. 201) — with a supporting quote from Garb (1989) citing Shortliffe, Buchanan & Feigenbaum (1979): "intelligent high school graduates, selected in large part because of poise and warmth of personality, can provide competent medical care for a limited range of problems when guided by protocols after only 4 to 8 weeks of training." (p. 201)

**Backgammon** — **[SNIPPET]** via the same secondary, citing Berliner (1980): a dynamic-reweighting program (not a fixed linear bootstrap) beat the 1979 world champion, offered by Camerer & Johnson as the one case in their review where a model that could *shift* weights (rather than use one fixed linear combination) actually surpassed, not just matched, top human performance: "a model that could shift weights during the game could possibly beat an expert, and one did: Berliner's (1980) backgammon program beat the 1979 world champion." (p. 213; full cite in refs, p. 215: Berliner, H. J. (1980). Backgammon computer program beats world champion. *Artificial Intelligence*, 14, 205–220.)

### Negative case #1 — rule-based expert-system extraction does NOT reliably beat simple linear models
This is a direct hit on the task's request for a documented **failure** of extraction, from inside the same literature that produced the successes above:

**[VERIFIED]**
> "Expert systems may predict less accurately than simple models because the systems are too much like experts. The main lesson from the regression-model literature is that large numbers of configural rules, which knowledge engineers take as evidence of expertise, do not necessarily make good predictions; simple linear combinations of variables (measured by experts) are better in many tasks." (p. 212–213)

The mechanism offered: experts' articulated ("configural") rules — e.g., a real rule from a Kleinmuntz (1968) MMPI-interpretation expert system built from clinicians' verbal protocols, "Call maladjusted if Pa ≥ 70 unless M ≤ 6, and K ≤ 65" (p. 204–205) — faithfully mirror what the expert *says* they do, but adding this configural complexity does not improve predictive accuracy over a plain weighted sum of the same cues, and can even hurt it because configural rules are "brittle": "small errors in measurement may have great impacts on configural rules... the linear rule that weights [cues] and combines them is less vulnerable to either error." (p. 210) So: **eliciting the expert's own stated decision rule and encoding it faithfully (classic expert-systems knowledge acquisition) is the extraction method that fails; eliciting the expert's implicit cue-weighting via paramorphic regression on their own judgments is the extraction method that succeeds.** This is a strong, citable contrast for the paper — same underlying judges, two different elicitation methodologies, opposite results.

### Negative case #2 — an articulated diagnostic "rule" that is simply wrong and resists correction (illusory correlation)
**[VERIFIED]**
> "...most clinicians and novices think that people who see male features or androgynous figures in Rorschach inkblots are more likely to be homosexual. They are not (Chapman & Chapman, 1967, 1969)." (p. 209–210)

Camerer & Johnson use this to argue that a stated/articulated expert rule can be actively counterproductive and durable despite being empirically false, and connect it to a feedback-based explanation for *why* bad rules persist even under a knowledge-elicitation program: "Inaccurate configural rules may persist because experts who get slow, infrequent, or unclear feedback will not learn that their rules are wrong... people tend to search instinctively for evidence that will confirm prior theories." (p. 210) Full citations: Chapman, L. J., & Chapman, J. P. (1967). Genesis of popular but erroneous psychodiagnostic observations. *Journal of Abnormal Psychology*, 73, 193–204; and (1969) Illusory correlation as an obstacle to the use of valid psychodiagnostic signs. *Journal of Abnormal Psychology* [journal volume printed as "46, 271-280" in this chapter's OCR'd reference list — that volume/page looks like an OCR or original typo, likely should be *Journal of Abnormal Psychology*, 74, 271–280; flagging rather than silently "fixing," since I have not independently verified the correct volume/page].

### Additional quantified findings from the same chapter worth keeping in reserve
- Training helps, experience mostly doesn't: "novices might classify 28% correctly, and experts 40%" on MMPI personality-disorder judgments (Garb 1989 review of 50+ studies, as summarized p. 201).
- One "outstanding individual expert beats the model" counter-example, explicitly flagged by the authors as possibly a fluke: Goldberg (1959) found one well-known (slow-working) expert scored 83% correct on organic-brain-damage diagnosis vs. 65% for other PhD clinical psychologists — "Whether such extraordinary expertise is a reliable phenomenon or a statistical fluke is a matter for further research." (p. 202)
- A quantified expert-calibration (not accuracy) result: on "kinetic family drawings," psychologists and secretaries scored 66%/61% overall, but among cases the *subject rated "positively certain,"* psychologists got 76% right vs. secretaries 59% — experts were better calibrated but "still overconfident" (Levenberg 1975, as summarized p. 202).

### What I could NOT verify (explicit, per verification rules)
- I could not retrieve or independently verify any exact number, quote, or page cite directly from Hoffman (1960), Slovic (1969), or Goldberg (1970) themselves — only their bibliographic metadata (title/journal/year/DOI, confirmed via OpenAlex/Crossref) and the numbers quoted above via Camerer & Johnson (1991). Do not attribute the Table 8.1 numbers to "Goldberg (1970), p. X" as if I'd read Goldberg directly — attribute them to Camerer & Johnson 1991, p. 201, citing Goldberg (1970).
- I could not retrieve Kundel & LaFollette (1972) directly; the radiology claim above is SNIPPET-via-secondary only.
- I did not find, within the time/tool budget, a primary source for the classic "Slovic 1969 stockbroker: stated cue-importance ranking vs. regression-derived cue-importance ranking diverge" claim specifically (this is the canonical statement of the result but I have no verified quote for it — flagging as **[LEAD]** only, sourced from general field knowledge of the lens-model literature, not from a document I actually opened).

---

## Repo-relevant note
The existing `biederman1987sexing` entry in `latex/refs-shared.bib` (lines 1705–1722) is currently LEAD-only and says "paywalled everywhere tried, NO quote obtained." That is now **stale** — see the updated entry below with full verified quotes and page cites. I did not delete the old entry's substance; the annote below supersedes it (this file also includes a drop-in replacement block).

---

## BibTeX entries (ready to paste)

```bibtex
% REPLACES the existing biederman1987sexing entry in latex/refs-shared.bib (lines 1705-1722).
% The old annote said "NO quote obtained" -- that is now false; full text was retrieved.
@article{biederman1987sexing,
  author  = {Biederman, Irving and Shiffrar, Margaret M.},
  title   = {Sexing Day-Old Chicks: A Case Study and Expert Systems Analysis of a Difficult Perceptual-Learning Task},
  journal = {Journal of Experimental Psychology: Learning, Memory, and Cognition},
  volume  = {13},
  number  = {4},
  pages   = {640--645},
  year    = {1987},
  keywords = {domain=cross-cutting; gap=stated-ne-used; type=experiment},
  annote  = {VERIFIED (full text retrieved via https://chrissnijders.com/materials/Biederman_Shiffrar_1987.pdf,
             a course-materials mirror; pdftotext -layout gave clean OCR with journal pagination 640-645).
             STIMULUS CAVEAT (popular retellings omit this): task used 18 STATIC PHOTOGRAPHS of
             deliberately rare/hard Canfield (1940/1941) cases, untimed for naive subjects -- not live,
             timed chicks. Cue: bead (cloacal eminence) convex=male vs. concave/flat=female, a
             viewpoint-invariant "nonaccidental" contour contrast (p. 641-642); ~1 min instruction sheet
             told subjects where to look (two cylindrical side lobes) and gave the shape words
             round/watermelon vs. pointed/pine-tree (Fig. 3, p. 643). NUMBERS (p.640, 643-644): naive
             pretest 60.5% (10.9/18) -> posttest 84% after ~1 min instruction; no-instruction control
             trial 1->2 DECLINED 59.0%->54.1%; professionals (n=5, mean 24.4 yrs exp.) 72% unweighted
             on the same hard photos (paper explicitly flags this as an underestimate of their true
             ~99% field accuracy, p.644); frequency-reweighted accuracy: experts 84.1%, instructed
             naive 89.8%, naive 65.3% (p.643); item-correlation with experts: pretest r=.21 ->
             posttest r=.82 (abstract, p.644); variance of expert judgments predictable from naive
             subjects: 4% before -> 67% after instruction (p.644, their strongest single claim).
             ARTICULATION-GAP NUANCE: not total inarticulacy. Sexers had a rough, partially-correct
             verbal descriptor ("round"/"pointy", p.641) and could name the target structure ("looking
             for the bead," p.643) but could not state the precise transferable rule; researchers
             extracted the actual cue by asking the master sexer to CIRCLE regions on images (p.641),
             not by taking his verbal report at face value. "Not a single sexer recalled being shown
             simple diagrams... Training was accomplished by sexing live birds, which were then
             checked by experts" (p.643). Self-reported field stats (interviews, not the photo
             experiment): 99.4% max accuracy (99.3-99.5% range), 960 birds/hr (900-1000), 2.4 months
             to 95% accuracy, 2-6 yrs to personal max (p.643).}
}
```

```bibtex
@article{horsey2002chicken,
  author  = {Horsey, Richard},
  title   = {The Art of Chicken Sexing},
  journal = {UCL Working Papers in Linguistics},
  volume  = {14},
  pages   = {107--117},
  year    = {2002},
  keywords = {domain=cross-cutting; gap=felt-not-stated; type=review},
  annote  = {VERIFIED (full text retrieved from Wayback Machine snapshot of the original UCL host:
             http://web.archive.org/web/20230314211825/https://www.phon.ucl.ac.uk/home/PUB/WPL/02papers/horsey.pdf
             -- the modern ucl.ac.uk mirror 403s / the paper is otherwise effectively delisted).
             Careful academic (not popular) treatment of the chicken-sexing folklore, arguing the
             skill is ordinary introspection-inaccessible categorization, not a special faculty.
             EXPLICITLY CORRECTS the popular retelling of Biederman & Shiffrar 1987: "untrained
             subjects perform slightly better than chance..., probably because they interpreted a
             prominent bead as an indication that the chick was male. Although this is not an
             accurate diagnostic, there is a weak statistical correlation" (fn.4, pp.109-110).
             Cites a second, independent knowledge-elicitation-failure case: WWII aircraft-spotter
             training (Allan 1958) -- "the experts had no idea how they had acquired their skills in
             the first place, or how to transmit those skills to others. Training regimens therefore
             had to be developed somewhat by trial-and-error" (p.108-109). Also documents
             birdwatchers' "jizz" gestalt-ID skill as a same-class case: "cannot say what the
             features are that make up the whole" (p.110). Good citable folklore quote (memoir, not
             data) that shows what the POPULAR version claims, for contrast: R.D. Martin quoted
             p.109-110 fn.5, "There was nothing there but I knew it was a cockerel."}
}
```

```bibtex
@incollection{camererjohnson1991paradox,
  author    = {Camerer, Colin F. and Johnson, Eric J.},
  title     = {The Process-Performance Paradox in Expert Judgment: How Can Experts Know So Much and Predict So Badly?},
  booktitle = {Toward a General Theory of Expertise: Prospects and Limits},
  editor    = {Ericsson, K. Anders and Smith, Jacqui},
  publisher = {Cambridge University Press},
  pages     = {195--217},
  year      = {1991},
  keywords  = {domain=cross-cutting; gap=stated-ne-used; type=review},
  annote    = {VERIFIED (full text retrieved open-access from CaltechAUTHORS,
               https://authors.library.caltech.edu/records/qrp4n-9az74/files/334945.pdf, found via
               OpenAlex API primary_location.pdf_url; pdftotext gave clean text, journal/book running
               heads confirm pp.195-217). THE gold quantified stated-vs-used table for the paper
               (Table 8.1, p.201; reproduced in full in lit2_chicken_extraction.md): across 5
               published studies (Goldberg 1970 psychosis-vs-neurosis; Dawes 1971 PhD admissions;
               Einhorn 1972 disease severity; Libby 1976 bankruptcy; Wiggins & Kohen 1971 grades), a
               linear "bootstrapping" model built from a JUDGE'S OWN cue inputs beats that SAME
               judge's own actual accuracy in every row (e.g. Goldberg 1970: judge r=.28 vs.
               bootstrap-of-judge r=.31; Wiggins & Kohen 1971: judge r=.33 vs. bootstrap r=.50).
               Names the mechanism "paramorphic" after Hoffman (1960): "It describes judgments in a
               purely statistical way, as if experts were weighing and combining cues in their heads;
               the process they use might be quite different" (p.199 fn.6). "For R_s=.6 ..., residual
               validity r_z must be about half as large as model accuracy for experts to outperform
               their own bootstrapping models. This rarely occurs" (p.200). POSITIVE case (radiology,
               via Kundel & LaFollette 1972, not independently verified -- SNIPPET only): "novices and
               first-year medical students were unable to detect lesions from radiographs of abnormal
               lungs, but fourth-year students (who had had some training) were as good as full-time
               radiologists" (p.201). NEGATIVE case #1 -- rule-based expert-system extraction of the
               SAME experts' stated configural rules does NOT reliably beat the linear bootstrap:
               "large numbers of configural rules, which knowledge engineers take as evidence of
               expertise, do not necessarily make good predictions; simple linear combinations of
               variables... are better in many tasks" (p.212-213); configural rules are "brittle" to
               small measurement error in a way linear models are not (p.210). NEGATIVE case #2 --
               an articulated diagnostic rule that is simply false and persists: clinicians' belief
               that Rorschach male/androgynous figures indicate homosexuality, refuted by Chapman &
               Chapman 1967/1969, persisting because of confirmation-seeking and poor feedback
               (p.209-210). One flagged "real outstanding expert beats the model" exception (Goldberg
               1959: one expert 83% vs. other PhDs 65% on brain-damage diagnosis, p.202, explicitly
               called possibly "a statistical fluke"). Calibration data: experts better calibrated
               than novices but still overconfident (Levenberg 1975, p.202).}
}
```

```bibtex
% LEAD only -- primary is APA-paywalled, DOI 10.1037/h0047807, confirmed closed via OpenAlex +
% Unpaywall (email-gated check, is_oa:false, best_oa_location:null). Numbers/quotes about this
% paper's finding are only available to me via camererjohnson1991paradox (p.199 fn.6); do NOT quote
% a page number from Hoffman 1960 itself without independently opening it.
@article{hoffman1960paramorphic,
  author  = {Hoffman, Paul J.},
  title   = {The Paramorphic Representation of Clinical Judgment},
  journal = {Psychological Bulletin},
  volume  = {57},
  number  = {2},
  pages   = {116--131},
  year    = {1960},
  doi     = {10.1037/h0047807},
  keywords = {domain=cross-cutting; gap=stated-ne-used; type=quantitative},
  annote  = {LEAD -- confirmed closed-access via OpenAlex + Unpaywall (no OA copy indexed anywhere).
             Coined "paramorphic representation": a linear regression fit to a judge's own inputs
             and outputs describes the judge's judgments statistically without claiming to describe
             the judge's actual cognitive process. Foundational citation for the whole
             bootstrapping/lens-model literature (see camererjohnson1991paradox for a verified,
             page-cited secondary quote and a full quantified table of studies in this tradition).
             Get the PDF before camera copy if a library-access route becomes available.}
}
```

```bibtex
% LEAD only -- primary is APA-paywalled, DOI 10.1037/h0029230, confirmed closed via OpenAlex +
% Unpaywall. Quantified table entry (judge r=.28 vs. bootstrap-of-judge r=.31 on psychosis-vs-
% neurosis diagnosis from MMPI) is available ONLY via camererjohnson1991paradox, Table 8.1, p.201 --
% cite that secondary for the number, not this entry, unless the primary is independently opened.
@article{goldberg1970man,
  author  = {Goldberg, Lewis R.},
  title   = {Man versus Model of Man: A Rationale, Plus Some Evidence, for a Method of Improving on Clinical Inferences},
  journal = {Psychological Bulletin},
  volume  = {73},
  number  = {6},
  pages   = {422--432},
  year    = {1970},
  doi     = {10.1037/h0029230},
  keywords = {domain=cross-cutting; gap=stated-ne-used; type=experiment},
  annote  = {LEAD -- confirmed closed-access via OpenAlex + Unpaywall (no OA copy indexed anywhere).
             Classic "bootstrapping" demonstration: a linear model built to mimic a clinician's own
             MMPI-based psychosis-vs-neurosis judgments outperforms the clinician. Quantified via
             camererjohnson1991paradox Table 8.1 (p.201): model fit R_s=.77, judge r_a=.28,
             bootstrap-of-judge r_m=.31, equal-weight r_ew=.34, best actuarial model R_e=.45 -- i.e.
             even a crude paramorphic model of the clinician's OWN stated cues beats the clinician,
             and a genuinely optimal actuarial model on the same cues beats both by a wide margin.
             Do not attribute this table to a page number in Goldberg 1970 itself -- I have not
             opened the primary.}
}
```

```bibtex
% LEAD only -- primary is APA-paywalled, DOI 10.1037/h0027773 (J. Applied Psychology, confirmed via
% Crossref bibliographic search), confirmed closed via OpenAlex + Unpaywall. UNLIKE the two entries
% above, I found NO secondary source in this search that quotes this paper's specific numbers with a
% page cite -- Camerer & Johnson 1991 mentions Slovic only via the unrelated Kahneman/Slovic/Tversky
% (1982) heuristics-and-biases volume, not this 1969 stockbroker study. Treat the "stated cue-
% importance ranking vs. regression-derived cue-importance ranking diverge" claim commonly attributed
% to this paper as UNVERIFIED folklore until a copy is opened.
@article{slovic1969stockbroker,
  author  = {Slovic, Paul},
  title   = {Analyzing the Expert Judge: A Descriptive Study of a Stockbroker's Decision Processes},
  journal = {Journal of Applied Psychology},
  volume  = {53},
  number  = {4},
  pages   = {255--263},
  year    = {1969},
  doi     = {10.1037/h0027773},
  keywords = {domain=cross-cutting; gap=stated-ne-used; type=quantitative},
  annote  = {LEAD ONLY -- no quote, no page cite, no verified secondary. Confirmed closed-access via
             OpenAlex + Unpaywall; no course-page or repository mirror found within the search
             budget (WebSearch tool was at its 200/200 session cap before this task started; DDG/Bing
             scraping via curl and WebFetch hit CAPTCHA/anti-abuse blocks). The commonly-cited claim
             (stated cue-importance rankings diverge from the stockbroker's own regression-captured
             weights) is NOT verified here -- do not quote it as fact without independently opening
             the primary or finding a secondary that cites it with a page number.}
}
```

```bibtex
% SNIPPET only, via camererjohnson1991paradox p.201 -- primary not retrieved.
@article{kundel1972radiographic,
  author  = {Kundel, Harold L. and LaFollette, Paul S.},
  title   = {Visual Search Patterns and Experience with Radiological Images},
  journal = {Radiology},
  volume  = {103},
  number  = {3},
  pages   = {523--528},
  year    = {1972},
  keywords = {domain=cross-cutting; gap=osmosis; type=experiment},
  annote  = {SNIPPET (via camererjohnson1991paradox, p.201, not independently opened). Positive
             extraction/training-substitutes-for-experience case parallel to chicken sexing: "novices
             and first-year medical students were unable to detect lesions from radiographs of
             abnormal lungs, but fourth-year students (who had had some training in radiography) were
             as good as full-time radiologists." Full citation reconstructed from Camerer & Johnson's
             reference list (p.213 in their chapter's bibliography) -- volume/issue/page not
             independently confirmed against the primary; verify before camera copy.}
}
```

```bibtex
% SNIPPET only, via camererjohnson1991paradox pp.209-210 -- primaries not retrieved. Two papers,
% listed together since both are cited jointly for the same finding.
@article{chapman1967genesis,
  author  = {Chapman, Loren J. and Chapman, Jean P.},
  title   = {Genesis of Popular but Erroneous Psychodiagnostic Observations},
  journal = {Journal of Abnormal Psychology},
  volume  = {73},
  number  = {3},
  pages   = {193--204},
  year    = {1967},
  keywords = {domain=cross-cutting; gap=confabulation; type=experiment},
  annote  = {SNIPPET (via camererjohnson1991paradox, pp.209-210, not independently opened). NEGATIVE
             CASE for the paper: a widely-held, confidently articulated diagnostic "rule" among
             clinicians (Rorschach responses showing male/androgynous figures indicate homosexuality)
             is false, per this and the companion 1969 study -- an articulated cue can be wrong and
             durable, not just tacit-and-correct. Camerer & Johnson's gloss: such illusory
             correlations persist because "experts who get slow, infrequent, or unclear feedback will
             not learn that their rules are wrong" and because people "search instinctively for
             evidence that will confirm prior theories" (p.210).}
}
```

```bibtex
@article{chapman1969illusory,
  author  = {Chapman, Loren J. and Chapman, Jean P.},
  title   = {Illusory Correlation as an Obstacle to the Use of Valid Psychodiagnostic Signs},
  journal = {Journal of Abnormal Psychology},
  volume  = {74},
  number  = {3},
  pages   = {271--280},
  year    = {1969},
  keywords = {domain=cross-cutting; gap=confabulation; type=experiment},
  annote  = {SNIPPET (via camererjohnson1991paradox, pp.209-210, not independently opened). Companion
             to chapman1967genesis; NOTE the volume/page here (74, 271-280) is my best reconstruction
             from standard citation databases, since the OCR'd reference list inside
             camererjohnson1991paradox itself prints "46, 271-280" for this entry, which looks like an
             OCR or original-print error (vol. 46 of J. Abnormal Psychology does not correspond to
             1969) -- FLAG AND VERIFY the exact volume/page before camera copy rather than trusting
             either number blindly.}
}
```
