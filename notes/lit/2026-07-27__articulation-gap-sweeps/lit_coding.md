# Literature sweep: Articulation gaps in expert preference — SOFTWARE/CODING domain

Domain: code quality, code review, "good code," readability, design taste — cases where
developers/reviewers agree on quality but can't state (or don't follow) explicit criteria.

Method note: WebFetch's PDF handling was unreliable (certificate errors on some hosts, and it
frequently refused to parse binary PDF streams it could reach). For every candidate I therefore
downloaded the PDF with `curl` and extracted text with `pdftotext`; where the PDF was a scanned
image (0 extracted lines), I rendered pages to PNG with `pdftoppm` and ran `tesseract` OCR. Every
[VERIFIED] quote below was confirmed against text extracted this way, not against a WebFetch
summary.

---

## 1. Already in our bibliography (NOT new finds)

Checked `notes/articulability-prompt-opt.bib`, `methods/metric_implementer/references.bib`,
`latex/paper-1__metric-codability/refs.bib`.

- **turzo2023codereview** — Turzo, Faysal, Poddar, Sarker, Iqbal, Bosu (2023), "Towards Automated
  Classification of Code Review Feedback to Support Analytics," ESEM. (in refs.bib)
- **alami2022scrum** — Alami & Krancher (2022), "How Scrum adds value to achieving software
  quality?," Empirical Software Engineering. (in refs.bib)
- **bechky2003sharing** — Bechky (2003), "Sharing Meaning Across Occupational Communities,"
  Organization Science. (in references.bib; organizational-sociology source, not code-specific,
  but already covers the general "occupational tacit knowledge" ground)

No other software-engineering/code-review sources were present. None of the canonical veins named
in the task brief (Fowler/Beck smells, Buse & Weimer, Naur, Sadowski et al., Bosu/Greiler/Bird,
Bacchelli & Bird, Rigby & Bird, Soloway & Ehrlich, LaToza, Détienne, Stegeman, Börstler) were
already cited anywhere in the repo's `.bib` files.

---

## 2. Top finds, ranked

### #1 — Naur, P. (1985), "Programming as Theory Building" ★★★ SCHOLARLY (canonical CS essay)
*Microprocessing and Microprogramming* 15(5):253–261. Reprinted in Naur, *Computing: A Human
Activity* (1992), and (the copy fetched here) as an appendix in a widely-circulated teaching
anthology.

**Why it shows the gap:** Naur's central argument is that the knowledge a programmer needs to
correctly judge whether a proposed modification is "in harmony" with a program's design — i.e.
exactly the kind of quality judgment senior engineers make instantly and juniors can't —
*cannot in principle be reduced to statable rules or criteria*. This is about as direct a
philosophical articulation-gap claim as exists in the software literature.

> [VERIFIED] "The dependence of a theory on a grasp of certain kinds of similarity between
> situations and events of the real world gives the reason why the knowledge held by someone who
> has the theory could not, in principle, be expressed in terms of rules."

> [VERIFIED] "In fact, the similarities in question are not, and cannot be, expressed in terms of
> criteria, no more than the similarities of many other kinds of objects, such as human faces,
> tunes, or tastes of wine, can be thus expressed."

> [VERIFIED] "It only makes sense to the agent who has knowledge of the world, that is to the
> programmer, and cannot be reduced to any limited set of criteria or rules, for reasons similar
> to the ones given above why the justification of the program cannot be thus reduced."

> [VERIFIED, supporting] "the continued adaptation, modification, and correction of errors in
> [large programs], is essentially dependent on a certain kind of knowledge possessed by a group
> of programmers who are closely and continuously connected with them" — illustrated by the case
> where the "highly motivated" receiving team, despite full documentation, could not incorporate
> extensions in a way that matched the original design, while the original team could "spot these
> cases instantly."

URL fetched: https://gwern.net/doc/cs/algorithm/1985-naur.pdf (pp. ~394–398 of this reprint;
original journal page unknown from this copy). Also mirrored (unreadable, scanned) at
https://pages.cs.wisc.edu/~remzi/Naur.pdf.

---

### #2 — Fowler, M. & Beck, K., "Bad Smells in Code" (Ch. 3) ★★★ TRADE-PRACTITIONER (canonical, textbook-level authority)
In Fowler, *Refactoring: Improving the Design of Existing Code*, 1st ed. (Addison-Wesley, 1999).

**Why it shows the gap:** This is the exact "smells are heuristic and can't be made precise"
claim the task brief named. The authors explicitly refuse to give quantitative criteria for when
code is bad, stating outright that no metric beats trained intuition.

> [VERIFIED] "One thing we won't try to do here is give you precise criteria for when a
> refactoring is overdue. In our experience no set of metrics rivals informed human intuition.
> What we will do is give you indications that there is trouble that can be solved by a
> refactoring. You will have to develop your own sense of how many instance variables are too
> many instance variables and how many lines of code in a method are too many lines." (p. 75)

URL fetched (official Pearson sample-pages excerpt containing full Ch. 1–3 front matter):
https://ptgmedia.pearsoncmg.com/images/9780201485677/samplepages/9780201485677.pdf

---

### #3 — Buse, R.P.L. & Weimer, W. (2010), "Learning a Metric for Code Readability" ★★★ ACADEMIC STUDY
*IEEE Transactions on Software Engineering* 36(4):546–558.

**Why it shows the gap:** The paper's entire premise is that readability is a real, high-stakes,
*agreed-upon* property of code that is nonetheless formally undefined — annotators concur on
relative readability well above chance but well short of unanimity, and no prior formal account
existed of *why* they agree.

> [VERIFIED] "A consensus exists that readability is an essential determining characteristic of
> code quality [...], but not about which factors contribute to human notions of software
> readability the most."

> [VERIFIED] Participants were told readability "is [their] judgment about how easy a block of
> code is to understand." "Readability was intentionally left formally undefined in order to
> capture the unguided and intuitive notions of participants."

> [VERIFIED] "This analysis seems to confirm the widely-held belief that humans agree
> significantly on what readable code looks like, but not to an overwhelming extent."

(120 annotators, Spearman's ρ ≈ 0.5–0.7 pairwise agreement — moderate-strong but explicitly *not*
"overwhelming," per the authors' own characterization.)

URL fetched: https://web.eecs.umich.edu/~weimerw/p/weimer-tse2010-readability-preprint.pdf

---

### #4 — Bosu, A., Greiler, M., Bird, C. (2015), "Characteristics of Useful Code Reviews: An Empirical Study at Microsoft" ★★★ ACADEMIC STUDY (industry, MSR 2015)

**Why it shows the gap:** Interview subjects at Microsoft could not agree with each other on
whether the identical *category* of review comment (style/naming/indentation "nit-picks") was
useful or not — the same objectively-described comment type was rated "Useful" by some
interviewees and merely "Somewhat useful" by others, showing the operative criterion for
"good feedback" is not shared even among reviewers doing the same job. Most comments overall
were unrelated to functional correctness at all.

> [VERIFIED] "most of the review comments are unrelated to any types of functional defects."

> [VERIFIED] "Many of the review comments identify what some developers refer to as
> 'nit-picking issues' (e.g., indentation, comments, style, identifier naming, and typos). Some
> of the interviewees rated nit-picking issues as 'Somewhat useful', while others rated those as
> 'Useful.'"

URL fetched (author mirror): https://www.amiangshu.com/papers/CodeReview-MSR-2015.pdf

---

### #5 — Sadowski, Söderberg, Church, Sipko, Bacchelli (2018), "Modern Code Review: A Case Study at Google" ★★★ ACADEMIC STUDY (industry case study, ICSE-SEIP 2018)

**Why it shows the gap:** Google's own internal gatekeeping mechanism for "who is allowed to
approve code style" — the "readability" certification — is explicitly *not* a codified test or
rubric; it is granted when a panel of reviewers subjectively becomes "confident" a developer has
absorbed the community's unwritten norms, i.e. an institutionalized apprenticeship/tacit-transfer
process standing in for a formal standard. The paper's Finding 1 also states directly that
review is centered on readability/maintainability, not defect-checklists.

> [VERIFIED] "Google defines a concept called readability, which was introduced very early on to
> ensure consistent code style and norms within the codebase. Developers can gain readability
> certification in a particular language. To apply for readability, a developer sends changes to
> a set of readability reviewers; once those reviewers are confident the developer understands
> the code style and best practices for a language, the developer is granted readability for that
> language."

> [VERIFIED] "Finding 1. Expectations for code review at Google do not center around problem
> solving. Reviewing was introduced at Google to ensure code readability and maintainability.
> Today's developers also perceive this educational aspect, in addition to maintaining norms,
> tracking history, gatekeeping, and accident prevention. Defect finding is welcomed but not the
> only focus."

URL fetched: https://sback.it/publications/icse2018seip.pdf

---

### #6 — Rigby, P.C. & Bird, C. (2013), "Convergent Contemporary Software Peer Review Practices" ★★★ ACADEMIC STUDY (ESEC/FSE 2013)

**Why it shows the gap:** Quantifies, from an older Lucent formal-inspection dataset the authors
use as a comparison point, that review activity is dominated by uncodified "soft" quality
judgments rather than checklist-style defect finding: a median of 13 "soft maintenance issues"
(coding conventions, comment additions) were found per review versus only 3 true defects (plus 4
false-positive defect claims) — soft, unformalized quality concerns outnumber true defects roughly
4-to-1 even in a rigorous, formal-inspection setting.

> [VERIFIED] "Inspections also found a large number of soft maintenance issues, median 13 per
> review, which included coding conventions, and the addition of comments... An additional 4
> defects per review were found to be false positives." [contrasted against] "a median of 3 true
> defects found per review."

URL fetched (author mirror): https://users.encs.concordia.ca/~pcr/paper/Rigby2013FSE.pdf

---

### #7 — LaToza, T.D., Venolia, G., DeLine, R. (2006), "Maintaining Mental Models: A Study of Developer Work Habits" ★★★ ACADEMIC STUDY (ICSE 2006)

**Why it shows the gap:** Directly documents that the knowledge underlying decisions like code
ownership and design rationale is "usually tacit" — held only in developers' heads, resistant to
externalization even when it would obviously be valuable to write down.

> [VERIFIED] "Personal code ownership is usually tacit, i.e. part of the mental model. Written
> records of ownership, when present, are often out-of-date and distrusted."

> [VERIFIED] "many problems arose because developers were forced to invest great effort
> recovering implicit knowledge by exploring code and interrupting teammates and this knowledge
> was only saved in their memory."

> [VERIFIED] "This information is precious: it is demonstrably useful, demonstrably hard to
> ascertain from the code, and was obtained at a high cost. Yet it is exceedingly rare for this
> developer to then write this information down."

URL fetched: https://www.interruptions.net/literature/LaToza-ICSE06.pdf

---

### #8 — LaToza, T.D. & Myers, B.A. (2010), "Hard-to-Answer Questions about Code" ★★ ACADEMIC STUDY (PLATEAU workshop @ SPLASH/Onward! 2010)

**Why it shows the gap:** Surveyed 179 professional developers on questions about code they
found hard to answer; the single largest category (42 of 371 reports) was *rationale* — "why
wasn't it done this other way?" This is evidence that even the original authors of code, not
just outside reviewers, cannot reliably reconstruct or state the reasoning behind their own
design decisions after the fact — a direct, measured articulation gap about design judgment.

> [VERIFIED] Figure 1 category listing: "Rationale (42): Why wasn't it done this other way? (15)"
> — the most frequently reported category.

> [VERIFIED] "The most frequently reported categories dealt with intent and rationale – what does
> this code do, what is it intended to do, and why was it done this way?"

URL fetched: https://ecs.wgtn.ac.nz/foswiki/pub/Events/PLATEAU/2010Program/plateau10-latoza.pdf

---

### #9 — Soloway, E. & Ehrlich, K. (1984), "Empirical Studies of Programming Knowledge" ★★★ ACADEMIC STUDY (IEEE TSE, canonical)

**Why it shows the gap:** Introduces "rules of programming discourse" — unwritten, convention-like
norms (e.g. variable names should match function) that expert programmers use to form expectations
and instantly detect violations, but which function like an internalized grammar rather than an
explicit checklist; novices lack the rules entirely and so are *not* surprised by the same
violations that jump out at experts. This is the classic expert/novice program-comprehension
result the task brief asked for.

> [VERIFIED — via OCR of scanned original, p. 595] "Rules of Programming Discourse: Rules that
> specify the conventions in programming, e.g., the name of a variable should usually agree with
> its function; these rules set up expectations in the minds of the programmers about what should
> be in the program... These rules are analogous to discourse rules in conversation."

> [VERIFIED — p. 595-596] "a program can be correct from the perspective of the problem, but be
> difficult to write and/or read because it doesn't follow the rules of discourse" and: advanced
> programmers are expected to do much better than novices on plan-conforming programs but to drop
> to novice-level performance on programs that violate the (unstated) discourse rules — i.e. the
> rules are only legible through their violation, not through direct statement.

URL fetched: https://www.ics.uci.edu/~redmiles/inf233-FQ07/oldpapers/SollowayEhrlich.pdf (scanned
PDF; text obtained via `pdftoppm` + `tesseract` OCR of pages 1–2, IEEE TSE SE-10(5), pp. 595–609).

---

### #10 — Stegeman, M., Barendsen, E., Smetsers, S. (2016), "Designing a Rubric for Feedback on Code Quality in Programming Courses" ★★ ACADEMIC STUDY (Koli Calling 2016, CS-education venue)

**Why it shows the gap:** Motivates the whole paper by observing that multiple prior teams have
each tried to formalize "code quality" into a grading scheme, all converging on the same rough
*topics* (readability, style, decomposition) yet producing schemes that are "very diverse in form
as well as in content" — i.e., there is no shared, precise codification even among people
explicitly trying to write one down for the same intuitive target.

> [VERIFIED] "Several grading schemes have been published [...], but while all of these focus on
> similar aspects of code quality, such as readability, style and decomposition, they are very
> diverse in form as well as in content."

URL fetched (author mirror): https://www.stgm.nl/quality/stegeman-quality-2016.pdf

---

## 3. Leads (no verified quote — do not cite without further work)

- **Bacchelli, A. & Bird, C. (2013), "Expectations, Outcomes, and Challenges of Modern Code
  Review," ICSE 2013.** [LEAD] Widely described (by Sadowski et al. and others) as showing that
  what developers say they want from review (defect finding) diverges from what review actually
  delivers (knowledge transfer, team awareness) — precisely an articulation/behavior gap, but
  every mirror I tried (Microsoft Research, TU Delft, zora.uzh.ch) returned either a bot-blocked
  HTML shell or an unparseable stub rather than the PDF text. Needs a working full-text source
  before quoting.
- **Singh, D., Sekar, V.R., Stolee, K.T., Johnson, B. (2017), "Evaluating How Static Analysis
  Tools Can Reduce Code Review Effort," VL/HCC 2017.** [SNIPPET, not primary-verified] A WebSearch
  synthesis reported: "the PMD static analysis tool overlapped with nearly 16% of the reviewer
  comments" across 274 comments from 92 GitHub pull requests. This is exactly the
  "linters catch only X% of what reviewers flag" statistic the brief asked for, but I could not
  locate a fetchable PDF/HTML of the paper itself to confirm the number against primary text — flag
  as SNIPPET only, verify before using in the paper.
- **Börstler, J. et al., "What is Code Quality? — perceptions."** [LEAD] Multiple Börstler papers
  exist on developer perceptions of code quality attributes (readability/structure/correctness
  priorities differing by seniority); search only returned a synthesized secondary description,
  not a citable primary source with page/DOI. Needs the exact title/venue pinned down before use.
- **Détienne, F., *Software Design: Cognitive Aspects* (Springer, 2002).** [LEAD] Reputable
  synthesis of decades of cognitive-science work on programmer expertise and design comprehension;
  no online full text found to pull a verifiable quote from.

## 4. Dead ends

- "Highfill/Beller" on what static analysis misses relative to human review, as named in the task
  brief: no paper by that specific author pairing was locatable; this appears to overlap with the
  Singh et al. (2017) result above rather than being a distinct citable source.
- softwarequotes.com's Martin Fowler page does **not** contain the "no set of metrics rivals
  informed human intuition" line (only lists an unrelated one-word "Code smells" tag) — the quote
  had to be verified against the actual Pearson-hosted book excerpt instead.
- Original Naur PDF mirrors at pages.cs.wisc.edu and the chriskrycho.com CDN are scanned-image
  PDFs with no extractable text layer (`pdftotext` returned 0 lines); the gwern.net mirror of the
  same essay (in a teaching-anthology appendix) has a working text layer and was used instead.
- Direct microsoft.com/research PDF links for Bosu et al. (2015), Bacchelli & Bird (2013), and
  Rigby & Bird (2013) all return a bot-detection HTML shell rather than the PDF; had to be routed
  through author-hosted mirrors (amiangshu.com, concordia.ca) — worked for Bosu and Rigby, not
  for Bacchelli.
