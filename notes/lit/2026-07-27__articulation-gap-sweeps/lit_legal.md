# Literature sweep: articulation gaps in legal argument / judicial judgment

Domain: legal argument, judicial judgment, "thinking like a lawyer," persuasive legal writing,
judges' evaluation of advocacy. Patents excluded (owned by a separate agent).

## (1) Already in our bib — NOT new finds

Checked `notes/articulability-prompt-opt.bib`, `methods/metric_implementer/references.bib`,
`latex/paper-1__metric-codability/refs.bib`.

- `schauer1991playing` — Schauer, *Playing by the Rules* (1991)
- `kaplow1992rules` / `kaplow1992rulesa` (duplicate keys, methods/metric_implementer/references.bib) — Kaplow, "Rules Versus Standards: An Economic Analysis," 42 Duke L.J. 557 (1992)
- `ehrlich1974rules` (latex/paper-1__metric-codability/refs.bib) — Ehrlich & Posner, "An Economic Analysis of Legal Rulemaking," 3 J. Legal Stud. 257 (1974)
- `garner1999winningbrief` — Garner, *The Winning Brief* (1999)
- `balkin1998canons` — Balkin & Levinson, "The Canons of Constitutional Law," 111 Harv. L. Rev. 963 (1998)
- `landes1998judicial` — Landes, Lessig & Solimine, "Judicial Influence: A Citation Analysis of Federal Courts of Appeals Judges," 27 J. Legal Stud. 271 (1998)
- `summers_good_faith_1968` — Summers, "'Good Faith' in General Contract Law...," 54 Va. L. Rev. 195 (1968)
- `restatement2contracts33` — Restatement (Second) of Contracts § 33 (Certainty)
- `ucc2204`, `ucc2305`, `ucc_1_304`, `ucc_2_306` — UCC §§ 2-204, 2-305, 1-304, 2-306
- `varney1916` — *Varney v. Ditmars*, 217 N.Y. 223 (1916)

**Note:** the task brief mentioned a "Vance judicial opinion writing bibliography" as something already cited. I could not find any `vance*` key or "opinion writing" string in any of the three bib files — this appears to be either stale/mistaken in the brief, or filed under a different key/file I didn't check. Flagging rather than guessing.

---

## (2) Top new finds (ranked)

### 1. Hutcheson, "Judgment Intuitive: The Function of the Hunch in Judicial Decision" — the single best find
**Joseph C. Hutcheson Jr., "Judgment Intuitive: The Function of the Hunch in Judicial Decision," 14 Cornell L. Rev. 274 (1929).** JUDICIAL SCHOLARSHIP (a sitting federal appellate judge's own account of his method).
Why it's a gap: a judge explicitly confesses that he reaches the *result* by intuitive "hunch" first and only afterward searches out the doctrinal reasoning to justify it in the written opinion — i.e., the stated reasons in published opinions are demonstrably NOT the actual decision procedure.
- Citation itself **[VERIFIED]** — confirmed directly on Cornell's own repository page: `https://scholarship.law.cornell.edu/clr/vol14/iss3/2/` (title, author, volume/page all match).
- The quotes below are **[SNIPPET]** — they recur verbatim, word-for-word identical, across multiple independent secondary sources that have read the primary text (a University of Baltimore Law Forum piece, T.J. Capurso's "How Judges Judge," and others), but every attempt to fetch the primary PDF directly returned HTTP 403 (Cornell repository, core.ac.uk mirror, UNLV repository, Scribd all blocked the fetch tool). No fabricated page pin is given below because I could not confirm the exact in-text page beyond the article's start page (274).
  > "the judge really decides by feeling and not by judgment, by hunching and not by ratiocination, such ratiocination appearing only in the opinion."
  > "The vital, motivating impulse for the decision is an intuitive sense of what is right or wrong in the particular case."
- Background anecdote (also SNIPPET, consistent across sources): Hutcheson recounts that as a young lawyer he heard a federal judge announce he would "wait for his hunch" before deciding a case, thought it a joke — then, eleven years into his own judgeship, confessed in this essay that he decides the same way.
- **Action for the paper:** worth a second pass with a tool that can get past the 403s (e.g., HeinOnline access, or a direct PDF download via curl with a browser user-agent) to pull the exact page pin before quoting in the paper.

### 2. Jacobellis v. Ohio (Stewart, J., concurring) — "I know it when I see it"
**Jacobellis v. Ohio, 378 U.S. 184, 197 (1964) (Stewart, J., concurring).** JUDICIAL OPINION.
Why it's a gap: a Supreme Court Justice explicitly declines to articulate a workable definition of the legal category at issue (hard-core pornography) and substitutes a bare claim of unarticulable recognition — an admission, on the record, that the legal standard he applies cannot be stated as a rule.
- **[VERIFIED]** — fetched via `https://caselaw.findlaw.com/court/us-supreme-court/378/184.html`.
  > "I shall not today attempt further to define the kinds of material I understand to be embraced within that shorthand description; and perhaps I could never succeed in intelligibly doing so. But I know it when I see it, and the motion picture involved in this case is not that."
- Pin cite: 378 U.S. at 197.

### 3. Cardozo, *The Nature of the Judicial Process* — subconscious forces in judging
**Benjamin N. Cardozo, *The Nature of the Judicial Process* (Yale Univ. Press 1921).** SCHOLARLY BOOK (Lectures; Cardozo was a sitting judge, later Supreme Court Justice).
Why it's a gap: Cardozo (writing as a judge about his own craft) argues that judges are governed by forces "beneath the surface" that they do not consciously access or articulate, and that these subconscious forces — not the stated doctrine — explain why judges are internally consistent yet diverge from one another.
- **[VERIFIED]** — fetched full text via `https://constitution.org/1-Constitution/cmt/cardozo/jud_proc.htm` (Lecture I).
  > "More subtle are the forces so far beneath the surface that they cannot reasonably be classified as other than subconscious. It is often through these subconscious forces that judges are kept consistent with themselves, and inconsistent with one another."
  > "Deep below consciousness are other forces, the likes and the dislikes, the predilections and the prejudices, the complex of instincts and emotions and habits and convictions, which make the man, whether he be litigant or judge."

### 4. Neumann, "Donald Schön, The Reflective Practitioner, and The Comparative Failures of Legal Education" — richest single source found
**Richard K. Neumann Jr., "Donald Schön, The Reflective Practitioner, and The Comparative Failures of Legal Education," 6 Clinical L. Rev. 401 (2000).** LEGAL SCHOLARSHIP (law professor synthesizing Schön's tacit-knowledge framework specifically for legal practice/education).
Why it's a gap: directly imports Polanyi/Schön's "tacit knowing" into legal practice; argues legal skill/judgment is not reducible to the doctrinal rules taught in law school, that "thinking like a lawyer" ≠ "solving problems like a professional," and that law teachers cannot articulate to students what constitutes competent performance until the students have already done it.
- **[VERIFIED]** — fetched and OCR'd directly (pp. 401–413) via `https://redclinicasjuridicas.ar/wp-content/uploads/2021/12/Donald-A.-Schon-Educating-the-Reflective-Legal-Practitioner-2-Clinical-L.-Rev.-231-245-1995..pdf` (the file actually served was Neumann's 2000 Clinical L. Rev. piece, not the underlying 1995 Schön essay it's titled after — verified by reading the extracted pages, which carry Hofstra/Scholarly Commons headers and Neumann's byline).
  > p.404 — "In a typical law school classroom, 'there is presumed to be a right answer for every situation.'" [Neumann quoting Schön, *Educating the Reflective Practitioner*, at 39]
  > p.405 — "'Thinking like a lawyer' is a label used by doctrinal teachers for a collection of textual interpretation skills and heightened forms of skepticism. ... In fact, so much energy has been devoted to textual interpretation and skepticism that we actually know very little about how effective lawyers go about solving problems."
  > p.406 — "[A]ll doctors will tell you that some percentage of the patients that come into the office are not in the book. By this they mean that the standard repertoire of diagnostic and treatment categories does not include this set of patients. Therefore, they need to invent and experiment on the spot..." [quoted from Schön 1995, at 239 — offered as the general professional-judgment analogue Neumann extends to lawyers]
  > p.407 — "So outstanding practitioners are not said to have more professional knowledge than others, but more 'wisdom,' 'talent,' 'intuition,' or 'artistry' — all terms that 'serve not to open up inquiry but to close it off.'" [quoting *Educating the Reflective Practitioner*, at 13]
  > p.407 — "Tacit knowing or knowing-in-action has this property: we exhibit it by the competent behavior we carry out but we are unable to describe what it is that we do." [quoting Schön, "Educating the Reflective Legal Practitioner," 2 Clin. L. Rev. 231, 243 (1995)]
  > p.408 — "in the midst of their education for practice there was a profound sense of mystery. This feeling resulted from the fact that the students literally did not know what they were doing, and their teachers could not tell them — because what the teachers knew how to say the students could not at that point in their experience understand." [quoting *Educating the Reflective Practitioner*, at 166]
  > p.408 — "If asked, would this stonemason have been able to explain how to figure out where in the rock to hit? Probably not. It is 'the kind of knowing that is exhibited by what we do' and not by what we think we know." [Neumann's own illustrative case, built on Schön's framework]
- This source both stands alone as a legal-scholarship find AND gives verified pin-cites into the underlying primary Schön essay, **Donald A. Schön, "Educating the Reflective Legal Practitioner," 2 Clin. L. Rev. 231 (1995)** — worth citing directly as well; direct fetch of that primary PDF failed (scanned/image PDF, non-OCRable by the fetch tool), but every quote above that is attributed to it is reproduced with quotation marks and specific page numbers by Neumann, a named legal-scholarship secondary source, so treat those as **[SNIPPET]**, not [VERIFIED] against the primary.

### 5. "Reasonable man" / "man on the Clapham omnibus" standard — applied, not evidenced
**Healthcare at Home Ltd v. The Common Services Agency [2014] UKSC 49, ¶3 (Lord Reed).** JUDICIAL OPINION.
Why it's a gap: an appellate court explicitly holds that the reasonable-person standard — the workhorse standard of negligence law — is *not* established by evidence (i.e., not something witnesses can testify to or that can be empirically pinned down) but is instead applied by the court's own unarticulated judgment/feel.
- **[SNIPPET]** — quote recovered via a WebFetch summary of the Wikipedia article "Man on the Clapham omnibus" (`https://en.wikipedia.org/wiki/Man_on_the_Clapham_omnibus`), which attributes it to Lord Reed's UKSC judgment; direct fetch of the primary judgment (BAILII `uk/cases/UKSC/2014/49.html` and supremecourt.uk PDF) both failed (403 / not in indexed content). Recommend a follow-up direct fetch of the BAILII or UKSC PDF before final citation.
  > "The behaviour of the reasonable man is not established by the evidence of witnesses, but by the application of a legal standard by the court."

### 6. Llewellyn, *The Common Law Tradition* — "situation sense"
**Karl N. Llewellyn, *The Common Law Tradition: Deciding Appeals* (1960), esp. pp. 121–122 and ch. on "situation sense."** SCHOLARLY BOOK (legal-realist jurisprudence).
Why it's a gap: Llewellyn's "situation sense" is explicitly a description of judges reaching outcomes via an unarticulated, trained perception of "what kind of case this really is" (informed by tacit knowledge of commercial/social realities and a lifetime of case-immersion) rather than by rule-application; it is widely described in the secondary literature as one of Llewellyn's most "obscure" constructs — i.e., even legal scholars cannot fully cash it out into explicit criteria.
- **[LEAD]** — no primary quote fetched (PDF/OCR failures on the Bramble Bush scan; Google Books snippet view not directly fetchable). Secondary characterization only, via a legal-scholarship page (`gongfa.com`) discussing situation sense as identifying "the truly operative facts, and the appropriate analytical framework... on both of which the proper disposition of the case will turn," and citing William Twining's commentary (Twining, *Karl Llewellyn and the Realist Movement*) at pp. 217–226 calling situation sense "one of the more obscure and controversial teachings of The Common Law Tradition."
- Companion earlier work, also a lead: **Llewellyn, *The Bramble Bush: On Our Law and Its Study* (1930)** — legal-realist claim that law students must learn a tacit "feel" for legal reasoning that cannot be taught by rule-recitation; PDF fetch attempts (Purdue-hosted copy) failed due to binary/compressed-stream parsing, so no verbatim quote — [LEAD] only.

### 7. Jerome Frank, *Law and the Modern Mind* (1930) and the "judicial hunch" lineage
**Jerome Frank, *Law and the Modern Mind* (1930).** SCHOLARLY BOOK (legal realism).
Why it's a gap: companion thesis to Hutcheson's — Frank argues the "basic legal myth" (that judges mechanically deduce outcomes from rules) is false, and that decisions are driven by an intuitive "hunch" arising from the judge's reaction to the facts, with the doctrinal reasoning constructed afterward.
- **[SNIPPET]** (via secondary characterization synthesized from search of `kancelaria-skarbiec.pl`/"First the Hunch, Then the Law" and Wikipedia's "Law and the Modern Mind" entry; direct primary-text fetch not attempted successfully — Wikipedia fetch explicitly declined to manufacture quotes it couldn't see in the article body):
  > Frank described how, after studying the record, a judge "gives his imagination play and waits for 'the feeling, the hunch — that intuitive flash of understanding which makes the jump-spark connection between question and decision.'"
- Treat this quote fragment cautiously — it is once-removed (a paraphrase-with-embedded-quote from a secondary web source), not independently cross-checked against a second citation of the same passage. Recommend verifying against a HathiTrust/archive.org scan of the actual 1930 text before use.

### 8. Guthrie, Rachlinski & Wistrich, "Blinking on the Bench: How Judges Decide Cases," 93 Cornell L. Rev. 1 (2007)
ACADEMIC STUDY (empirical/experimental jurisprudence — judges as subjects).
Why it's a gap: a real experimental study of sitting judges showing that judicial decisions are frequently produced by fast, intuitive ("System 1"-style) processing rather than the deliberative legal reasoning judges report using, with judges often unable to fully separate the intuitive component from the reasoned one.
- **[LEAD]** — every fetch attempt (NAWJ-hosted PDF, Vanderbilt repository PDF, SSRN abstract page) either 403'd or returned an unparseable scanned/compressed PDF. Only characterization available: the paper's central thesis is the deliberative/intuitive dichotomy in judging, published Cornell L. Rev. 2007, co-authored by Chris Guthrie, Jeffrey J. Rachlinski, and Andrew J. Wistrich. Companion/earlier paper by the same trio, also a lead only: "Inside the Judicial Mind," 86 Cornell L. Rev. 777 (2001).
- **Action:** this is very likely worth a dedicated re-fetch attempt (e.g., via SSRN full-text download rather than abstract page, or Vanderbilt's non-viewcontent URL) since it is exactly the kind of controlled empirical evidence (real judges, real cases/vignettes) the paper wants, and only the abstract page 403'd rather than being unavailable in principle.

### 9. Richard Posner, *How Judges Think* (Harvard Univ. Press 2008)
SCHOLARLY BOOK (sitting federal appellate judge's account of judicial reasoning).
Why it's a gap: Posner argues that in the "open area" of law (vague statutes, contested constitutional questions — arguably most appellate argument that matters), judges decide based on "unconscious preconceptions," personal experience, and temperament, using articulated legal reasoning as a post-hoc "tiebreaker" justification rather than the actual mechanism of decision.
- **[LEAD]** — no verbatim quote fetched (only paraphrased characterizations from secondary book-review sources: SLU law review review article, Case Western law review piece). Worth a targeted fetch of Google Books preview or a book review that quotes directly.

### 10. Elizabeth Mertz, *The Language of Law School: Learning to "Think Like a Lawyer"* (Oxford Univ. Press 2007)
ACADEMIC STUDY (linguistic ethnography of eight U.S. law school classrooms).
Why it's a gap: empirically documents that "thinking like a lawyer" is transmitted through Socratic-method classroom language practices — a tacit linguistic/ideological retraining (stripping moral/emotional/social context from case facts) — rather than through explicit, statable rules; students absorb it by immersion, not instruction.
- **[LEAD]** — no verbatim quote fetched. Secondary paraphrase only, via Edinburgh Law Review abstract, law-and-language blog posts, and a Cambridge/Language in Society review notice. Full text available at `archive.org/details/languageoflawsch0000mert` — worth a direct fetch/borrow attempt in a follow-up pass.

### 11. Sullivan et al. (Carnegie Foundation), *Educating Lawyers: Preparation for the Profession of Law* (2007)
ACADEMIC STUDY / policy report (the "Carnegie Report" on legal education).
Why it's a gap: formalizes legal competence into three "apprenticeships" — cognitive, practical-skills, and professional-identity — with the latter two explicitly characterized as learned through supervised practice/observation (apprenticeship) rather than propositional instruction, i.e., an official acknowledgment that core lawyering judgment is not rule-transmissible.
- **[LEAD]** — paraphrase-level only (three-apprenticeships framework, and the "combine... into the capacity for judgment guided by a sense of professional responsibility" formulation), no page-pinned verbatim quote fetched successfully.

---

## (3) Additional leads (not yet verified even at SNIPPET level — worth a follow-up pass)

- **Karl Llewellyn, "situation sense"** in *The Common Law Tradition* — see #6 above; needs a HathiTrust/archive.org full-text fetch.
- **Jerome Frank / Hutcheson secondary syntheses** — "First the Hunch, Then the Law" (`kancelaria-skarbiec.pl/en/psychology-court/`) is a decent single round-up piece connecting Hutcheson, Frank, and modern judicial-hunch scholarship; worth a direct fetch pass (not attempted due to search-budget exhaustion mid-session).
- **Douglas H. Ginsburg, "Of Hunches and Mere Hunches: Two Cheers for Terry"** (SSRN) — a sitting D.C. Circuit judge's own essay on the judicial hunch literature, turned up in search but not fetched.
- **Timothy J. Capurso, "How Judges Judge: Theories on Judicial Decision Making"** — law-review synthesis piece that appears (per search snippets) to directly quote Hutcheson's "ratiocination" line; a good secondary target to re-attempt fetching (Scribd copy 403'd; look for a law-school repository mirror instead).
- **Wistrich, Guthrie & Rachlinski, "Inside the Judicial Mind," 86 Cornell L. Rev. 777 (2001)** — predecessor empirical study to "Blinking on the Bench"; not independently searched this session (budget ran out) but almost certainly relevant and probably easier to access.
- **"Bramble Bush Revisited" symposium piece** (`jle.aals.org/cgi/viewcontent.cgi?article=1040&context=home`) — turned up in search on Llewellyn but not fetched; likely has direct Bramble Bush quotes on legal-education tacitness.
- **Grading/inter-rater reliability of legal writing** — I was not able to locate a specific empirical study documenting low inter-rater reliability in law-school essay/brief grading before the web-search budget for this session was exhausted (200/200 WebSearch calls used, shared across the session). This vein (moot-court judge scoring agreement, bar-exam essay grader reliability, legal-writing-rubric validity studies) is very likely to exist in the legal-education/psychometrics literature and is worth a dedicated follow-up search — I did not get to run it.

## (4) Dead ends

- **Cornell Law Review / core.ac.uk / UNLV / Scribd copies of Hutcheson's primary 1929 text** — all returned HTTP 403 to the fetch tool; the citation is solid but the exact in-article page pin for either quoted line could not be independently confirmed against the primary.
- **web.archive.org** — WebFetch tool explicitly cannot access this domain in this environment; blocks off the usual fallback route for paywalled/403'd primary sources.
- **BAILII and UKSC.gov.uk direct judgment fetches for Healthcare at Home v. Common Services Agency** — both 403'd or returned only an index page, not full text; the Clapham-omnibus quote rests on a Wikipedia intermediary, not the primary judgment.
- **Guthrie/Rachlinski/Wistrich "Blinking on the Bench" full text** (NAWJ PDF, Vanderbilt repository PDF, SSRN abstract) — all three attempts either 403'd or returned an unparseable scanned/compressed-stream PDF; only the paper's existence and thesis, not any verbatim text, could be confirmed.
- **Llewellyn's *Bramble Bush* PDF (Purdue-hosted copy)** — fetched successfully (200 OK) but the tool could not parse the compressed PDF stream into readable text; no quote recoverable this way.
- **Donald Schön's own 1995 "Educating the Reflective Legal Practitioner" primary PDF** (`redclinicasjuridicas.ar` copy) — the URL actually served Neumann's 2000 secondary article instead (mislabeled file/host mismatch); Schön's primary text itself remains unfetched directly, though richly quoted secondhand by Neumann with page pins.
