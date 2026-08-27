# Lit sweep: articulation gaps in expert humor/comedy judgment

Domain: HUMOR / COMEDY — professional comedians, comedy editors (New Yorker cartoons), sitcom
writers, comedy theory, whether experts agree on funniness while failing to state the rule.

## 1. Already in our bib (NOT new finds)

Checked `notes/articulability-prompt-opt.bib`, `methods/metric_implementer/references.bib`,
`latex/paper-1__metric-codability/refs.bib`.

- `freud1916wit` — Freud, *Wit and Its Relation to the Unconscious* (trans. Brill), 1905/1916 —
  in `latex/paper-1__metric-codability/refs.bib`.
- `alnajjar2006undescribable` — Al-Najjar, Anderlini & Felli, "Undescribable Events," *Review of
  Economic Studies* 73(4), 2006 — in `methods/metric_implementer/references.bib` (this is an
  economics/decision-theory cite, not humor-specific — presumably used for the general
  "undescribability" framing, not comedy content).
- `npr2014getit` — NPR, "`New Yorker' Cartoon Editor Explores What Makes Us Get It," *All Things
  Considered*, 24 March 2014 — in `latex/paper-1__metric-codability/refs.bib`. (Bob Mankoff
  interview; I re-attempted to fetch this for a stronger pull-quote — two fetch attempts timed
  out, so I could not add a new verbatim line from it. It remains cited only as already present.)

No other humor/comedy/cartoon/joke entries exist in any of the three bibs.

---

## 2. Top new finds (ranked)

### (1) Amir, O. (2016), "The Frog Test: A Tool for Measuring Humor Theories' Validity and Humor Preferences" — ACADEMIC STUDY

*Frontiers in Human Neuroscience*, 10:40. https://doi.org/10.3389/fnhum.2016.00040
https://pmc.ncbi.nlm.nih.gov/articles/PMC4746281/

**Why it shows a gap:** This is a rare piece of academic humor-theory methodology that directly
confronts the problem this paper cares about — that competing theoretical accounts of *why*
something is funny (incongruity-resolution, superiority, relief, benign violation, etc.) cannot be
adjudicated by asking people to introspect and report which theory matches their experience,
because people are bad at introspecting on their own humor judgments. It proposes an indirect,
behavioral test ("the frog test") explicitly because direct articulation is unreliable.

> **[VERIFIED]** "Proponents of the different theoretical accounts often show a high degree of
> conviction, suggesting introspection might not be the best tool for judging the validity of
> humor theories." — Amir (2016), *Frontiers in Human Neuroscience* 10:40, page unknown (fetched
> via PMC full text, https://pmc.ncbi.nlm.nih.gov/articles/PMC4746281/)

> **[VERIFIED]** "I do not propose the method is superior over the other approaches, rather it is
> a qualitatively different method, and as such, it might yield novel insights." — same source.

Note: the paper opens by invoking a paraphrase of the E.B. White frog-dissection line (see #2
below) and explicitly extends the metaphor: "if the surgeon is competent the frog might survive.
That is to say, if a theory explains with some accuracy why a particular joke is funny, such
explanation might not reduce the perceived funniness of the joke to the same extent an invalid
theoretical explanation might." [VERIFIED, same source] — i.e., the paper takes seriously, as a
testable premise, the idea that *correct* explanation is less corrosive to funniness than
*incorrect* explanation, which is itself evidence the field treats "explaining kills it" as a live
empirical question rather than folklore.

---

### (2) E. B. White & Katharine S. White, "Some Remarks on Humor," preface to *A Subtreasury of American Humor* (Coward-McCann, 1941), p. xvii — TRADE / LITERARY (the canonical epigram)

**Why it shows a gap:** This is the classic statement that dissecting humor to state its rules
destroys the thing itself — precisely the "explicit criteria kill or fail to capture the
phenomenon" claim. Widely misattributed and mis-paraphrased, so verification matters.

> **[SNIPPET, via secondary scholarly-quotation source]** "Humor can be dissected, as a frog can,
> but the thing dies in the process and the innards are discouraging to any but the purely
> scientific mind." — E. B. White and Katharine S. White, *A Subtreasury of American Humor*
> (1941), Preface, p. xvii. Verified via Quote Investigator's sourced writeup
> (https://quoteinvestigator.com/2014/10/14/frog/), which traces it to the 1941 Coward-McCann
> preface and to a near-identical printing in *The Saturday Review of Literature* (18 Oct 1941). I
> did not myself obtain a scan of the 1941 book page, so I label this SNIPPET rather than
> VERIFIED-primary; it should be treated as reliably sourced but not primary-checked.

**Verification notes (important for citation accuracy):**
- Correct attribution is **joint** — E. B. White *and* Katharine S. White, not E. B. White alone
  (Quote Investigator explicitly credits both as co-editors/co-authors of the preface).
- The word is **"purely"** scientific mind in the primary printing, not "pure" — the version
  circulating as "pure scientific mind" (as flagged in the task prompt) is a common corruption.
- The much shorter, punchier "Analyzing humor is like dissecting a frog. Few people are interested
  and the frog dies of it." is a *paraphrase* that circulates independently (and is itself quoted,
  e.g., by Amir 2016 above, attributed there simply to "Amir, 2016" recounting the folk paraphrase)
  — do not conflate the two wordings when quoting.
- The frog-dissection line is also sometimes misattributed to Mark Twain; Quote Investigator finds
  no substantive evidence for that attribution.

---

### (3) Hessel, Marasović, Hwang, Lee, Da, Zellers, Mankoff & Choi, "Do Androids Laugh at Electric Sheep? Humor 'Understanding' Benchmarks from The New Yorker Caption Contest," ACL 2023 (Best Paper) — ACADEMIC STUDY

arXiv:2209.06293 / https://aclanthology.org/2023.acl-long.41/ — co-authored with Bob Mankoff
(former New Yorker cartoon editor), built on 14 years / 700+ New Yorker Caption Contests.

**Why it shows a gap:** Three escalating tasks — match caption to cartoon, pick the funnier
caption, explain why the winning caption is funny — are precisely a "can you state the rule"
ladder. State-of-the-art multimodal/LLM systems fail hard, and even when given ground-truth scene
descriptions (i.e., the "hard part" of visual understanding handed to them), their *explanations*
of the joke still lose to human-written ones in a supermajority of head-to-head comparisons. This
is evidence that whatever New Yorker editors and readers use to recognize a winning caption is not
something current explicit/statistical decomposition captures, even at large scale.

> **[VERIFIED, fetched via ar5iv full-text render, https://ar5iv.labs.arxiv.org/html/2209.06293]**
> "We find that both types of models struggle at all three tasks. For example, our best
> multimodal models fall 30 accuracy points behind human performance on the matching task, and,
> even when provided ground-truth visual scene descriptors, human-authored explanations are
> preferred head-to-head over the best machine-authored ones (few-shot GPT-4) in more than 2/3 of
> cases." — Hessel et al. (2023), abstract, arXiv:2209.06293, page unknown (arXiv preprint, no
> pagination in the render I fetched).

Caveat on framing: this is a human-vs-model gap paper, not a paper that directly documents *human
editors failing to articulate their own rule* — I could not get readable text out of the actual
PDF (binary-only fetch) to check for passages about editors' own inarticulacy, only the abstract
via ar5iv. Use it for the "funniness resists explicit/statistical decomposition even under heavy
optimization pressure" angle, not for a direct editor-admission quote.

---

### (4) Shahaf, Horvitz & Mankoff, "Inside Jokes: Identifying Humorous Cartoon Captions," KDD 2015 — ACADEMIC STUDY — LEAD (existence/topic confirmed, no verbatim quote obtained)

ACM SIGKDD '15; co-authored with Bob Mankoff. https://dl.acm.org/doi/10.1145/2783258.2783388 ;
PDF at https://erichorvitz.com/phumor.pdf (fetch returned binary-only, unreadable) and Microsoft
Research listing at
https://www.microsoft.com/en-us/research/publication/inside-jokes-identifying-humorous-cartoon-captions/
(abstract not rendered in that page's text either).

**Why it's relevant (from search-result synthesis, not a direct quote — mark LEAD):** builds a
supervised pairwise "which caption is funnier" classifier over a large crowdsourced corpus of
New Yorker contest captions, using features like sentiment, perplexity, readability, and
keyword-based description of the cartoon's visual anomaly — i.e., an explicit attempt to formalize
funniness features for exactly the domain where the editor (Mankoff, a co-author) presumably
judges by feel. I was not able to fetch readable full text to pull a direct quote about editors'
inarticulacy; flagging as LEAD until someone can pull the PDF text directly (try the ACM DL PDF
behind institutional access, or Google Scholar cached HTML).

---

### (5) Sam Friedman, *Comedy and Distinction: The Cultural Currency of a 'Good' Sense of Humour* (Routledge/CRESC, 2014) — SCHOLARLY BOOK — LEAD, with an important complicating finding

First large-scale empirical sociological study of British comedy taste (survey + interviews at the
Edinburgh Festival Fringe). This was the single most-recommended source in the prompt, so I pushed
hardest on it, with mixed results.

**What I could verify:** I could not get readable text out of the LSE eprints PDF itself (binary
fetch both from `eprints.lse.ac.uk` and its `researchonline.lse.ac.uk` redirect target). I did get
readable text from a review of the book:

> **[VERIFIED, via review, https://journals.openedition.org/lectures/19758?lang=en]** Describing
> Friedman's high-cultural-capital ("HCC") comedy audiences: they "emphasize cleverness and
> difficulty. A good comedy should be hard to understand: you need to be 'working for your
> laughter'" — whereas low-cultural-capital ("LCC") audiences state that comedy "equates ... to
> laughter, pleasure, and relaxation. Comedy must be funny."

**Important complication for our thesis:** this review's summary suggests Friedman's respondents
*do* offer explicit, articulate, class-patterned criteria for what makes comedy good (cleverness/
difficulty vs. pleasure/relaxation) — i.e., the book's headline finding (as far as I could verify
secondhand) is about **stratified explicit taste criteria**, not about inarticulacy or a
stated-criteria/actual-behavior gap. That doesn't rule out relevant passages elsewhere in the book
(e.g., interviewees falling back on "it's just funny" or "I don't know why, I just like him" when
pressed past their headline justification, which is a very common finding in taste/distinction
ethnography generally, per Bourdieu's tradition) — but I do not have a verified quote for that
specific claim from this book. Recommend: pull the book directly (not just a review) before citing
it for the articulation-gap claim specifically; it is very likely still useful for the paper's
domain generally (empirical sociology of comedy taste) even if the "inarticulacy" angle needs
sourcing from elsewhere in it.

---

## 3. Leads (no verified quote obtained — do not cite without further verification)

- **Attardo & Raskin, General Theory of Verbal Humor (GTVH)** — a structural theory of joke
  similarity that is widely acknowledged (informally) to explain *taxonomic* relationships between
  jokes but not to predict *degree of funniness* or quality ranking. I could not find a
  Wikipedia page at the URL I tried (404) or pull a direct scholarly critique text in the time
  available. Worth pursuing directly via Attardo's own later work (he has written self-critically
  about GTVH's limits) or via Ritchie's critiques.
- **Graeme Ritchie, *The Linguistic Analysis of Jokes* (2004)** — not fetched/verified this pass.
- **Binsted & Ritchie** computational humor generation work — not fetched.
- **Mihalcea & Strapparava**, computational humor recognition papers — not fetched.
- **Willibald Ruch**, 3WD model / humor appreciation structure — not fetched.
- **Warren & McGraw, benign violation theory** — I fetched the Wikipedia "Benign violation theory"
  page directly; it describes the original 2010 church/Hummer-raffle study but contains **no**
  cited critique of predictive failure or falsifiability (see Dead Ends below). A direct critique
  would need to come from primary journal sources (e.g., commentaries in *Psychological Science*
  or *HUMOR* responding to McGraw & Warren 2010), not yet located.
- **Mike Sacks, *Poking a Dead Frog: Conversations with Today's Comedy Writers* (2014)** — trade
  interview collection, exactly the "you either have it or you don't" flavor of material the
  prompt wants; not fetched this pass (Wikipedia page 404'd). Recommend direct excerpt-hunting
  (e.g., Google Books "look inside") next round.
- **Judd Apatow, *Sick in the Head* (2015)** — same category, not fetched.
- **Michael Billig, *Laughter and Ridicule* (2005)** — sociological/critical theory of humor and
  power; not fetched this pass.
- **John Limon, *Stand-up Comedy in Theory, or, Abjection in America* (2000)** — not fetched.
- **Bielby & Bielby, "'All Hits Are Flukes': Institutionalized Decision Making and the Rhetoric of
  Network Prime-Time Program Development," *American Journal of Sociology* 99(5), 1994** —
  classic sociology-of-culture paper on TV executives who explicitly disavow ability to predict/
  articulate what makes a show succeed ("nobody knows anything," per William Goldman's screenwriting
  aphorism). Not comedy-specific (covers all prime-time programming) and not fetched/verified this
  pass, but structurally exactly the "professional consensus that success criteria are
  inarticulable" pattern the prompt wants; flag as a lead for the "TV/entertainment industry
  professional judgment" angle rather than "what's funny" per se.
- **Sitcom writers'-room ethnographies** — the prompt suggested "Phalen & Osellame" and
  "Henderson, *Ethnographies of the Writers' Room*" by name; I could not locate or verify either
  as real published works in the time/search budget available (WebSearch quota was exhausted
  mid-task — see Dead Ends). Treat these two names as unverified leads only; do not cite them
  without independently confirming they exist.

---

## 4. Dead ends

- **Wikipedia "General theory of verbal humor"** — page does not exist at the URL I tried (404).
- **Wikipedia "Benign violation theory"** — page exists but contains no predictive-failure
  critique; only the original supportive 2010 finding.
- **Wikipedia "Poking a Dead Frog"** — page does not exist (404).
- **Wikipedia "Robert Mankoff"** — exists but has no quoted material on his judgment process
  beyond a bare mention that he wrote process essays for The New Yorker; no verbatim inarticulacy
  quote.
- **NPR transcripts (2017 "Outgoing 'New Yorker' Cartoon Editor Says 'Being Funny Is Being
  Awake'" and 2012 "Not Funny Enough, New Yorker Gives Seinfeld Cartoon a Second Chance")** —
  fetched (or attempted; several timed out) without turning up a usable inarticulacy quote from
  Mankoff. A WebSearch synthesis surfaced a promising paraphrase — Mankoff reportedly saying
  something like "sometimes things ... just seem weird and funny, and you can't explain" — but I
  could not confirm this against the actual source page text (fetch attempts on the likely source,
  npr.org/transcripts/523992723, timed out twice), so it is **[LEAD] only, unverified**, not safe
  to quote yet.
- **degruyterbrill.com HUMOR journal review of Friedman's book** — 405 Method Not Allowed, could
  not fetch.
- **LSE Review of Books / academia.edu copies of the Friedman book review** — 404 / 403, could not
  fetch.
- **Shahaf/Horvitz/Mankoff (KDD 2015) PDF, both at erichorvitz.com and Microsoft Research** —
  fetch tool returned only binary/unreadable PDF structure both times; no verbatim text obtained
  despite the paper's existence and topic being well-confirmed via search.
- **WebSearch budget exhausted mid-task** (session-wide cap, shared with earlier work in this
  conversation) — after roughly 10 targeted queries I lost the ability to issue new searches and
  had to fall back on WebFetch against URLs found earlier or guessed from domain knowledge. This
  is why several of the prompt's named leads (Phalen & Osellame, Henderson, Ritchie, Ruch,
  Mihalcea/Strapparava, Attardo's self-critique of GTVH) remain unverified rather than resolved.
