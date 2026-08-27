# Lit recon: measuring linguistic register / lexical stratification

Purpose: survey the measurement literature for a "register-height index" over criterion
names — i.e., can we score whether "help" vs "assist" vs "facilitate" (or "teacher" vs
"professor") sit at different lexical register heights, and can we do this computationally
and validate it. Sources below were confirmed live this session (WebFetch against ACL
Anthology, PubMed, Wikipedia, NeurIPS proceedings, DuckDuckGo) unless flagged [UNVERIFIED].

---

## 1. Corson's "Lexical Bar" thesis (education / class barrier)

### Corson, D. (1985). *The Lexical Bar*. Oxford: Pergamon Press.
**Summary:** Corson's central claim is that Graeco-Latin (G-L) vocabulary populates the
knowledge categories of the English-medium school curriculum "almost to the exclusion of
Anglo-Saxon words," and that this creates a durable barrier to educational success for
students (working-class, some ethnic-minority) whose home register is Anglo-Saxon/everyday
English rather than the Latinate academic register. He built a "Graeco-Latin (G-L) Instrument"
to score vocabulary complexity and used it to track lexical change in students aged 12-15,
cross-cut by social class, region, and ethnicity, across 144 semantic subdivisions
corresponding to areas of the school curriculum (human intellectual activity).
**Relevance to us:** This is the founding framing citation for "register height as a social
barrier" — directly motivates why a Latinate/Germanic split is not just etymological trivia
but a live class/access signal. The G-L Instrument is the closest historical precedent to a
"register-height index" we're trying to build, though its exact scoring rule is not
recoverable from secondary sources alone (would need to track down the original book chapter
for the instrument's item-level scoring criteria).

### Coxhead, A. (2000). "A New Academic Word List." *TESOL Quarterly*, 34(2), 213-238.
**Summary:** Introduces the Academic Word List (AWL): 570 word families extracted from a
3.5M-word corpus of academic writing, specifically the words that appear frequently in
academic texts but fall outside the 2,000 most common general-English words (building on
West's 1953 General Service List). AWL words are heavily Latinate/Greek in origin (e.g.
"constitute," "hypothesis," "acquisition").
**Relevance to us:** Gives a ready-made, hand-curated binary feature — "is this criterion
name (or its content morphemes) an AWL headword?" — as a cheap, well-validated proxy for
"sits in the academic/Latinate register," complementary to and partially overlapping with
Corson's thesis but with an off-the-shelf downloadable list.

---

## 2. Etymological stratification (Germanic / Latinate / Greek layers)

### de Melo, G. (2014). "Etymological Wordnet: Tracing the History of Words." *LREC 2014*, Reykjavik. ACL Anthology: L14-1063. Project site: etym.org.
**Summary:** Introduces the first large machine-readable, cross-lingual network of word-origin
(etymology) relations, mined largely from Wiktionary. It links words across languages via
etymological relations (derived-from, etymologically-related, has-derived-form, etc.),
letting you trace an English word's descent back through Old English/Germanic vs.
Latin/French/Greek donor languages.
**Relevance to us:** The most directly reusable **method** for computing a per-word
"etymological stratum" label (Germanic vs. Latinate vs. Greek vs. other) at scale — this is
feature #1 on our operationalization shortlist below.

---

## 3. Formality measurement (lexical formality scores, Latinate ↔ formality link)

### Brooke, J., Wang, T., & Hirst, G. (2010). "Automatic Acquisition of Lexical Formality." *Coling 2010: Posters*, pp. 90-98, Beijing. ACL Anthology: C10-2011.
**Summary:** Builds an automatically-acquired lexicon scoring individual words for formality,
using a semi-supervised approach that propagates formality labels through a word
co-occurrence/similarity graph seeded from a small set of formal/informal word pairs (this
lineage continues in the authors' related 2013-2014 papers on "lexical style" and "continuous
lexical attributes," found alongside this one on ACL Anthology).
**Relevance to us:** Directly gives a **word-level formality lexicon** we could either reuse
(if released) or replicate the graph-propagation method on our own criterion-name vocabulary —
this is the clearest prior art for "assign every word a scalar formality score."

### Pavlick, E., & Tetreault, J. (2016). "An Empirical Analysis of Formality in Online Communication." *TACL*, 4:61-74. DOI: 10.1162/tacl_a_00083. ACL Anthology: Q16-1005.
**Summary:** Collects human formality judgments across four genres (news, blogs, email,
question-answer forums — the "PT16" formality corpus) and trains a statistical formality
predictor evaluated across feature settings and genres; applies the model to formality
coordination effects in online discussion forums.
**Relevance to us:** The canonical **sentence/document-level formality classifier + training
corpus**; useful as a validation target (does our Latinate-ratio/etymology-based score
correlate with PT16-style formality judgments?) even though PT16 was built for sentences, not
isolated criterion-name phrases.

### Heylighen, F., & Dewaele, J-M. (1999). "Formality of Language: Definition, Measurement and Behavioral Determinants." Internal Report, Center "Leo Apostel," Vrije Universiteit Brussel.
**Summary:** Proposes a foundational distinction between **deep formality** (minimizing
context-dependence/ambiguity via explicit, precise language) and **surface formality**
(the observable linguistic correlates of deep formality — more nouns/prepositions, fewer
pronouns/verbs — the basis of their computable "F-score"), and argues formality is "the most
important dimension of variation between styles or registers." Shows formality scores vary
systematically across professional domains and across Dutch/English texts.
**Relevance to us:** Framing citation distinguishing "deep" (semantic/social) vs. "surface"
(lexico-grammatical, computable) formality — useful for arguing our lexical register-height
index measures *surface* formality as a proxy for the deeper construct, and the F-score
formula (noun+adjective+preposition+article ratio minus pronoun+verb+adverb+interjection
ratio) is a directly reusable POS-based formula, independent of etymology.

---

## 4. Psycholinguistic norms as register-height proxies

### Kuperman, V., Stadthagen-Gonzalez, H., & Brysbaert, M. (2012). "Age-of-acquisition ratings for 30,000 English words." *Behavior Research Methods*, 44, 978-990.
**Summary:** Crowdsourced age-of-acquisition (AoA) ratings — the age at which a word is
typically learned — for ~30,000 English words, superseding smaller adult-rated AoA norm sets.
Early-acquired words are known to be processed faster (e.g., in picture naming) than
late-acquired words.
**Relevance to us:** AoA is a strong candidate proxy for register height: words acquired late
(often Latinate/academic, e.g. "facilitate," "articulate") plausibly index higher/more formal
register than early-acquired synonyms (e.g. "help"). Coverage is single-word norms, so
multi-word criterion names need decomposition.

### Brysbaert, M., & New, B. (2009). "Moving beyond Kučera and Francis: A critical evaluation of current word frequency norms and the introduction of a new and improved word frequency measure for American English." *Behavior Research Methods*, 41(4), 977-990. DOI: 10.3758/brm.41.4.977. (SUBTLEX-US)
**Summary:** Critiques traditional written-corpus frequency norms (e.g. Kučera & Francis) and
introduces SUBTLEX-US, word frequency counts derived from ~51M words of film/TV subtitles,
shown to better predict lexical-decision and naming latencies than older written-text-based
norms.
**Relevance to us:** Canonical **log-frequency covariate**. Frequency correlates with but is
distinct from register/formality (common words skew informal; rare words skew formal or
technical) — best used as a control variable alongside a register-specific score rather than
as the register measure itself.

### Brysbaert, M., Warriner, A. B., & Kuperman, V. (2014). "Concreteness ratings for 40 thousand generally known English word lemmas." *Behavior Research Methods*, 46(3), 904-911. DOI: 10.3758/s13428-013-0403-5.
**Summary:** Crowdsourced concreteness ratings (>4,000 raters) for 37,058 English word lemmas
and 2,896 two-word expressions, restricted to lemmas recognized by ≥85% of raters; ratings
draw on all sensory/motor experience though visual/haptic experience dominated raters'
judgments.
**Relevance to us:** Concreteness is a plausible confound/covariate for register height —
abstract nominalizations ("facilitation," "articulability") are both more Latinate and more
abstract than their concrete Germanic counterparts, so concreteness ratings let us check
whether an apparent register effect is really just an abstractness effect.

---

## 5. Sociolinguistic prestige (overt vs. covert, Labov/Trudgill)

### Labov, W. (1966). *The Social Stratification of English in New York City*. Washington, DC: Center for Applied Linguistics.
**Summary:** Classic department-store study (Saks Fifth Avenue vs. Macy's vs. S. Klein,
proxying high/mid/low-status retail) showing postvocalic /r/ pronunciation — the prestige
variant — increases with store status and with more careful/formal speech style, establishing
the overt-prestige paradigm: standard variants index power/status and are consciously valued.
**Relevance to us:** Canonical operationalization of **overt prestige** via observed
production correlated with an independent status/formality proxy (store class here; register
of the eliciting context there) — a template for eliciting register judgments in different
"formality contexts."

### Trudgill, P. (1972). "Sex, covert prestige and linguistic change in the urban British English of Norwich." *Language in Society*, 1(2), 179-195.
**Summary:** Found Norwich working-class male speakers *under-reported* their own use of
standard variants relative to observed speech (claiming to use more non-standard forms than
they did), which Trudgill attributes to **covert prestige** — non-standard variants indexing
in-group solidarity/toughness rather than status, valued especially by men.
**Relevance to us:** Establishes that "prestige"/register-height is not unidimensional or
universally agreed — a word can rank high on an overt (formal/institutional) prestige scale
while a *different*, locally-valued dimension (solidarity/authenticity) favors the
lower-register synonym. Important caveat for interpreting any single register-height score:
it captures overt/formal prestige, not covert/in-group prestige, and the two can diverge.

### Joos, M. (1961). *The Five Clocks*. New York: Harcourt, Brace and World.
**Summary:** Proposes five discrete formality "styles"/registers along a single cline —
frozen (fixed, e.g. liturgical text), formal (one-way, technical vocabulary salient),
consultative (two-way, background supplied), casual (in-group, slang/ellipsis), intimate
(private, minimal explicit content) — still the standard folk-taxonomy reference point for
"register level" in sociolinguistics and language teaching.
**Relevance to us:** Gives us discrete, human-interpretable **anchor labels** (frozen →
intimate) we could use as a rubric/scale for an LLM-judge register-height annotation task,
rather than only a continuous computed score.

---

## 6. Codability lineage (naming agreement, communication accuracy, Zipf/Pitman-Yor)

### Brown, R. W., & Lenneberg, E. H. (1954). "A study in language and cognition." *Journal of Abnormal and Social Psychology*, 49(3), 454-462. PMID: 13174309.
**Summary:** Introduces **codability** — operationalized via color-naming experiments,
testing whether English speakers more easily remember/recognize color chips that have short,
high-agreement, high-frequency names (vs. chips requiring long, hesitant, low-agreement
descriptions), and whether naming-ease differs across languages (English vs. Zuni) with
different color-term inventories, in a test of the Whorfian "weak" hypothesis (structural
differences in language parallel non-linguistic cognitive differences).
**Relevance to us:** This is the founding **codability measurement paradigm**: quantify
naming ease/agreement (response latency, name length, inter-subject naming agreement) as a
proxy for how "entrenched"/available a concept-word pairing is — directly analogous to asking
whether one register variant of a criterion name is more "codable"/natural than another.

### Lantz, D., & Stefflre, V. (1964). "Language and cognition revisited." *Journal of Abnormal and Social Psychology*, 69(5), 472-481. DOI: 10.1037/h0043769.
**Summary:** Extends Brown & Lenneberg's codability paradigm with **communication accuracy**:
rather than just naming-agreement/latency, they measure how accurately a listener can pick
the correct referent (e.g., a specific color chip) out of an array, given only another
speaker's verbal description — a behavioral, task-grounded successor metric to raw codability.
**Relevance to us:** Suggests a **validation task** for any register-height index: does a
"more codable"/lower-register name actually communicate the target concept more accurately to
a naive listener than a rarer, higher-register synonym? This gives us a downstream behavioral
check, not just an intrinsic lexical score.

### Snodgrass, J. G., & Vanderwart, M. (1980). "A standardized set of 260 pictures: Norms for name agreement, image agreement, familiarity, and visual complexity." *Journal of Experimental Psychology: Human Learning and Memory*, 6(2), 174-215. DOI: 10.1037/0278-7393.6.2.174.
**Summary:** The classic modern **naming-norms** dataset: 260 standardized line drawings
normed for name agreement (do independent namers converge on the same word?), image
agreement, familiarity, and visual complexity; the four measures were found to be largely
uncorrelated, i.e. independent attributes of a picture-concept pairing.
**Relevance to us:** "Name agreement" is a direct, modern operationalization of Brown &
Lenneberg's codability, and the finding that the four norm dimensions are uncorrelated is a
methodological caution: a register-height score should not be assumed to track familiarity or
complexity automatically — they need to be measured/controlled separately.

### Duñabeitia, J. A., Crepaldi, D., Meyer, A. S., New, B., Pliatsikas, C., Smolka, E., & Brysbaert, M. (2018). "MultiPic: A standardized set of 750 drawings with norms for six European languages." *Quarterly Journal of Experimental Psychology*, 71(4), 808-816. DOI: 10.1080/17470218.2017.1310261.
**Summary:** Modern, multilingual (6-language) successor to Snodgrass & Vanderwart: 750
colored drawings of concrete concepts normed for name agreement, image agreement,
familiarity, visual complexity, image variability, and age of acquisition, hosted openly by
BCBL.
**Relevance to us:** Shows the naming-agreement/codability paradigm is still actively
maintained and cross-linguistically extended — a template if we ever want cross-language
register-height comparison, and its AoA-norm bundling reinforces AoA as a standard companion
measure to naming-agreement/codability.

### Goldwater, S., Johnson, M., & Griffiths, T. L. (2005). "Interpolating between types and tokens by estimating power-law generators." *Advances in Neural Information Processing Systems 18 (NeurIPS 2005)*.
**Summary:** Introduces a Bayesian framework in which a **Pitman-Yor process** is used as an
"adaptor" on top of a base generator, reproducing the realistic power-law (Zipfian) type/token
frequency statistics of natural language; shown to improve an unsupervised morphology-learning
model.
**Relevance to us:** Canonical citation for treating word/synonym frequency as a **Pitman-Yor
/ power-law-generated** quantity rather than raw counts — relevant if we want a
principled smoothing/rarity model for how "marked" a rare register variant's frequency is,
beyond a flat log-frequency covariate (cf. SUBTLEX above).

---

## 7. Register differences between synonyms (formality/simplification pairs — ready-made method + validation data)

### Pavlick, E., & Callison-Burch, C. (2016). "Simple PPDB: A Paraphrase Database for Simplification." *ACL 2016 (Short Papers)*, pp. 143-148, Berlin. ACL Anthology: P16-2024.
**Summary:** Filters/re-scores the large Paraphrase Database (PPDB) down to **SimplePPDB**: 4.5M paraphrase rules annotated with a supervised "simplification" score indicating which side
of a pair is simpler, making it (at the time) the largest available lexical-simplification
resource; scores are competitive with then-state-of-the-art dedicated simplification systems.
**Relevance to us:** The single most directly reusable **synonym-pair register/complexity
dataset**: for any criterion-name pair that appears in PPDB, we get an existing
complexity-direction label we can use to validate our own register-height score against an
independent source, on the overlap set.

### Rao, S., & Tetreault, J. (2018). "Dear Sir or Madam, May I Introduce the GYAFC Dataset: Corpus, Benchmarks and Metrics for Formality Style Transfer." *NAACL-HLT 2018*, pp. 129-140, New Orleans. ACL Anthology: N18-1012.
**Summary:** Introduces GYAFC (Grammarly's Yahoo Answers Formality Corpus), the largest
parallel corpus of matched **formal ↔ informal sentence rewrites** for a single content
(built from Yahoo Answers text with human rewrites into both registers), plus MT-style
baselines and metrics for the formality-style-transfer task.
**Relevance to us:** Gives literal **parallel same-meaning, different-register sentence
pairs** — the closest existing analogue to what we want for criterion names (same concept,
different register). Useful both as a method template (how to elicit register-paired
rewrites from annotators) and as anchor items for calibrating an LLM judge's sense of
"formal register."

### Paetzold, G., & Specia, L. (2016). "SemEval 2016 Task 11: Complex Word Identification." *SemEval-2016*, pp. 560-569, San Diego. DOI: 10.18653/v1/S16-1085.
**Summary:** Shared task where systems predict, for words in context, whether a native or
non-native reader would find the word "complex" (i.e., needing simplification) — establishing
Complex Word Identification (CWI) as a distinct sub-task from full lexical substitution, with
a shared benchmark and multiple competing feature-based/model-based systems.
**Relevance to us:** CWI systems' feature sets (frequency, length, syllable count, AoA,
etc. — used by competing systems in this task and its follow-ups, e.g. Yimam et al.'s later
multilingual CWI shared tasks) are a ready-made **feature bank** for "how complex/high-register
does this word look," largely overlapping with our own operationalization shortlist below.

### Specia, L., Jauhar, S. K., & Mihalcea, R. (2012). "SemEval-2012 Task 1: English Lexical Simplification." In *\*SEM 2012*, pp. 347-355, Montréal. ACL Anthology: S12-1046.
**Summary:** The original lexical-simplification shared task: given a target word in a
sentence, systems must rank a set of candidate substitute words from simplest to most
complex, evaluated against human simplicity rankings.
**Relevance to us:** Establishes the **ranking-not-classification** framing for register/
complexity comparison between synonyms — directly matches our need to rank criterion-name
variants against each other rather than assign an absolute score, and its human-ranking
protocol is a template for eliciting our own register-height ground truth via LLM judges.

---

## Operationalization shortlist

Ranked by how directly we can compute it per criterion-name today, given the sources above.

1. **Etymological stratum (Germanic vs. Latinate vs. Greek) via Etymological Wordnet (de
   Melo 2014, etym.org).** Most theoretically load-bearing feature — this *is* the Corson
   "lexical bar" axis directly. Caveat: many everyday words have mixed, contested, or
   Wiktionary-incomplete etymologies (e.g. loanwords fully nativized centuries ago); need a
   rule for multi-morpheme criterion names (e.g. majority stratum by content-morpheme count,
   or "highest" stratum present) and manual spot-checks against coverage gaps.

2. **Formality score via a trained lexicon/classifier** — either reconstruct the Brooke,
   Wang & Hirst (2010) graph-propagation lexicon on our own vocabulary, or run the Pavlick &
   Tetreault (2016) PT16-style formality model. Caveat: both are trained/validated at
   sentence or document level (news/blogs/email/QA-forum genres), not on isolated
   noun-phrase criterion names — may need in-domain recalibration or to fall back on
   Heylighen & Dewaele's POS-ratio F-score, which is at least word/phrase-computable.

3. **Age-of-acquisition (Kuperman, Stadthagen-Gonzalez & Brysbaert 2012).** Cheap, one
   lookup per content word, strong prior literature linking late-AoA to Latinate/formal
   register. Caveat: coverage is single common words only; multi-word or technical criterion
   names will have low or missing coverage and need decomposition/averaging across words.

4. **Log word frequency (SUBTLEX-US, Brysbaert & New 2009), used as covariate not standalone
   index.** Cheap, well-validated, near-universal coverage. Caveat: frequency conflates
   register with topical rarity/technicality — a rare *concrete* technical term (e.g.
   "spectrophotometer") is not "high register" the way an abstract Latinate nominalization
   is, so must be paired with concreteness (Brysbaert, Warriner & Kuperman 2014) as a
   de-confounder, not used alone.

5. **Nominalization / Latinate-suffix morphology (hand-built suffix list: -tion, -ity,
   -ance/-ence, -ology, -ism, etc., informed by Biber-style academic-register features and
   Coxhead's 2000 Academic Word List).** Directly targets the morphological signature Corson
   and Coxhead both point to. Caveat: no off-the-shelf tool — this is a small hand-built
   rule/list we'd need to construct and validate ourselves (false positives on
   non-Latinate words ending in similar strings, e.g. "-ance" in native-feeling
   "grievance"-type borrowings that are fully nativized).

6. **Synonym-pair cross-check against SimplePPDB (Pavlick & Callison-Burch 2016) and/or GYAFC
   (Rao & Tetreault 2018) overlap.** Not a per-criterion-name feature but a **validation**
   step: for any criterion-name pair that also appears as a paraphrase pair in SimplePPDB or
   as a formal/informal rewrite pair in GYAFC, check that our computed register-height
   ordering agrees with their independently-labeled direction. Caveat: coverage on our
   specific criterion-name vocabulary will likely be sparse (both resources are corpus-mined
   from general text, not evaluation/quality-criterion jargon), so this validates the method
   on a small overlap subset rather than scoring most of our items directly.
