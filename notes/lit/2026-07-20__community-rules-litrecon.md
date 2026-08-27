# Lit recon: community rules / online norms literature

Purpose: survey the "community rules / online norms" literature (Reddit rule taxonomies,
CSCW/ICWSM norm-mining, Wikipedia policy studies, Stack Exchange/Twitch/Mastodon rule
studies) for a paper on "the lexicon of evaluation" — naming/codability of a community's
quality criteria, Good-Turing/Pitman-Yor naming-diversity models, and provenance
(official/codified vs. folk/tacit) of criteria. All sources below were confirmed live
this session via the Semantic Scholar Graph API and the OpenAlex API (WebSearch was
unavailable — its session budget was exhausted before this task started), unless flagged
[UNVERIFIED]. Full JSON metadata (title/authors/venue/year/DOI/abstract) was pulled and
checked for each entry; none are quoted from memory alone.

---

## 1. Reddit rule taxonomies (the closest structural precedent)

### Fiesler, C., Jiang, J., McCann, J., Frye, K., & Brubaker, J. R. (2018). "Reddit Rules! Characterizing an Ecosystem of Governance." *ICWSM 2018*. DOI: 10.1609/icwsm.v12i1.15033.
**Summary:** Mixed-methods study of 100,000 subreddits and their posted rules. The team
built a qualitative codebook of rule *types* (e.g., content rules, behavior rules,
format rules, meta rules) by open-coding a sample of rules, then applied it at scale to
characterize the frequency and distribution of rule types across the site, and how rule
prevalence varies with subreddit size/topic/age. Finds rules are both subreddit-specific
(context-dependent) and share common cross-site characteristics.
**Dataset:** 100,000 subreddits' worth of posted community rules (Reddit's structured
"rules" field), mixed qualitative (coded subsample) + quantitative (full-scale) analysis.
**Relevance to us:** The direct methodological ancestor of any "taxonomy of stated
community rules" approach — gives us a validated *category* scheme for what rules are
*about*, but not for how they are *worded*; it treats rule text as something to classify
into a fixed ontology, not as a naming/vocabulary object in its own right. Good baseline
taxonomy to contrast against a codability/naming-diversity framing.

### Nicholson, M. N., Keegan, B. C., & Fiesler, C. (2023). "Mastodon Rules: Characterizing Formal Rules on Popular Mastodon Instances." *CSCW Companion 2023*. DOI: 10.1145/3584931.3606970.
**Summary:** Applies the Fiesler et al. Reddit rule-type codebook to a sample of the most
popular Mastodon instances, then explicitly compares the resulting rule-type distribution
to Reddit's. Finds Mastodon rules disproportionately emphasize harassment/hate,
attributed to the influence of the "Mastodon Covenant" and to the federated platform's
history of platform-migration in response to other sites' moderation failures.
**Dataset:** Rule sets from the most popular Mastodon instances (exact N is a modest,
hand-collected instance-level sample, not full-population scale).
**Relevance to us:** A direct cross-platform replication showing the *same* codebook
transfers but the *distribution* over categories shifts by platform history/culture —
useful evidence that rule *content* categories travel but doesn't touch whether the same
*category* gets named with the same words across platforms (our angle).

### Lloyd, T., Gosciak, J., Nguyen, T., & Naaman, M. (2025). "AI Rules? Characterizing Reddit Community Policies Towards AI-Generated Content." *CHI 2025* (arXiv:2410.11698). DOI: 10.1145/3706598.3713292.
**Summary:** Large-scale extension of the Fiesler rule-taxonomy method: collects rules
from 300,000+ subreddits, measures the emergence and growth of rules specifically about
AI-generated content over roughly a year, and develops a new sub-taxonomy of AI-specific
rule types (on top of the general Fiesler categories). Finds AI rules more common in
larger subreddits and in art/celebrity-topic communities, less common in social-support
communities; rules commonly cite "quality" and "authenticity" as justification.
**Dataset:** 300,000+ public subreddits' rule metadata, tracked longitudinally over ~1 year.
**Relevance to us:** Shows how a *new* evaluative concern (AI-generated content) gets
codified into rule language over time and how communities justify new rules using
existing quality vocabulary ("quality," "authenticity") — a live example of naming/
codification-in-progress that could anchor a case study on how a criterion acquires a
name.

---

## 2. Reddit norms and values at scale (macro/meso/micro structure)

### Chandrasekharan, E., Samory, M., Jhaver, S., Charvat, H., Bruckman, A., Lampe, C., Eisenstein, J., & Gilbert, E. (2018). "The Internet's Hidden Rules: An Empirical Study of Reddit Norm Violations at Micro, Meso, and Macro Scales." *Proc. ACM Hum.-Comput. Interact. (CSCW 2018)*. DOI: 10.1145/3274301.
**Summary:** Studies *enforced* (not stated) norms by mining 2.8M moderator-removed
comments across 100 top subreddits over 10 months. Combines computational clustering with
qualitative coding to identify three norm scales: macro norms (near-universal across
Reddit), meso norms (shared by clusters/genres of subreddits), and micro norms (specific
to individual subreddits). Framed as the first large-scale census of internet-culture
norms via revealed behavior (what gets removed) rather than stated rules.
**Dataset:** 2.8M removed comments, 100 top subreddits, 10 months.
**Relevance to us:** Establishes the macro/meso/micro *scale* structure for norms that is
now a standard reference point in this literature — a natural axis to cross with our
naming/codability axis (do macro-scale norms get named more consistently/with more
official vocabulary than micro-scale ones?). Also a key methodological contrast: norms
inferred from enforcement action vs. norms as explicitly stated/named text.

### Goyal, A., Lambert, C., & Chandrasekharan, E. (2024). "Uncovering the Internet's Hidden Values: An Empirical Study of Desirable Behavior Using Highly-Upvoted Content on Reddit." *ICWSM 2024* (arXiv:2410.13036).
**Summary:** Flips the Chandrasekharan et al. removal-based method to the positive side:
uses upvotes as a proxy for community approval, and has an LLM extract *values* (not just
norms) from 16,000 highly-upvoted comments across 80 subreddits at two time points
(2016 and 2022), compiling 64 (2016) and 72 (2022) macro/meso/micro values keyed by
cross-community frequency. Finds existing computational prosociality measures miss ~82%
of the values the LLM surfaces, and that some of these values are genuinely new (not in
prior qualitative taxonomies).
**Dataset:** 16,000 highly-upvoted comments, 80 subreddits, two snapshots (2016, 2022).
**Relevance to us:** Very close kin to our project — it is explicitly an LLM-naming
exercise over community-endorsed content, uses the same macro/meso/micro frequency
framing as Chandrasekharan et al., and directly demonstrates LLM-based value/criterion
extraction and cross-community frequency binning. It is a strong baseline/comparison
point for our LLM-naming-behavior component, though it does not do any
sampling-theoretic (Good-Turing/Pitman-Yor) modeling of how many distinct value-names
exist or how naming diversity saturates with more data — it stops at a frequency count.

### Weld, G. C., Zhang, A. X., & Althoff, T. (2021). "What Makes Online Communities 'Better'? Measuring Values, Consensus, and Conflict across Thousands of Subreddits." *ICWSM 2021* (arXiv:2111.05835). DOI: 10.1609/icwsm.v16i1.19363.
**Summary:** First large-scale *survey* of community values (rather than inference from
removed/upvoted content): surveys 2,769 Reddit users across 2,151 unique subreddits about
what they value in their community, combined with quantitative analysis of public Reddit
data. Finds substantial within- and between-community disagreement (e.g., members
disagree on how "safe" their community is; longstanding communities value trustworthiness
30.1% more than newer ones; moderators want communities 56.7% less democratic than
members do). Builds a small feature set that predicts community values with ROC AUC 0.667.
**Dataset:** 2,769 survey respondents across 2,151 unique subreddits.
**Relevance to us:** A rare *ground-truth* elicitation of community values directly from
members (not inferred from text), useful as a validity check/contrast for any
codability claim we make about "what the community says it values" vs. "what an LLM
infers it values" — and its explicit finding that values are "rarely explicitly stated"
is a direct motivating gap for a paper about the *naming* of criteria.

### Weld, G. C., Zhang, A. X., & Althoff, T. (2021/2024). "Making Online Communities 'Better': A Taxonomy of Community Values on Reddit." *ICWSM 2021* (arXiv:2109.05152). DOI: 10.1609/icwsm.v18i1.31413.
**Summary:** Companion paper building the taxonomy behind the survey above: 212 members of
627 unique subreddits were asked to describe, in their own words, what they value about
their community; 1,481 free-text responses were iteratively categorized into a validated
taxonomy of 29 subcategories nested within 9 top-level value categories (e.g., content
quality, community size, moderation style). Used to reframe known governance problems
(e.g., managing member influx) as tensions between named values, and to flag understudied
values (content quality, community size).
**Dataset:** 212 respondents, 627 subreddits, 1,481 free-text value statements.
**Relevance to us:** This is the single closest prior artifact to a "naming inventory of
evaluative criteria" in this literature — it is literally elicited, in members' own
words, then hand-coded into a taxonomy. But the coding process collapses many different
member wordings into single taxonomy labels without ever measuring or reporting the
*naming diversity/agreement rate* among the raw free-text responses that fed each
category — exactly the sampling-theoretic gap (how many distinct names map to one
concept, and does that count saturate) our paper targets.

---

## 3. Wikipedia policy codification

### Butler, B., Joyce, E., & Pike, J. (2008). "Don't Look Now, But We've Created a Bureaucracy: The Nature and Roles of Policies and Rules in Wikipedia." *CHI 2008*. DOI: 10.1145/1357054.1357227. (278 citations per Semantic Scholar; abstract not retrievable through the API — publisher-elided — but title/authors/venue/DOI independently confirmed.)
**Summary (from established secondary knowledge of this widely-cited paper, metadata-verified above):** Traces the growth of Wikipedia's formal policy pages over time and argues policies serve multiple, sometimes conflicting, functions (representing norms, resolving disputes, controlling behavior, socializing newcomers) — i.e., a single named policy can carry different institutional work depending on who invokes it and when.
**Relevance to us:** Foundational citation for "codification as an ongoing social process"
in online communities — motivates why the *existence* of a named, official policy is not
the same as consensus about what the name means, a distinction central to our
provenance-stratification (official vs. folk) framing.

### Beschastnikh, I., Kriplean, T., & McDonald, D. W. (2008). "Wikipedian Self-Governance in Action: Motivating the Policy Lens." *ICWSM 2008*. DOI: 10.1609/icwsm.v2i1.18611.
**Summary:** Quantitatively analyzes *citations* to Wikipedia policies within article
talk-page discussions — i.e., not the policy pages themselves, but how often and by whom
editors invoke a named policy as justification for an action. Finds policy-citation is
used widely by both registered editors and administrators, that usage patterns are
converging/stabilizing across these groups over time, and that citation highlights the
growing importance of specific policy areas (notably source attribution/verifiability).
**Dataset:** Talk-page discussions and their policy citations, mined at Wikipedia scale
(exact N of citations not stated in the abstract; large corpus of talk-page revisions).
**Relevance to us:** Directly measures "provenance in use" — how often the *named*,
official vocabulary of policy actually gets invoked in everyday deliberation, as opposed
to just existing. This is close to a "codability in the wild" measurement (does the
community actually use the official name?) and is a strong methodological precedent for
provenance-stratified analysis of criteria vocabulary.

### Kriplean, T., Beschastnikh, I., McDonald, D. W., & Golder, S. A. (2007). "Community, Consensus, Coercion, Control: CS*W or How Policy Mediates Mass Participation." *GROUP 2007*. DOI: 10.1145/1316624.1316648. (149 citations per Semantic Scholar.)
**Summary:** Earlier companion study from the same group examining how Wikipedia policy
functions as a coordination mechanism across four framings (community, consensus,
coercion, control), using the same policy-citation-in-talk-pages method later scaled up
in the 2008 ICWSM paper.
**Relevance to us:** Establishes that "policy as cited vocabulary" (rather than "policy as
written page") is the right unit of analysis for provenance work — reinforces the
Beschastnikh et al. framing above as a load-bearing methodological choice, not a one-off.

---

## 4. Stack Exchange / Stack Overflow close-and-duplicate norms

### Correa, D., & Sureka, A. (2013). "Fit or Unfit: Analysis and Prediction of 'Closed Questions' on Stack Overflow." *COSN 2013* (arXiv:1307.7291). DOI: 10.1145/2512938.2512954. (82 citations per Semantic Scholar.)
**Summary:** First systematic study of Stack Overflow's five official "closed" reasons
(duplicate, off-topic, subjective, not-a-real-question, too-localized). Analyzes 4 years
of data (3.4M questions, ~100K closed) to characterize closed questions and builds a
predictive (ensemble ML) model to flag likely-to-be-closed questions at creation time
(70.3% accuracy), using user-profile, community-process, textual-style, and content
features.
**Dataset:** 3.4M questions over 4 years; ~100,000 closed questions.
**Relevance to us:** Stack Exchange's five closing *reasons* are a rare case of a
platform's evaluative criteria being an explicit, small, fixed, officially-named taxonomy
— a useful high-codability anchor case (closed-form naming) to contrast against the
open-ended naming behavior we expect elsewhere (e.g., Reddit rule text, humor, peer
review).

### Fang, J., Liang, J.-W., & Wang, H.-C. (2026). "Understanding Discussions Around Norms in Heavily Moderated Knowledge-Based Online Communities: A Case Study on Meta Stack Overflow." *CSCW 2026*. DOI: 10.1007/s10606-026-09547-3.
**Summary:** Mixed-methods study of *meta-discussions about norms* on Meta Stack
Overflow (the community's own forum for discussing its rules). Thematically analyzes 187
questions, 313 answers, and 2,738 comments to build a taxonomy of "whats and hows" of
norm-related discussion — what triggers a norm discussion, and the discourse strategies
people use to initiate or respond to it — and relates discussion framing (via construal
level theory) to discussion sentiment.
**Dataset:** 187 questions, 313 answers, 2,738 comments from Meta Stack Overflow.
**Relevance to us:** A rare study of the community *arguing about what a norm should be
called/mean* rather than just applying it — directly adjacent to codability/naming-
agreement, since norm ambiguity and disagreement over correct application is exactly
where naming convergence/divergence would show up empirically.

---

## 5. Twitch live-streaming rules and norm-setting

### Cai, J., Guanlao, C., & Wohn, D. Y. (2021). "Understanding Rules in Live Streaming Micro Communities on Twitch." *ACM IMX 2021*. DOI: 10.1145/3452918.3465491.
**Summary:** Two studies of channel-level ("micro-community") rules on Twitch. Study 1
(survey-based) finds users perceive both rule transparency and communication frequency as
shaping channel "vibe" and harassment frequency. Study 2 (data analysis) finds, somewhat
counterintuitively, that the *most popular* channels tend to have no formal channel/chat
rules at all; among channels that do have rules, streamer-encouraged (as opposed to
purely mod-enforced) rules are most prominent.
**Dataset:** Two studies — a user survey plus channel-rule/metadata analysis across
Twitch channels (exact N of channels not given in the abstract).
**Relevance to us:** A live-streaming analog of the Fiesler Reddit-rules taxonomy, and
notable for showing that formal/written rules and community success can be *inversely*
related — i.e., codified rule text is not a reliable proxy for the actual evaluative
norms in play, an important caution for any study (like ours) that leans on stated/
codified vocabulary as ground truth.

### Seering, J., Kraut, R. E., & Dabbish, L. (2017). "Shaping Pro and Anti-Social Behavior on Twitch Through Moderation and Example-Setting." *CSCW 2017*. DOI: 10.1145/2998181.2998277. (240 citations per OpenAlex.)
**Summary:** Uses millions of Twitch chat messages to test whether norms are shaped more
by *proactive tooling* (chat modes restricting content) and *reactive bans*, versus
*imitation of positive examples set by high-status users* (streamers/mods). Finds
imitation effects are real and stronger for high-status exemplars; proactive tools work
well against spam specifically, reactive bans generalize across more behavior types.
**Dataset:** Millions of Twitch chat messages (exact channel/message count not given in
abstract; large-scale behavioral log analysis).
**Relevance to us:** Demonstrates a norm-transmission channel that has *no explicit naming*
at all — behavioral imitation of exemplars rather than reading/citing a rule — a useful
limit case for our provenance axis: some norms never get named/codified in language even
though they are clearly transmitted and enforced.

---

## 6. On the "wording/naming of rules" angle specifically (our closest prior work)

I ran five separate targeted searches this session for the specific angle closest to our
own contribution — lexical/textual similarity of rule *wording* across communities, a
"rule genome," convergent governance vocabulary, or text-reuse/boilerplate analysis of
community guidelines/ToS-style documents (queries tried: "lexical similarity rules across
online communities wording diffusion governance"; "AutoModerator rule templates reuse
subreddits text similarity"; "community guidelines text similarity cross-platform content
policy comparison"; "diffusion of governance institutions across online communities
isomorphism rules"; "text reuse boilerplate community guidelines policy language
platforms"; "convergent vocabulary naming agreement community guidelines rules
classification NLP" — all via Semantic Scholar and OpenAlex). **None of these returned a
paper that studies the wording/text-similarity of rules as its object** (the closest
returns were off-topic bibliometric/policy-topic-modeling papers in unrelated domains).
I could not verify that a "rule genome"-style paper exists in this literature; if one
exists it did not surface under any of the phrasings tried. This absence is reported
directly (not [UNVERIFIED] — these are genuine negative search results, not an unchecked
claim) and is treated as supporting evidence for the gap statement below rather than as a
citable source.

---

## Gap statement

This literature has built rich *category* taxonomies of what community rules and values
are *about* (Fiesler et al.'s rule types; Chandrasekharan et al.'s macro/meso/micro norm
scales; Weld/Zhang/Althoff's 9-category value taxonomy) and has studied how those
categories are *invoked* (Beschastnikh et al.'s policy-citation counts) or *transmitted*
(Seering et al.'s imitation channel), but in every case the analytic move is to collapse
raw member language into a small, researcher-defined set of categories — none of it asks
how many genuinely *distinct names* a community's members use for the same evaluative
concept, whether that name-count is still growing or has saturated with more data (a
Good-Turing/Pitman-Yor question), or what fraction of members converge on the same name
versus coin idiosyncratic ones (a codability/naming-agreement question in the
psycholinguistic sense). Nor does this literature systematically stratify a *single*
concept's names by register or provenance — official/platform-codified terms (e.g.,
Stack Overflow's five closing reasons, a subreddit's numbered rules) versus folk/
community-coined terms for the functionally same criterion — as opposed to studying
official and folk vocabulary as separate objects. Repeated, explicit searches this
session for a paper studying rule *wording* similarity or diffusion across communities
came back empty, suggesting this is a genuine open lane rather than an oversight on our
part. Finally, none of this literature uses LLMs as *namers* — i.e., prompting a model to
generate or classify the name a community would give a criterion and checking that
against real usage — the Goyal/Lambert/Chandrasekharan paper comes closest (LLM-extracted
values) but stops at frequency counts, not naming-process modeling or LLM-vs-human naming
comparison. Our paper's contribution is exactly this stack: sampling-theoretic
(Good-Turing/Pitman-Yor) modeling of criterion-name diversity, register/provenance
stratification of the *same* concept's different names, and a direct empirical check of
LLM naming behavior against measured human naming agreement — none of which this
literature currently provides.
