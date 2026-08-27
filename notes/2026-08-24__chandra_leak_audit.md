# Chandra cells leak audit (chandra_humor / chandra_cw) — 2026-08-24

User-ordered (2026-08-24): "Start with a manual audit, then something like Logistic
Regression to look at the top features." Question: dense Llama-8B LoRA gets
T=.849 (humor: eval .845-.850 / test .837-.842) and T=.911 (cw: eval .906-.910 /
test .905-.906) while the articulated bank gets VA .693/.579, and the build-time
within-sub char-ngram probe was already .806/.834 — is the surface-visible signal
pipeline/survival artifact or real content?

Design asymmetry under audit: removed side = Chandrasekharan 2017 research corpus
(`datasets/prior_norms/reddit-removal-log.csv`, columns **body, subreddit only** — no
ids, no timestamps), kept side = Arctic Shift API fetch (2026) of era-window comments;
one shared renderer (`build_removal_v2_normalized.norm` + %0A decode) applied to both
at build. Populations `datasets/prior_norms_cells/chandra_{humor,cw}_population.csv.gz`
(sk3). AUDIT ONLY — nothing modified or deleted; all new artifacts under
`/lfs/skampere3/0/alexspan/chandra_leak_audit/` (tell_battery.csv, manual_sample.jsonl,
lr_results.json, temporal_results.json, audit scripts + audit_b.log).

## Headline verdict

Mixed evidence, but the **measurable leak channels are individually small (each
≤ ~.01 AUC on the LR probes) while the top-feature mass is moderation-relevant
content** (slurs/insults, immersion-breaking meta in nosleep, piracy asks, mod-talk).
The quantified leak controls (survival-marker ablation + named-event ablation +
nosleep two-week temporal holdout) together account for ≤ ~.02 of probe AUC;
against dense-vs-bank gaps of .156 (humor) and .332 (cw), **>90% of the gap
survives every control we could run**. The one channel we could NOT test —
era-scale (11-month) drift for the 8 subs whose kept side spans 1-6 days — remains
open and is the concrete v2 fix (recollect kept side uniformly over the window).

## Part 0 — structural finding: kept-side time window is 1-6 days for 8/9 subs

The kept collector (`collect_chandra_kept.py`) walks a `before` cursor backward from
2017-03-31 and STOPS at target count. Per-row created_utc recovered for **100% of
kept population rows** (rendered-text-hash rejoin to kept_<sub>.jsonl.gz):

| sub | kept n | min | median | max | distinct days |
|---|---|---|---|---|---|
| funny | 11,620 | 2017-03-30 | 03-30 | 03-30 | **1** |
| Showerthoughts | 8,996 | 03-29 | 03-30 | 03-30 | 2 |
| tifu | 272 | 03-30 | 03-30 | 03-30 | **1** |
| nottheonion | 6,224 | 03-28 | 03-30 | 03-30 | 3 |
| me_irl | 3,674 | 03-27 | 03-29 | 03-30 | 4 |
| asoiaf | 5,101 | 03-25 | 03-28 | 03-30 | 6 |
| books | 7,316 | 03-25 | 03-28 | 03-30 | 6 |
| gameofthrones | 4,206 | 03-27 | 03-29 | 03-30 | 4 |
| nosleep | 14,685 | **03-02** | 03-15 | 03-30 | **29** |

Removed side spans the full study window 2016-05..2017-03 by provenance (timestamps
not recoverable — the log carries body+subreddit only). So class ⊥̸ date everywhere,
catastrophically so for funny/tifu (single day). Any date-specific content is a
free class feature. Also noted: kept_tifu.jsonl.gz has a torn gzip tail — only ~327
of 10,561 collected rows are readable, which is why the tifu stratum is 272/class.

## Part 1 — manual audit (108 rows read, stratified sub x class x train/heldout)

Sample file: chandra_leak_audit/manual_sample.jsonl. What I saw:

**Content-real (exonerating).** Removed rows are heavy in exactly what mods remove:
- insults/toxicity: "You're a stupid piece of shit." (books, chandra_cw:29979-adjacent
  sample row 32185/23); "You're a fucking retard xD" (tifu); "Are you quite literally
  insane, or just mostly deaf?" (asoiaf)
- slur-bait jokes: "What's a word that starts with 'n' ... neighbor" (funny)
- off-topic politics in humor subs: "the Deplorables are all over the default subs"
  (funny), "Through the power of white supremacy is how" (me_irl)
- repost/meta policing: "What about the last time this was posted?" (me_irl),
  "Mods just removed my earlier posting of the same article" (nottheonion)
- nosleep immersion-breaking: "Escorts don't kiss. Write what you know!"
Kept nosleep rows are rule-FOLLOWING in-immersion support ("Keep an eye on that bite
mark OP!", "My guess is sometime in your girlfriends past a demon locked onto her.").
This is the norm signal the cell was built to carry, and it is visible to the eye.

**Artifact channels found (damning, each low-prevalence):**
1. **Moderation-notice contamination on BOTH sides.** Removed r/funny row = a mod
   removal notice ("Hi LatvianGiant, your post ... breaks the rules of /r/funny and
   has been removed for the following reason s :"); removed nosleep row = AutoModerator
   formatting notice with mangled URL residue ("Note+to+moderation"). But a KEPT tifu
   row is ALSO an AutoModerator removal notice ("Thank you for submitting to /r/tifu.
   Your submission has been removed because it violates our rules") — bot comments
   themselves survive, so the channel points BOTH ways. Strict-pattern rates:
   humor kept 0.93% / removed 0.69%; cw kept 0.66% / removed 1.74%.
2. **Survival edit markers concentrate on kept.** "edit:" ~0.9% of kept vs 0.06-0.15%
   of removed (6-16x ratio, direction = post-hoc edits survive only on kept posts);
   `^^`-caret bot residue present both sides (0.1-0.5%).
3. **Era tells.** Removed rows mention harambe, "if trump wins" (pre-election),
   "Rickon doesnt make it through this season" (GoT S6 airing, mid-2016); kept rows
   sit on late-March-2017 topics. Confirms the Part-0 window asymmetry in text.
4. No encoding/escaping tells found in the read sample (renderer did its job):
   no &amp;-entities (0.0% both sides), no %0A residue asymmetry worth noting,
   quote styles uniform. Length: humor removed mean 182 vs kept 143 chars (medians
   82 vs 76); cw matched (208/206).

## Part 2 — LR probes on dense train rows (join split/train.csv -> population)

AUCs (LR, TF-IDF; grouped-OOF = 5-fold GroupKFold by sub; within-sub = per-sub
5-fold StratifiedKFold OOF, n-weighted mean):

| cell | probe | grouped-OOF by sub | within-sub | per-sub |
|---|---|---|---|---|
| humor | word 1-2gram | .641 | .778 | ST .794 / funny .751 / me_irl .735 / notonion .829 / tifu .793 |
| humor | char 3-5gram | .653 | .794 | ST .813 / funny .762 / me_irl .764 / notonion .841 / tifu .835 |
| cw | word 1-2gram | .609 | .855 | asoiaf .815 / books .861 / got .782 / nosleep .888 |
| cw | char 3-5gram | .607 | .860 | asoiaf .817 / books .876 / got .786 / nosleep .888 |

(Dense T: humor .849 / cw .911 — dense adds only ~.05 over a plain within-sub LR.)

**Top-40 word features, classified** (full lists in lr_results.json):

- humor REMOVED side: retarded 5.01, tranny 4.79, "kill yourself" 4.50, kill 4.00,
  troll 3.92, mods 3.83, black 3.82, suicide 3.81, cunt 3.67, fucking 3.65, bitch,
  fuck, gay, fat, dumb, muslims, jews, idiot, nigga, moron, retard, pussy, faggot,
  islam ... -> **CONTENT** (~30/40: toxicity/identity-attack/meta-mod vocabulary =
  the removal norms themselves). TEMPORAL: 2016 (3.61), harambe (2.74).
  AMBIGUOUS: lordtuts (username, 3.68), hello (3.61), com (3.25 — bare-domain
  residue; link-drop comments are also genuinely removable).
- humor KEPT side: **edit -4.35 = SURVIVAL-ARTIFACT (top kept feature!)**; then a
  wall of datable late-March-2017 event tokens -> **TEMPORAL** (~18/40): curry,
  statue + ronaldo (the Mar-29-2017 Ronaldo bust unveiling), golf/golfing (Trump
  golf coverage), spicer, bamboozle + kenneth (the March-2017 me_irl meme), tinder,
  conan, buzzfeed, girl scouts, obama/trump/president/white house (front-page news
  of those days). Char probe agrees: `edit:` variants + golf/curry/tind/scout/atue.
- cw REMOVED side: story 15.65(!), trump 7.52, fake 6.86, iia 6.80 (u/iia, nosleep
  author — AMBIGUOUS), writing 6.74, bullshit, paragraph(s), mods, gay, proof,
  fiction, porn, hitler, retarded, real, "made up", "good story", tldr, torrent,
  pdf -> **CONTENT** (~28/40): immersion-breaking story-talk is nosleep's #1 removal
  norm; piracy asks (pdf/torrent) are books/GoT removals; plus toxicity/politics.
  TEMPORAL-mixed: episode/spoiler/leak (S6 airing + leak era on removed side only).
- cw KEPT side: edit -3.41 (**SURVIVAL**), dragonstone -3.27 + trailer -2.52
  (**TEMPORAL smoking gun: the GoT S7 "Dragonstone" trailer dropped 2017-03-30 —
  the exact kept-fetch day**), scalzi/dylan/prize/april (**TEMPORAL**: March-2017
  book news), doll/dolls/lily (that month's popular nosleep series — temporal-topical),
  vs "so sorry", "good luck", "your daughter", "made me", op, "re right" ->
  **CONTENT** (supportive in-immersion register), ~8-12/40 temporal.

## Ablations (within-sub probes, train rows)

| control | humor word | humor char | cw word | cw char |
|---|---|---|---|---|
| baseline | .7779 | .7939 | .8552 | .8597 |
| survival-marker strip (edit:/update:/eta:/gold-thanks/rip-inbox/blew-up; text truncated at marker) | .7775 (-.0004) | .7931 (-.0008) | .8549 (-.0003) | .8595 (-.0002) |
| named-event-token strip (ronaldo/statue/spicer/bamboozle/golf/dragonstone/trailer/... ~3% rows touched) | .7740 (-.0039) | .7900 (-.0039) | .8521 (-.0031) | .8564 (-.0033) |

Survival-marker rows changed: humor 291 (kept 244 / removed 47), cw 269 (kept 250 /
removed 19) — the asymmetry is real (5-13x) but prevalence ~0.5% caps its AUC value
at ~nothing. Grouped-OOF AUCs unchanged under both ablations.

**Nosleep temporal holdout** (only sub with kept spread: 29 days; kept split at
median date, removed split randomly; train removed₁-vs-kept_early, test
removed₂-vs-kept_late; control = same with random kept halves):

| | control (random kept split) | temporal holdout | drop |
|---|---|---|---|
| word 1-2 | .8806 | .8689 | .0117 |
| char 3-5 | .8832 | .8738 | .0093 |

Date-from-text within kept-only (early-vs-late, word LR): AUC .593 — kept texts are
only weakly datable at 2-week scale. So nosleep's .89 separation transfers across
disjoint kept fortnights nearly intact: at the scales testable here, it is content,
not date.

## Timestamp comparison per class

Kept: fully recovered (table in Part 0). Removed: **not recoverable** — the removal
log has no id/timestamp columns; provenance says 2016-05..2017-03. The per-class
comparison is therefore design-level: removed ≈ 11 months vs kept 1-29 days ending
2017-03-30. This channel is real (era tokens appear in BOTH classes' top features:
2016/harambe/episode on removed, ronaldo/spicer/dragonstone on kept) but its
quantified share, where measurable, was small (see ablations); its day-scale share
for funny/tifu/Showerthoughts is **not measurable with this data** — no kept spread.

## Bottom line

- Fraction of the dense-vs-bank gap surviving leak controls: **~all of it**.
  Quantified leak channels sum to ≤ ~.02 probe-AUC (survival ≤.001, named-event
  ≤.004, week-scale temporal ≤.012); gaps are .156 (humor) and .332 (cw). The LR
  probes' own top features say the recoverable surface signal is dominated by
  moderation-relevant content (slurs, immersion-breaking, piracy, mod-meta) that a
  richer bank SHOULD articulate — i.e. the bank shortfall looks like bank coverage,
  not a fake dense number.
- Remaining risk, stated plainly: the era/day-scale channel for the 8 subs with
  1-6-day kept windows is untested and untestable in-place. The "Dragonstone"
  feature proves the model CAN see the fetch date; the ablations only bound the
  named-token part of that channel.
- **Recommendation:** kept-side v2 recollect for both cells — sample Arctic Shift
  uniformly over 2016-05..2017-03 (stratified `before` cursors, ~10-30 dates per
  sub) instead of walking back from the window end; also re-pull tifu (torn gzip
  left 272/class of 10.5K collected) and drop mod/AutoModerator-notice rows from
  BOTH classes (strict pattern, 0.7-1.7%). Then re-run the within-sub probe + dense.
  Given nosleep's temporal holdout (-.01) and the tiny ablation deltas, the
  expected outcome is probe ≈ unchanged and dense T within ~.01-.03 of current —
  but until v2, quote chandra dense T as "upper bound pending kept-side era-uniform
  rebuild". Do NOT retract: no evidence here supports the pipeline-fingerprint story
  (encoding tells absent, grouped-OOF probes only .61-.65, survival channel ≈ .001).

Artifacts: sk3 `/lfs/skampere3/0/alexspan/chandra_leak_audit/` (audit_a_facts.py,
audit_b_lr.py + audit_b.log, audit_c_temporal.py, tell_battery.csv,
manual_sample.jsonl, lr_results.json, temporal_results.json). Populations and dense
outputs untouched.

## POSTSCRIPT — v2 rebuild executed same day (era-uniform kept side)

Recommendation above was executed (coordinator-ordered): collect_chandra_kept_v2.py
(22 evenly spaced ~fortnight strata over 2016-05-01..2017-03-31, author field
persisted, tifu re-pulled) + build_chandra_cells_v2.py (same gated build + mod/
AutoMod-notice strip BOTH classes [4,023 kept + notice-matching removals dropped]
+ per-row kept ts + kept author_hash sha1[:16]; removed side author untestable —
corpus anonymized). v2 populations/manifests alongside v1 on sk3; v1 untouched.

Era confound BROKEN: kept ts coverage now min 2016-05-14/15, med 2016-10-14,
max 2017-03-30 (28/47 distinct anchor days across the full 11-month window vs
v1's 1-6 days). tifu restored to 12,650 rows (v1 544, torn gzip).
n: humor 73,268 (v1 61,572), cw 64,788 (62,616).

**Headline probe comparison (v1 gate config: char_wb 2-4gram, 30k, 70/30 within-sub
mean) — v1 -> v2:**

| cell | v1 within-sub | v2 within-sub | delta | v2 per-sub |
|---|---|---|---|---|
| chandra_humor | .806 | **.760** | -.046 | funny .717 / ST .767 / tifu .821 / notonion .757 / me_irl .735 |
| chandra_cw | .834 | **.808** | -.026 | nosleep .850 / books .860 / got .739 / asoiaf .783 |

Audit-config 5-fold within-sub (v1 numbers were train-rows-only, v2 full-pop —
near-identical sampling frame): humor word12 .778->.742, char35 .794->.757;
cw word12 .855->.826, char35 .860->.832. Grouped-OOF-by-sub (v1cfg):
humor .652->.684, cw .605->.628.

Reading (results, not verdicts): removing the era window + notice rows costs the
surface probe ~.03-.05 — same order as the audit's channel-by-channel estimates —
and the remaining .74-.83 within-sub separability is the signal whose top features
the audit classified as predominantly moderation-relevant content. Author channel:
kept-side author_hash coverage .992/.994, 27,065/18,572 unique authors (author-
disjoint/grouped folds now possible on the kept side only; removed side anonymized).

STOPPED here per coordinator: no dense rescore / no bank VA / no GPU work queued.
Next gates for VAT-table incorporation: dense T + bank VA on v2 populations, then
era-stratified and author-grouped readouts.
