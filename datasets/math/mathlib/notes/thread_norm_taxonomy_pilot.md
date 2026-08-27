# Mathlib4 PR Review-Thread Norm Taxonomy — Pilot (600-thread stratified sample)

**Date:** 2026-06-11
**Corpus:** `sk3:/lfs/skampere3/0/alexspan/norm-research/datasets/math/mathlib/pr_thread_comments.jsonl`
(15,962 PRs / 86,001 threads / 139,238 comments)
**Sample:** 600 threads, stratified by year tercile (≤2024 / 2025 / 2026) × `isResolved`,
100 per cell, capped at 2 threads per PR (26,831 candidate threads after the cap).
Sample file: `notes/thread_norm_pilot_sample.jsonl.gz` (this dir); hand labels:
`notes/thread_norm_pilot_labels.json` (index in sample → category code).
Every thread was read and hand-labeled with a single primary category (first comment = the
reviewer's norm articulation; replies = negotiation).

**Why this domain matters for V/A/T:** mathlib correctness is machine-checked — Lean + CI
guarantee every proof compiles. So *nothing* in these threads is about whether the math is
true. The entire review corpus is norms **beyond correctness**: the purest available sample
of what experts can articulate about quality once verifiability is fully outsourced to a
machine. The question per category is whether the norm could be re-mechanized (V), needs
LLM-level judgment (A), or is genuinely contested / case-by-case (T).

---

## 1. Taxonomy summary table

| # | Category (code) | Share | Definition (one line) | V/A/T |
|---|---|---|---|---|
| 1 | Proof golfing & tactic style (GOLF) | 19.7% (118) | Shorter/cleaner/more robust proofs: golf suggestions, term vs tactic, squeeze `simp`s, avoid `erw`/`change`/defeq abuse, avoid duplicated subproofs, prefer sturdy tactics | **A**, T at the margin |
| 2 | Documentation & comments (DOC) | 11.7% (70) | Add/fix docstrings, explain magic constants, module docs, comment accuracy & placement | **A** (presence is V) |
| 3 | API completeness & lemma-worthiness (API) | 9.2% (55) | Add the symm/iff/equiv/`ofNat` variants, factor out sublemmas, "is this lemma worth having", instance vs lemma vs abbrev | **A**, T at the margin |
| 4 | Naming conventions (NAME) | 8.5% (51) | Lemma/def names: casing, dot-notation, `Foo.of_bar` patterns, name↔statement sync, `to_additive` names | **V+A**, T for contested rules |
| 5 | Statement form & binder hygiene (SIG) | 6.3% (38) | Implicit vs explicit arguments, `variable` sections, canonical spelling (`0 < k` vs `k ≠ 0`), iff direction, `omit`/section scoping | **A**, some V |
| 6 | simp & attribute hygiene (SIMP) | 6.2% (37) | Should this be `@[simp]`; simp-normal form; loops; `norm_cast`/`gcongr`/`fun_prop` tagging; rewrite direction | **V+A** |
| 7 | Formatting, typos & grammar (STYLE) | 5.3% (32) | Whitespace, indentation, line breaks, English grammar, `<|` vs `$`, notation spacing; incl. lint-bot threads | **V** |
| 8 | Statement generality (GEN) | 4.5% (27) | Weaken hypotheses ("AddZeroClass is enough"), generalize typeclass/universe, avoid "fake generality" | **A** |
| 9 | File/import/namespace organization (ORG) | 4.3% (26) | Right file/section/namespace, import minimization, lemma placement & ordering | **A**, some V |
| 10 | Deprecation & migration mechanics (DEPR) | 4.3% (26) | `deprecated` aliases, `#align` (porting era), porting notes, don't-delete-just-deprecate | **V** |
| 11 | Metaprogramming robustness (META) | 3.5% (21) | Review of tactic/linter code: monad/cache discipline, `Expr` handling, test robustness | **A** |
| 12 | Social / pointer / unclassifiable (MISC) | 3.3% (20) | Praise ("nice!", "Powergolf!"), "same here"/"ditto" pointers, bot crossrefs | — |
| 13 | CI/build/scripts infrastructure (INFRA) | 3.3% (20) | GitHub workflows, cache, shell/python scripts, lake/toolchain mechanics | **A** (generic SWE review) |
| 14 | Diff vigilance (VIGIL) | 3.0% (18) | "Is this intended?", bad merges, accidentally deleted lemmas, unexplained changes | **V+A** |
| 15 | API duplication / use-existing (DUP) | 2.7% (16) | "This already exists as `Foo.bar`", "special case of X", duplicate PRs | **V-assisted A** |
| 16 | Mathematical design choices (MATH) | 1.8% (11) | Definition choice, junk values, typeclass diamonds, which notion is canonical | **T** |
| 17 | Performance (PERF) | 1.2% (7) | `!bench` requests, instance perf, `maxHeartbeats`, elaboration speed | **V** |
| 18 | PR scoping & process (SCOPE) | 1.2% (7) | Don't mix refactor into feat PR, split unrelated changes, follow-up-PR etiquette | **V+A** |

Aggregate V/A/T (share-weighted, primary verdict): roughly **V ≈ 20%** (STYLE, DEPR, PERF,
SIMP-half, NAME-half, VIGIL-half), **A ≈ 70%**, **T ≈ 10%** (MATH, plus the contested tails
of GOLF/API/NAME). Even in the most machine-checked domain on earth, the articulated norm
layer is dominated by judgment calls that are *statable but not mechanizable* — strong
support for a wide A-band between V and taste.

### Fit to the expected families

All ten expected families appeared. Mapping: naming→NAME, docstring→DOC, statement
generality→GEN, API duplication→DUP, simp/tactic hygiene→SIMP, proof style→GOLF,
file/import org→ORG, mathematical content→MATH, typos/grammar→STYLE, CI/build→INFRA.
Notable deviations from the prior:

- **Proof golfing is the single biggest category (~20%)** — far larger than "proof style"
  suggests. A third of these arrive as ready-made GitHub ```suggestion``` blocks: reviewers
  don't just state the norm, they *perform* it.
- **API duplication is small (2.7%)** — much rarer than expected, probably because authors
  search before PRing and `exact?`/loogle already catch the easy cases.
- **Mathematical content is tiny (1.8%)** — at thread level, reviewers almost never argue
  about the mathematics. The proof assistant has fully absorbed that concern.
- **Falls outside the expected families** (not anticipated): API completeness/lemma-worthiness
  (9.2%, the "library-design" norm), binder/statement-form hygiene (6.3%), deprecation
  mechanics (4.3%), diff vigilance (3.0%), metaprogramming code review (3.5%),
  PR scoping (1.2%), performance (1.2%).

---

## 2. Category details

Verbatim quotes lightly truncated; PR numbers refer to leanprover-community/mathlib4.

### 1. GOLF — Proof golfing & tactic style (118/600, 19.7%)

The norm: a proof should be as short, transparent, and maintenance-robust as possible.
Sub-norms observed: squeeze `simp`s to reveal dependencies; avoid `erw`/`change`/defeq
abuse; don't produce the same proof twice (`wlog`/`suffices` over `<;>`); prefer "sturdy"
tactics; extract sublemmas from long proofs; remove dead lines; avoid tactics inside `def`s.

> PR#32906 (j-loreaux): "This proof is hideous. Instead of fixing the whitespace issue, I'd like to just replace the proof. It was likely written very long ago when some people preferred term-mode proofs. Here's a short and sweet one-line proof to replace it"
>
> PR#16877 (eric-wieser): "Instead of using `<;>`, can you use a `wlog` or a `suffices` for the inequality? That way you produce one proof and use it twice, rather than producing the same proof twice."
>
> PR#30213 (eric-wieser): "These lemmas all look suspicious to me; can you squeeze the `simp`s to make clear what existing lemmas they are using? My guess would be that many of these results are already in core with different names, which that will reveal."
>
> PR#34705 (chrisflav): "Please avoid `erw` and `change`, but instead (add and) use the correct API lemmas."

**V/A/T: A, with a T tail.** Detecting `erw`/`change`/unsqueezed `simp`/dead lines is V
(linters exist or are trivial). Whether a 5-line tactic proof beats a 1-line term proof is
A; threads like PR#16105 ("I prefer not having unnecessary definitions" vs "you have this
definition anyway") and PR#33602 ("I don't know if there's clear guidance on which form is
preferred") show a genuine taste residue.

### 2. DOC — Documentation & comments (70/600, 11.7%)

Docstring presence and *quality*: explain abbreviations and magic constants, keep
comments synchronized with code, docstring placement and audience.

> PR#6570 (kim-em): "Could we have a doc-string? At least telling me what `CS` stands for. :-)"
>
> PR#19204 (kim-em): "What is the meaning of this magic constant 100?"
>
> PR#20040 (YaelDillies): "My point is that docstrings will mostly be read *after the fact*. People will see them either when hovering over the definition in a later file or in the docs"

**V/A/T:** presence is **V** (mathlib's `docBlame` linter already enforces it for defs);
content quality, audience-appropriateness, and doc/code synchronization are **A**.

### 3. API — API completeness & lemma-worthiness (55/600, 9.2%)

The library-design norm: a definition is incomplete without its lemma cloud. Reviewers ask
for symm/iff/`ofNat`/equiv/dual variants, instances for common cases, induction principles
— and conversely push back on lemmas not worth having.

> PR#23946 (b-mehta): "There's a few other lemmas I can think of for the new definition, basically just describing the interaction with other defs in this file (I have a rough mental heuristic that there should be about 1/2 n^2 lemmas for n definitions, describing how each def relates to each other (within reason, of course!))."
>
> PR#9925 (eric-wieser): "Can you add the equiv versions too, where `e ⁻¹' Icc x y = Icc (e.symm x) (e.symm y)` or maybe also `e.symm ⁻¹' Icc x y = Icc (e x) (e y)`?"
>
> PR#7337 (YaelDillies): "I would drop this lemma since it is the combination of two much simpler rewrites and the statement is not in simp normal form" — countered by fpvandoorn: "I still think it's useful to keep … this is a simple statement that will probably occur more frequently"

**V/A/T: A** (an LLM can be taught the ½n² heuristic and the variant checklist), with a
**T** tail — lemma-worthiness disputes (PR#7337, PR#28818, PR#39892 "I do think it's
valuable to have named theorems/identities") are openly value-laden.

### 4. NAME — Naming conventions (51/600, 8.5%)

The most codified norm family: mathlib has a written naming convention, and reviewers
enforce it constantly — casing (`Finite` vs `finite`), dot-notation enablement, name must
parse compositionally from the statement, `to_additive`-generated names.

> PR#11926 (erdOne): "And maybe incorporate `Module.End` into the name? … Though I think `Finite` should be lowercase according to the naming convention."
>
> PR#28708 (j-loreaux): "`pdist` is impenetrable. I had absolutely no idea what it meant until I read both the lemma *and* its docstring."
>
> PR#26502 (sgouezel): "I thought that the convention was: if one writes `Monotone foo` then the name is `foo_monotone`, while if one writes `lemma … (hxy : x ≤ y) : foo x ≤ foo y` then the name is `foo_mono`. Have I dreamt this convention?" — YaelDillies: "It's certainly more of a dream than of a statu quo."
>
> PR#32103 (YaelDillies, dissenting from a rename): "Using `Foo.of_bar` to mean `Foo _ -> Bar _ -> Foo _` is a common and understood pattern. As such, this rename is both out of scope for this PR and (IMO) a regression."

**V/A/T: split.** Mechanical sub-rules (casing, `'`-suffix policy, name↔`to_additive`
consistency) are **V** — some already linted. Whether a name "reads well" or matches the
statement's semantics is **A**. And a visible slice is **T**: the `_mono` thread and the
`Splits.of_dvd` thread are reviewers *discovering they disagree about what the convention
is*. Naming is where articulated norms most clearly under-determine practice.

### 5. SIG — Statement form & binder hygiene (38/600, 6.3%)

How a statement should be *spelled*, independent of its content: implicit vs explicit
binders, `variable` section scoping, hypothesis spelling (`0 < k` lets you write `hk.ne'`),
which side of an iff/eq goes first, strict-implicit binders, `private` markers.

> PR#27193 (eric-wieser): "Strictly speaking the arguments to these lemmas should be explicit"
>
> PR#30484 (plp127): "Why are these arguments implicit? The other ones are all explicit."
>
> PR#16448 (ocfnash): "Nitpick but I slightly prefer this (even though it is defeq) … (e.g., not relevant here but you can only write `hk.ne` or `hk.ne'` with the `<` spelling)"

**V/A/T: A** dominant. There are semi-formal rules (arguments inferable from later
arguments should be implicit) that could be V-checked, but most threads turn on predicted
downstream usage ("the caller will want…"), which needs judgment.

### 6. SIMP — simp & attribute hygiene (37/600, 6.2%)

The simp-set is a shared global resource; tagging is reviewed like a commons. Sub-norms:
simp-normal form (more complex expression on the LHS), loop avoidance, confluence with
existing lemmas, and the growing attribute zoo (`norm_cast`, `gcongr`, `fun_prop`,
`nontriviality`, `reassoc`, `grind`).

> PR#21160 (jcommelin): "Usually the more complicated expression goes on the left hand side, because then you can simplify expressions by rewriting with this lemma from left to right. Could you please swap the two sides of the equation (and the name)?"
>
> PR#14832 (j-loreaux): "It can't be a `simp` lemma without reversing the order because then it would just keep adding `ᗮ` repeatedly."
>
> PR#31692 (jcommelin, on a potential loop): "Why is the linter not complaining, if this creates a loop?"

**V/A/T: V+A.** simp-NF and loop checking are partially mechanized today (`simpNF`
linter); reviewers are doing the *remainder* — whether a rewrite direction matches global
library convention, whether the simp-set's total behavior changes. Those need A.

### 7. STYLE — Formatting, typos & grammar (32/600, 5.3%)

Whitespace, indentation conventions, semantic line breaks, English grammar in docs,
notational style rules. Includes the 5 bot-authored threads (reviewdog lint-style/
lint-bib/imports) in the sample.

> PR#6816 (eric-wieser): "The style guide says `<|` should be used … its use in mathlib is disallowed in favor of `<|` for consistency as well as because of the symmetry with `|>`"
>
> PR#7919 (sgouezel): "When you have a multiline tactic like `simp` or `rw`, all lines but the first one are indented by two more spaces to help locate the beginning and the end of the tactic."
>
> PR#19880 (grunweg): "style nit: English grammer starts lower-cased after a colon"

**V/A/T: V.** Almost all of this is already linted or trivially lintable; human threads
here are the residue the linters don't yet cover (and reviewers say so explicitly,
e.g. PR#38361 "Which linter complained? We should fix that linter.").

### 8. GEN — Statement generality (27/600, 4.5%)

The canonical mathlib norm: state every result at its natural maximal generality — but no
further ("fake generality" is also flagged).

> PR#9343 (urkud): "It's enough to assume that the domain is an `AddZeroClass`. Please rewrite assuming `AddZeroClass`."
>
> PR#21159 (ADedecker): "Could you instead generalize `TopologicalGroup.isInducing_iff_nhds_one` to `MonoidHomClass`, so that it applies directly to ring homomorphisms?"
>
> PR#28817 (YaelDillies): "Not without strengthening `Semiring` to `Ring`, but I agree the former is fake generality, so I'll change it"

**V/A/T: A.** Lean could in principle search for the minimal typeclass (this is
semi-decidable, and tools have been prototyped), but knowing which generalization is
*mathematically natural* vs fake is exactly the articulable-expert layer. Note this
category is about generality of the *statement*; disagreement about the right generality
of a *definition* lands in MATH.

### 9. ORG — File/import/namespace organization (26/600, 4.3%)

Placement of declarations, import-graph weight, namespace and section design, lemma
ordering within a file.

> PR#24804 (joelriou): "I would suggest moving this part to a new file `WithTerminal.Discrete`, so as to reduce imports."
>
> PR#9869 (sgouezel): "I'm not sure this section belongs to this file: does it involve Inv?"
>
> PR#24914 (b-mehta): "Is it really sensible to be importing material about torsion to a file which just does synonym stuff?"

**V/A/T: A with V assists.** Import minimization is tool-supported (`#min_imports`,
`#find_home` — authors cite them in threads, e.g. PR#28669); "where does this concept
*belong*" is judgment.

### 10. DEPR — Deprecation & migration mechanics (26/600, 4.3%)

Era-specific but persistent: during the mathlib3→4 port (2022–23) this was `#align`
bookkeeping and porting notes; now it's `@[deprecated (since := …)]` aliases on every
rename/removal.

> PR#17295 (j-loreaux): "deprecation missing"
>
> PR#6875 (mcdoll): "You should not delete #aligns (same below)"
>
> PR#18315 (grunweg): "I would say: yes, out of principle. Dealing with deprecations is *much* nicer than dealing with removed definitions."

**V/A/T: V.** Scripts already generate deprecations (a thread in the sample, PR#17295, is
literally about a bug in the deprecation-generating script); the norm is fully
mechanizable, the threads are mostly about tooling gaps.

### 11. META — Metaprogramming robustness (21/600, 3.5%)

Review of Lean metaprograms (tactics, linters): proper monad discipline, `Expr`
normalization, cache safety, test coverage. Reads like systems code review, by a handful
of experts.

> PR#36841 (JovanGerb): "This function should be in `MetaM`, because otherwise it may misuse the `SimpM` cache. … you can only share a cache safely if every use of the cache is computing the same thing."
>
> PR#25501 (b-mehta): "Isn't `consumeMData` the usual way to ignore metadata? In particular, I'm anxious about whnf doing more things (eg slower things) than just discarding metadata"

**V/A/T: A** (expert software judgment; not mathlib-specific).

### 12. MISC — Social / pointer / unclassifiable (20/600, 3.3%)

Praise ("nice!" PR#18145; "Powergolf! 🏌️ ⛳" PR#31895; "Very nice golf!" PR#34357), "ditto"/
"same here" pointer comments whose content lives in another thread, reminders, test
comments. For the full-scale pass these need an `OTHER/POINTER` bucket — pointer comments
(~2-3%) can only be classified by joining the sibling threads of the same review.

### 13. INFRA — CI/build/scripts (20/600, 3.3%)

Threads on `.github/workflows/*`, `Cache/*`, `scripts/*`, lakefile: token permissions,
cache correctness, cross-platform behavior, toolchain pinning. Ordinary software review.

> PR#23868 (grunweg): "Do you also need to disable the `build`, `lint` and `test` steps explicitly? The lean-action README is not clear to me, can you check?"
>
> PR#21822 (eric-wieser, Cache): "This only works on linux, right? The shell doesn't expand globs on windows." (PR#1430)

**V/A/T: A** (generic code review; out of scope for math-norm extraction — keep as a
separate bucket and consider filtering by `path` prefix).

### 14. VIGIL — Diff vigilance (18/600, 3.0%)

Reviewers audit the diff for *unintended* content: bad merges, accidentally deleted
lemmas, changes that don't match the PR's stated purpose. This is the closest the corpus
gets to correctness review — and it's about repository state, not mathematics.

> PR#19506 (kim-em): "I'm guessing this is a bad merge. Could someone try reverting this file?"
>
> PR#9176 (eric-wieser): "Is this some merge weirdness, or genuinely intended as part of this patch?"
>
> PR#31313 (erdOne): "Why is this lemma removed?" — plp127: "Must have been accidentally eliminated in resolving the merge conflict"

**V/A/T: V+A.** Declaration-diff tooling (mathlib's `decl_diff` bot) already mechanizes
detection; deciding intent needs the human/LLM.

### 15. DUP — API duplication / use-existing (16/600, 2.7%)

"This already exists." Smaller than expected — likely because `exact?`, loogle, and Zulip
searching catch most duplicates pre-review.

> PR#14458 (urkud): "Do we have `{y | ∀ x ∈ m, B x y = 0}` (a.k.a. the intersection of all `(B x).ker`) as a submodule for any set (can't find it)? If yes, then we should have a lemma saying that `polar` is equal to this set instead of adding a new definition."
>
> PR#2498 (eric-wieser): "I PRd these already as #2049..."
>
> PR#38218 (gasparattila): "Special case of `Measurable.of_discrete`."

**V/A/T: V-assisted A.** Candidate-retrieval is V (semantic search over the library);
the up-to-definitional-equivalence judgment ("special case of", "should be canonical
spelling") is A. Notably the *least resolved* category (see §3).

### 16. MATH — Mathematical design choices (11/600, 1.8%)

The rare threads about mathematical substance: which definition is canonical, junk-value
conventions, typeclass-diamond safety, whether a TODO's proposed generalization even makes
sense.

> PR#25042 (ocfnash): "Do we really want to have the power to control the junk values?"
>
> PR#21371 (MichaelStollBayreuth): "I'm not sure that `IsDiscrete` is a good name; the point is (as far as I can see) that the discrete valuation is *normalized*. The discreteness is captured by the target group already." (naming dispute that is really a dispute about the mathematical concept)
>
> PR#14210 (kbuzzard): "If the base isn't a field then P isn't a point, it's a section, so 'point...is nonsingular' perhaps does not even make sense."

**V/A/T: T.** These regularly escalate to Zulip and end in judgment calls by senior
maintainers; the community itself treats them as not settleable by rule.

### 17. PERF — Performance (7/600, 1.2%)

Compile-time performance of instances/proofs; benchmark culture (`!bench`).

> PR#19181 (YaelDillies): "I am pretty sure `AlgHom.map_add` was used for performance reasons... Can you please run !bench once CI passes?" — kim-em: "The difference here is 5000 heartbeats, again inconsequential."
>
> PR#22842 (eric-wieser): "I worry this instance and the `isDomain` one below might be a performance drag; could you split them (together) to a separate PR so that we can benchmark them in isolation?"

**V/A/T: V.** Fully benchmarkable; the only judgment is the threshold (5000 heartbeats =
"inconsequential").

### 18. SCOPE — PR scoping & process (7/600, 1.2%)

One-PR-one-change; refactors don't ride along with features; follow-up-PR etiquette.

> PR#9403 (kmill): "For a `feat` PR, it's better practice to not mix refactoring into it, unless you can find some reviewer who thinks it's immediately a good idea … If you limit the PR to just adding `equivPresentedGroup` you'll have a better shot at getting it quickly reviewed."
>
> PR#19775 (Ruben-VandeVelde): "This seems unrelated, and there's more unrelated changes in the PR. Would you mind splitting them out or reverting them?"

**V/A/T: V+A.** Diff-vs-title consistency is checkable; "is this related enough" is A.

---

## 3. Thread-metadata patterns worth modeling

1. **Resolution skews hard by category.** The sample is 50/50 resolved/unresolved by
   design, so the cross-category *relative* differences are the signal:
   GOLF 70% resolved, API 64%, DOC 60%, STYLE 56%, NAME 55% — concrete, actionable asks
   get closed. DUP 6%(!), INFRA 10%, ORG 27%, VIGIL 28%, GEN 30% — open-ended/design
   threads stay unresolved. Caveat: `isResolved` is a button people forget; unresolved ≠
   rejected. Still, "resolvedness" is a usable weak label for *norm concreteness*.
   (Population base rate: ~78% of threads are resolved — the sample oversamples
   unresolved 3.5×.)

2. **Era shifts (within-era share):**
   - DEPR collapses 12.6% (≤2023) → 5.3% (2024) → 2.2% (2025-26): the `#align`/porting
     bookkeeping was an era artifact. Any extraction over the full corpus must either
     model era or expect a DEPR-heavy 2022-23 slab.
   - META is 8.0% in ≤2023 (porting tactics) then ~1-3% after.
   - GOLF grows 12.6% → 16.8% → 22.0%: as the library matured (and `grind`, `fun_prop`,
     `gcongr` landed), more review bandwidth goes to proof quality.
   - SIG grows 2.3% → 8.2% and GEN 1.1% → 5.2%: rising attention to optimal statement
     form once content is in place. Norms are *thickening* over time.

3. **36% of opening comments contain a GitHub ```suggestion``` block** — the reviewer
   supplies the exact replacement code. These threads are gold for extraction: the norm
   comes with a before/after diff pair. They also resolve at a much higher rate.

4. **Reviewer concentration:** the top 12 reviewers author 55% of opening comments
   (eric-wieser alone 12%). Norm articulation is an oligarchy; per-reviewer norm profiles
   (e.g. eric-wieser → API/bundling/simp-NF; grunweg → style/docs; joelriou → category
   theory API design) are feasible and would let us separate "community norm" from
   "individual taste" — directly relevant to the subjective-vs-intersubjective question.

5. **Zulip is the escalation valve (~2% of threads link it).** When a thread hits a
   contested norm (naming conventions, simp policy, new definitions), the move is "ask on
   Zulip". A thread ending in a Zulip link is a strong signal of a **T-zone or
   norm-vacuum** — worth flagging as a feature in the full pass.

6. **50% of threads are single-comment** (no reply): either silently accepted
   (suggestion applied) or ignored. Reply count and whether the author pushed back
   ("negotiation depth") is cheap to compute and separates *settled* norms from
   *negotiated* ones — e.g. NAME threads have visibly more pushback than STYLE threads.

7. **Path prefix is a strong prior:** `.github/`, `scripts/`, `Cache/`, `lakefile` →
   INFRA; `Mathlib/Tactic/` → META-heavy; `MathlibTest/`, `test/` → META. Consider
   routing these separately rather than diluting the math-norm extraction.

8. **~1% of opening comments are bots** (reviewdog lint suggestions, crossref bots) —
   filter by `author.login` ∈ {github-actions, …} or treat as their own AUTOMATED class.

---

## 4. Extraction prompt skeleton for the full 139K-comment vLLM pass

Recommended unit: **thread** (not comment) — classify on the first comment, with replies
as context for the `resolution`/`negotiation` fields. Output one JSON object per thread.

```text
SYSTEM:
You are analyzing code-review threads from mathlib4, the Lean 4 mathematical library.
Correctness is machine-checked by the Lean compiler and CI, so review comments express
quality norms BEYOND correctness. Classify the norm articulated by the FIRST comment of
the thread. Replies are provided as context only.

Categories (choose exactly one primary; optionally one secondary):

- PROOF_STYLE: shorter/cleaner/more robust proof requested: golfing, term vs tactic
  mode, squeezing simp calls, avoiding erw/change/defeq abuse, avoiding duplicated
  subproofs, extracting sublemmas, preferring sturdy tactics.
- DOCUMENTATION: add/fix/relocate a docstring or comment; explain a magic constant;
  keep comments in sync with code; module-doc content.
- API_COMPLETENESS: add missing variants (symm/iff/ofNat/equiv/dual), instances,
  induction principles; OR push back that a lemma/def is not worth having.
- NAMING: declaration name choice: casing, dot-notation, naming-convention patterns
  (Foo.of_bar, _mono, primes), name↔statement consistency, to_additive names.
- STATEMENT_FORM: how the statement is spelled: implicit vs explicit binders, variable
  sections, hypothesis spelling (0 < k vs k ≠ 0, Ne vs Not), iff/eq direction,
  canonical spelling of an expression, private/protected.
- SIMP_ATTRS: attribute hygiene: should/shouldn't be @[simp]; simp-normal form; simp
  loops; norm_cast/gcongr/fun_prop/nontriviality/reassoc/grind tagging.
- FORMATTING: whitespace, indentation, line breaks, English grammar/typos, notation
  spacing, style-guide mechanics.
- GENERALITY: weaken/strengthen hypotheses or typeclasses of a STATEMENT ("monoid
  suffices"), universe generality, fake-generality pushback.
- ORGANIZATION: which file/section/namespace a declaration belongs in; import
  minimization; ordering of declarations.
- DEPRECATION: deprecated aliases, #align (mathlib3 port), porting notes,
  don't-delete-just-deprecate.
- METAPROGRAMMING: review of tactic/linter/meta code internals: monads, Expr handling,
  caches, elaboration, test robustness of tactics.
- CI_INFRA: GitHub workflows, build scripts, cache system, toolchain — non-Lean
  infrastructure.
- DIFF_VIGILANCE: questioning whether a change is intended: bad merges, accidental
  deletions, unexplained or unrelated diff hunks.
- DUPLICATION: the result/definition already exists in mathlib or core, or is a special
  case of an existing one; duplicate PR.
- MATH_DESIGN: choice of mathematical definition itself: canonical notion, junk-value
  conventions, typeclass-diamond safety, whether a generalization is mathematically
  meaningful.
- PERFORMANCE: compile-time/elaboration performance, benchmarks, heartbeats, instance
  cost.
- PR_PROCESS: PR scoping (split unrelated changes), follow-up-PR etiquette, review
  workflow (bors, delegation).
- OTHER_SOCIAL: praise, thanks, pointer comments ("same as above", "ditto"), bot
  notifications, unclassifiable.

Also output:
- norm_statement: one sentence stating the norm as a general rule (not the instance).
- vat: "V" if a program/linter/benchmark could check compliance; "A" if it requires
  natural-language/mathematical judgment an LLM could render; "T" if the thread shows
  the norm is contested or case-by-case among experts.
- has_suggestion_block: bool (first comment contains ```suggestion).
- author_response: one of {accepted, pushed_back, negotiated, no_reply}.
- escalated_to_zulip: bool.

USER:
PR #{number} ({year}), file: {path}, resolved: {isResolved}
[REVIEWER] {comment_0}
[REPLY 1] {comment_1}
...
JSON:
```

Operational notes for the full pass:
- **Few-shots:** include one per category drawn from §2's quoted threads (they are
  verbatim, with PR numbers, and were hand-validated here). Per
  `feedback_local_explanations_per_task_fewshots`, register them alongside the task.
- **Routing:** pre-route by path prefix (CI_INFRA, METAPROGRAMMING candidates) and
  pre-tag bot authors; this removes ~7% of threads from the math-norm prompt.
- **Era covariate:** keep `year`; expect the DEPR slab in 2022-23 and re-normalize
  shares within era when reporting trends.
- **Pointer comments:** "ditto/same as above" threads (~2-3%) need the sibling threads
  of the same PR review in context, or an explicit POINTER output that triggers a join.
- **Validation:** the 600 hand labels in `notes/thread_norm_pilot_labels.json` (+ the
  gzipped sample JSONL) are a ready-made dev set; target ≥0.7 macro-agreement before
  scaling (validate before scaling — inspect LLM outputs on ~50 threads first).

---

## 5. One-paragraph V/A/T takeaway

With correctness fully delegated to the machine, mathlib reviewers spend their words on:
how proofs are *written* (20%), how the library is *documented* (12%), what the API
*should contain* (9%), what things are *called* (8.5%), and how statements are *spelled*
(6%). Roughly a fifth of articulated norms are re-mechanizable (and the community
visibly keeps mechanizing them — linters, bots, deprecation scripts, `!bench`), the
large majority sit squarely in the A-band (statable as rules an LLM judge could apply,
but requiring contextual judgment), and a thin but important T-residue (mathematical
design choices, contested conventions, lemma-worthiness) is openly negotiated through
Zulip and maintainer fiat. The corpus also shows norms migrating V-ward over time: what
reviewers said by hand in 2022 (style, imports, deprecations) is said by bots in 2026,
while human bandwidth shifts up-stack to golfing, generality, and API design.
