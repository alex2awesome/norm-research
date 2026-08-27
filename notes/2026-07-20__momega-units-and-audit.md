# 2026-07-20 — M_ω unit audit: every shipped unit, source, and the 0/1-unit diagnosis

User directives (2026-07-20): paper-exact everything; report LLM-derived units separately;
note the actual units; audit why some runs compile 0-1 units.

## Shipped units, verbatim, with source attribution

### paper-exact hover / Qwen3-8B — SHIPPED, beat GEPA (.517 vs .450, p=.0037)

| # | module | source | unit |
|---|---|---|---|
| 1 | create_query_hop2 | trajectory-mined | You must use the context from the claim and summary to deduce these unstated entities, as the target evidence documents rely on these specific names. |
| 2 | create_query_hop2 | **LLM-suggested** | "Dave Evans" (in the context of rock bands) refers to the singer **Dave Evans**. |
| 3 | summarize1 | **LLM-suggested** | Explicitly state which specific entities, dates, or numerical figures from the claim are missing from the retrieved passages. This clearly flags the exact knowledge gaps that future search iterations must target. |

2 of 3 winning units are LLM-suggested (incl. one entity-fact unit that DID transfer on a
300-item test, unlike T1-hover's Arkapaw unit on 200 items). Clean source-level attribution
of the +.067 margin needs a trajectory-only ablation compile (not yet run).

### T1 hotpotqa / GLM-4.7 (NOT paper-exact; superseded as headline) — 7 units, +.030 n.s.

1. Include introductory prepositions in your answer if they accurately describe the relationship, such as 'from', 'in', or 'during'.
2. Preserve exact article usage (e.g., 'the', 'a'), quotes, and modifiers exactly as they appear in the context.
3. Include necessary articles such as 'the', 'a', or 'an' at the beginning of your answer.
4. Pay close attention to grammatical completeness and exact phrasing based on the question asked:
5. If the question asks "was formerly known as what", "is what", or similar constructs that expect a noun phrase, prefix your answer with "the" if the context names a specific entity (e.g., answer "the North Atlantic Conference" instead of "North Atlantic Conference").
6. Include descriptive qualifiers and nouns if they are part of the specific named phrase or answer in the text (e.g., answer '"Teach the Controversy" campaign' instead of just "Teach the Controversy", or '3,677 seated' instead of just '3,677').
7. Always provide a complete geographic designation, such as including both the specific location and its enclosing city or state.

(pool was 12 LLM-suggested + 79 mined; per-unit source tags were not logged in this run — harness now logs them)

### T1 hover / GLM-4.7 (NOT paper-exact) — 1 unit, did not transfer (-.020 n.s.)

1. DIRECTOR/CINEMATOGRAPHER MISMATCH: Adam Arkapaw was the cinematographer for the 2015 horror film "The Witch".

### paper-exact AIME (both LM columns) — 0 units (guard shipped GEPA init; correct 4/4)

## The 0-unit / 1-unit audit (user question)

**AIME-Qwen, 0 units.** The 27-item select panel has NO ranking signal: Spearman(panel
marginal, 150-item test rescore) over the 24 unit candidates = **+0.186, p=.38** (~zero).
The candidate that is best on test (.4667) got panel score .518 vs init .556 — a NEGATIVE
panel marginal — so selection never even considered it; the single unit that cleared the
panel failed confirm and the guard shipped GEPA's prompt. Selection-instrument failure,
not unit-pool failure (top-3 of the whole rescored pool are unit compiles .467/.460/.447).

**T1-hover, 1 unit.** Many positive marginals (top-5: +.150/.133/.133/.133/.133 with
12-3-style win-loss records) but the cumulative-prefix sweep plateaued at k=1: the units
are REDUNDANT (entity-fact style, overlapping coverage) so conditional value collapses
after the first accept. Contrast T1-hotpotqa: smaller marginals (+.100...+.033) but
ADDITIVE format/phrasing rules -> prefix kept improving to k=7.

**Reading: unit COUNT is a symptom, not a lever.** >=3 units appear exactly where (a) GEPA
left headroom and (b) units are additive process-rules; 0-1 units appear where the panel
has no signal (AIME n=27) or units are redundant single-fact patches (hover-T1). The
binding constraint is selection power (D3, third confirmation).

## Certified-bound ladder per paper-exact dataset (state as of this note)

| dataset (LM) | best shipped | pool max | union oracle (pool-certified single-prompt cap) | EVT endpoint (process-conditional, est.) |
|---|---|---|---|---|
| aime (Qwen3-8B) | .367 (GEPA=M_omega) | .4667 | **.5933** | GPD .4667 [.440,.491] / Pickands .518 [wide] - unstable, do not quote single number |
| aime (GLM-5.2, fixed) | .467 (seed; no arm gained) | pending rescore | pending | pending |
| hover (Qwen3-8B) | .517 (M_omega v2) | pending rescore | pending | pending |
| hotpot (Qwen3-8B) | running | - | - | - |

Certificate strengths: union oracle is certified ONLY relative to the pool (no single
prompt selected from this pool can exceed it); EVT is a process-conditional ESTIMATE (i.i.d.
violated, undershoots the all-prompt truth - gestalt argument); the only all-process bound
is rung 0 (DPI/Bayes), which is non-trivially certifiable only with extra task structure -
for these benchmarks no non-trivial rung-0 number exists and we do not fabricate one.
