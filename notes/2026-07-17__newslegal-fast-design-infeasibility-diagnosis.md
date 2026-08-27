# Why news-homepages + legal-outcome-prediction fail v14 FAST design

**2026-07-17 (overnight).** Diagnosis of the 37/37 deterministic design failures when expanding
FAST to the two built-but-unused Tier-B domains (news-homepages 25, legal-outcome-prediction 12).
Analysis run read-only against sk2 (`outputs/fast_expand_newslegal/fast_design_exclusions.json` +
Tier-B bootstraps); scratch scripts `~/.claude/jobs/395f1ee1/tmp/diag_newslegal{,2}.py`.

## One root cause, two symptoms

Both failure reasons — `teaching_label_balance_infeasible` and `panel_infeasible` — are the same
underlying problem at different severities: **extreme label skew under the frozen Llama-3.1-8B
executor**. Labels are `P(YES) > 0.5` over the 300 frozen probes; design needs ≥3 probes per class
population-wide (`ensure_teaching_label_balance`, `v14_panel_design.py:122`), then enough minority
probes that 50 hard-balanced panels can cover every teaching probe under same-label-swap repair
(`v14_panel_design.py:370`).

Sampled label distributions (n=300 probes each):

| metric | mean p_yes | std | n_pos | reason |
|---|---|---|---|---|
| news_R3_m12 | .095 | .042 | **0** | balance_infeasible |
| news_R3_m10 | .052 | .082 | **2** | balance_infeasible |
| news_R3_m0 | .199 | .102 | **4** | panel_infeasible |
| news_R3_m1 | .378 | .092 | **25** | panel_infeasible |
| legal_R3_m0 | .061 | .023 | **0** | balance_infeasible |
| legal_R3_m2 | .310 | .071 | **0** | balance_infeasible |
| legal_R3_m1 | .399 | .077 | **35** | panel_infeasible |
| legal_R3_m5 | .343 | .111 | **31** | panel_infeasible |

Aggregate over ALL Tier-B metrics (minority-class size):

| task | n | median minority | frac <3 | frac <40 |
|---|---|---|---|---|
| **news-homepages** | 25 | **0** | .68 | .96 |
| **legal-outcome-prediction** | 12 | **0** | .67 | 1.00 |
| creative-writing | 67 | 76 | .00 | .21 |
| math-stackexchange | 11 | 74 | .00 | .27 |
| peer-review | 13 | 30 | .23 | .77 |

The executor's P(YES) never crosses 0.5 for most news/legal criteria (means .05–.40) — the
documented Llama-8B weak-executor discrimination floor (R2 notes), now shown to be
**domain-graded**: severe on news/legal, mild on creative-writing/math. This is also consistent
with the 152/220 Tier-B executor-feasibility census from 2026-07-16.

**Asset vintage ruled out**: all 7 tasks share the same vintage (300 probes, one probe_sha per
task, same manifest schema). Label problem, not asset problem.

## Fixability verdict

- **17 news + 8 legal metrics with minority ≤2 (many 0/300): genuinely degenerate under the
  frozen 8B executor — PARK.** No probe extension helps a metric where the executor answers NO on
  everything; swapping executors would change the frozen estimand (trade-off flagged, not
  recommended).
- **8 news + 4 legal metrics with minority 4–35: fixable-with-extend-probes.** Make the existing
  extend-probes phase **class-targeted**: mine extension texts the executor itself scores >0.5
  (executor-score selection — NOT gold-label-aware, so it stays label-blind per standing rules).
  Roughly +50–300 targeted probes to lift minority counts toward the empirically-feasible ~40+
  zone (peer-review succeeded at minority ≈30, so signature margins also matter; treat 40 as
  approximate).

**Next action per domain**: class-targeted extend-probes pass for the 12 panel_infeasible metrics
(news 8, legal 4) → re-run design; formally park the 25 balance-infeasible metrics with the
degeneracy stats above as the recorded reason.
