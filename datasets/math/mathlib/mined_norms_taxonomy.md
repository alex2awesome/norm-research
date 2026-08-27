# mathlib4 style norms — mined from ~1800 review comments (12 sonnet subagents)

Consolidated & deduped across chunks. Each norm tagged **[V-checkable]** (deterministic regex/AST) or
**[A-semantic]** (needs reading/understanding). **NEW** = not in current 11 metrics (doc_cov, naming camelCase,
line>100, tactic_auto ratio, import order, sorry, ...).

## 1. NAMING (largest cluster)
- **Avoid prime `'` suffix in decl names** (foo'→foo_alt) — NEW [V-checkable]. 5 chunks, ~18 mentions.
- **Avoid redundant namespace prefix** (already in namespace) — NEW [A-semantic: needs namespace context]. 6 chunks.
- **Instance naming**: use auto `inst*`, no type suffix (Prod.instAdd not instAddProd), don't name instances — NEW [V-checkable-ish]. 5+ chunks.
- **Predicate `Is`/`Has` prefix** — NEW [V-checkable].
- **camelCase not snake_case (rule 4)** — [V-checkable] (current naming_ok partially covers).
- **Name matches formal statement** — NEW [A-semantic]. 3 chunks.
- **"of" = implication only; iff-before-of; left-to-right; name-order matches symbol-order** — NEW [A-semantic].
- **US-English spelling** (fiber not fibre) — NEW [A-semantic/lexical].
- **capitalize type, lowercase verb in theorem names** — NEW [A-semantic].
- **no trailing `_of` / `fun_` prefix** — NEW [V-checkable].

## 2. TACTIC / PROOF STYLE
- **Avoid non-terminal `simp` / squeeze it** — NEW [V-checkable: detect bare simp mid-proof]. 5 chunks, ~20 mentions — HUGE.
- **Combine consecutive `rw`** — NEW [V-checkable].
- **`change` tactic is a code smell** (prefer rw/simp lemma) — NEW [V-checkable].
- **Avoid `try` tactic; avoid `by exact` boilerplate; avoid semicolons in proofs; no blank lines in proofs; one tactic per line** — NEW [V-checkable]. ~4 chunks.
- **Use `letI` not `have` for instances; `let` not `letI` for non-typeclass** — NEW [V-checkable-ish].
- **Bullet (·) structure for cases; bullet only easy cases** — NEW [partially V].
- **Use `mt` for modus tollens; avoid defeq abuse; avoid backwards assoc** — [A-semantic].

## 3. FORMATTING / INDENTATION (most checkable, MOST MISSED)
- **4-space indent for theorem STATEMENT lines, 2-space for tactic PROOF lines** — NEW [V-checkable]. ~8/11 chunks — THE most-mentioned norm, totally absent from current metrics.
- **multiline tactic: indent continuation +2; signature line-break +2; return type on new line** — NEW [V-checkable].
- **`match`/`calc` no extra indent; braces-continuation rules** — NEW [V-checkable].
- **space after `←`; spaces around binary operators; no consecutive spaces** — NEW [V-checkable].
- **no extra indent inside docstring blocks** — NEW [V-checkable].

## 4. IMPORTS / ORGANIZATION
- **Remove unused/redundant imports** (`#min_imports`) — NEW [V-checkable-heuristic]. 3 chunks.
- **Alphabetical import order** — [V-checkable] (current import_ok weakly covers). 5 chunks.
- **Minimize `public` imports; no unnecessary `open`** — NEW [V-checkable].
- **Use `section` for organization/navigation** — NEW [partially V].
- **Move general lemmas to appropriate files; split files by topic** — [A-semantic].

## 5. GENERALITY / API DESIGN (the single most FREQUENT norm)
- **Generalize / weaken assumptions / `Type`→`Type*`** — NEW [A-semantic]. ~24 mentions in 2 chunks alone — TOP frequency.
- **Remove redundant hypotheses** — NEW [A-semantic].
- **Avoid duplicate lemmas; reuse existing API** — NEW [A-semantic: needs library knowledge].
- **`variable` declarations to reduce repetition** — NEW [partially V].
- **make inferable args implicit; typeclass arg ordering** — NEW [A-semantic-ish].
- **`abbrev` vs `def` choice; prefer unbundled; `Finite` over `Fintype`** — [A-semantic].

## 6. DOCSTRINGS
- **complete sentences + terminal period** — NEW [V-checkable].
- **explain the concept, don't restate/copy-paste the type** — NEW [A-semantic].
- **no abbreviations on first use; Lean over LaTeX; single backticks** — NEW [V-checkable-ish].
- **module docstring with "Main definitions" section** — NEW [V-checkable-ish].
- **document junk values; no context-dependence; no TODO** — NEW [mixed].

---
## KEY TAKEAWAYS
1. **Current V misses the 3 most-checkable, highest-frequency norms**: INDENTATION (4/2 rule), PRIME-suffix avoidance, and NON-TERMINAL simp. Adding these (all regex/AST) should lift matched-V well above 0.82.
2. **The articulability gap (V→A) is the semantic cluster**: generalize/weaken statements, name-matches-statement, iff-of ordering, duplicate-lemma detection, abbrev-vs-def. These are crisply stated (articulable) but require reading = exactly A-judge territory.
3. **Top frequency norm overall = "generalize the statement"** (~24+ mentions) — pure A, no deterministic metric can touch it.
