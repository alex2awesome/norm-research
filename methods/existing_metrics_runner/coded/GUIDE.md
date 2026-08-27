# Metric Implementer — Per-Norm Verification Programs

## What this is

A library of **per-aspect** Python programs that each measure ONE evaluation
norm using real verification tools (linters, AST parsers, complexity analyzers,
test runners), not regex heuristics on the artifact text.

Each metric is a measurement, not a classifier. Downstream code (RF, LR, etc.)
combines metric outputs into label predictions. This is the pattern that worked
for Tier 3 (`lizard` features lifted code_review AUC from 0.616 → 0.627);
generalizing it to per-norm coverage is the goal.

## Why this exists (and why it is NOT `verification_library/`)

`methods/verification_library/` predicts the **end label** (PR accept/reject,
paper accept/reject) directly from a single LLM-written Python program per
example. Empirically that hit ensemble AUC = 0.500 on peer_review across 7+
runs because the prompt restricted programs to stdlib-only regex heuristics
and never invoked actual tools.

`methods/existing_metrics_runner/coded/` instead writes **one program per norm**, each
program required to call a real verification tool. The output is a
norm-conformance score, not a label prediction. Many such scores feed a
downstream model.

## File layout

```
methods/existing_metrics_runner/coded/
├── GUIDE.md                # this file — read first
├── runner.py               # batches metrics over diffs, builds feature matrix
├── sandbox.py              # subprocess wrappers with timeout + tool whitelist
├── metrics/
│   ├── __init__.py
│   ├── {aspect_id}_{slug}.py   # one file per metric, see template below
│   └── ...
└── fixtures/
    └── sample_prs.json     # 22 (diff, label) tuples for fast local iteration
```

## Metric contract

Every `metrics/{id}_{slug}.py` must export:

```python
ASPECT_ID = "a316"
ASPECT_NAME = "Python linter diagnostics compliance"
TIER = 3                      # see tier table below
TOOLS = ["ruff"]              # subprocess tools used; empty for pure-AST
APPLIES_TO_LANGS = ["Python"] # router hint
CLASSIFICATION = "THIN"       # "THIN" | "PARTIALLY_THIN" | "THICK"

def applies(diff_text: str) -> bool:
    """True iff this metric can produce a meaningful score for this diff."""
    ...

def score(diff_text: str) -> float | None:
    """Return [0,1] (1 = norm strongly satisfied). None = could not measure."""
    ...
```

**Three return states for `score()`:**

| return | meaning |
|---|---|
| float in [0,1] | measurement succeeded |
| None when applies()=False | norm doesn't apply (router-gated) |
| None when applies()=True | tool failed or signal unreliable — abstain |

The downstream feature matrix encodes this as **two columns per metric**:
`score` and `applied` (1/0). Abstain is `(applied=1, score=NaN)` and is
distinguishable from non-applicable `(applied=0)`.

## Tier table

| Tier | What it can call | Examples |
|---|---|---|
| 1 | stdlib only (`re`, `string`, …) | regex over diff text |
| 2 | `ast`, `tokenize`, structural parsing | imports, defs, syntax |
| 3 | external CLI tool via subprocess | `lizard`, `ruff`, `eslint`, `mypy`, `pylint`, `radon`, `pmd`, `checkstyle` |
| 4 | execution / test-running | `pytest --collect-only`, `pylint -E`, syntax compile, type-check, language-server diagnostics |

**Default to the highest tier you can reasonably reach.** Tier-1 regex
is a signal of last resort and should be paired with a clear `CLASSIFICATION =
"PARTIALLY_THIN"`.

## Required: write `applies()` before `score()`

`applies()` is the **applicability gate** the codegen_claude system was
missing. It must be cheap (run via diff parsing only, no subprocess) and
must over-abstain rather than over-apply. A norm about Prisma schemas should
return `False` on a Python-only PR. The downstream RF reads abstained metrics
as missing data, not as 0.5 noise.

## When to mark a metric `THICK`

If after honest exploration (including web search for tools) you find no
deterministic way to measure the norm, the metric body should be:

```python
CLASSIFICATION = "THICK"

def applies(diff_text): return False
def score(diff_text):   return None
```

The metric file still exists so the catalog is complete; it just records that
the norm is *not* deterministically verifiable. THICK metrics are NOT failures
— they are the most informative measurements of the articulability boundary.

Bias for *honest THICK* over *desperate Tier-1 regex*. A regex that returns
0.55 for everyone is worse than a True/None pair that says "I cannot measure
this."

## Tool registry — what the sandbox provides

The shared sandbox guarantees the following are on `$PATH` inside metric
execution:

- `lizard` — cross-language cyclomatic complexity & NLOC
- `ruff` — Python linter (fast, replaces flake8 + pylint subset)
- `radon` — Python maintainability index, raw metrics
- `ast` (Python stdlib) — Python syntax tree
- `python3 -m py_compile` — Python syntax check
- (optional, installed-if-needed) `eslint`, `mypy`, `pylint`, `bandit`,
  `checkstyle`, `pmd`, `semgrep`

If a tool is needed but not yet in the registry, add it to `sandbox.py`'s
`ALLOWED_TOOLS` list and document why.

## Implementation protocol

For each candidate norm:

1. **Read** the aspect description in `aspects.json`.
2. **Identify** the tool family that measures it. Web-search if unsure.
3. **Write** `applies()` first, restricted to diff parsing.
4. **Write** `score()` using the strongest available tool (highest tier).
5. **Run** on 22 fixtures (`fixtures/sample_prs.json`).
6. **Diagnose**: did `applies()` fire on at least 4 fixtures? Did `score()`
   produce non-constant output? Iterate if not.
7. **Mark THICK** if exploration genuinely fails — do not paper over with
   regex.

## What `score()` actually measures

`score()` returns conformance to the **norm**, not predicted label. A high
score means "this PR satisfies this norm well." Correlation with the
accept/reject label is the empirical question downstream models answer.

So `a181` ("Treat warnings/lints as errors") = 1.0 when zero lint
violations are added by the diff, decaying to 0.0 as violations pile up.
Whether that correlates with merge is the experiment, not the design goal.

## Coding-time tooling

You (the implementer) may web-search for tools, read tool docs, and inspect
the diff fixtures freely. The output is one tested `.py` file per metric.
The metric runs in the sandbox with no network access.

## Implementation discipline: library-first, regex when honest

**Why we prefer parsers / libraries over regex on code:**

The codegen_claude diagnostic showed that regex pretending to parse code
produces zero signal beyond text length. 1182 Python programs, each
regex-matching keywords against PR text, reached AUC = 0.59 — *worse* than
5 metadata features. Their "v2_holistic" survivors all converged to the
same comment-length proxy because pattern-matching against the wrong
abstraction level (text → code) collapses into measuring text length.

A real parser (tree-sitter, `ast`, language CLI tool) walks structure, so
"function named `equals` with `Object` parameter" stays a real measurement
even on a 5-line snippet. That is the difference between AUC 0.518 and 0.627
on code_review.

**The actual rule, in order of preference:**

1. **A real tool exists for this artifact** → use it. `lizard` for CCN,
   `ruff`/`pylint` for Python lint, `tree-sitter-X` for syntax, `radon` for
   Python maintainability, `prettier`/`gofmt` for format checks, `spacy`
   for prose. Search before implementing — most norms have a tool.
2. **No tool, but a parser exists for the artifact format** → use the parser.
   `whatthepatch` for diffs, `ast` for Python, `tree-sitter` for 40+
   languages, `spacy` for English prose, `mwparserfromhell` for wikitext.
3. **No parser, but the format is a well-defined text format** → regex with
   `# REGEX_OK: <reason>` is appropriate. Diff headers, robots.txt, file
   paths, conventional commit subjects, etc.
4. **Genuinely no formal structure** (creative-writing prose, free-form
   press releases) → regex on surface features is honest. Annotate with
   `# REGEX_OK: prose_surface` and move on. **Do not invent fake parsers
   for unparseable text.**

If after honest tool-searching nothing measures the norm, mark `THICK`.
THICK is *informative*, not a failure.

**The checker** (`check_no_regex_on_code.py`) is a soft check: it lists
every `re.` use without a `# REGEX_OK:` annotation, with a hint for that
file's language. Treat it as a prompt to **double-check you searched for a
library first**, not as a hard prohibition.

```
python3 methods/existing_metrics_runner/coded/check_no_regex_on_code.py
```

**Acceptable `# REGEX_OK:` reasons:**

| Reason | When |
|---|---|
| `tool_output` | Parsing a fixed CLI output format (ruff line format, lizard CSV) |
| `file_path` | Classifying file paths — paths aren't a language to parse |
| `format_header` | Diff headers, hunk markers, conventional commit subjects |
| `prose_surface` | Prose with no formal parser — creative writing, press releases |
| `binary_format` | Well-defined non-code text (FASTA, robots.txt, BibTeX) |

## Hard fields without parsers — prose and beyond

For tasks where no parser exists for the artifact (creative_writing,
press_release, news_homepage prose, humor):

- **NLP tools are tier-3 here**: `spacy` (dependency parse, POS, NER),
  `stanza`, `textstat`, `pyphen`, `readability`, LM perplexity/surprisal.
- **Regex on surface features is fine** when measuring something genuinely
  surface (e.g. "average sentence length", "third-person-singular rate").
  Annotate `# REGEX_OK: prose_surface`.
- **Mark THICK liberally** for norms that require taste, narrative
  coherence, voice, irony — that catalog of THICKs IS the deliverable. It
  measures the articulability boundary precisely.

## Aggregation note

The runner builds a (N × M) feature matrix where N = #PRs, M = 2 × #metrics
(score + applied flag). Metrics that abstain on a given PR contribute NaN;
the downstream model handles NaNs explicitly (RF: surrogate splits, LR:
median impute + applied flag interaction).
