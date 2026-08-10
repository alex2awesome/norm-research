"""
Hybrid metric channel for a54: "Organization and reader navigation"
(Math StackExchange answers).

Design rationale (see report for full writeup):
  - The v1_structure baseline (\\section/\\label/"outline"/"roadmap" keyword
    counting) scored baseline_train_rho=0.101 because this corpus is
    StackExchange prose/LaTeX, not sectioned documents -- those keywords are
    essentially absent regardless of quality (a keyword-presence proxy that
    doesn't fire).
  - Reading the 30 train examples: judge reward is NOT monotone in length or
    in generic transition-word density (both extremes -- a very long
    explicitly-partitioned answer, AND a very short answer that states its
    point immediately -- score highest; a medium/long answer that reads as
    one undifferentiated prose wall, even with lots of step language like
    "Show that... Next... Conclude that...", scores low-mid). The real
    driver looks like genuine addressable structure (named/labeled parts a
    reader could jump to) OR immediate front-loading of the claim -- neither
    of which is reliably regexable without risking the same keyword-count
    trap the baseline fell into (e.g. "then"/"so"/"therefore" appear in
    almost every proof regardless of organization quality).
  - We therefore keep code responsible for FORM signals that ARE reliably
    parseable (numbered/referenceable equations, formal theorem/proof/qed
    structure, run-on sentences, broken LaTeX delimiters -- all hurt or help
    skimmability mechanically) and use two LLM fields for the two semantic,
    non-regexable judgments identified above: explicit addressable
    partitioning, and upfront statement of the claim.
"""

import re

LLM_FIELDS = {
    "explicit_partition": (
        "Does the ANSWER explicitly partition into separately labeled, "
        "jump-to parts (named headers, 'Problem 1/2', 'Case A/B', numbered "
        "steps) rather than one flowing narrative? Reply yes, partial, or no."
    ),
    "answer_upfront": (
        "Does the ANSWER state its main claim or result in its very first "
        "sentence, before any justification? Reply yes, partial, or no."
    ),
}


def _yn(raw):
    """Map a short yes/partial/no-ish LLM answer to {0.0, 0.5, 1.0}."""
    s = str(raw or "").strip().lower()
    if not s or s in ("none", "n/a", "na", "unclear", "unknown"):
        return 0.0
    if s.startswith("y") or "yes" in s:
        return 1.0
    if "partial" in s or "somewhat" in s or "some " in s or s.startswith("some"):
        return 0.5
    if s.startswith("n") or "no" in s:
        return 0.0
    return 0.0


def _safe_sum_counts(d):
    total = 0
    try:
        for v in (d or {}).values():
            try:
                total += int(v)
            except Exception:
                try:
                    total += len(v)
                except Exception:
                    pass
    except Exception:
        pass
    return total


def score(text: str, extracted: dict, ops) -> float:
    try:
        if not text or not text.strip():
            return 0.0

        try:
            t = ops.normalize(text)
        except Exception:
            t = text
        if not t or not t.strip():
            return 0.0

        # Isolate the answer portion (this corpus's convention is
        # "Question: ...\n\nAnswer: ..."); fall back to whole doc if absent.
        idx = t.find("Answer:")
        ans = t[idx + len("Answer:"):].strip() if idx != -1 else t.strip()
        if len(ans) < 3:
            return 0.3

        # --- code-side FORM signals -------------------------------------
        n_sent, mean_wps, frac_long_words = 1, 0.0, 0.0
        try:
            n_sent, mean_wps, frac_long_words = ops.sent_stats(ans)
            n_sent = max(1, int(n_sent))
            mean_wps = float(mean_wps or 0.0)
        except Exception:
            pass

        n_display = n_inline = n_numbered = 0
        avg_tokens = 0.0
        try:
            n_display, n_inline, n_numbered, avg_tokens = ops.equation_stats(ans)
        except Exception:
            pass

        formal_markers = 0
        try:
            skel = ops.proof_skeleton(ans) or {}
            for cat in ("theorem", "definition", "proof", "qed"):
                formal_markers += len(skel.get(cat, []) or [])
        except Exception:
            pass

        delim_issues = 0
        try:
            delim_issues = _safe_sum_counts(ops.delimiter_health(ans))
        except Exception:
            pass

        # --- LLM-grounded semantic signals -------------------------------
        extracted = extracted or {}
        partition_score = _yn(extracted.get("explicit_partition", ""))
        upfront_score = _yn(extracted.get("answer_upfront", ""))

        # --- combine ------------------------------------------------------
        raw = 0.30
        raw += 0.28 * upfront_score
        raw += 0.25 * partition_score
        raw += 0.05 if n_numbered > 0 else 0.0
        raw += 0.03 * min(2, formal_markers)
        raw += 0.05 if n_sent >= 2 else 0.0

        if mean_wps > 45:
            raw -= 0.15
        elif mean_wps > 32:
            raw -= 0.06

        raw -= min(0.15, 0.03 * delim_issues)

        return max(0.0, min(1.0, raw))
    except Exception:
        return 0.5
