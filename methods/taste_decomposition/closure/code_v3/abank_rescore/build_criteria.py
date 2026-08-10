#!/usr/bin/env python3
"""Build the judge-portable code-review A-bank criteria file.

Source of truth for WHICH criteria: the 127 aspect IDs that the published
PR V/A/T ladder actually scored (columns of pr_a_metrics_full.parquet, the
coded backend). Source of truth for the CRITERION TEXT: the mined
code_review aspect catalog runs/validity_full/v2/code_review/aspects.json
(name + description). Aspects added later than the catalog (a4xx/a5xx) get
hand-written norm statements here, and the ones that have no text-judge form
at all (LeetCode execution outcomes, AST-fingerprint entropies) are marked
portable=False and excluded from the judge bank.
"""
import json, sys, os
from pathlib import Path

REPO = Path(os.environ.get("NR_REPO", "/lfs/skampere3/0/alexspan/norm-research"))
NAMES = json.load(open(sys.argv[1]))            # aspect_id -> ASPECT_NAME (coded module)
CATALOG = {x["aspect_id"]: x for x in json.load(open(REPO / "runs/validity_full/v2/code_review/aspects.json"))}

# Hand-written norm statements for the post-catalog aspects that ARE genuine
# code-review norms (a400-a409). Written from each module's ASPECT_NAME +
# docstring, phrased as a norm a reviewer could assert about a diff.
HAND = {
 "a400": "The change avoids gratuitously expensive algorithmic complexity: loops and data-structure operations scale reasonably with input size, and nested scans over the same collection are avoided where a cheaper access pattern is available.",
 "a401": "The change is decomposed into several small, single-purpose functions rather than concentrated in one long function that does many things.",
 "a402": "Where recursion is used it is appropriate and well-formed: a clear base case, bounded depth, and no recursion where a simple loop would be clearer or safer.",
 "a403": "The added code uses the language's idiomatic constructs (e.g. comprehensions, iteration helpers such as enumerate/zip, context managers, string interpolation, decorators) rather than non-idiomatic or transliterated equivalents.",
 "a404": "The change picks data structures appropriate to how the data is used (e.g. a set or dict for membership/lookup instead of repeated linear scans over a list).",
 "a405": "The added functions are mostly pure: they compute from their arguments and return results rather than mutating shared or global state as a side effect.",
 "a406": "The change avoids unexplained magic numbers and string literals; recurring constants are named or configurable rather than inlined at each use.",
 "a407": "Identifiers introduced by the change are expressive and informative: names describe what the thing is or does, rather than being single letters, abbreviations, or placeholder names.",
 "a408": "Error and exception handling in the change is deliberate: failures are caught at a level that can act on them, are not silently swallowed, and preserve enough context to diagnose the problem.",
 "a409": "The change carries explanatory comments where they are needed: comments state intent and rationale for non-obvious code, rather than restating what the line already says.",
}

# Aspects with NO text-judge form: they are execution outcomes on a different
# corpus (LeetCode candidate solutions) or opaque structural fingerprints
# imported from the code-attribution line. A judge reading a PR diff cannot
# assert these as norms, so they leave the bank (documented, not silently).
NON_PORTABLE_PREFIX_NOTE = {
 "a478": "AST-parse validity of a standalone Python file (not a norm about a PR)",
 "a479": "import-resolution fraction of a standalone Python file",
 "a480": "LeetCode test-pass outcome (execution on another corpus)",
 "a481": "LeetCode runtime percentile (execution on another corpus)",
 "a482": "LeetCode memory percentile (execution on another corpus)",
 "a500": "anonymized 4-gram entropy (authorship fingerprint)",
 "a501": "branching-density fingerprint",
 "a502": "max loop nesting depth fingerprint",
 "a503": "call-chain depth fingerprint",
 "a504": "AST shape proxy entropy (fingerprint)",
 "a517": "Chinese-character comment ratio (authorship fingerprint)",
 "a518": "AST node-type sequence entropy (fingerprint)",
 "a519": "heuristic Big-O bucket as ordinal fingerprint",
 "a520": "competitive-programming algorithm tag count",
 "a521": "I/O signature category (competitive-programming)",
 "a522": "function-definition density per 100 LOC (fingerprint)",
}


def main():
    import pandas as pd
    a = pd.read_parquet(REPO / "datasets/code-review/pr_test_execution/outputs/pr_a_metrics_full.parquet")
    ids = [c[:-6] for c in a.columns if c.endswith("_score")]
    cov = {i: float(a[f"{i}_applied"].mean()) for i in ids}

    rows = []
    for i in ids:
        cat = CATALOG.get(i)
        if cat:
            name, desc, src = cat["name"], cat["description"], "aspects.json"
        elif i in HAND:
            name, desc, src = NAMES.get(i, i), HAND[i], "hand_from_module"
        else:
            name, desc, src = NAMES.get(i, i), "", "none"
        portable = bool(desc) and i not in NON_PORTABLE_PREFIX_NOTE
        rows.append({
            "aspect_id": i, "name": name, "description": desc,
            "coded_coverage": round(cov[i], 4),
            "coded_cov_ge5pct": cov[i] >= 0.05,
            "text_source": src, "portable": portable,
            "drop_reason": NON_PORTABLE_PREFIX_NOTE.get(i, "" if portable else "no catalog text"),
        })
    outp = REPO / "datasets/code-review/dense_standard_v3/abank_rescore/criteria_code_abank.jsonl"
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    npor = sum(r["portable"] for r in rows)
    print(f"total coded aspects: {len(rows)}  portable: {npor}  "
          f"portable&cov>=5%: {sum(r['portable'] and r['coded_cov_ge5pct'] for r in rows)}")
    print(f"-> {outp}")


if __name__ == "__main__":
    main()
