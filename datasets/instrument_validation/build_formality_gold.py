#!/usr/bin/env python3
"""Build a normalized gold file for lexical (word/phrase-level) formality
validation.

Sources (all freely, directly downloaded -- see README.md for URLs/licenses):
  - pavlick_nenkova_2015 : Pavlick & Nenkova (2015 NAACL) "Inducing Lexical
      Style Properties for Paraphrase and Genre Differentiation" human
      MTurk-rated word/phrase formality scores
      (formality/human/phrase-scores in the style-scores.tar.gz release).
      This is the primary, largest, continuous-score source and the one
      named as the top candidate for this task.
  - brooke_seeds : Brooke, Wang & Hirst (2010) COLING "Automatic Acquisition
      of Lexical Formality" -- the manually curated formal/informal SEED
      WORD LISTS the authors bootstrapped their algorithm from
      (formal_seeds_100.txt / informal_seeds_100.txt). These are
      CATEGORICAL (author-assigned class membership), not a continuous
      human rating scale -- Brooke et al. never released the full
      continuously-scored output lexicon (~thousands of words) for direct
      download, only these seed lists and the CTRW synonym pairs. We map
      them onto the SAME 0-100 scale used by Pavlick & Nenkova using the
      scale's own endpoints (0 = most casual, 100 = most formal) so the
      two sources are at least scale-comparable, but this is a
      binary/categorical construction, NOT an independently-elicited
      numeric rating -- flagged clearly via the source tag and in README.

Output: formality_gold.tsv with columns term, formality_score, source
  formality_score: 0-100 scale, 100 = most formal, 0 = most casual
    (Pavlick & Nenkova's native scale; Brooke seeds mapped onto the same
    scale's endpoints as described above -- see README for caveats)
  term: the word or phrase
  source: pavlick_nenkova_2015_phrase | brooke_seeds_2010
"""
import csv
from pathlib import Path

RAW = Path(__file__).parent / "raw"
OUT = Path(__file__).parent / "formality_gold.tsv"

PN_PHRASE_SCORES = RAW / "style-scores" / "naacl-2015-style-scores" / "formality" / "human" / "phrase-scores"
BROOKE_FORMAL_SEEDS = RAW / "brooke_formality" / "formal_seeds_100.txt"
BROOKE_INFORMAL_SEEDS = RAW / "brooke_formality" / "informal_seeds_100.txt"


def load_pavlick_nenkova():
    rows = []
    with open(PN_PHRASE_SCORES, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            score, phrase = parts[0], parts[1]
            rows.append((phrase, float(score), "pavlick_nenkova_2015_phrase"))
    return rows


def load_brooke_seeds():
    rows = []
    with open(BROOKE_FORMAL_SEEDS, encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                rows.append((w, 100.0, "brooke_seeds_2010"))
    with open(BROOKE_INFORMAL_SEEDS, encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                rows.append((w, 0.0, "brooke_seeds_2010"))
    return rows


def main():
    rows = load_pavlick_nenkova() + load_brooke_seeds()

    # de-dup exact term+source duplicates
    seen = set()
    deduped = []
    for term, score, source in rows:
        key = (term, source)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((term, score, source))

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["term", "formality_score", "source"])
        for term, score, source in deduped:
            w.writerow([term, f"{score:.6f}", source])

    print(f"wrote {len(deduped)} rows to {OUT}")
    by_source = {}
    for _, _, source in deduped:
        by_source[source] = by_source.get(source, 0) + 1
    for source, n in sorted(by_source.items()):
        print(f"  {source}: {n}")


if __name__ == "__main__":
    main()
