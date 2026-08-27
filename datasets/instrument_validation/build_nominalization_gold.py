"""
Build nominalization_gold.tsv for INSTRUMENT B (nominalization: does a
term contain a derived abstract nominal, e.g. -tion/-ity/-ness/-ment/
-ance/-ence).

Positives: every distinct :ORTH entry in NOMLEX-2001.exp (NYU Proteus
Project nominalization lexicon, freely downloadable, 2001 version).

Negatives: sampled from a standard frequency list (Google 10000 English
words, already present in raw/google-10k.txt from a sibling task in this
shared directory), restricted to:
  - words tagged as noun or adjective in WordNet (so the negative pool is
    POS-matched to what a nominalization-detector would be applied to,
    not e.g. function words/verbs),
  - words that do NOT end in one of the canonical derivational-nominal
    suffixes (-tion/-sion/-ity/-ty/-ness/-ment/-ance/-ence/-ancy/-ency/
    -al/-ing used nominally is excluded on purpose -- see README caveat),
  - words not already in the NOMLEX positive set.
Negatives are drawn in frequency-rank order (highest-frequency qualifying
words first) until the negative pool is the same size as the positive
pool, giving a balanced 1:1 gold set. This is a deliberately simple,
reproducible sampling rule -- not a claim that frequency-matched
negatives are unbiased on every dimension (see README).

Run from datasets/instrument_validation/ (requires nltk wordnet corpus,
already present locally).
"""
import re
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

RAW = "raw"
OUT = "nominalization_gold.tsv"

SUFFIXES = (
    "tion", "sion", "ity", "ty", "ness", "ment", "ance", "ence",
    "ancy", "ency",
)


def has_nominalizing_suffix(word):
    w = word.lower()
    return any(w.endswith(suf) for suf in SUFFIXES)


# ---------------------------------------------------------------------
# Positives: NOMLEX-2001.exp :ORTH entries
# ---------------------------------------------------------------------
nomlex_path = f"{RAW}/nomlex/NOMLEX-2001.exp"
with open(nomlex_path, encoding="utf-8", errors="replace") as f:
    text = f.read()
positives = sorted(set(re.findall(r':ORTH\s+"([^"]+)"', text)))
positives = [w for w in positives if w.isalpha()]

# ---------------------------------------------------------------------
# Negatives: Google 10000 English word list, frequency-ranked,
# POS-filtered to noun/adjective via WordNet, excluding suffix matches
# and NOMLEX overlap.
# ---------------------------------------------------------------------
freq_path = f"{RAW}/google-10k.txt"
with open(freq_path, encoding="utf-8") as f:
    freq_words = [w.strip().lower() for w in f if w.strip()]

pos_set = set(positives)
negatives = []
seen = set()
for w in freq_words:
    if not w.isalpha() or len(w) < 3:
        continue
    if w in pos_set or w in seen:
        continue
    if w in STOPWORDS:
        continue
    if has_nominalizing_suffix(w):
        continue
    # In-context POS tag (NLTK averaged-perceptron tagger) in a neutral
    # noun-favoring frame, so function-word senses of e.g. "can"/"no"/
    # "has" (which have a marginal WordNet noun sense but are dominantly
    # non-noun in use) are excluded; only keep words the tagger calls a
    # common noun (NN/NNS) or adjective (JJ) in this frame.
    tag = nltk.pos_tag([w])[0][1]
    if tag not in ("NN", "NNS", "JJ"):
        continue
    negatives.append(w)
    seen.add(w)
    if len(negatives) >= len(positives):
        break

# ---------------------------------------------------------------------
# Write out
# ---------------------------------------------------------------------
with open(OUT, "w", encoding="utf-8") as f:
    f.write("word\tis_nominalization\tsource\n")
    for w in positives:
        f.write(f"{w}\t1\tNOMLEX-2001_NYUProteus\n")
    for w in negatives:
        f.write(f"{w}\t0\tGoogle10k_freqlist_WordNet_POS_filtered\n")

print(f"positives (NOMLEX): {len(positives)}")
print(f"negatives (freq-list, POS-filtered, suffix-excluded): {len(negatives)}")
print(f"TOTAL: {len(positives) + len(negatives)}")
