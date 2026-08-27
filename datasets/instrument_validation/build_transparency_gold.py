"""
Build transparency_gold.tsv for INSTRUMENT A (semantic transparency /
compositionality: is a phrase's meaning composable from its parts vs
idiomatic/opaque).

Reads from raw/ (already-downloaded free sources) and writes
transparency_gold.tsv with columns:
    item    score_or_label    scale    source

Run from datasets/instrument_validation/.
"""
import csv
import json
import os
import re
import statistics
from collections import defaultdict

RAW = "raw"
OUT = "transparency_gold.tsv"

rows = []  # (item, score_or_label, scale, source)

# ---------------------------------------------------------------------
# 1. LADEC (Gagne, Spalding & Schmidtke 2019) — noun-noun compounds,
#    whole-compound semantic transparency/compositionality rating.
# ---------------------------------------------------------------------
ladec_path = os.path.join(RAW, "ladec", "LADECv1-2019.csv")
n_ladec = 0
with open(ladec_path, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        stim = row.get("stim", "").strip()
        rating = row.get("ratingcmp", "").strip()
        if not stim or rating in ("", "NA"):
            continue
        try:
            val = float(rating)
        except ValueError:
            continue
        rows.append((stim, f"{val:.4f}",
                     "0-100 continuous, mean whole-compound compositionality rating (Amazon Mechanical Turk magnitude-style scale; higher = more semantically transparent/compositional)",
                     "LADEC_Gagne2019"))
        n_ladec += 1

# ---------------------------------------------------------------------
# 2. Reddy, McCarthy & Manandhar (2011) — 90 noun-noun compounds,
#    whole-compound compositionality mean (Cpd_mean), 0-5 Likert.
# ---------------------------------------------------------------------
reddy_path = os.path.join(RAW, "reddy_compositionality",
                           "ijcnlp_compositionality_data",
                           "MeanAndDeviations.clean.txt")
n_reddy = 0
with open(reddy_path, encoding="utf-8") as f:
    header = f.readline()
    for line in f:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        head, _, rest = line.partition("\t")
        nums = rest.split()
        if len(nums) < 6:
            continue
        compound_raw = head  # e.g. "end-n user-n"
        words = [w.rsplit("-", 1)[0] for w in compound_raw.split()]
        item = " ".join(words)
        cpd_mean = nums[4].strip()
        try:
            val = float(cpd_mean)
        except ValueError:
            continue
        rows.append((item, f"{val:.4f}",
                     "0-5 Likert mean (5=fully compositional/transparent, 0=opaque idiomatic), AMT annotators, from Reddy/McCarthy/Manandhar (2011) whole-compound judgments",
                     "Reddy2011_IJCNLP"))
        n_reddy += 1

# ---------------------------------------------------------------------
# 3. Venkatapathy & Joshi (2005) verb-noun / verb-adjective collocation
#    compositionality ratings (SVAJ2005, released via McCarthy site),
#    1-6 Likert, mean of 2 annotators.
# ---------------------------------------------------------------------
svaj_path = os.path.join(RAW, "reddy_compositionality",
                          "SVAJ2005compositionality_rating.txt")
n_svaj = 0
with open(svaj_path, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        item, a1, a2 = parts
        try:
            v1, v2 = float(a1), float(a2)
        except ValueError:
            continue
        mean_val = (v1 + v2) / 2.0
        rows.append((item.strip(), f"{mean_val:.4f}",
                     "1-6 Likert mean of 2 annotators (Roderick Saxey native, Pranesh Agarwal non-native; higher = more compositional/literal), Venkatapathy & Joshi (2005) V-N/V-Adj collocations",
                     "VenkatapathyJoshi2005_SVAJ"))
        n_svaj += 1

# ---------------------------------------------------------------------
# 4. McCarthy, Keller & Carroll (2003) — English phrasal verbs
#    ("verb+particle"), 0-10 compositionality scale, mean of 3 native-
#    speaker judges (excludes the released NonNativeSpeaker file).
# ---------------------------------------------------------------------
pv_dir = os.path.join(RAW, "mccarthy_phrasalverbs")
judge_scores = defaultdict(list)
pat = re.compile(r"\d+\s*:\s*(\S+)\s*:\s*\d+\s*:\s*(\d+)")
for fn in ("Judge1", "Judge2", "Judge3"):
    fp = os.path.join(pv_dir, fn)
    if not os.path.exists(fp):
        continue
    with open(fp, encoding="utf-8") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            verb_particle, score = m.group(1), float(m.group(2))
            item = verb_particle.replace("+", " ")
            judge_scores[item].append(score)

n_pv = 0
for item, scores in judge_scores.items():
    mean_val = statistics.mean(scores)
    rows.append((item, f"{mean_val:.4f}",
                 "0-10 mean of 3 native-English-speaker judges (higher = more compositional/literal, lower = idiomatic), McCarthy/Keller/Carroll (2003) phrasal verbs",
                 "McCarthyKellerCarroll2003_PhrasalVerbs"))
    n_pv += 1

# ---------------------------------------------------------------------
# 5. MAGPIE (Haagsma, Bos & Nissim 2020) — idiom TYPES aggregated
#    across all annotated token instances (filtered split, i.e.
#    confidence-filtered subset). Binary idiomatic(1)/literal(0)
#    majority label per idiom type + mean confidence.
# ---------------------------------------------------------------------
magpie_path = os.path.join(RAW, "magpie-corpus", "magpie-corpus-master",
                            "MAGPIE_filtered_split_random.jsonl")
type_labels = defaultdict(list)
type_conf = defaultdict(list)
if os.path.exists(magpie_path):
    with open(magpie_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            lab = d.get("label")
            if lab not in ("i", "l"):
                continue
            idiom = d.get("idiom", "").strip()
            if not idiom:
                continue
            type_labels[idiom].append(lab)
            type_conf[idiom].append(d.get("confidence", 0.0))

n_magpie = 0
for idiom, labels in type_labels.items():
    n_i = labels.count("i")
    n_l = labels.count("l")
    frac_idiomatic = n_i / len(labels)
    majority = 1 if n_i >= n_l else 0
    mean_conf = statistics.mean(type_conf[idiom])
    rows.append((idiom, str(majority),
                 f"binary (1=idiomatic/opaque majority label, 0=literal/transparent majority label; type-level aggregate over {len(labels)} token instance(s), fraction-idiomatic={frac_idiomatic:.3f}, mean annotator confidence={mean_conf:.3f}), Haagsma/Bos/Nissim (2020) MAGPIE",
                 "MAGPIE_Haagsma2020"))
    n_magpie += 1

# ---------------------------------------------------------------------
# Write out
# ---------------------------------------------------------------------
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    w.writerow(["item", "score_or_label", "scale", "source"])
    for item, score, scale, source in rows:
        w.writerow([item, score, scale, source])

print(f"LADEC: {n_ladec}")
print(f"Reddy2011: {n_reddy}")
print(f"Venkatapathy&Joshi SVAJ: {n_svaj}")
print(f"McCarthyKellerCarroll phrasal verbs: {n_pv}")
print(f"MAGPIE idiom types: {n_magpie}")
print(f"TOTAL rows: {len(rows)}")
