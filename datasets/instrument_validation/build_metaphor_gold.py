#!/usr/bin/env python3
"""Build a normalized gold file for binary metaphoricity validation.

Sources (all freely, directly downloaded — see README.md for URLs/licenses):
  - MOH        : original Mohammad/Shutova/Turney (2016) verb metaphor-emotion
                 release, term-sense-sentence-class-confidence format.
  - MOH-X      : cleaned MOH-X subset (Mohammad et al. 2016 / Shutova) as
                 redistributed by Gao et al. (2018) metaphor-in-context repo.
  - TroFi      : Birke & Sarkar TroFi human-labeled subset, as redistributed
                 by Gao et al. (2018).
  - TroFi-X    : TroFi subset with parsable subject-verb-object triples,
                 as redistributed by Gao et al. (2018).
  - VUA        : verb-classification subset derived from the VU Amsterdam
                 Metaphor Corpus (Steen et al. 2010) for the NAACL 2018
                 Metaphor Shared Task, as redistributed by Gao et al. (2018).
                 NOTE: this is a derived verb-level slice, not the full
                 VUAMC (which requires registration - see README).
  - Yulia      : test set from Tsvetkov et al., as redistributed by
                 Gao et al. (2018).
  - MAGPIE     : Multi-Genre AnnotatIon of Potentially Idiomatic Expressions
                 (Haagsma et al. 2020), BNC-based. This is PHRASE-level
                 idiomaticity (idiomatic vs literal use of a multi-word
                 idiom span), not single-word verb metaphoricity like the
                 sources above -- included because it is a phrase-level
                 figurative-vs-literal resource, and downstream users may
                 wish to filter it out if they need pure single-word
                 metaphor labels. Only the binary-label filtered subset
                 (label in {i=idiomatic, l=literal}; confidence>=0.75) is
                 used, deduplicated against the random/typebased split
                 files (same 48,395 instances, just split differently).

Output: metaphor_gold.tsv with columns item, label, context, source
  label: 1 = metaphorical, 0 = literal
  item: target word/phrase (verb lemma as annotated)
  context: full sentence (empty if the dataset is type-level only; none of
           our sources are type-level, all are usage-in-context)
  source: short source tag
"""
import csv
import json
import re
from pathlib import Path

BASE = Path(__file__).parent
RAW = BASE / "raw"
OUT = BASE / "metaphor_gold.tsv"

rows = []  # (item, label, context, source)
counts = {}


def add(item, label, context, source):
    item = (item or "").strip()
    context = (context or "").strip()
    if item == "" or label not in ("0", "1"):
        return
    rows.append((item, label, context, source))
    counts[source] = counts.get(source, 0) + 1


# --- MOH (original Mohammad et al. release) ---
moh_path = RAW / "moh_emotion" / "Metaphor-Emotion-Data-Files" / "Data-metaphoric-or-literal.txt"
with open(moh_path, encoding="utf-8") as fh:
    r = csv.DictReader(fh, delimiter="\t")
    for row in r:
        if not row.get("class"):
            continue  # skip trailing summary lines in the source file
        cls = row["class"].strip().lower()
        label = "1" if cls == "metaphorical" else ("0" if cls == "literal" else None)
        if label is None:
            continue
        sentence = re.sub(r"</?b>", "", row["sentence"])
        add(row["term"], label, sentence, "MOH")

# --- MOH-X (cleaned, via Gao et al. 2018 redistribution) ---
mohx_path = RAW / "gao_data" / "data" / "MOH-X" / "MOH-X_formatted_svo_cleaned.csv"
with open(mohx_path, encoding="utf-8") as fh:
    r = csv.DictReader(fh)
    for row in r:
        add(row["verb"], row["label"].strip(), row["sentence"], "MOH-X")

# --- TroFi (via Gao et al. 2018 redistribution) ---
trofi_path = RAW / "gao_data" / "data" / "TroFi" / "TroFi_formatted_all3737.csv"
with open(trofi_path, encoding="utf-8") as fh:
    r = csv.DictReader(fh)
    for row in r:
        add(row["verb"], row["label"].strip(), row["sentence"], "TroFi")

# --- TroFi-X (via Gao et al. 2018 redistribution) ---
trofix_path = RAW / "gao_data" / "data" / "TroFi-X" / "TroFi-X_formatted_svo.csv"
with open(trofix_path, encoding="utf-8") as fh:
    r = csv.DictReader(fh)
    for row in r:
        add(row["verb"], row["label"].strip(), row["sentence"], "TroFi-X")

# --- VUA verb-classification subset (via Gao et al. 2018 redistribution) ---
vua_path = RAW / "gao_data" / "data" / "VUA" / "VUA_formatted.csv"
with open(vua_path, encoding="latin-1") as fh:
    r = csv.DictReader(fh)
    for row in r:
        add(row["verb"], row["label"].strip(), row["sentence"], "VUA")

# --- Yulia (Tsvetkov et al. test set, via Gao et al. 2018 redistribution) ---
yulia_path = RAW / "gao_data" / "data" / "Yulia" / "Yulia_formatted_svo.csv"
with open(yulia_path, encoding="utf-8") as fh:
    r = csv.DictReader(fh)
    for row in r:
        add(row["verb"], row["label"].strip(), row["sentence"], "Yulia")

# --- MAGPIE (Haagsma et al. 2020), phrase-level idiomaticity, CC-BY 4.0 ---
magpie_path = (
    RAW / "magpie-corpus" / "magpie-corpus-master" / "MAGPIE_filtered_split_random.jsonl"
)
with open(magpie_path, encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        lbl = rec.get("label")
        if lbl == "i":
            label = "1"
        elif lbl == "l":
            label = "0"
        else:
            continue  # skip 'f' (figurative-non-idiom), 'o' (other), '?' (undecided)
        context_sentences = rec.get("context", [])
        idx = len(context_sentences) // 2  # target sentence is the middle one
        context = context_sentences[idx] if context_sentences else ""
        add(rec["idiom"], label, context, "MAGPIE")

# --- dedup exact duplicate rows (same item+context+source) ---
seen = set()
deduped = []
for item, label, context, source in rows:
    key = (item, context, source)
    if key in seen:
        continue
    seen.add(key)
    deduped.append((item, label, context, source))

dupes_removed = len(rows) - len(deduped)

with open(OUT, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(["item", "label", "context", "source"])
    for item, label, context, source in deduped:
        w.writerow([item, label, context, source])

print(f"wrote {len(deduped)} rows to {OUT} ({dupes_removed} exact dupes dropped)")
print("per-source counts (pre-dedup):")
for k, v in sorted(counts.items()):
    print(f"  {k}: {v}")

# post-dedup per-source counts
post_counts = {}
label_counts = {}
for item, label, context, source in deduped:
    post_counts[source] = post_counts.get(source, 0) + 1
    label_counts.setdefault(source, {"0": 0, "1": 0})[label] += 1
print("per-source counts (post-dedup) with label breakdown:")
for k in sorted(post_counts):
    print(f"  {k}: total={post_counts[k]} literal(0)={label_counts[k]['0']} metaphor(1)={label_counts[k]['1']}")
