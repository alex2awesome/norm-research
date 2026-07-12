"""Build a confound-controlled `is_scary` corpus: 1000 short stories, 50% scary / 50% not,
with SETTING and #CHARACTERS held in equal proportion across the two classes.

Confound control is by construction via MATCHED PAIRS. For each (setting, n_characters) cell
we emit equal numbers of scary and non-scary stories that share the same setting, the same
character count, AND the same character names and neutral skeleton — the two members of a
pair differ ONLY in whether their planted markers are scary (from `cues.MARKERS`) or calm
(from `cues.CALM_MARKERS`). So setting and #characters are exactly balanced across the label
and cannot be a shortcut; length is matched because scary/calm fragments are length-similar.

Deterministic (seeded ``random.Random``); re-running reproduces the corpus byte-for-byte.

    python -m methods.metric_implementer.synthetic_examples.test_metric_scary.build_dataset
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from . import cues

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

SETTINGS: List[str] = [
    "the old house on Pine Street",
    "the forest trail past the ridge",
    "the late-night subway platform",
    "the cabin by the lake",
    "the rooftop of the apartment building",
]
CHAR_COUNTS: List[int] = [1, 2, 3, 4]
NAMES: List[str] = [
    "Mara", "Jonah", "Priya", "Eli", "Wen", "Sofia", "Diego", "Aisha",
    "Tomas", "Greta", "Noor", "Kai", "Lena", "Owen", "Ravi", "Bex",
]
CONNECTIVES: List[str] = [
    "For a moment, no one spoke.",
    "Time seemed to stretch out.",
    "They paused at the threshold.",
    "Then they stepped inside.",
    "Nothing moved for a while.",
]


def _names_clause(names: List[str]) -> str:
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def _sentence(fragment: str) -> str:
    return fragment[0].upper() + fragment[1:] + "."


def _scary_fragments(rng: random.Random) -> tuple:
    """Pick the planted scary categories (DREAD + a random 1-3 of the others) and return 4
    DISTINCT scary marker fragments, every chosen category represented at least once."""
    others = [c for c in cues.CATEGORIES if c != "DREAD"]
    k = rng.randint(1, 3)
    chosen = ["DREAD"] + rng.sample(others, k)
    rng.shuffle(chosen)
    frags: List[str] = []
    for cat in chosen:                                   # one distinct marker per category
        frags.append(rng.choice(cues.MARKERS[cat]))
    while len(frags) < 4:                                # fill remaining slots, no repeats
        cat = rng.choice(chosen)
        pick = rng.choice([m for m in cues.MARKERS[cat] if m not in frags] or cues.MARKERS[cat])
        if pick not in frags:
            frags.append(pick)
    rng.shuffle(frags)
    return frags, sorted(set(chosen))


def _calm_fragments(rng: random.Random) -> List[str]:
    return rng.sample(cues.CALM_MARKERS, 4)


def _assemble(names: List[str], setting: str, connective: str, frags: List[str]) -> str:
    intro = f"{_names_clause(names)} arrived at {setting} just as the light was fading."
    body = [_sentence(f) for f in frags]
    return " ".join([intro, body[0], body[1], connective, body[2], body[3]])


def build(n_examples: int = 1000, seed: int = 7) -> List[Dict]:
    """Return ``n_examples`` records (half scary, half not), perfectly balanced on
    (setting, n_characters). ``n_examples`` must be divisible by 2*len(SETTINGS)*len(CHAR_COUNTS)."""
    cells = [(s, c) for s in SETTINGS for c in CHAR_COUNTS]
    n_pairs = n_examples // 2
    if n_pairs % len(cells) != 0:
        raise ValueError(
            f"n_examples={n_examples} -> {n_pairs} pairs not divisible by {len(cells)} "
            f"(setting x char-count) cells; pick a multiple of {2 * len(cells)}.")
    pairs_per_cell = n_pairs // len(cells)
    rng = random.Random(seed)

    records: List[Dict] = []
    pair_id = 0
    for setting, n_char in cells:
        for _ in range(pairs_per_cell):
            names = rng.sample(NAMES, n_char)
            connective = rng.choice(CONNECTIVES)
            scary_frags, categories = _scary_fragments(rng)
            calm_frags = _calm_fragments(rng)
            scary_text = _assemble(names, setting, connective, scary_frags)
            calm_text = _assemble(names, setting, connective, calm_frags)
            # self-check: the planted label must match the construction (guards drift)
            assert cues.is_scary_label(scary_text) == 1, scary_text
            assert cues.is_scary_label(calm_text) == 0, calm_text
            for label, text, cats in ((1, scary_text, categories), (0, calm_text, [])):
                records.append({
                    "id": f"scary_{pair_id:04d}_{label}",
                    "text": text,
                    "label": label,                    # ground-truth is_scary
                    "setting": setting,
                    "n_characters": n_char,
                    "pair_id": pair_id,
                    "categories": cats,                # planted scary categories ([] if calm)
                    "n_words": len(text.split()),
                })
            pair_id += 1
    rng.shuffle(records)
    return records


def write(records: List[Dict], data_dir: Path = DATA_DIR) -> Dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    pool_path = data_dir / "scary_pool.jsonl"
    labels_path = data_dir / "scary_labels.csv"
    with pool_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    cols = ["id", "label", "setting", "n_characters", "pair_id", "n_words", "categories"]
    with labels_path.open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in records:
            row = [str(r["id"]), str(r["label"]), f"\"{r['setting']}\"",
                   str(r["n_characters"]), str(r["pair_id"]), str(r["n_words"]),
                   f"\"{'|'.join(r['categories'])}\""]
            f.write(",".join(row) + "\n")
    return {"pool": pool_path, "labels": labels_path}


def _balance_report(records: List[Dict]) -> str:
    import collections
    n = len(records)
    n_scary = sum(r["label"] for r in records)
    by_setting = collections.Counter((r["setting"], r["label"]) for r in records)
    by_nchar = collections.Counter((r["n_characters"], r["label"]) for r in records)
    wc1 = [r["n_words"] for r in records if r["label"] == 1]
    wc0 = [r["n_words"] for r in records if r["label"] == 0]
    lines = [f"n={n}  scary={n_scary}  non-scary={n - n_scary}",
             f"mean words: scary={sum(wc1)/len(wc1):.1f}  non-scary={sum(wc0)/len(wc0):.1f}"]
    lines.append("setting x label (must be equal across labels):")
    for s in SETTINGS:
        lines.append(f"  {s:<40} scary={by_setting[(s,1)]:>4} non={by_setting[(s,0)]:>4}")
    lines.append("n_characters x label:")
    for c in CHAR_COUNTS:
        lines.append(f"  {c} chars   scary={by_nchar[(c,1)]:>4} non={by_nchar[(c,0)]:>4}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    records = build(args.n, args.seed)
    paths = write(records)
    print(_balance_report(records))
    print(f"\nwrote {paths['pool']}\nwrote {paths['labels']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
