"""Loader for SemEval-2017 Task 6 #HashtagWars data.

Yields (hashtag, tweet_text, label) tuples.
Labels: 0 = not in top-10, 1 = in top-10 (not winner), 2 = winner,
as judged by the @midnight TV show staff. Evaluation split tweets are
unlabeled in evaluation_data/ (label=None); use split="gold" for the
same 6 hashtags with labels.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPLITS = {
    "train": "train_data",
    "trial": "trial_data",
    "eval": "evaluation_data",   # unlabeled
    "gold": "gold_labels",       # labeled version of eval hashtags
}


def load(split="train"):
    """Yield (hashtag, tweet_text, label) for the given split.

    split: one of "train", "trial", "eval", "gold".
    label is an int in {0, 1, 2}, or None for the unlabeled "eval" split.
    """
    d = os.path.join(_HERE, _SPLITS[split])
    for fname in sorted(os.listdir(d)):
        if not fname.endswith(".tsv"):
            continue
        hashtag = fname[:-4]  # e.g. "Cereal_Songs"
        with open(os.path.join(d, fname), encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                text = parts[1]
                label = int(parts[2]) if len(parts) >= 3 and parts[2] in "012" else None
                yield hashtag, text, label


if __name__ == "__main__":
    from collections import Counter
    for s in _SPLITS:
        rows = list(load(s))
        print(s, len(rows), Counter(l for _, _, l in rows))
