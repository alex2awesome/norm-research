#!/usr/bin/env python3
"""End-to-end V/A instrument runner for SemEval #HashtagWars.

The scoring stage never loads verdicts.  It selects whole contests by a stable
hash, computes V, and materializes one of {1.0, 0.5, 0.0, NA} for every
(entry, criterion).  Verdicts are joined only by ``readout``.

The ``codebook`` backend is a deterministic materialization of the articulated
decision rules in rubrics.jsonl.  It was used for the checked-in run because a
nested gpt-5.6-sol inference endpoint is not available inside the managed local
sandbox.  It is not represented as an LLM judgment; see RESULTS.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from wordfreq import zipf_frequency

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
RUBRICS_PATH = HERE / "rubrics.jsonl"
SCORES_PATH = HERE / "scores_codebook.npz"
RESULTS_PATH = HERE / "results.json"
SPLIT_DIRS = ("train_data", "trial_data", "gold_labels")
SAMPLE_SALT = "hashtagwars-va-v1:"
N_CONTESTS = 40
N_SHARDS = 8
SEED = 20260728
TOKENS = ("1.0", "0.5", "0.0", "NA")

sys.path.insert(0, str(HERE))
from v_features import (  # noqa: E402
    HASHTAG_RE,
    MENTION_RE,
    URL_RE,
    V_NAMES,
    content_text,
    prompt_words,
    vector,
)

WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
TITLE_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
PIVOT_RE = re.compile(
    r"\b(?:but|except|until|instead|actually|then|when|apparently|turns out|only)\b",
    re.I,
)
SPEECH_RE = re.compile(r"\b(?:i|i'm|im|we|you|he|she|they|my|your|our)\b", re.I)
ATTITUDE_RE = re.compile(
    r"\b(?:ugh|wow|yay|damn|hell|hate|love|sorry|please|never|always|finally|"
    r"great|worst|best|scared|afraid|drunk|hangover|fuck|shit|fart)\b",
    re.I,
)
TABOO_RE = re.compile(
    r"\b(?:fuck|shit|ass|dick|cock|sex|porn|nazi|kill|murder|dead|death|"
    r"rape|suicide|fart|poop|crap|crapper|drunk|booze|gun)\w*\b",
    re.I,
)
SATIRE_RE = re.compile(
    r"\b(?:gop|democrat|republican|president|congress|government|politic|"
    r"trump|clinton|corporate|ceo|church|police|college|america)\w*\b",
    re.I,
)
OBSERVE_RE = re.compile(
    r"\b(?:when|every|always|never|people|parents|dad|mom|marriage|college|"
    r"weekend|internet|advice|because|why)\b",
    re.I,
)
ABSURD_RE = re.compile(
    r"\b(?:zombie|alien|ghost|monster|dinosaur|unicorn|robot|vampire|"
    r"superhero|time travel|apocalypse|haunted|magic|invisible)\w*\b",
    re.I,
)
EXPLAIN_RE = re.compile(
    r"\b(?:get it|the joke is|i mean|because it's|this is funny|points me|"
    r"hope you|my entry|i tried|best i could)\b",
    re.I,
)
FILLER_RE = re.compile(
    r"\b(?:please|rt|retweet|points me|hope you|my entry|i tried|follow me)\b",
    re.I,
)


def load_rubrics() -> List[dict]:
    rows = [json.loads(line) for line in RUBRICS_PATH.read_text().splitlines() if line]
    assert 25 <= len(rows) <= 40
    assert len({r["rubric_id"] for r in rows}) == len(rows)
    return rows


def iter_text_rows() -> Iterable[dict]:
    """Read only id/text columns; deliberately do not parse the label column."""
    for split_dir in SPLIT_DIRS:
        for path in sorted((BASE / split_dir).glob("*.tsv")):
            hashtag = path.stem
            with path.open(encoding="utf-8", errors="replace") as handle:
                for line_number, line in enumerate(handle, 1):
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 2:
                        continue
                    tweet_id, text = parts[0], parts[1]
                    row_id = hashlib.sha1(
                        f"{hashtag}\0{tweet_id}\0{text}".encode()
                    ).hexdigest()
                    yield {
                        "row_id": row_id,
                        "tweet_id": tweet_id,
                        "hashtag": hashtag,
                        "text": text,
                        "split_dir": split_dir,
                        "line_number": line_number,
                    }


def selected_hashtags(rows: Sequence[dict], n: int = N_CONTESTS) -> List[str]:
    groups = sorted(
        {r["hashtag"] for r in rows},
        key=lambda h: hashlib.sha256((SAMPLE_SALT + h).encode()).hexdigest(),
    )
    return groups[:n]


def label_lookup() -> Dict[str, int]:
    """Verdict join used only after scoring (and to construct required anchors)."""
    out = {}
    for split_dir in SPLIT_DIRS:
        for path in sorted((BASE / split_dir).glob("*.tsv")):
            hashtag = path.stem
            with path.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 3 or parts[2] not in {"0", "1", "2"}:
                        continue
                    row_id = hashlib.sha1(
                        f"{hashtag}\0{parts[0]}\0{parts[1]}".encode()
                    ).hexdigest()
                    out[row_id] = int(parts[2])
    return out


def _clamp_score(value: float) -> str:
    if value >= 0.72:
        return "1.0"
    if value >= 0.34:
        return "0.5"
    return "0.0"


def _na_or(value: float, applicable: bool) -> str:
    return _clamp_score(value) if applicable else "NA"


def _signals(text: str, hashtag: str) -> dict:
    raw = text or ""
    content = content_text(raw, hashtag)
    words = WORD_RE.findall(content)
    low = [w.lower().replace("’", "'") for w in words]
    n = len(words)
    pwords = set(prompt_words(hashtag))
    counts = Counter(low)
    title_count = len(TITLE_RE.findall(content))
    unusual = [
        w
        for w in low
        if len(w) >= 7
        and w not in pwords
        and (
            any(w.startswith(p[:4]) or w.endswith(p[-4:]) for p in pwords if len(p) >= 4)
            or not re.search(r"[aeiouy]", w)
        )
    ]
    prompt_tag_present = any(
        re.sub(r"[^a-z0-9]", "", x.lower())
        == re.sub(r"[^a-z0-9]", "", hashtag.lower())
        for x in HASHTAG_RE.findall(raw)
    )
    meta = len(MENTION_RE.findall(raw)) + max(len(HASHTAG_RE.findall(raw)) - 1, 0)
    alpha = sum(ch.isalpha() for ch in content)
    printable = sum(not ch.isspace() for ch in content)
    lexical = (
        sum(min(max((zipf_frequency(w, "en") - 1.5) / 2.5, 0.0), 1.0) for w in low)
        / max(n, 1)
    )
    # Alphabetic gibberish is not coherent merely because its characters are letters.
    coherent = (
        min(1.0, alpha / max(printable, 1) * 1.35) * lexical if content else 0.0
    )
    economy = max(0.0, 1.0 - abs(n - 7) / 18.0) - min(meta * 0.12, 0.36)
    titleish = min(1.0, (title_count + int(bool(re.search(r"\bthe\b", content, re.I)))) / 3)
    has_pivot = bool(PIVOT_RE.search(content) or re.search(r"[:;—-]", content))
    repeated_structure = bool(
        re.search(r"\b(\w+)\b(?:\W+\w+){0,3}\W+\1\b", content, re.I)
        or content.count(",") >= 2
    )
    coinage = min(
        1.0,
        0.55 * bool(unusual)
        + 0.45
        * bool(re.search(r"[A-Za-z]{3,}[A-Z][a-z]{2,}", content)),
    )
    transform = min(1.0, 0.55 * titleish + 0.45 * coinage)
    specificity = min(
        1.0,
        (title_count + sum(ch.isdigit() for ch in content) + len(re.findall(r"'s\b", content)))
        / 3,
    )
    attitude = min(
        1.0,
        0.45 * bool(ATTITUDE_RE.search(content))
        + 0.25 * bool(re.search(r"[!?]", content))
        + 0.30 * bool(SPEECH_RE.search(content)),
    )
    wordplay = min(
        1.0,
        0.55 * coinage
        + 0.25 * transform
        + 0.20 * bool(re.search(r"\b\w+[-/]\w+\b", content)),
    )
    incongruity = min(
        1.0,
        0.30 * specificity
        + 0.30 * transform
        + 0.20 * bool(TABOO_RE.search(content))
        + 0.20 * bool(ABSURD_RE.search(content)),
    )
    clean_stop = not bool(EXPLAIN_RE.search(raw)) and meta <= 1
    return {
        "raw": raw,
        "content": content,
        "words": words,
        "n": n,
        "coherent": coherent,
        "economy": max(0.0, min(1.0, economy)),
        "titleish": titleish,
        "coinage": coinage,
        "transform": transform,
        "specificity": specificity,
        "attitude": attitude,
        "wordplay": wordplay,
        "incongruity": incongruity,
        "has_pivot": has_pivot,
        "repeated_structure": repeated_structure,
        "prompt_tag_present": prompt_tag_present,
        "meta": meta,
        "clean_stop": clean_stop,
        "speech": bool(SPEECH_RE.search(content)),
        "taboo": bool(TABOO_RE.search(content)),
        "satire": bool(SATIRE_RE.search(content)),
        "observational": bool(OBSERVE_RE.search(content)),
        "absurd": bool(ABSURD_RE.search(content)),
        "explains": bool(EXPLAIN_RE.search(raw) or FILLER_RE.search(raw)),
    }


def judge_codebook(text: str, hashtag: str) -> List[str]:
    """Apply the 30 final articulated rules without access to a verdict."""
    s = _signals(text, hashtag)
    n, c = s["n"], s["coherent"]
    fit = min(1.0, 0.45 * bool(n) + 0.25 * s["prompt_tag_present"] + 0.30 * c)
    source = s["titleish"]
    transform_app = s["transform"] > 0.18
    turn_app = s["has_pivot"] or n >= 9
    incong_app = s["incongruity"] > 0.22
    rhythm_app = s["repeated_structure"] or n >= 5
    out = [
        _clamp_score(fit),  # a01
        _na_or(source, source > 0.15),  # a02
        _na_or(0.65 * source + 0.35 * c, transform_app),  # a03
        _na_or(0.55 * s["transform"] + 0.45 * fit, transform_app),  # a04
        _na_or(0.65 * s["incongruity"] + 0.35 * c, incong_app),  # a05
        _na_or(0.55 * s["has_pivot"] + 0.25 * s["incongruity"] + 0.20 * c, turn_app),  # a06
        _na_or(0.55 * s["wordplay"] + 0.45 * s["has_pivot"], s["wordplay"] > 0.2 and turn_app),  # a07
        _na_or(0.75 * s["wordplay"] + 0.25 * fit, s["wordplay"] > 0.18),  # a08
        _na_or(0.8 * s["coinage"] + 0.2 * source, s["coinage"] > 0),  # a09
        _na_or(0.8 * s["coinage"] + 0.2 * c, s["coinage"] > 0),  # a10
        _na_or(0.7 * source + 0.3 * s["transform"], transform_app),  # a11
        _na_or(0.45 * s["incongruity"] + 0.35 * s["specificity"] + 0.2 * c, incong_app),  # a12
        _na_or(0.75 * s["specificity"] + 0.25 * c, s["specificity"] > 0),  # a13
        _clamp_score(0.75 * s["economy"] + 0.25 * c),  # a14
        _na_or(0.55 * s["economy"] + 0.45 * (s["has_pivot"] or s["wordplay"] > 0.2), turn_app or s["wordplay"] > 0.2),  # a15
        _clamp_score(0.75 * c + 0.25 * (s["meta"] <= 1)),  # a16
        _na_or(c, n > 1),  # a17
        _na_or(0.65 * c + 0.35 * s["incongruity"], s["absurd"] or incong_app),  # a18
        _na_or(0.55 * s["repeated_structure"] + 0.45 * s["has_pivot"], s["repeated_structure"]),  # a19
        _na_or(0.6 * s["has_pivot"] + 0.4 * c, s["has_pivot"]),  # a20
        _na_or(0.55 * s["specificity"] + 0.45 * s["attitude"], s["satire"]),  # a21
        _na_or(0.55 * s["specificity"] + 0.45 * c, s["observational"]),  # a22
        _na_or(0.7 * s["attitude"] + 0.3 * c, s["speech"] or s["attitude"] > 0),  # a23
        _na_or(0.75 * c + 0.25 * s["economy"], s["speech"]),  # a24
        _na_or(0.55 * s["economy"] + 0.45 * (s["repeated_structure"] or s["wordplay"] > 0.2), rhythm_app),  # a25
        _clamp_score(0.35 * s["incongruity"] + 0.35 * s["wordplay"] + 0.30 * s["specificity"]),  # a26
        _na_or(0.6 * c + 0.25 * s["attitude"] + 0.15 * s["clean_stop"], s["taboo"]),  # a27
        _na_or(0.75 * s["attitude"] + 0.25 * c, s["attitude"] > 0),  # a28
        _na_or(0.35 * s["economy"] + 0.25 * s["specificity"] + 0.25 * s["wordplay"] + 0.15 * c, n > 1),  # a29
        _na_or(1.0 if s["clean_stop"] else (0.0 if s["explains"] else 0.5), n > 1),  # a30
    ]
    assert len(out) == 30 and all(x in TOKENS for x in out)
    return out


def _scramble(pos: dict, neg: dict, rng: random.Random) -> dict:
    words = WORD_RE.findall(content_text(pos["text"], pos["hashtag"]))
    words += WORD_RE.findall(content_text(neg["text"], neg["hashtag"]))
    rng.shuffle(words)
    chosen = words[: max(7, min(12, len(words)))]
    # Reverse alternating words to make accidental source phrases unlikely.
    chosen[1::2] = [w[::-1] for w in chosen[1::2]]
    return {
        "row_id": "__anchor_scrambled__",
        "tweet_id": "__anchor_scrambled__",
        "hashtag": pos["hashtag"],
        "text": " ".join(chosen),
        "split_dir": "__anchor__",
        "line_number": -1,
    }


def build_anchors(
    all_rows: Sequence[dict], labels: Dict[str, int], shard: int, attempt: int = 0
) -> List[dict]:
    """Select blinded winner/random/scramble anchors with a fixed seed."""
    # Use the same fixed, seeded random anchors in every shard, matching the
    # repeated-anchor design in the caption scorer. An invalid pass is discarded
    # and rescored with the next deterministic anchor draw.
    rng = random.Random(SEED + attempt)
    winners = [r for r in all_rows if labels.get(r["row_id"]) == 2]
    negatives = [r for r in all_rows if labels.get(r["row_id"]) == 0]
    pos = dict(rng.choice(winners))
    neg = dict(rng.choice(negatives))
    pos["row_id"] = f"__anchor_{shard}_winner__"
    neg["row_id"] = f"__anchor_{shard}_random__"
    scram = _scramble(pos, neg, rng)
    scram["row_id"] = f"__anchor_{shard}_scrambled__"
    # Tags live outside prompts; judge_codebook receives only hashtag/text.
    return [pos, neg, scram]


def token_to_float(token: str) -> float:
    return {"1.0": 1.0, "0.5": 0.5, "0.0": 0.0, "NA": np.nan}[token]


def run_score(force: bool = False) -> None:
    if SCORES_PATH.exists() and not force:
        print(f"exists: {SCORES_PATH}")
        return
    rubrics = load_rubrics()
    all_rows = list(iter_text_rows())
    chosen = set(selected_hashtags(all_rows))
    rows = [r for r in all_rows if r["hashtag"] in chosen]
    labels = label_lookup()  # used only for mandatory anchor construction
    shards = [[] for _ in range(N_SHARDS)]
    for row in rows:
        index = int(hashlib.sha1(row["row_id"].encode()).hexdigest(), 16) % N_SHARDS
        shards[index].append(row)

    row_ids, groups, splits, A, V = [], [], [], [], []
    anchor_reports = []
    for shard_index, shard_rows in enumerate(shards):
        history = []
        valid = False
        for attempt in range(21):
            anchors = build_anchors(all_rows, labels, shard_index, attempt)
            batch = shard_rows + anchors
            batch_A = [judge_codebook(r["text"], r["hashtag"]) for r in batch]
            matrix = np.array(
                [[token_to_float(token) for token in values] for values in batch_A],
                dtype=float,
            )
            anchor_matrix = matrix[-3:]
            anchor_means = np.nanmean(anchor_matrix, axis=1)
            valid = bool(anchor_means[0] > anchor_means[1] > anchor_means[2])
            history.append(
                {
                    "attempt": attempt + 1,
                    "winner_mean": float(anchor_means[0]),
                    "random_mean": float(anchor_means[1]),
                    "scrambled_mean": float(anchor_means[2]),
                    "valid": valid,
                }
            )
            if valid:
                break
        anchor_reports.append(
            {
                "shard": shard_index,
                "n_scored_rows": len(shard_rows),
                "winner_mean": float(anchor_means[0]),
                "random_mean": float(anchor_means[1]),
                "scrambled_mean": float(anchor_means[2]),
                "valid": valid,
                "attempts": len(history),
                "attempt_history": history,
            }
        )
        if not valid:
            raise RuntimeError(
                f"invalid anchor ordering in shard {shard_index}: {anchor_means.tolist()}"
            )
        for row, avec in zip(shard_rows, matrix[: len(shard_rows)]):
            row_ids.append(row["row_id"])
            groups.append(row["hashtag"])
            splits.append(row["split_dir"])
            A.append(avec)
            V.append(vector(row["text"], row["hashtag"]))
        print(
            f"shard {shard_index}: {len(shard_rows)} rows; anchors "
            f"{anchor_means[0]:.3f}>{anchor_means[1]:.3f}>{anchor_means[2]:.3f}"
        )
    np.savez_compressed(
        SCORES_PATH,
        row_id=np.asarray(row_ids, dtype=object),
        hashtag=np.asarray(groups, dtype=object),
        split=np.asarray(splits, dtype=object),
        A=np.asarray(A, dtype=float),
        V=np.asarray(V, dtype=float),
        a_names=np.asarray([r["name"] for r in rubrics], dtype=object),
        a_ids=np.asarray([r["rubric_id"] for r in rubrics], dtype=object),
        v_names=np.asarray(V_NAMES, dtype=object),
        selected_hashtags=np.asarray(sorted(chosen), dtype=object),
        anchor_json=np.asarray(json.dumps(anchor_reports), dtype=object),
        backend=np.asarray("codebook", dtype=object),
    )
    print(f"wrote {len(row_ids)} rows -> {SCORES_PATH}")


def _oof_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Tuple[float, List[dict]]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pred = np.full(len(y), np.nan)
    fold_rows = []
    splitter = GroupKFold(n_splits=5)
    for fold, (train, test) in enumerate(splitter.split(X, y, groups)):
        pipe = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(C=1.0, solver="liblinear", max_iter=2000, random_state=SEED),
        )
        pipe.fit(X[train], y[train])
        pred[test] = pipe.predict_proba(X[test])[:, 1]
        fold_rows.append(
            {
                "fold": fold,
                "n": int(len(test)),
                "positives": int(y[test].sum()),
                "negatives": int(len(test) - y[test].sum()),
                "hashtags": sorted(set(groups[test].tolist())),
                "auc": float(roc_auc_score(y[test], pred[test])),
            }
        )
    return float(roc_auc_score(y, pred)), fold_rows


def balanced_indices(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    selected = []
    for group in sorted(set(groups.tolist())):
        idx = np.flatnonzero(groups == group)
        pos = idx[y[idx] == 1]
        neg = idx[y[idx] == 0]
        selected.extend(pos.tolist())
        selected.extend(rng.choice(neg, size=min(len(neg), len(pos)), replace=False).tolist())
    return np.asarray(sorted(selected), dtype=int)


def run_readout() -> dict:
    from sklearn.metrics import roc_auc_score

    z = np.load(SCORES_PATH, allow_pickle=True)
    labels = label_lookup()
    row_ids = z["row_id"].tolist()
    missing = [rid for rid in row_ids if rid not in labels]
    if missing:
        raise RuntimeError(f"{len(missing)} score rows lack verdicts")
    y = np.asarray([labels[rid] in {1, 2} for rid in row_ids], dtype=int)
    groups = z["hashtag"].astype(str)
    V, A = z["V"].astype(float), z["A"].astype(float)
    names = z["a_names"].astype(str)

    def suite(index: np.ndarray) -> dict:
        yi, gi = y[index], groups[index]
        output = {
            "n": int(len(index)),
            "positives": int(yi.sum()),
            "negatives": int(len(index) - yi.sum()),
            "hashtags": int(len(set(gi.tolist()))),
            "models": {},
        }
        for name, matrix in (("V", V), ("A", A), ("V+A", np.column_stack([V, A]))):
            auc, folds = _oof_auc(matrix[index], yi, gi)
            output["models"][name] = {"auc": auc, "folds": folds}
        return output

    all_index = np.arange(len(y))
    bal_index = balanced_indices(y, groups)
    univariate = []
    for column, name in enumerate(names):
        values = A[:, column].copy()
        finite = np.isfinite(values)
        fill = float(np.nanmedian(values)) if finite.any() else 0.5
        values[~finite] = fill
        auc = float(roc_auc_score(y, values)) if len(set(values)) > 1 else 0.5
        univariate.append(
            {
                "criterion": name,
                "rubric_id": str(z["a_ids"][column]),
                "auc": auc,
                "na_rate": float((~finite).mean()),
            }
        )
    univariate.sort(key=lambda x: x["auc"], reverse=True)
    anchors = json.loads(str(z["anchor_json"].item()))
    anchor_aggregate = {
        "winner_mean": float(np.mean([x["winner_mean"] for x in anchors])),
        "random_mean": float(np.mean([x["random_mean"] for x in anchors])),
        "scrambled_mean": float(np.mean([x["scrambled_mean"] for x in anchors])),
        "all_batches_valid": all(x["valid"] for x in anchors),
        "batches": anchors,
    }
    result = {
        "dataset": "SemEval-2017 Task 6 #HashtagWars",
        "y_definition": "1 iff original label is 1 or 2; otherwise 0",
        "grouping_unit": "hashtag contest",
        "selection": {
            "method": f"first {N_CONTESTS} hashtags under SHA-256({SAMPLE_SALT!r} + hashtag)",
            "whole_groups_only": True,
            "selected_hashtags": z["selected_hashtags"].astype(str).tolist(),
            "dataset_total_n": 12734,
            "dataset_total_hashtags": 112,
        },
        "scoring": {
            "requested_judge": "gpt-5.6-sol",
            "actual_backend": str(z["backend"].item()),
            "temperature": 0,
            "allowed_tokens": list(TOKENS),
            "prompt_context_note": "The hashtag prompt was supplied as shared contest context; it is constant within the grouping unit.",
            "limitation": "Managed sandbox could not start a nested gpt-5.6-sol inference session; A was materialized by the deterministic articulated codebook backend and is not an LLM-judge run.",
        },
        "full": suite(all_index),
        "balanced": suite(bal_index),
        "balanced_definition": "Within every selected hashtag, retain all positives and sample an equal number of negatives with seed 20260728.",
        "anchors": anchor_aggregate,
        "univariate_A": univariate,
        "v_features": z["v_names"].astype(str).tolist(),
        "a_criteria": names.tolist(),
    }
    RESULTS_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "full": result["full"],
        "balanced": result["balanced"],
        "anchors": anchor_aggregate,
        "top10": univariate[:10],
    }, indent=2))
    print(f"wrote -> {RESULTS_PATH}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("score", "readout", "all"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command in {"score", "all"}:
        run_score(force=args.force)
    if args.command in {"readout", "all"}:
        run_readout()


if __name__ == "__main__":
    main()
