"""OSL anchor battery — the DECLARED capability instrument (spec: notes/2026-07-07__osl-executor-scaling-spec.md).

Every item is (criterion, text, truth) with truth computed BY CODE — zero label noise, no human or
LLM labels (instrument calibration: the blinded-anchor discipline promoted to a measurement). The
battery is built once, frozen (sha256 recorded in the spec), and scored identically for every
executor. Declared capability scalar: z_E = logit(AUC(P_YES, truth)) — threshold-free readout.

Item families (all code-truth, ~50/50 balanced by construction):
  threshold    "longer/shorter than K words" — difficulty graded by margin |wc−K|/wc
  presence     "contains a question mark / digit / the word 'W'"
  negation     "does not contain any digits / the word 'W'"
  composite    "longer than K words AND contains a question mark" (both clauses must hold)
  paraphrase   the same predicate under a second surface form (truth-bearing, enters the scalar)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re

import numpy as np

from . import alpha_probe as ap

_STOP = set("the and for that with this from have will your they there what when been were "
            "them then than because about into just like also over such only".split())


def _wc(t: str) -> int:
    return len(t.split())


def _words(t: str):
    return [w for w in re.findall(r"[a-zA-Z']{4,}", t.lower()) if w not in _STOP]


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _item(crit, text, truth, family, difficulty):
    return {"criterion": crit, "text": text, "truth": bool(truth),
            "family": family, "difficulty": difficulty}


def build_battery(texts, seed: int = 0, per_text_cap: int = 4, target: int = 240):
    """Deterministic battery from a frozen text window. Texts with <8 words are skipped."""
    rng = np.random.default_rng(seed)
    pool = [t for t in texts if _wc(t) >= 8]
    items = []
    for ti, t in enumerate(pool):
        wc = _wc(t)
        made = []
        # -- threshold (hard margin ~7%, easy margin ~50%): K placed BELOW wc (-> mostly true)
        # and ABOVE wc (-> mostly false), so the family carries both truth values --------------
        for margin, diff in ((0.07, "hard"), (0.5, "easy")):
            k_lo = max(1, int(round(wc * (1 - margin))))
            k_hi = int(round(wc * (1 + margin))) + 1
            made.append(_item(f"This text is longer than {k_lo} words.", t, wc > k_lo,
                              "threshold", diff))
            made.append(_item(f"This text is longer than {k_hi} words.", t, wc > k_hi,
                              "threshold", diff))
        # -- presence: punctuation / digit --------------------------------------------------
        made.append(_item("This text contains at least one question mark.", t, "?" in t,
                          "presence", "easy"))
        made.append(_item("This text contains at least one digit (0-9).", t,
                          bool(re.search(r"\d", t)), "presence", "easy"))
        # -- presence: word (true = word in text; false = word from elsewhere, absent here) --
        ws = _words(t)
        if ws:
            w_in = ws[int(rng.integers(len(ws)))]
            made.append(_item(f"This text mentions the word '{w_in}'.", t, True,
                              "presence", "medium"))
            other = pool[int(rng.integers(len(pool)))]
            cand = [w for w in _words(other) if w not in set(_words(t))]
            if cand:
                w_out = cand[int(rng.integers(len(cand)))]
                made.append(_item(f"This text mentions the word '{w_out}'.", t, False,
                                  "presence", "medium"))
        # -- negation ------------------------------------------------------------------------
        made.append(_item("This text does not contain any digits.", t,
                          not re.search(r"\d", t), "negation", "medium"))
        # -- composite (both clauses must hold) ----------------------------------------------
        k = max(1, int(round(wc * (1 - 0.15))))
        made.append(_item(f"This text is longer than {k} words AND contains a question mark.",
                          t, (wc > k) and ("?" in t), "composite", "hard"))
        # -- paraphrase surface form of the hard threshold (truth-bearing; direction alternates
        # by text index so the family carries both truth values) ------------------------------
        k2 = max(1, int(round(wc * (1 - 0.07) if ti % 2 == 0 else wc * (1 + 0.07) + 1)))
        made.append(_item(f"The passage above runs to more than {k2} words in total length.",
                          t, wc > k2, "paraphrase", "hard"))
        picks = rng.permutation(len(made))[:per_text_cap]
        items.extend(made[i] for i in picks)
        if len(items) >= target * 2:
            break
    # balance to ~50/50 by dropping surplus of the majority class (deterministic order)
    rng2 = np.random.default_rng(seed + 1)
    order = rng2.permutation(len(items))
    pos = [items[i] for i in order if items[i]["truth"]]
    neg = [items[i] for i in order if not items[i]["truth"]]
    n = min(len(pos), len(neg), target // 2)
    out = pos[:n] + neg[:n]
    out = [out[i] for i in np.random.default_rng(seed + 2).permutation(len(out))]
    for i, it in enumerate(out):
        it["id"] = i
    return out


def _auc(scores, labels) -> float:
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    m = np.isfinite(s)
    s, y = s[m], y[m]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def score_battery(executor, items, max_chars: int) -> dict:
    """One batched forced-logprob pass over all items; AUC overall + per family + per half."""
    prompts = [ap._YESNO_TEXTFIRST.format(text=it["text"][:max_chars], rubric=it["criterion"])
               for it in items]
    py = np.asarray(executor.score_binary(prompts), float)
    truth = np.array([it["truth"] for it in items], int)
    fams = sorted({it["family"] for it in items})
    per_family = {f: _auc(py[[i for i, it in enumerate(items) if it["family"] == f]],
                          truth[[i for i, it in enumerate(items) if it["family"] == f]])
                  for f in fams}
    auc = _auc(py, truth)
    ev, od = np.arange(0, len(items), 2), np.arange(1, len(items), 2)
    out = {"auc": auc, "z": float(np.log(max(auc, 1e-6) / max(1 - auc, 1e-6))),
           "per_family": per_family, "nan_rate": float(np.mean(~np.isfinite(py))),
           "auc_even_items": _auc(py[ev], truth[ev]), "auc_odd_items": _auc(py[od], truth[od]),
           "pyes": [None if not np.isfinite(v) else round(float(v), 5) for v in py]}
    return out


def main(argv=None):
    from .run_real_test import _load_texts
    from .. import config as cfgmod
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="humor")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--text-window", default="360:600",
                   help="manifest slice for battery texts (disjoint from eval probes 60:360)")
    a = p.parse_args(argv)
    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), a.task)
    lo, hi = (int(x) for x in a.text_window.split(":"))
    texts, _ = _load_texts(a.task, hi, cfg)
    if len(texts) < hi:
        raise SystemExit(f"corpus too small: {len(texts)} < {hi}")
    items = build_battery(texts[lo:hi], seed=a.seed)
    blob = {"meta": {"task": a.task, "seed": a.seed, "text_window": a.text_window,
                     "n_items": len(items)}, "items": items}
    js = json.dumps(blob, indent=1, sort_keys=True)
    open(a.out, "w").write(js)
    print(f"[battery] {len(items)} items -> {a.out}  sha256={_sha(js)}")


if __name__ == "__main__":
    main()
