"""Signal-leakage detection for articulation-STaR rationales.

Operationalizes "signal leakage" as: the rationale conveys the label to the
judge via cues that are NOT specific articulation of artifact content. We
estimate this two ways:

  (A) Automated proxies (cheap):
      - specificity: # of quoted/paraphrased substrings from artifact per bullet
      - templatey:   # of formulaic phrases ("normative principle of X",
                     "principle of \\w+", "engagement with the surroundings")
      - sentiment_polarity: pos-word-count minus neg-word-count, normalized
      - sentiment_alone_acc: a logreg trained on (sent_pos_count,
                             sent_neg_count) to predict the judge's
                             prediction. If this matches judge accuracy,
                             sentiment is doing the work -> leakage.

  (B) LLM-judged leakage (more expensive, per-stage sample):
      - Sample N rationales per stage.
      - Ask an LLM judge to rate each rationale on a small rubric:
          * cites_specific (0/1): references concrete content from artifact
          * uses_template (0/1): uses formulaic phrasing
          * sentiment_only (0/1): could be decoded by sentiment alone
          * hallucinates (0/1): describes content not in the artifact
      - Aggregate to a per-stage "leakage rate" = 1 - cites_specific +
        uses_template + sentiment_only + hallucinates, averaged.

Usage:
  --mode auto    -> compute automated proxies for one stage
  --mode aggregate -> print per-stage table from auto outputs
  --mode llm     -> LLM-judged sample (uses OpenAI API key on sk3)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean, median

from .config import LoopConfig, TASKS


# ── lexicons ───────────────────────────────────────────────────────
POS_WORDS = {
    "immersive", "vivid", "engaging", "thoughtful", "thought-provoking",
    "evocative", "compelling", "nuanced", "rich", "captivating", "elegant",
    "poignant", "skillful", "masterful", "effective", "strong", "successful",
    "powerful", "creative", "original", "authentic", "relatable", "moving",
    "atmospheric", "well-developed", "polished", "tight", "clean", "sharp",
    "satisfying", "resonant", "memorable", "impressive", "striking",
}
NEG_WORDS = {
    "clichéd", "cliched", "cliche", "uneven", "rushed", "convenient",
    "unrealistic", "underdeveloped", "thin", "shallow", "generic", "stilted",
    "weak", "flat", "predictable", "contrived", "awkward", "forced",
    "unclear", "muddled", "confused", "disjointed", "lacking", "missing",
    "abrupt", "jarring", "unnatural", "implausible", "stereotypical",
    "derivative", "trite", "boring", "tedious", "incoherent",
}

TEMPLATE_PATTERNS = [
    re.compile(r"normative principle of \w+", re.IGNORECASE),
    re.compile(r"the principle of (showing|using|engaging|effective) ", re.IGNORECASE),
    re.compile(r"engagement with the surroundings?", re.IGNORECASE),
    re.compile(r"intellectual curiosity", re.IGNORECASE),
    re.compile(r"empower(ment|ing) of the protagonist", re.IGNORECASE),
]


BULLET_RE = re.compile(r"^\s+-\s+(.+?)(?:\s+—\s+(.+?))?\s*$", re.MULTILINE)


def _bullets(text: str) -> list[tuple[str, str]]:
    """Return [(content_part, norm_part), ...] for all bullets in text."""
    out = []
    for m in BULLET_RE.finditer(text):
        content = (m.group(1) or "").strip()
        norm = (m.group(2) or "").strip()
        if content:
            out.append((content, norm))
    return out


def _ngram_overlap_count(rationale: str, artifact: str, min_len: int = 5) -> int:
    """Count of >=min_len-token n-grams from rationale that appear verbatim
    (case-insensitive) in artifact. Cheap specificity proxy."""
    art_lower = artifact.lower()
    tokens = re.findall(r"[a-zA-Z']+", rationale)
    if len(tokens) < min_len:
        return 0
    count = 0
    seen = set()
    for i in range(len(tokens) - min_len + 1):
        ng = " ".join(t.lower() for t in tokens[i:i + min_len])
        if ng in seen:
            continue
        seen.add(ng)
        if ng in art_lower:
            count += 1
    return count


def _quoted_substring_count(rationale: str, artifact: str) -> int:
    """Count of double-quoted strings in rationale that appear verbatim in
    artifact. Stricter specificity proxy."""
    art_lower = artifact.lower()
    out = 0
    for m in re.finditer(r'"([^"]{8,})"', rationale):
        if m.group(1).lower() in art_lower:
            out += 1
    return out


def _sentiment_counts(rationale: str) -> tuple[int, int]:
    toks = [t.lower() for t in re.findall(r"[a-zA-Z'-]+", rationale)]
    pos = sum(1 for t in toks if t in POS_WORDS)
    neg = sum(1 for t in toks if t in NEG_WORDS)
    return pos, neg


def _template_hits(rationale: str) -> int:
    return sum(len(p.findall(rationale)) for p in TEMPLATE_PATTERNS)


# ── automated proxy mode ───────────────────────────────────────────

def run_auto(cfg: LoopConfig, stage: str, n_train_iters: int = 3) -> Path:
    """Compute automated proxies for one stage's test-eval rationales."""
    task = TASKS[cfg.task]
    test_dir = Path(cfg.output_root) / cfg.task / cfg.run_name / "test_eval"
    arts = {r["row_id"]: r for r in (
        json.loads(l) for l in (test_dir / "test_artifacts.jsonl").open())}
    rats = [json.loads(l) for l in (test_dir / f"rationales_{stage}.jsonl").open()]
    scores = {p["row_id"]: p for p in (
        json.loads(l) for l in (test_dir / f"scores_{stage}.jsonl").open())}

    out = []
    for r in rats:
        art = arts[r["row_id"]]["text"]
        rationale = r["completion"]
        quoted_hits = _quoted_substring_count(rationale, art)
        ngram_hits = _ngram_overlap_count(rationale, art, min_len=5)
        pos_n, neg_n = _sentiment_counts(rationale)
        templates = _template_hits(rationale)
        bullets = _bullets(rationale)
        rec = {
            "row_id": r["row_id"],
            "y": r["y"],
            "stage": stage,
            "quoted_hits": quoted_hits,
            "ngram_hits": ngram_hits,
            "n_bullets": len(bullets),
            "specificity_per_bullet": (quoted_hits + ngram_hits) / max(len(bullets), 1),
            "pos_words": pos_n,
            "neg_words": neg_n,
            "sentiment_polarity": (pos_n - neg_n) / max(pos_n + neg_n, 1),
            "template_hits": templates,
            "judge_pred": scores.get(r["row_id"], {}).get("judge_pred"),
        }
        out.append(rec)

    p = test_dir / f"leakage_auto_{stage}.jsonl"
    with p.open("w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")

    print(f"[leakage_auto:{stage}] n={len(out)} "
          f"avg_specificity={mean(r['specificity_per_bullet'] for r in out):.2f} "
          f"avg_templates={mean(r['template_hits'] for r in out):.2f}")
    return p


def run_aggregate(cfg: LoopConfig, stages: list[str]) -> None:
    """Per-stage table of automated leakage proxies + sentiment-alone
    classifier accuracy vs the judge."""
    test_dir = Path(cfg.output_root) / cfg.task / cfg.run_name / "test_eval"
    print("\n" + "=" * 95)
    print(f"LEAKAGE PROXIES  ({cfg.task} / {cfg.run_name})")
    print("=" * 95)
    print(f"{'stage':<10} {'spec/blt':>9} {'quoted':>7} {'ngram':>6} "
          f"{'pos_w':>6} {'neg_w':>6} {'sent_pol':>9} "
          f"{'tmpl':>5} {'sent_only_acc':>14}")
    print("-" * 95)
    for s in stages:
        p = test_dir / f"leakage_auto_{s}.jsonl"
        if not p.exists():
            print(f"{s:<10}  (missing {p.name})")
            continue
        rows = [json.loads(l) for l in p.open()]
        spec = mean(r["specificity_per_bullet"] for r in rows)
        q = mean(r["quoted_hits"] for r in rows)
        n = mean(r["ngram_hits"] for r in rows)
        pw = mean(r["pos_words"] for r in rows)
        nw = mean(r["neg_words"] for r in rows)
        sp = mean(r["sentiment_polarity"] for r in rows)
        tm = mean(r["template_hits"] for r in rows)
        # "sentiment-only" classifier: predict pos iff sentiment_polarity > 0,
        # compare to judge prediction. High agreement => judge is mostly
        # using sentiment cues from rationale.
        agree = 0
        total = 0
        for r in rows:
            if r["judge_pred"] is None:
                continue
            sent_pred = 1 if r["sentiment_polarity"] >= 0 else 0
            if sent_pred == r["judge_pred"]:
                agree += 1
            total += 1
        sa = agree / total if total else 0.0
        print(f"{s:<10} {spec:>9.2f} {q:>7.2f} {n:>6.2f} "
              f"{pw:>6.2f} {nw:>6.2f} {sp:>9.3f} "
              f"{tm:>5.2f} {sa:>13.1%}")
    print("=" * 95)


# ── LLM-judged leakage rate ────────────────────────────────────────

LLM_RUBRIC_PROMPT = """\
You are evaluating a reviewer's notes about a creative-writing story for
signs that the notes leak the verdict via surface cues rather than via
substantive articulation. You have:

  ARTIFACT: the actual story
  RATIONALE: the reviewer's notes (positive and negative aspects)

Score the RATIONALE on each axis (0 = not at all, 1 = somewhat, 2 = clearly):

  cites_specific  : The rationale references concrete content actually
                    present in the artifact (quoted phrases, specific
                    character/place names, specific scenes/events).
  uses_template   : The rationale contains formulaic, repeated phrasings
                    that could apply to any artifact ("normative principle
                    of X", "engagement with the surroundings", generic
                    abstract praise).
  sentiment_only  : The rationale's verdict could be guessed purely from
                    overall positive vs negative sentiment, without
                    reading what's specifically said.
  hallucinates    : The rationale describes story content that does not
                    appear in the artifact.

Respond with a single JSON object, no preamble, no markdown:
{"cites_specific": 0|1|2, "uses_template": 0|1|2, "sentiment_only": 0|1|2, "hallucinates": 0|1|2}

ARTIFACT:
{artifact}

RATIONALE:
{rationale}
"""


def run_llm_judge(cfg: LoopConfig, stage: str, n_sample: int = 60) -> Path:
    """Sample n_sample rationales for one stage and have OpenAI rate them.

    Reads OpenAI key from `[[reference_openai_key_sk3]]` (SALT lab key)."""
    import os
    import random

    # Per [[reference_openai_key_sk3]].
    key_path = "/lfs/skampere3/0/alexspan/.openai-salt-lab-key.txt"
    if os.path.exists(key_path):
        with open(key_path) as f:
            os.environ["OPENAI_API_KEY"] = f.read().strip()

    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("openai package not installed; pip install openai")

    client = OpenAI()
    model = os.environ.get("LEAKAGE_LLM_MODEL", "gpt-5-mini")

    test_dir = Path(cfg.output_root) / cfg.task / cfg.run_name / "test_eval"
    arts = {r["row_id"]: r for r in (
        json.loads(l) for l in (test_dir / "test_artifacts.jsonl").open())}
    rats = [json.loads(l) for l in (test_dir / f"rationales_{stage}.jsonl").open()]
    random.seed(91)
    sample = random.sample(rats, min(n_sample, len(rats)))

    out = []
    for r in sample:
        art = arts[r["row_id"]]["text"]
        prompt = (LLM_RUBRIC_PROMPT
                  .replace("{artifact}", art[:4000])
                  .replace("{rationale}", r["completion"][:3000]))
        try:
            # gpt-5-mini only supports temperature=1 (the default), so we
            # omit the parameter and rely on JSON-only instructions for
            # structured output.
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.choices[0].message.content.strip()
            # find first {...} block
            m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
            judged = json.loads(m.group(0)) if m else {}
        except Exception as e:
            judged = {"_error": str(e)}
        rec = {
            "row_id": r["row_id"], "y": r["y"], "stage": stage,
            "judged": judged,
        }
        out.append(rec)

    p = test_dir / f"leakage_llm_{stage}.jsonl"
    with p.open("w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    print(f"[leakage_llm:{stage}] n={len(out)} written to {p}")
    return p


def run_llm_aggregate(cfg: LoopConfig, stages: list[str]) -> None:
    test_dir = Path(cfg.output_root) / cfg.task / cfg.run_name / "test_eval"
    print("\n" + "=" * 95)
    print(f"LLM-JUDGED LEAKAGE  ({cfg.task} / {cfg.run_name})")
    print("=" * 95)
    print(f"{'stage':<10} {'n':>4} {'cites':>7} {'tmpl':>6} {'sentonly':>9} "
          f"{'halluc':>7} {'leakage_score':>14}")
    print("-" * 95)
    for s in stages:
        p = test_dir / f"leakage_llm_{s}.jsonl"
        if not p.exists():
            print(f"{s:<10}  (missing {p.name})")
            continue
        rows = [json.loads(l) for l in p.open()]
        valid = [r for r in rows if "_error" not in r["judged"] and r["judged"]]
        if not valid:
            print(f"{s:<10}  (no valid LLM judgments)")
            continue
        def avg(k):
            vals = [r["judged"].get(k) for r in valid if isinstance(r["judged"].get(k), (int, float))]
            return mean(vals) if vals else float("nan")
        cs = avg("cites_specific")
        tm = avg("uses_template")
        so = avg("sentiment_only")
        ha = avg("hallucinates")
        # leakage_score: higher = more leakage. Reward cites, penalize the rest.
        # Each axis is 0/1/2, normalize to [0,1]. leakage = (template + sentonly + halluc - cites) / 4 + 0.5
        leakage = ((tm + so + ha - cs) / 8) + 0.5
        print(f"{s:<10} {len(valid):>4} {cs:>7.2f} {tm:>6.2f} {so:>9.2f} "
              f"{ha:>7.2f} {leakage:>14.3f}")
    print("=" * 95)


# ── CLI ────────────────────────────────────────────────────────────

def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="creative_writing")
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--mode", choices=["auto", "aggregate", "llm", "llm_aggregate"],
                    required=True)
    ap.add_argument("--stage", default=None)
    ap.add_argument("--stages", default="base,iter00,iter01,iter02")
    ap.add_argument("--n_sample", type=int, default=60)
    return ap.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    cfg = LoopConfig(task=a.task, run_name=a.run_name)
    if a.mode == "auto":
        if a.stage is None:
            raise SystemExit("--stage required for auto")
        run_auto(cfg, a.stage)
    elif a.mode == "aggregate":
        run_aggregate(cfg, a.stages.split(","))
    elif a.mode == "llm":
        if a.stage is None:
            raise SystemExit("--stage required for llm")
        run_llm_judge(cfg, a.stage, n_sample=a.n_sample)
    elif a.mode == "llm_aggregate":
        run_llm_aggregate(cfg, a.stages.split(","))
