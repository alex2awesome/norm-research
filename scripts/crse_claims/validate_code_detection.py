"""
validate_code_detection.py — Fix-2 validation: run the new code-block
extractor over REAL sampled CR.SE answers (raw HTML bodies from the SE data
dump) and report detection rates vs. the old stripped-text regex.

Usage (sk3):
  python scripts/crse_claims/validate_code_detection.py \
      --posts-parquet datasets/codereview_se/posts.parquet \
      --pilot-claims outputs/crse_claims_pilot/claims.jsonl \
      --n-random 200 --out outputs/crse_claims_pilot_v2/code_detection_validation.json
"""
from __future__ import annotations
import argparse, json, random, re, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from code_blocks import (extract_code_blocks, detect_format, guess_language)

# the OLD pilot extractor (markdown fences + indents on stripped text)
OLD_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def strip_html(html_text: str) -> str:
    """Same stripper as build_crse_pool*.py — what the pool `text` field saw."""
    if not isinstance(html_text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", html_text)
    text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    text = text.replace("&quot;", '"').replace("&#39;", "'")
    return " ".join(text.split()).strip()


def old_extract(text: str) -> str | None:
    m = OLD_CODE_BLOCK_RE.search(text or "")
    if m:
        return m.group(1).strip()
    lines = (text or "").split("\n")
    block = []
    for L in lines:
        if L.startswith("    "):
            block.append(L[4:])
        elif block and not L.strip():
            block.append("")
        elif block:
            break
    return ("\n".join(block).strip()) or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts-parquet", required=True)
    ap.add_argument("--pilot-claims", default=None,
                    help="claims.jsonl from the pilot — validates on the SAME "
                         "300 answers, plus --n-random extra")
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_parquet(args.posts_parquet, columns=["Id", "PostTypeId", "Body"])
    ans = df[df.PostTypeId == 2].copy()
    ans["Id"] = ans["Id"].astype(str)

    pilot_ids = []
    if args.pilot_claims and Path(args.pilot_claims).exists():
        for L in open(args.pilot_claims):
            try:
                pilot_ids.append(str(json.loads(L)["answer_id"]))
            except Exception:
                pass
    pilot_set = set(pilot_ids)

    rng = random.Random(args.seed)
    rand_ids = rng.sample(ans["Id"].tolist(), args.n_random)
    groups = {"pilot_300": sorted(pilot_set), "random": rand_ids}

    body_map = dict(zip(ans["Id"], ans["Body"].fillna("")))
    report = {"groups": {}}
    shown = 0
    for gname, ids in groups.items():
        ids = [i for i in ids if i in body_map]
        fmt_dist = Counter()
        lang_dist = Counter()
        n_new, n_old, n_blocks_total = 0, 0, 0
        per_example = []
        for aid in ids:
            body = body_map[aid]
            fmt_dist[detect_format(body)] += 1
            blocks = extract_code_blocks(body)
            stripped = strip_html(body)
            old = old_extract(stripped)
            if blocks:
                n_new += 1
                n_blocks_total += len(blocks)
                lang_dist[guess_language(blocks[0])] += 1
            if old:
                n_old += 1
            per_example.append({"answer_id": aid, "new_n_blocks": len(blocks),
                                "old_found": bool(old)})
            if blocks and shown < args.show:
                shown += 1
                print(f"--- example {aid} ({gname}) ---")
                print(f"  format={detect_format(body)} "
                      f"lang_guess={guess_language(blocks[0])}")
                print("  block head:", blocks[0][:200].replace("\n", " ⏎ "))
        n = len(ids) or 1
        report["groups"][gname] = {
            "n_answers": len(ids),
            "format_dist": dict(fmt_dist),
            "new_detection_rate": round(n_new / n, 4),
            "old_detection_rate": round(n_old / n, 4),
            "avg_blocks_when_found": round(n_blocks_total / max(1, n_new), 2),
            "first_block_lang_guess_dist": dict(lang_dist),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["groups"], indent=2))


if __name__ == "__main__":
    main()
