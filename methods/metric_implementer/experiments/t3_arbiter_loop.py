"""Tier-3 arbiter wave (2026-08-14): gpt-5.6-sol via Codex companion judges each
(criterion, full-document) item blind to mention-y and to all judge scores.

Batches: outputs/analyses/t3_arbiter/t3_batches/batch_NN.json (6 items each; 18 sealed
mechanical anchors are shuffled in — the key lives ONLY on sk3, unsealed at analysis).
Resumable: a batch is skipped iff its verdict file exists and parses with 6 items.
Per standing rule, arbiter waves ride Codex (never Claude credits).
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
T3 = REPO / "outputs/analyses/t3_arbiter"
CODEX = Path.home() / ".claude/plugins/cache/openai-codex/codex/1.0.5/scripts/codex-companion.mjs"

INSTR = """You are an expert scientific arbiter. The file {batch} contains a JSON list of
items, each with: item_id, criterion (an evaluation criterion), definition (may be empty),
and document (a paper's text, possibly truncated).

For EACH item, read the document carefully and judge: does the document, as written,
satisfy / exhibit the criterion? Judge only from the document text. Be strict but fair;
a criterion counts as satisfied if the document substantively exhibits the property, not
merely mentions related words.

Write your verdicts to {out} as a JSON list, one object per item, in the same order:
  {{"item_id": "...", "score": <integer 0-10>, "applies": <true|false>}}
score 0 = clearly does not satisfy, 10 = clearly and strongly satisfies; applies = your
binary call. Do not read any other files. Reply with just the count of items judged."""


def done(out, want):
    try:
        v = json.load(open(out))
        return isinstance(v, list) and len(v) >= want and all("item_id" in x for x in v)
    except Exception:
        return False


def main(limit=None, batch_dir="t3_batches", verdict_dir="verdicts"):
    batches = sorted((T3 / batch_dir).glob("batch_*.json"))
    if limit:
        batches = batches[:int(limit)]
    (T3 / verdict_dir).mkdir(exist_ok=True)
    for b in batches:
        out = T3 / verdict_dir / (b.stem + "_verdicts.json")
        want = len(json.load(open(b)))
        if done(out, want):
            print(f"SKIP {b.name} (complete)")
            continue
        prompt = INSTR.format(batch=b, out=out)
        print(f"=== {b.name} {time.strftime('%H:%M:%S')} ===", flush=True)
        try:
            r = subprocess.run(["node", str(CODEX), "task", "--write", prompt, "--fresh"],
                               timeout=1200, capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            print(f"{b.name}: TIMEOUT — skipping (re-run loop to retry)", flush=True)
            continue
        except Exception as e:
            print(f"{b.name}: ERROR {e} — skipping", flush=True)
            continue
        if not done(out, want):                     # stdout code-fence fallback
            m = re.findall(r"```(?:json)?\n(.*?)```", r.stdout or "", re.S)
            for cand in m[::-1]:
                try:
                    v = json.loads(cand)
                    if isinstance(v, list) and all("item_id" in x for x in v):
                        json.dump(v, open(out, "w")); break
                except Exception:
                    continue
        print(f"{b.name}: {'OK' if done(out, want) else 'INCOMPLETE'}", flush=True)
    n_ok = sum(1 for b in batches if done(T3 / verdict_dir / (b.stem + "_verdicts.json"),
                                          len(json.load(open(b)))))
    print(f"WAVE DONE: {n_ok}/{len(batches)} batches complete")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "-" else None,
         sys.argv[2] if len(sys.argv) > 2 else "t3_batches",
         sys.argv[3] if len(sys.argv) > 3 else "verdicts")
