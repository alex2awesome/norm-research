"""EXP-EAP-1 arbiter wave (prereg a545e0c3f). gpt-5.6-sol via Codex companion judges
exemplar-anchored items: 8 anchor excerpts sharing ONE unnamed quality + a candidate
document. NO definitions/reconstructions/metric names in input (circularity guard).
Resumable: batch skipped iff verdict exists and parses with the right count."""
import json
import subprocess
import sys
import time
from pathlib import Path

EAP = Path(__file__).resolve().parent
CODEX = Path.home() / ".claude/plugins/cache/openai-codex/codex/1.0.5/scripts/codex-companion.mjs"

INSTR = """You are an expert literary/scientific arbiter. The file {batch} contains a
JSON list of items. Each item has: item_id; anchors — 8 short excerpts from documents
that expert readers flagged as ALL exhibiting one specific shared quality (the quality
is deliberately not named); and document — a candidate document.

For EACH item: (1) infer from the anchors alone what single SPECIFIC quality they share
— name the property precisely, not a generic label like "funny" or "well-written";
(2) judge whether the candidate document exhibits that same quality, based only on the
document text. Be strict but fair.

Write your verdicts to {out} as a JSON list, one object per item, same order:
  {{"item_id": "...", "quality": "<your inferred quality, <=12 words>",
    "score": <integer 0-10>, "applies": <true|false>}}
score 0 = clearly lacks the quality, 10 = exhibits it clearly and strongly; applies =
your binary call. Do not read any other files. Reply with just the count judged."""


def ok(out, want):
    if not out.exists():
        return False
    try:
        v = json.load(open(out))
        return isinstance(v, list) and len(v) == want and all("applies" in r for r in v)
    except Exception:
        return False


def main():
    batches = sorted((EAP / "batches").glob("batch_*.json"))
    vd = EAP / "verdicts"
    vd.mkdir(exist_ok=True)
    for b in batches:
        want = len(json.load(open(b)))
        out = vd / (b.stem + "_verdict.json")
        if ok(out, want):
            continue
        prompt = INSTR.format(batch=b, out=out)
        for attempt in range(3):
            r = subprocess.run(["node", str(CODEX), "task", "--write", prompt, "--fresh"],
                               timeout=1800, capture_output=True, text=True)
            if ok(out, want):
                print(f"{b.stem} OK", flush=True)
                break
            print(f"{b.stem} attempt {attempt + 1} failed: {(r.stdout or '')[-120:]}",
                  flush=True)
            time.sleep(20)
    done = sum(1 for b in batches
               if ok(vd / (b.stem + "_verdict.json"), len(json.load(open(b)))))
    print(f"DONE {done}/{len(batches)}")


if __name__ == "__main__":
    main()
