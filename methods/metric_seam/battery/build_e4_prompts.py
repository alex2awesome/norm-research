"""E4 LOCUS prompt builder — few-shot COMPLETION prompts (base-vs-instruct arm).

One prompt file per task, run identically through Llama-3.1-8B (base) and
Llama-3.1-8B-Instruct (later 70B pair) via llama_base_score_sk3.py, so the format is
held fixed and only the checkpoint varies. Readout (eval_scale.py --e4): fm_base vs
fm_instruct on the frozen programs. base ~ instruct -> field competence lives in
pretraining; base ~ blank -> it is installed by post-training.

Few-shot exemplars are TRAIN items with the certified Gemma extraction's own field
values as demonstrations (2 distinct non-empty + 1 NONE where available) — extractor
outputs only, never judge labels. Exemplar docs 1500 chars, target doc 12000 chars.

Usage: python3 build_e4_prompts.py <task>   (press_releases|creative_writing|math|humor)
-> <task outdir>/e4_prompts.jsonl   aspect_id = "{aid}.e4__{field}"
"""
import json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from battery_common import load_ctx, BASE  # noqa: E402

EX_CHARS = 1500
TGT_CHARS = 12000

HEADER = """Below are documents, each followed by an answer extracted from it.
Extraction task: from each document, {instruction}
If the document does not contain the answer, the answer is NONE.
"""

BLOCK = """
Document:
{doc}

Answer: {ans}
"""


def pick_exemplars(f_aid, items, train, field):
    """2 distinct non-empty answers + 1 NONE (if available), deterministic order."""
    nonempty, none_d = [], None
    seen_vals = set()
    for d in sorted(train):
        if d not in items:
            continue
        v = f_aid.get(d, {}).get(field)
        if v is None:
            continue
        if v == "":
            if none_d is None:
                none_d = d
        elif v.lower() not in seen_vals and len(nonempty) < 2:
            seen_vals.add(v.lower())
            nonempty.append((d, v))
    if len(nonempty) < 2:
        return None
    ex = list(nonempty)
    if none_d is not None:
        ex.append((none_d, "NONE"))
    else:  # third distinct non-empty instead
        for d in sorted(train):
            v = f_aid.get(d, {}).get(field)
            if v and v.lower() not in seen_vals:
                ex.append((d, v))
                break
    return ex


def main():
    task = sys.argv[1]
    ctx = load_ctx(task)
    inv = json.load(open(BASE / "battery/inventory.json"))[task]
    outdir = ctx["outdir"]
    test = sorted(ctx["test"])

    n = skipped = 0
    with open(outdir / "e4_prompts.jsonl", "w") as f:
        for aid in sorted(inv):
            f_aid = ctx["f_orig"].get(aid, {})
            for field, fmeta in inv[aid].get("fields", {}).items():
                ex = pick_exemplars(f_aid, ctx["items"], ctx["train"], field)
                if ex is None:
                    skipped += 1
                    continue
                head = HEADER.format(instruction=fmeta["instruction"])
                shots = "".join(BLOCK.format(doc=ctx["items"][d][:EX_CHARS], ans=a)
                                for d, a in ex)
                for d in test:
                    prompt = (head + shots
                              + BLOCK.format(doc=ctx["items"][d][:TGT_CHARS],
                                             ans="").rstrip() + " ")
                    f.write(json.dumps({"channel": "field",
                                        "aspect_id": f"{aid}.e4__{field}",
                                        "datapoint_id": d,
                                        "prompt": prompt}) + "\n")
                    n += 1
    print(f"{task}: {n} e4 prompts ({skipped} fields skipped, <2 exemplars) "
          f"-> {outdir / 'e4_prompts.jsonl'}")


if __name__ == "__main__":
    main()
