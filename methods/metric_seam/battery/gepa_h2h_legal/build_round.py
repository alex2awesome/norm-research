"""GEPA-H2H round-N prompts-file builder (seam note Sec 9, arm G).

For each criterion whose state.json "round" == <round>, renders that criterion's current
BODY template against its 40 fixed dev items (ctx["items"], TRAIN only) and appends the
fixed reply-format footer. One row per (criterion, dev item), v2 prompts-file format.

Usage: python3 build_round.py <round>
-> gepa_round<round>_prompts.jsonl  (12 x 40 = 480 rows at round 0)
   + prints the queue2 submit hint (orchestrator runs this on sk3; this script does not).

Never touches ctx["test"].
"""
import json, sys

from common import CRITERIA, HERE, build_doc_prompt, crit_key, load_ctx, load_state


def main():
    if len(sys.argv) != 2:
        print("usage: python3 build_round.py <round>"); sys.exit(1)
    rd = int(sys.argv[1])
    state = load_state()

    ctxs = {}
    out_path = HERE / f"gepa_round{rd}_prompts.jsonl"
    n = 0
    skipped = []
    with open(out_path, "w") as f:
        for task, aid in CRITERIA:
            key = crit_key(task, aid)
            c = state["criteria"][key]
            if c["round"] != rd:
                skipped.append((key, c["round"]))
                continue
            if task not in ctxs:
                ctxs[task] = load_ctx(task)
            items = ctxs[task]["items"]
            for dpid in c["dev_ids"]:
                text = items.get(dpid, "")
                row = {"channel": "field", "aspect_id": f"{key}.g{rd}",
                      "datapoint_id": dpid, "prompt": build_doc_prompt(c["prompt"], text)}
                f.write(json.dumps(row) + "\n")
                n += 1

    if skipped:
        print(f"SKIPPED (state round != {rd}, run ingest_round.py/propose.py for the "
              f"pending round first): {skipped}")
    print(f"wrote {n} rows ({n // 40 if n else 0} criteria x {40}) -> {out_path}")
    print()
    print("orchestrator queue2 submit hint (sk3, gemma4 env, GPU N):")
    out_results = HERE / f"gepa_round{rd}_results.jsonl"
    print(f"  echo '{out_path} {out_results}' > "
          f"$QDIR2/{rd:03d}_gepa_round{rd}.job")
    print("  # scorer/pybin omitted -> queue2 defaults apply (gemma_score_v1.py, gemma4 env)")
    print(f"  # after it completes: python3 ingest_round.py {rd} {out_results}")


if __name__ == "__main__":
    main()
