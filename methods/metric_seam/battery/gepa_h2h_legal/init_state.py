"""GEPA-H2H round-0 state builder (seam note Sec 9).

For each of the 12 criteria: seeds the round-0 scoring-prompt BODY (rendered from the
improver pack's criterion_name/criterion_description, seam-note template) and samples a
FIXED 40-item dev set from ctx["train"] (seed 13, stable sorted-then-sampled order --
never re-sampled, never touches ctx["test"]).

Usage: python3 init_state.py
-> state.json (round 0 for all 12 criteria; refuses to overwrite an existing state.json --
   delete it first if you really want to restart from scratch)
"""
import random, sys

from common import (CRITERIA, N_DEV, SEED, STATE_PATH, TASK_DOMAIN, SEED_TEMPLATE,
                     DOC_MARKER, crit_key, load_improver_pack, load_ctx, save_state)


def dev_sample(ctx, aid):
    """40 fixed TRAIN items with judge coverage, seeded stable sample (never re-shuffled)."""
    judge = ctx["judge"].get(aid, {})
    avail = sorted(d for d in ctx["train"] if d in judge)   # stable order first
    rng = random.Random(SEED)
    n = min(N_DEV, len(avail))
    return sorted(rng.sample(avail, n))                     # sorted again for readability


def main():
    if STATE_PATH.exists():
        print(f"REFUSING: {STATE_PATH} already exists. Delete it first to reinit.")
        sys.exit(1)

    ctxs = {}
    criteria_state = {}
    for task, aid in CRITERIA:
        if task not in ctxs:
            ctxs[task] = load_ctx(task)
        ctx = ctxs[task]
        pack = load_improver_pack(task, aid)
        dev_ids = dev_sample(ctx, aid)
        if len(dev_ids) < N_DEV:
            print(f"WARN {task}.{aid}: only {len(dev_ids)} judge-covered train items "
                  f"(<{N_DEV})")
        body0 = SEED_TEMPLATE.format(domain=TASK_DOMAIN[task],
                                     criterion_name=pack["criterion_name"],
                                     criterion_description=pack["criterion_description"],
                                     marker=DOC_MARKER)
        assert DOC_MARKER in body0
        key = crit_key(task, aid)
        criteria_state[key] = {
            "task": task, "aid": aid,
            "criterion_name": pack["criterion_name"],
            "criterion_description": pack["criterion_description"],
            "dev_ids": dev_ids,
            "round": 0,
            "prompt": body0,
            "history": [],
            "proposed_through": -1,
            "best": None,
        }
        print(f"{key}: dev_n={len(dev_ids)} seed prompt len={len(body0)} chars")

    state = {"meta": {"seed": SEED, "n_dev": N_DEV,
                      "n_criteria": len(criteria_state)},
             "criteria": criteria_state}
    save_state(state)
    print(f"-> {STATE_PATH}  ({len(criteria_state)} criteria, round 0)")


if __name__ == "__main__":
    main()
