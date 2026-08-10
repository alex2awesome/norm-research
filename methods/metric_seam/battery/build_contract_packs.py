"""WS1a: build contract-authoring packs for the effort-ladder panel (runbook 2026-07-10).

For each frozen panel unit, emit a pack the contract-authoring fleet consumes. The authored
CONSTRUCT CONTRACT (frozen thereafter, reused at every effort rung) must contain:
  1. construct_definition — verbatim from the criterion description (no paraphrase)
  2. cf_probes — 4-6 executable minimal pairs {text_pos, text_neg, why} the construct must
     separate (label-free, authored from the definition, NOT from train labels)
  3. discrimination_checks — bounds any candidate must satisfy on TRAIN scores
     (std >= 0.05 raw-scale, frac_at_mode <= 0.85)
Validation gates (central, before freeze): probes executable; h0 passes >= half its probes
where h0 is certified; no probe references labels/metadata/length.

Source: improver_packs/<aid>.json (tasks/<task>/ or v2/ for press_releases).
Usage: python3 build_contract_packs.py
-> outputs/metric_seam_pilot/battery/effort_ladder/contract_packs/<task>__<aid>.json
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/metric_seam_pilot"
EL = BASE / "battery/effort_ladder"

PACK_DIR = {t: BASE / "tasks" / t / "improver_packs"
            for t in ("creative_writing", "math", "humor", "legal_title_vii")}
PACK_DIR["press_releases"] = BASE / "v2/improver_packs"

AUTHOR_INSTRUCTIONS = """\
You are authoring a FROZEN CONSTRUCT CONTRACT for one evaluative criterion. This contract
guards a multi-round compiler crew against construct drift: every candidate implementation
must keep passing it, at every effort rung. Author from the criterion definition ONLY —
never from item labels or score patterns.
Return JSON: {"construct_definition": <verbatim criterion description>,
 "cf_probes": [{"text_pos": ..., "text_neg": ..., "why": ...} x4-6]  (minimal pairs: the two
   texts differ ONLY in the construct; short, realistic for this task's genre),
 "discrimination_checks": {"min_std": 0.05, "max_frac_at_mode": 0.85},
 "boundary_notes": <2-3 sentences: what neighboring constructs this one is NOT>}"""


def main():
    panel = json.load(open(EL / "panel_freeze.json"))
    out_dir = EL / "contract_packs"
    out_dir.mkdir(exist_ok=True)
    n_ok = n_missing = 0
    for task, groups in panel["panel"].items():
        for role, rows in groups.items():
            for r in rows:
                aid = r["aspect"]
                src = PACK_DIR[task] / f"{aid}.json"
                pack = {"task": task, "aspect_id": aid, "panel_role": role,
                        "r_hyb_frozen": r["r_hyb"], "r_base_frozen": r["r_base"],
                        "author_instructions": AUTHOR_INSTRUCTIONS}
                if src.exists():
                    ip = json.load(open(src))
                    pack.update({k: ip.get(k) for k in
                                 ("criterion_name", "criterion_description", "contract",
                                  "judge_reliability", "attenuation_ceiling")})
                    n_ok += 1
                else:
                    pack["MISSING_SOURCE"] = str(src)
                    n_missing += 1
                json.dump(pack, open(out_dir / f"{task}__{aid}.json", "w"), indent=1)
    print(f"packs: {n_ok} sourced, {n_missing} missing-source -> {out_dir}")


if __name__ == "__main__":
    main()
