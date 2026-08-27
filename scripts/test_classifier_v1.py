"""
Test the v1 classifier prompt on ~20 rubrics drawn from the parquet.

Inputs: stratified sample including (a) the 15 hand-labeled rubrics from
manual_labels_15.md (as ground-truth pairs) and (b) ~5 fresh rubrics from
tasks not yet labeled (to probe generalization).

Calls GPT-5-mini via the OpenAI API with strict json_schema response_format.

Reports per-rubric: model output (target/actor/action/keep) + agreement with
my hand-label (where available).
"""

from __future__ import annotations
import asyncio, json, sys, time
from pathlib import Path

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
sys.path.insert(0, str(ROOT / "scripts"))

from classify_rubric_v1_prompt import SYSTEM_PROMPT_CLASSIFY, JSON_SCHEMA_CLASSIFY
from extract_rubric_features import _load_api_key

OPENAI_KEY = _load_api_key()

OUT_DIR = ROOT / "logs/rubric_labeling/classifier_v1_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# My hand-labels from manual_labels_15.md, keyed by (page_match, rubric_name_match)
# Format: (target, actor, action, keep)
GROUND_TRUTH = {
    ("kirkus_indie", "How Kirkus chooses a reviewer"):              ("service_or_logistics", "platform", "describe", "drop"),
    ("kirkus_indie", "Honest and impartial evaluation"):            ("evaluation_judgment", "evaluator", "judge", "keep"),
    ("kirkus_indie", "Review lengths by option"):                   ("meta_artifact", "platform", "constrain", "drop"),
    ("kirkus_indie", "Turnaround times / due dates"):               ("service_or_logistics", "platform", "transact", "drop"),
    ("kirkus_indie", "Publication control / confidentiality"):      ("meta_artifact", "consumer", "distribute", "drop"),
    ("kirkus_indie", "Reviewer qualifications and pool size"):      ("actor_attribute", "platform", "describe", "drop"),
    ("kirkus_indie", "Accepted formats and language scope"):        ("submission_form", "platform", "constrain", "borderline"),  # borderline keep
    ("aaai_23",      "Two-phase review: Phase 1"):                  ("service_or_logistics", "gatekeeper", "transact", "borderline"),
    ("mpep_chapter_2100", "Two criteria for subject matter"):       ("work", "producer", "constrain", "keep"),
    ("aaas_communicating", "Define your audience"):                 ("production_process", "producer", "produce", "keep"),
    ("aaai_23", "Two-phase review: Phase 2"):                       ("service_or_logistics", "gatekeeper", "transact", "drop"),
    ("aaai_23", "First-page content and anonymity"):                ("submission_form", "producer", "constrain", "keep"),
    ("mpep_chapter_2100", "Statutory category requirement"):        ("work", "producer", "constrain", "keep"),
    ("mpep_chapter_2100", "Step 2B"):                               ("evaluation_judgment", "evaluator", "judge", "keep"),
    ("aaas_communicating", "Apply the goal-audience-message"):      ("production_process", "producer", "produce", "keep"),
}


def load_test_rubrics():
    """Load 15 hand-labeled + 5 fresh rubrics."""
    import pandas as pd
    df = pd.read_parquet(ROOT / "notebooks/_explore_cache/rubrics.parquet")
    out = []
    # Match each hand-label to a real rubric row
    for (page_match, name_match), gt in GROUND_TRUTH.items():
        sub = df[df['page_id'].str.contains(page_match, case=False, na=False)
                 & df['rubric_name'].str.contains(name_match, case=False, na=False)]
        if len(sub) == 0:
            print(f"  GT NOT MATCHED: {page_match} / {name_match}")
            continue
        r = sub.iloc[0]
        out.append({
            "task": r["task"], "page_id": r["page_id"], "subtask_short": r["subtask_short"],
            "rubric_name": r["rubric_name"], "rubric_description": r["rubric_description"],
            "rubric_guidance": r["rubric_guidance"], "ground_truth": gt,
        })
    # 5 fresh from untested tasks: pick diverse rubrics
    FRESH_PICKS = [
        ("humor", "sarah_silverman_ironic_persona", "Clear ironic persona"),     # KEEP expected
        ("code-review", "phase2_110_developer_hashicorp_com_terraform", "Resources should represent a single API object"),  # KEEP expected
        ("legal-outcome-prediction", "scotus_rule_24_lii", "Citations to the appendix"),  # KEEP expected
        ("press-releases", "sec_regulation_fd_ecfr", "Definition of \"intentional\" disclosure"),  # KEEP expected
        ("news-homepages", "waveh1_6badfa426042", "IFCN badge usage"),    # likely DROP (about meta-artifact / org status)
    ]
    for task, page_match, name_match in FRESH_PICKS:
        sub = df[(df['task']==task) & df['page_id'].str.contains(page_match, case=False, na=False)
                 & df['rubric_name'].str.contains(name_match, case=False, na=False)]
        if len(sub) == 0:
            print(f"  FRESH NOT MATCHED: {task} / {page_match} / {name_match}")
            continue
        r = sub.iloc[0]
        out.append({
            "task": r["task"], "page_id": r["page_id"], "subtask_short": r["subtask_short"],
            "rubric_name": r["rubric_name"], "rubric_description": r["rubric_description"],
            "rubric_guidance": r["rubric_guidance"], "ground_truth": None,
        })
    return out


async def call_classifier(client, item: dict) -> dict:
    user_msg = (
        f"PAGE CONTEXT:\n"
        f"  task: {item['task']}\n"
        f"  page_id: {item['page_id']}\n"
        f"  subtask_short: {item['subtask_short']}\n\n"
        f"RUBRIC TO CLASSIFY:\n"
        f"  name: {item['rubric_name']}\n"
        f"  description: {item['rubric_description']}\n"
        f"  guidance: {item['rubric_guidance']}\n"
    )
    t0 = time.perf_counter()
    try:
        resp = await client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_CLASSIFY},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_schema", "json_schema": JSON_SCHEMA_CLASSIFY},
        )
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "elapsed_s": time.perf_counter() - t0}
    elapsed = time.perf_counter() - t0
    try:
        parsed = json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {"ok": False, "error": f"json_parse: {e}", "raw": resp.choices[0].message.content, "elapsed_s": elapsed}
    return {"ok": True, "extracted": parsed, "elapsed_s": elapsed,
            "input_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "output_tokens": resp.usage.completion_tokens if resp.usage else 0}


async def main():
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=OPENAI_KEY)
    items = load_test_rubrics()
    print(f"loaded {len(items)} test rubrics ({sum(1 for i in items if i['ground_truth'])} hand-labeled, {sum(1 for i in items if not i['ground_truth'])} fresh)")

    n_correct_target = n_correct_action = n_correct_keep = 0
    n_total_gt = 0
    results = []

    for i, item in enumerate(items):
        print(f"\n[{i:>2}] {item['task']:<22s}  {item['rubric_name'][:60]}")
        r = await call_classifier(client, item)
        out_path = OUT_DIR / f"{i:02d}_{item['rubric_name'].replace('/','_')[:40]}.json"
        out_path.write_text(json.dumps({"input": item, "output": r}, indent=2, default=str))
        if not r["ok"]:
            print(f"  !! ERROR: {r['error']}")
            results.append({"i": i, "name": item['rubric_name'], "error": r['error']})
            continue
        ex = r["extracted"]
        gt = item.get("ground_truth")
        line = f"  → t={ex['target']:<22s}  a={ex['actor']:<12s}  act={ex['action']:<10s}  k={ex['keep']}"
        if gt:
            n_total_gt += 1
            t_ok = ex['target'] == gt[0]
            a_ok = ex['action'] == gt[2]
            k_ok = ex['keep'] == gt[3]
            n_correct_target += t_ok; n_correct_action += a_ok; n_correct_keep += k_ok
            line += f"   GT t={gt[0]} a={gt[2]} k={gt[3]}   {'✓' if t_ok else '✗'}T {'✓' if a_ok else '✗'}A {'✓' if k_ok else '✗'}K"
        print(line)
        print(f"     just: {ex.get('justification','')[:140]}")
        results.append({"i": i, "name": item['rubric_name'], "task": item['task'],
                        "target": ex['target'], "actor": ex['actor'], "action": ex['action'],
                        "keep": ex['keep'], "ground_truth": gt})

    print(f"\n========== AGREEMENT WITH HAND-LABELS (n={n_total_gt}) ==========")
    print(f"  target agreement: {n_correct_target}/{n_total_gt} ({100*n_correct_target/n_total_gt:.0f}%)")
    print(f"  action agreement: {n_correct_action}/{n_total_gt} ({100*n_correct_action/n_total_gt:.0f}%)")
    print(f"  keep   agreement: {n_correct_keep}/{n_total_gt} ({100*n_correct_keep/n_total_gt:.0f}%)")


if __name__ == "__main__":
    asyncio.run(main())
