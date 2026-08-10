"""
Test the per-task Llama-3.3 classifier on the same 19 rubrics used for
gpt-5-mini v2 (15 hand-labeled + 4 fresh) so we can compare head-to-head.
"""

from __future__ import annotations
import asyncio, json, re, sys, time
from pathlib import Path

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
sys.path.insert(0, str(ROOT / "scripts"))

from classify_rubric_llama_prompt import build_prompt_for_task

OPENROUTER_KEY = (Path.home() / ".openrouter-api-key.txt").read_text().strip()
OUT_DIR = ROOT / "logs/rubric_labeling/classifier_llama_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Same hand-labels and fresh picks as test_classifier_v2.py
GROUND_TRUTH = {
    ("kirkus_indie", "How Kirkus chooses a reviewer"):              ("service_or_logistics", "platform", "describe", "drop"),
    ("kirkus_indie", "Honest and impartial evaluation"):            ("evaluation_judgment", "evaluator", "judge", "keep"),
    ("kirkus_indie", "Review lengths by option"):                   ("meta_artifact", "platform", "constrain", "drop"),
    ("kirkus_indie", "Turnaround times / due dates"):               ("service_or_logistics", "platform", "transact", "drop"),
    ("kirkus_indie", "Publication control / confidentiality"):      ("meta_artifact", "consumer", "distribute", "drop"),
    ("kirkus_indie", "Reviewer qualifications and pool size"):      ("actor_attribute", "platform", "describe", "drop"),
    ("kirkus_indie", "Accepted formats and language scope"):        ("submission_form", "platform", "constrain", "borderline"),
    ("aaai_23",      "Two-phase review: Phase 1"):                  ("service_or_logistics", "gatekeeper", "transact", "borderline"),
    ("mpep_chapter_2100", "Two criteria for subject matter"):       ("work", "producer", "constrain", "keep"),
    ("aaas_communicating", "Define your audience"):                 ("production_process", "producer", "produce", "keep"),
    ("aaai_23", "Two-phase review: Phase 2"):                       ("service_or_logistics", "gatekeeper", "transact", "drop"),
    ("aaai_23", "First-page content and anonymity"):                ("submission_form", "producer", "constrain", "keep"),
    ("mpep_chapter_2100", "Statutory category requirement"):        ("work", "producer", "constrain", "keep"),
    ("mpep_chapter_2100", "Step 2B"):                               ("evaluation_judgment", "evaluator", "judge", "keep"),
    ("aaas_communicating", "Apply the goal-audience-message"):      ("production_process", "producer", "produce", "keep"),
}

FRESH_PICKS = [
    ("humor", "sarah_silverman_ironic_persona", "Clear ironic persona"),
    ("code-review", "phase2_110_developer_hashicorp_com_terraform", "Resources should represent a single API object"),
    ("legal-outcome-prediction", "scotus_rule_24_lii", "Citations to the appendix"),
    ("press-releases", "sec_regulation_fd_ecfr", "Definition of \"intentional\" disclosure"),
    ("news-homepages", "waveh1_6badfa426042", "IFCN badge usage"),
]


def salvage_json(raw: str):
    if not raw or not raw.strip(): return None
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try: return json.loads(s)
    except: pass
    start = s.find("{")
    if start < 0: return None
    for end in range(len(s), start, -1):
        if s[end-1] == "}":
            try: return json.loads(s[start:end])
            except: continue
    return None


def load_test_rubrics():
    import pandas as pd
    df = pd.read_parquet(ROOT / "notebooks/_explore_cache/rubrics.parquet")
    out = []
    for (page_match, name_match), gt in GROUND_TRUTH.items():
        sub = df[df['page_id'].str.contains(page_match, case=False, na=False)
                 & df['rubric_name'].str.contains(name_match, case=False, na=False)]
        if len(sub) == 0:
            print(f"  GT NOT MATCHED: {page_match} / {name_match}"); continue
        r = sub.iloc[0]
        out.append({"task": r["task"], "page_id": r["page_id"], "subtask_short": r["subtask_short"],
                    "rubric_name": r["rubric_name"], "rubric_description": r["rubric_description"],
                    "rubric_guidance": r["rubric_guidance"], "ground_truth": gt})
    for task, page_match, name_match in FRESH_PICKS:
        sub = df[(df['task']==task) & df['page_id'].str.contains(page_match, case=False, na=False)
                 & df['rubric_name'].str.contains(name_match, case=False, na=False)]
        if len(sub) == 0: print(f"  FRESH NOT MATCHED: {task} / {name_match}"); continue
        r = sub.iloc[0]
        out.append({"task": r["task"], "page_id": r["page_id"], "subtask_short": r["subtask_short"],
                    "rubric_name": r["rubric_name"], "rubric_description": r["rubric_description"],
                    "rubric_guidance": r["rubric_guidance"], "ground_truth": None})
    return out


async def call_classifier(client, item, pin_provider=False):
    sys_prompt = build_prompt_for_task(item['task'])
    user_msg = (
        f"PAGE CONTEXT:\n  task: {item['task']}\n  page_id: {item['page_id']}\n  subtask_short: {item['subtask_short']}\n\n"
        f"RUBRIC TO CLASSIFY:\n  name: {item['rubric_name']}\n  description: {item['rubric_description']}\n  guidance: {item['rubric_guidance']}\n"
    )
    extra = {}
    if pin_provider:
        extra["extra_body"] = {"provider": {"order": ["fireworks", "together", "deepinfra"], "allow_fallbacks": False}}
    t0 = time.perf_counter()
    try:
        resp = await client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct",
            messages=[{"role":"system","content":sys_prompt},{"role":"user","content":user_msg}],
            temperature=0.0, max_tokens=1024,
            response_format={"type": "json_object"},
            **extra,
        )
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "elapsed_s": time.perf_counter()-t0}
    elapsed = time.perf_counter() - t0
    raw = resp.choices[0].message.content or ""
    parsed = salvage_json(raw)
    if parsed is None:
        return {"ok": False, "error": "json_salvage_failed", "raw": raw, "elapsed_s": elapsed}
    return {"ok": True, "extracted": parsed, "elapsed_s": elapsed,
            "in_tok": resp.usage.prompt_tokens if resp.usage else 0,
            "out_tok": resp.usage.completion_tokens if resp.usage else 0}


async def main():
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY)
    items = load_test_rubrics()
    print(f"loaded {len(items)} test rubrics ({sum(1 for i in items if i['ground_truth'])} hand-labeled, {sum(1 for i in items if not i['ground_truth'])} fresh)")

    n_correct_target = n_correct_action = n_correct_keep = 0
    n_total_gt = 0
    tot_in = tot_out = 0

    for i, item in enumerate(items):
        print(f"\n[{i:>2}] {item['task']:<22s}  {item['rubric_name'][:60]}")
        r = await call_classifier(client, item, pin_provider=False)
        if not r["ok"]:
            print(f"  retry pinned ({r['error']})...")
            r = await call_classifier(client, item, pin_provider=True)
        out_path = OUT_DIR / f"{i:02d}_{item['rubric_name'].replace('/','_')[:40]}.json"
        out_path.write_text(json.dumps({"input": item, "output": r}, indent=2, default=str))
        if not r["ok"]:
            print(f"  !! ERROR: {r['error']}")
            continue
        ex = r["extracted"]
        # Llama might omit fields — be defensive
        target = ex.get('target','?'); actor = ex.get('actor','?'); action = ex.get('action','?'); keep = ex.get('keep','?')
        gt = item.get("ground_truth")
        line = f"  → t={target:<22s}  a={actor:<12s}  act={action:<10s}  k={keep}"
        if gt:
            n_total_gt += 1
            t_ok = target == gt[0]; a_ok = action == gt[2]; k_ok = keep == gt[3]
            n_correct_target += t_ok; n_correct_action += a_ok; n_correct_keep += k_ok
            line += f"   GT t={gt[0]} a={gt[2]} k={gt[3]}   {'✓' if t_ok else '✗'}T {'✓' if a_ok else '✗'}A {'✓' if k_ok else '✗'}K"
        print(line)
        print(f"     just: {ex.get('justification','')[:140]}")
        tot_in += r.get("in_tok",0); tot_out += r.get("out_tok",0)

    print(f"\n========== Llama-3.3-70B classifier — agreement (n={n_total_gt}) ==========")
    print(f"  target: {n_correct_target}/{n_total_gt} ({100*n_correct_target/n_total_gt:.0f}%)")
    print(f"  action: {n_correct_action}/{n_total_gt} ({100*n_correct_action/n_total_gt:.0f}%)")
    print(f"  keep  : {n_correct_keep}/{n_total_gt} ({100*n_correct_keep/n_total_gt:.0f}%)")
    print(f"\ntokens: in={tot_in:,} out={tot_out:,}; ~ ${(tot_in*0.12 + tot_out*0.30)/1e6:.4f} per 19-rubric run")
    print(f"projected for 361K rubrics: ~ ${(tot_in*0.12 + tot_out*0.30)/1e6 * 361_000/19:.0f}")


if __name__ == "__main__":
    asyncio.run(main())
