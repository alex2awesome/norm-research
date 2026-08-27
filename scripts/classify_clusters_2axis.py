"""Two-axis classification of rubric clusters (R1-refined children = dedup units).

Operates at the CLUSTER level, not the merged_group level: merged_groups washed
out to a near-constant articulability=3 because they blend thin + thick leaves.
A cluster (R1 child) is the dedup-clustering output — one deduped rubric concept,
carrying a medoid name + description.

Axis 1 — Articulability (1-4): how hard the criterion is to apply, on the
  nested spectrum code-checkable -> language-tacit -> defensible-judgment -> tacit.
Axis 2 — Surface-vs-substance (1-4): whether the criterion governs form or content.

CALIBRATION-MODE script for prompt workshopping:
  python scripts/classify_clusters_2axis.py --sample 33 --tag v2
  python scripts/classify_clusters_2axis.py --print --tag v2

Output: outputs/analyses/twoaxis_calib_<tag>.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from collections import Counter
from pathlib import Path

from openai import AsyncOpenAI

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
HIER = ROOT / "outputs" / "hierarchy"
OUT = ROOT / "outputs" / "analyses"
KEY_PATH = Path("/Users/spangher/.openai-salt-lab-key.txt")

MODEL = "gpt-5-mini"
SAMPLE_SEED = 14

TASKS = [
    "code-review", "creative-writing", "grant-funding", "humor",
    "legal-outcome-prediction", "math-stackexchange", "news-homepages",
    "notice-and-comment", "patents", "peer-review", "press-releases",
]

CLASSIFY_SCHEMA = {
    "name": "rubric_classification",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        # Property order matters: the model generates fields in schema order, so
        # the two diagnostic components come first and articulability last —
        # decompose-then-aggregate.
        "properties": {
            "reasoning_depth": {"type": "integer", "minimum": 1, "maximum": 4},
            "reasoning_depth_why": {"type": "string"},
            "indeterminacy": {"type": "integer", "minimum": 1, "maximum": 4},
            "indeterminacy_why": {"type": "string"},
            "surface_vs_substance": {"type": "integer", "minimum": 1, "maximum": 4},
            "surface_why": {"type": "string"},
            "articulability": {"type": "integer", "minimum": 1, "maximum": 4},
            "articulability_why": {"type": "string"},
        },
        "required": ["reasoning_depth", "reasoning_depth_why",
                     "indeterminacy", "indeterminacy_why",
                     "surface_vs_substance", "surface_why",
                     "articulability", "articulability_why"],
    },
}

SYSTEM_PROMPT = """You classify one evaluation criterion on four ordinal axes.

The criterion is a deduped rubric concept from an evaluation taxonomy (e.g. "the methods section reports a power analysis"). You see its name, description, and — if it bundled near-duplicates — a few source rubrics.

Classify in this order. The first two are DIAGNOSTIC COMPONENTS; assess them first because they ground the final ARTICULABILITY call. Higher always means harder / more resistant to capture.

=== 1. REASONING_DEPTH (1-4) ===
ASSUMING the criterion is perfectly clear, how much inference does APPLYING it take?
  1 = MECHANICAL. Count, match, look up, compare values. No interpretation.
      "Abstract is under 250 words." "No comma splices."
  2 = SHALLOW-SEMANTIC. One step of interpretation: classify by type, locate a section, judge a surface property.
      "Every numerical claim has a citation." "The register is formal."
  3 = INFERENCE. Relate multiple parts, evaluate an argument, weigh evidence, synthesize across a document.
      "The argument addresses the strongest counter-claim." "Conclusions follow from the results."
  4 = DEEP EXPERT REASONING / HOLISTIC. Sustained domain reasoning, or apprehending the whole work at once as a gestalt.
      "The proof is elegant." "The piece has a distinctive voice."

=== 2. INDETERMINACY (1-4) ===
INDEPENDENT of how hard it is to apply — how fully does the criterion SPECIFY what it is asking?

CRITICAL: mere threshold / degree vagueness — "how much is ENOUGH" — is UNIVERSAL. Every gradable criterion has it ("concrete ENOUGH", "substantive ENOUGH", "clear ENOUGH", "ruthless ENOUGH editing"). Threshold vagueness does NOT count as indeterminacy — it is level 2. Only STRUCTURAL under-specification counts: a parameter that changes WHAT QUESTION is asked, or a contested core concept with no agreed definition.

  1 = FULLY SPECIFIED. Agreed terms, no free parameters, not even a degree threshold.
      "Use Oxford commas." "Cite a source for each numerical claim."
  2 = THRESHOLD-VAGUE ONLY. The core concept is clear; only "how much is enough" is open. THIS IS THE NORMAL CASE for any gradable criterion — most criteria are level 2.
      "Provide concrete detail." "Substantive, not boilerplate, quotes." "Tight, well-edited prose." "Acknowledge intellectual debts."
  3 = STRUCTURAL FREE PARAMETER. The criterion asks a genuinely DIFFERENT question depending on something it leaves unstated; fixing that thing changes what is being evaluated.
      "Conform to the local court's norms" (which court?). "Tailor tone to the audience" (when the audience is genuinely unstated).
  4 = CONTESTED CORE CONCEPT. The central term has no agreed definition; experts dispute what it even means.
      "Avoid stereotypes." "The piece has voice." "The proof is elegant." "The connection is unexpected."

TEST for 3 vs 2: would two competent people be asking DIFFERENT QUESTIONS (3), or the SAME question and only drawing the pass/fail line in slightly different places (2)? Different question → 3. Same question, different threshold → 2.

=== 3. SURFACE_VS_SUBSTANCE (1-4) ===
What does the criterion GOVERN — form or content? Independent of the other axes.
  1 = PURE SURFACE / FORM. Syntax, formatting, mechanics, layout. "Use Oxford commas."
  2 = MOSTLY SURFACE, SOME SUBSTANCE. A form rule whose point is to constrain meaning. "Headlines title-case AND accurate."
  3 = MOSTLY SUBSTANCE, SOME FORM. A content rule with formal scaffolding. "Methods section reports a power analysis."
  4 = PURE SUBSTANCE. Meaning, validity, truth, soundness. "The thesis is falsifiable."

=== 4. ARTICULABILITY (1-4) — THE PRIMARY AXIS ===
This axis is about how the criterion can be CONVEYED and SHARED: how codifiable it is, how much description + tacit knowledge transmitting it takes, and whether trained experts converge on it. A criterion's use of technical vocabulary does NOT by itself set its level — sort it with the four tests below.

  1 = CODIFIABLE. A reasonably large code library could reliably identify this feature, with minimal disagreement between independent implementations. The topic may be complex or jargon-heavy — that does not matter — what matters is that the criterion reduces to something a program computes (near-)deterministically. Includes counts/lengths, fixed lexical lists, format/structure presence, citation mechanics, near-duplicate detection, AND complex-but-defined computations (readability formulas, cyclomatic complexity, type-token ratio, "the statistical test matches the data type", "the citation graph has a cycle").
      TEST: could a competent team, given the rubric, write a library function resolving it with minimal disagreement? Yes → 1, even if the rubric is technical.
      Typically reasoning_depth 1.
  2 = BRIEFLY ARTICULABLE. Not codifiable, but you can describe the feature in a reasonable number of words such that a NON-EXPERT understands what to look for and can apply it, drawing on only LOW tacit knowledge. Below a moderate complexity threshold.
      "Every numerical claim cites a source." "The tone suits a general audience." "Headlines are not clickbait." "The comment stays on the topic of the rule."
      Typically reasoning_depth 2-3, indeterminacy 1-2.
  3 = DENSE / APPRENTICESHIP-ARTICULABLE. The feature CAN be described in principle, but a faithful description would take too many words and rests on too much tacit knowledge — a non-expert could not reliably apply it from any reasonable description; it takes a deep apprenticeship (training, sustained practice) to internalize. CRUCIALLY, trained experts who HAVE that apprenticeship DO converge — the criterion is intersubjective and shared, not personal. Dense and semi-inarticulable, but real and agreed.
      "Whether a story's pacing works." "Whether a legal brief is genuinely persuasive." "Whether a mathematical proof is rigorous." "Whether a patent claim is definite (its scope clear in the patent-law sense)." "Whether a comedic act sustains a coherent persona."
      TEST: would conveying this to a capable non-expert need an apprenticeship rather than a paragraph — AND would two well-trained experts converge? Both yes → 3.
      Typically reasoning_depth 3-4.
  4 = PERSONAL / INARTICULABLE / NOISY. So tacit, and so shot through with personal taste, that even well-trained experts each given the best possible brief would NOT reliably agree. Description largely fails; personalization dominates; the signal is noisy. This exists in EVERY field — wherever applying the criterion comes down to individual taste: an elegant proof, a moving passage, a "valuable" or "important" invention, a joke that lands "for me", a beautiful turn of phrase. Do NOT confine level 4 to creative domains — formal/technical fields have personal-taste criteria too.
      TEST: would two well-trained experts, each fully briefed, still disagree because it comes down to personal taste? Yes → 4.
      Typically reasoning_depth 4 and/or indeterminacy 4.

The two hard boundaries:
- 2 vs 3 — APPRENTICESHIP: can a non-expert apply it from a reasonable written description (→2), or does internalizing it require a deep apprenticeship (→3)? The length and tacit-knowledge depth of a faithful description is the test — NOT whether the topic is technical.
- 3 vs 4 — EXPERT CONVERGENCE: dense but trained experts AGREE → 3; dense and trained experts DISAGREE because personal taste dominates → 4.

=== WORKED EXAMPLES (all four axes) ===

"Use Oxford commas"
  reasoning_depth=1 · indeterminacy=1 · surface_vs_substance=1 · articulability=1

"Cyclomatic complexity stays under 10"
  reasoning_depth=1 · indeterminacy=1 · surface_vs_substance=2 · articulability=1
  ("cyclomatic complexity" is jargon, but it is a defined, computable metric — a library resolves it deterministically. CODIFIABLE → 1. Jargon does not gate this.)

"Every numerical claim cites a source"
  reasoning_depth=2 · indeterminacy=1 · surface_vs_substance=3 · articulability=2
  (Briefly describable to a non-expert, low tacit knowledge. Not purely codifiable — matching "claim" to "source" needs reading — so 2, not 1.)

"The tone suits the intended audience"
  reasoning_depth=2 · indeterminacy=2 · surface_vs_substance=3 · articulability=2
  (A non-expert understands and can apply this from a sentence. Briefly articulable → 2.)

"The story's pacing works" (scenes and beats sustain momentum)
  reasoning_depth=3 · indeterminacy=2 · surface_vs_substance=3 · articulability=3
  (Conveying what good pacing IS takes far more than a paragraph plus a developmental editor's apprenticeship to internalize — but trained editors DO converge. Dense, intersubjective → 3.)

"The patent claim is definite (scope clear and unambiguous in the §112(b) sense)"
  reasoning_depth=3 · indeterminacy=2 · surface_vs_substance=3 · articulability=3
  (Genuinely needs patent-law apprenticeship — a non-expert cannot apply "definiteness" from a description — but trained patent examiners converge. Apprenticeship + expert agreement → 3. The jargon is not the reason; the apprenticeship is.)

"The mathematical proof is rigorous"
  reasoning_depth=4 · indeterminacy=2 · surface_vs_substance=4 · articulability=3
  (Rigor takes a mathematician's apprenticeship to apply, but mathematicians converge on it → 3.)

"The proof is elegant"
  reasoning_depth=4 · indeterminacy=4 · surface_vs_substance=4 · articulability=4
  (Even trained mathematicians do not agree on elegance — personal taste dominates → 4.)

"The invention is valuable / important"
  reasoning_depth=3 · indeterminacy=4 · surface_vs_substance=4 · articulability=4
  (A formal-domain L4: experts genuinely disagree on what makes an invention "important" — personalization dominates. L4 is not only for creative fields.)

"The piece has a distinctive voice"
  reasoning_depth=4 · indeterminacy=4 · surface_vs_substance=3 · articulability=4
  (Personal, experts diverge → 4.)

"Conform to the local court's norms (some judges require footnotes, others forbid them)"
  reasoning_depth=2 · indeterminacy=3 · surface_vs_substance=1 · articulability=3
  (STRUCTURAL free parameter — the criterion asks a different question per court. Shallow once the court is fixed, but the unstated parameter → 3.)

"The argument addresses the strongest counter-claim"
  reasoning_depth=3 (identify counter-claims, judge engagement) · indeterminacy=2 · surface_vs_substance=4 · articulability=3
  (Domain reasoning to find the strongest counter-claim and judge engagement — expertise-gated.)

"Avoid stereotypes"
  reasoning_depth=2 · indeterminacy=4 ("stereotype" is a contested core concept) · surface_vs_substance=4 · articulability=3
  (Contested concept — but it has a working operational definition raters can be briefed on, so application DOES converge → 3, not 4.)

"News reporting is objective"  [contested concept -> still 3]
  reasoning_depth=3 · indeterminacy=4 ("objectivity" is contested — the field debates whether it is even achievable) · surface_vs_substance=4 · articulability=3
  (Contested AS A CONCEPT, but it has working operational proxies — source balance, fact/opinion separation, loaded-language absence. Brief raters on those and they converge. The contestation is meta-level; application is convergent → 3, NOT 4.)

"The connection drawn is genuinely unexpected / surprising"  [3-vs-4 BOUNDARY]
  reasoning_depth=3 · indeterminacy=4 · surface_vs_substance=4 · articulability=4
  (Contrast with "Avoid stereotypes": BOTH have a contested core concept (indeterminacy 4). But "unexpected/surprising" is AESTHETIC — briefing does NOT produce convergence; what astonishes one expert is obvious to another, and no shared definition fixes that. So → 4, not 3. The 3-vs-4 line is: does a shared brief make raters converge? Stereotype: yes → 3. Surprise/elegance/feel: no → 4.)

"The comic's page proportions feel harmonic" / "The proof is elegant"
  reasoning_depth=4 (holistic aesthetic apprehension) · indeterminacy=4 · surface_vs_substance=2 · articulability=4
  (Even briefed experts do not converge — briefing does not fix the disagreement. Aesthetic 'feel' → 4.)

"The piece has a distinctive voice"
  reasoning_depth=4 · indeterminacy=4 · surface_vs_substance=3 · articulability=4

Output JSON: the four integer scores, each with a one-sentence reasoning. Assess reasoning_depth and indeterminacy first, then surface_vs_substance, then articulability last."""


def build_user_msg(cl: dict) -> str:
    parts = [
        f"CRITERION: {cl.get('medoid_name', '')}",
        f"DESCRIPTION: {cl.get('medoid_description', '')}",
    ]
    # If the cluster bundled >1 near-duplicate source rubric, show a couple as
    # context (they are near-dupes of the medoid, not distinct concepts).
    members = [r.get("name", "") for r in cl.get("rubrics", []) if r.get("name")]
    if len(members) > 1:
        shown = "; ".join(members[:3])
        parts.append(f"(cluster of {len(members)} near-duplicate source rubrics, e.g.: {shown})")
    parts.append("")
    parts.append("Classify this criterion on both axes.")
    return "\n".join(parts)


def sample_clusters(n: int, seed: int = SAMPLE_SEED) -> list[dict]:
    """Sample R1-refined children (dedup clusters) from the general bucket,
    stratified across tasks."""
    random.seed(seed)
    per_task = max(1, n // len(TASKS))
    out = []
    for task in TASKS:
        p = HIER / f"{task}_general_r1_refined.json"
        if not p.exists():
            continue
        clusters = []
        for par in json.loads(p.read_text()).get("parented_trees", []):
            for ch in par.get("children", []):
                clusters.append(ch)
        if not clusters:
            continue
        picks = random.sample(clusters, min(per_task, len(clusters)))
        for c in picks:
            out.append({"task": task, **c})
    return out


async def call_llm(client, user, sem, timeout_sec=120.0):
    async with sem:
        for attempt in range(3):
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user},
                        ],
                        response_format={"type": "json_schema", "json_schema": CLASSIFY_SCHEMA},
                        service_tier="flex",
                    ),
                    timeout=timeout_sec,
                )
                return json.loads(resp.choices[0].message.content or "{}")
            except asyncio.TimeoutError:
                if attempt == 2:
                    return {"_error": "timeout"}
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                if attempt == 2:
                    return {"_error": str(e)[:200]}
                await asyncio.sleep(2 ** attempt)
    return {}


async def run_sample(n: int, tag: str, seed: int = SAMPLE_SEED):
    sample = sample_clusters(n, seed)
    print(f"sampled {len(sample)} clusters (seed={seed})")
    client = AsyncOpenAI(api_key=KEY_PATH.read_text().strip())
    sem = asyncio.Semaphore(50)

    async def one(cl):
        res = await call_llm(client, build_user_msg(cl), sem)
        return {
            "task": cl["task"],
            "medoid_name": cl.get("medoid_name", ""),
            "medoid_description": cl.get("medoid_description", ""),
            "cluster_size": len(cl.get("rubrics", [])),
            "result": res,
        }

    t0 = time.perf_counter()
    results = await asyncio.gather(*[one(mg) for mg in sample])
    print(f"done in {time.perf_counter()-t0:.0f}s")

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"twoaxis_calib_{tag}.jsonl"
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out_path}")
    return out_path


def pretty_print(tag: str):
    path = OUT / f"twoaxis_calib_{tag}.jsonl"
    if not path.exists():
        print(f"no file {path}")
        return
    rows = [json.loads(l) for l in path.open()]
    art_dist, surf_dist, rd_dist, indet_dist = Counter(), Counter(), Counter(), Counter()
    for i, r in enumerate(rows, 1):
        res = r.get("result", {})
        if "_error" in res:
            print(f"\n[{i}] [{r['task']}] {r['medoid_name']}  ERROR: {res['_error']}")
            continue
        rd = res.get("reasoning_depth", "?")
        ind = res.get("indeterminacy", "?")
        s = res.get("surface_vs_substance", "?")
        a = res.get("articulability", "?")
        rd_dist[rd] += 1; indet_dist[ind] += 1; surf_dist[s] += 1; art_dist[a] += 1
        sz = r.get("cluster_size", "?")
        print(f"\n[{i}] [{r['task']}] {r['medoid_name']}  (cluster size {sz})")
        print(f"    desc: {(r.get('medoid_description') or '')[:140]}")
        print(f"    reasoning_depth={rd}  | {res.get('reasoning_depth_why','')[:120]}")
        print(f"    indeterminacy  ={ind}  | {res.get('indeterminacy_why','')[:120]}")
        print(f"    surface/subst  ={s}  | {res.get('surface_why','')[:120]}")
        print(f"    ARTICULABILITY ={a}  | {res.get('articulability_why','')[:120]}")
    print(f"\n{'='*70}")
    print(f"reasoning_depth dist: {dict(sorted(rd_dist.items()))}")
    print(f"indeterminacy dist:   {dict(sorted(indet_dist.items()))}")
    print(f"surface dist:         {dict(sorted(surf_dist.items()))}")
    print(f"ARTICULABILITY dist:  {dict(sorted(art_dist.items()))}")


FULL_OUT = OUT / "cluster_2axis_full.jsonl"


def all_clusters() -> list[dict]:
    """Enumerate every general-bucket cluster with a stable uid.
    uid = {task}::p{parent_idx}::c{child_idx} — deterministic from the
    static R1-refined files, so the run is resumable."""
    out = []
    for task in TASKS:
        p = HIER / f"{task}_general_r1_refined.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for pi, par in enumerate(d.get("parented_trees", [])):
            for ci, ch in enumerate(par.get("children", [])):
                out.append({"cluster_uid": f"{task}::p{pi}::c{ci}", "task": task, **ch})
    return out


def load_done_uids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open() as f:
        for line in f:
            try:
                done.add(json.loads(line)["cluster_uid"])
            except Exception:
                continue
    return done


async def run_full(concurrency: int = 200):
    OUT.mkdir(parents=True, exist_ok=True)
    clusters = all_clusters()
    done = load_done_uids(FULL_OUT)
    todo = [c for c in clusters if c["cluster_uid"] not in done]
    print(f"{len(clusters)} general-bucket clusters; {len(done)} done; {len(todo)} to classify")
    if not todo:
        print("nothing to do")
        return

    client = AsyncOpenAI(api_key=KEY_PATH.read_text().strip())
    sem = asyncio.Semaphore(concurrency)
    t0 = time.perf_counter()

    with FULL_OUT.open("a") as fp:
        async def one(c):
            res = await call_llm(client, build_user_msg(c), sem, timeout_sec=300.0)
            rec = {
                "cluster_uid": c["cluster_uid"],
                "task": c["task"],
                "medoid_name": c.get("medoid_name", ""),
                "medoid_description": c.get("medoid_description", ""),
                "cluster_size": len(c.get("rubrics", [])),
                "result": res,
            }
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fp.flush()
            return rec

        coros = [one(c) for c in todo]
        for i, fut in enumerate(asyncio.as_completed(coros)):
            await fut
            if (i + 1) % 500 == 0 or i == 0:
                el = time.perf_counter() - t0
                rate = (i + 1) / el
                eta = (len(todo) - i - 1) / max(rate, 0.001)
                print(f"  {i+1}/{len(todo)}  rate={rate:.1f}/s  eta={eta:.0f}s")

    print(f"done in {time.perf_counter()-t0:.0f}s")
    # summary
    rows = [json.loads(l) for l in FULL_OUT.open()]
    n_err = sum(1 for r in rows if "_error" in r.get("result", {}))
    art = Counter(r["result"].get("articulability") for r in rows if "_error" not in r.get("result", {}))
    print(f"total {len(rows)}, errors {n_err}")
    print(f"articulability dist: {dict(sorted((k, v) for k, v in art.items() if k is not None))}")


RECLASS_OUT = OUT / "cluster_2axis_l3_v7.jsonl"


def load_l3_clusters() -> list[dict]:
    """All articulability=3 clusters from the v6 full run, deduped by name+desc."""
    seen, out = set(), []
    for line in FULL_OUT.open():
        try:
            r = json.loads(line)
        except Exception:
            continue
        res = r.get("result", {})
        if "_error" in res or not res or res.get("articulability") != 3:
            continue
        key = (r.get("medoid_name", ""), r.get("medoid_description", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "task": r["task"],
            "medoid_name": r.get("medoid_name", ""),
            "medoid_description": r.get("medoid_description", ""),
            "cluster_size": r.get("cluster_size"),
            "v6_articulability": 3,
        })
    return out


def load_done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    for line in path.open():
        try:
            r = json.loads(line)
            done.add((r.get("medoid_name", ""), r.get("medoid_description", "")))
        except Exception:
            continue
    return done


async def run_reclassify_l3(concurrency: int = 200):
    """Re-judge every v6 articulability=3 cluster with the (v7) prompt — the
    audit found L3 systematically inflated by a technical-vocabulary confound."""
    OUT.mkdir(parents=True, exist_ok=True)
    clusters = load_l3_clusters()
    done = load_done_keys(RECLASS_OUT)
    todo = [c for c in clusters if (c["medoid_name"], c["medoid_description"]) not in done]
    print(f"{len(clusters)} v6-L3 clusters; {len(done)} done; {len(todo)} to reclassify with v7")
    if not todo:
        print("nothing to do")
        return
    client = AsyncOpenAI(api_key=KEY_PATH.read_text().strip())
    sem = asyncio.Semaphore(concurrency)
    t0 = time.perf_counter()
    with RECLASS_OUT.open("a") as fp:
        async def one(c):
            res = await call_llm(client, build_user_msg(c), sem, timeout_sec=300.0)
            rec = {**{k: c[k] for k in ("task", "medoid_name", "medoid_description",
                                        "cluster_size", "v6_articulability")},
                   "result": res}
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fp.flush()
            return rec
        for i, fut in enumerate(asyncio.as_completed([one(c) for c in todo])):
            await fut
            if (i + 1) % 500 == 0 or i == 0:
                el = time.perf_counter() - t0
                print(f"  {i+1}/{len(todo)}  rate={(i+1)/el:.1f}/s  eta={(len(todo)-i-1)/max((i+1)/el,0.001):.0f}s")
    print(f"done in {time.perf_counter()-t0:.0f}s")
    rows = [json.loads(l) for l in RECLASS_OUT.open()]
    moved = Counter(r["result"].get("articulability") for r in rows if "_error" not in r.get("result", {}))
    print(f"v7 re-judgement of {len(rows)} v6-L3 clusters: {dict(sorted((k,v) for k,v in moved.items() if k))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--seed", type=int, default=SAMPLE_SEED)
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--full", action="store_true", help="classify all general-bucket clusters (resumable)")
    ap.add_argument("--reclassify-l3", action="store_true", help="re-judge v6 L3 clusters with v7 prompt")
    ap.add_argument("--concurrency", type=int, default=200)
    args = ap.parse_args()
    if args.reclassify_l3:
        asyncio.run(run_reclassify_l3(args.concurrency))
    elif args.full:
        asyncio.run(run_full(args.concurrency))
    elif args.sample:
        asyncio.run(run_sample(args.sample, args.tag, args.seed))
        pretty_print(args.tag)
    elif args.print:
        pretty_print(args.tag)
    else:
        print("specify --sample N, --print, or --full")


if __name__ == "__main__":
    main()
