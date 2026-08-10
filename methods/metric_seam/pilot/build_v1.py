"""Seam pilot v1 — corrected + robust build.

Fixes over v0:
 F1  SAME canonical text for BOTH channels (v0 bug: code saw full text, LLM saw first-8k —
     apples-to-apples violation; contact blocks live in the tail).
 F2  head+tail truncation so footers (contacts, boilerplate) survive.
 F3  two judge passes (original + paraphrased template) -> per-aspect reliability.
 F4  document-scoping channel ("is this actually a press release?").
 F5  counterfactual probe for a86: inject a generic excited-CEO quote into quote-free items;
     presence-proxy code should jump, a quality judge should not.
"""
import json, random, pathlib, statistics as st

ROOT = pathlib.Path(__file__).resolve().parents[3]
V2 = ROOT / "runs/validity_full/v2/press_releases"
V0 = ROOT / "outputs/metric_seam_pilot"
OUT = V0 / "v1"
OUT.mkdir(parents=True, exist_ok=True)

ASPECTS = ["a79", "a80", "a110", "a100", "a101", "a86", "a105", "a118", "a117", "a73"]
HEAD, TAIL = 5000, 2500
SEED = 0

T1 = """You are an experienced newswire editor grading ONE press release against ONE criterion.

Criterion: {name}
Definition: {description}

Press release (may be truncated; '[...]' marks an elided middle):
<press_release>
{text}
</press_release>

Grade how well the press release satisfies this criterion on a 0-10 integer scale \
(0 = completely fails, 10 = exemplary). If the criterion genuinely does not apply to this \
kind of release, answer NA. Reply with exactly one line in the form:
SCORE: <integer 0-10 or NA>"""

T2 = """As a senior wire-service editor, evaluate the document below on a single quality \
criterion, ignoring all other qualities.

The criterion — {name}: {description}

Document ('[...]' = elided middle):
<press_release>
{text}
</press_release>

Give an integer 0-10 (0 = criterion badly violated or absent, 10 = criterion fully exemplified). \
If this criterion simply cannot apply to a document of this kind, answer NA. \
Your entire reply must be one line:
SCORE: <integer 0-10 or NA>"""

TSCOPE = """Look at the following text scraped from the web.

<text>
{text}
</text>

Is this an actual press release (a statement issued by an organization announcing its own news, \
possibly via a wire service), as opposed to a news article, blog post, product/marketing page, \
website navigation chrome, archive listing, or other content? Reply with exactly one line:
SCORE: <integer 0-10>  (0 = clearly NOT a press release, 10 = clearly a press release)"""

INJECT_QUOTE = ('"We are thrilled and excited about this important milestone, and we look '
                'forward to continuing our journey," said John Smith, Chief Executive Officer.')


def canonical(text):
    if len(text) <= HEAD + TAIL + 500:
        return text
    return text[:HEAD] + "\n[...]\n" + text[-TAIL:]


def main():
    aspects = {x["aspect_id"]: x for x in json.load(open(V2 / "aspects.json"))}
    items = json.load(open(V0 / "items.json"))          # same 250 items as v0
    for it in items:
        it["ctext"] = canonical(it["text"])
    json.dump(items, open(OUT / "items_v1.json", "w"))

    # F5: pick CF items from v0 results (a86 judge <=2, v0 keyword code below its median)
    v0_llm, v0_code = {}, json.load(open(V0 / "code_scores.json"))["a86_v0_keyword"]
    for line in open(V0 / "results.jsonl"):
        r = json.loads(line)
        if r["aspect_id"] == "a86" and isinstance(r["score"], int):
            v0_llm[r["datapoint_id"]] = r["score"]
    cmed = st.median(v for v in v0_code.values() if v is not None)
    cands = [d for d, s in v0_llm.items()
             if s <= 2 and v0_code.get(d) is not None and v0_code[d] <= cmed]
    random.seed(SEED)
    cf_ids = random.sample(cands, min(30, len(cands)))
    by_id = {it["datapoint_id"]: it for it in items}
    cf_items = []
    for d in cf_ids:
        t = by_id[d]["ctext"]
        cut = t.find("\n", 1500)
        cut = cut if cut > 0 else min(1500, len(t))
        cf_items.append({"datapoint_id": d,
                         "ctext": t[:cut] + "\n\n" + INJECT_QUOTE + "\n" + t[cut:]})
    json.dump(cf_items, open(OUT / "cf_items_a86.json", "w"))

    n = 0
    with open(OUT / "prompts_v1.jsonl", "w") as f:
        def emit(channel, aid, dpid, prompt):
            nonlocal n
            f.write(json.dumps({"channel": channel, "aspect_id": aid,
                                "datapoint_id": dpid, "prompt": prompt}) + "\n")
            n += 1
        for aid in ASPECTS:
            a = aspects[aid]
            for it in items:
                emit("pass1", aid, it["datapoint_id"],
                     T1.format(name=a["name"], description=a["description"], text=it["ctext"]))
                emit("pass2", aid, it["datapoint_id"],
                     T2.format(name=a["name"], description=a["description"], text=it["ctext"]))
        for it in items:
            emit("scope", "scope", it["datapoint_id"], TSCOPE.format(text=it["ctext"]))
        a = aspects["a86"]
        for it in cf_items:
            emit("cf_a86", "a86", it["datapoint_id"],
                 T1.format(name=a["name"], description=a["description"], text=it["ctext"]))
    print(f"wrote {n} prompts, {len(cf_items)} CF items -> {OUT}")


if __name__ == "__main__":
    main()
