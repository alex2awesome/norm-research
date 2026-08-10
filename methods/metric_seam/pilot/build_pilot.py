"""Build the seam-pilot inputs: sample items + per-(aspect x item) judge prompts.

Seam pilot v0 (press_releases, 10 aspects, N=250 items):
- LLM channel  = Gemma-4-31B batch-scored 0-10/NA per (aspect, item)  [gemma_score_sk3.py]
- code channel = existing codegen flavors v0/v1/v2 per aspect          [run_code_flavors.py]
- readouts     = per-aspect kappa_e(L) profile, disagreement mass, distribution checks
"""
import json, random, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
V2 = ROOT / "runs/validity_full/v2/press_releases"
OUT = ROOT / "outputs/metric_seam_pilot"
OUT.mkdir(parents=True, exist_ok=True)

ASPECTS = ["a79", "a80", "a110", "a100", "a101", "a86", "a105", "a118", "a117", "a73"]
N_ITEMS = 250
TRUNC = 8000  # chars; URL-dense texts hit ~2 chars/token, keep well under max_model_len
SEED = 0

PROMPT = """You are an experienced newswire editor grading ONE press release against ONE criterion.

Criterion: {name}
Definition: {description}

Press release (may be truncated):
<press_release>
{text}
</press_release>

Grade how well the press release satisfies this criterion on a 0-10 integer scale \
(0 = completely fails, 10 = exemplary). If the criterion genuinely does not apply to this \
kind of release, answer NA. Reply with exactly one line in the form:
SCORE: <integer 0-10 or NA>"""


def main():
    aspects = {x["aspect_id"]: x for x in json.load(open(V2 / "aspects.json"))}
    data = json.load(open(V2 / "datapoints.json"))
    pool = [d for d in data if len(d.get("text", "")) >= 1000]
    random.seed(SEED)
    items = random.sample(pool, N_ITEMS)
    json.dump(items, open(OUT / "items.json", "w"))

    with open(OUT / "prompts.jsonl", "w") as f:
        for aid in ASPECTS:
            a = aspects[aid]
            for it in items:
                prompt = PROMPT.format(
                    name=a["name"], description=a["description"],
                    text=it["text"][:TRUNC])
                f.write(json.dumps({
                    "aspect_id": aid, "datapoint_id": it["datapoint_id"],
                    "prompt": prompt}) + "\n")
    n = len(ASPECTS) * len(items)
    print(f"wrote {len(items)} items, {n} prompts -> {OUT}")


if __name__ == "__main__":
    main()
