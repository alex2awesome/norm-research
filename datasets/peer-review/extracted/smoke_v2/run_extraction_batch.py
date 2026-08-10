"""Smoke v2 (Batch-API variant): submit all 20 reviews to Anthropic Message Batches.

Uses the USC API key (~/.anthropic-usc-key.txt) so it has batch scope and 50% discount.
Polls until the batch completes, then writes results to output_part_1..N.jsonl in chunks
of 5 reviews to mirror v1's directory layout, plus merged output_all.jsonl.

Same prompt/schema as run_extraction.py.
"""
import os
import json
import time
import anthropic
from run_extraction import (
    SMOKE_DIR,
    INPUT_PATH,
    RUBRIC_PATH,
    MODEL,
    SYSTEM_PROMPT_TEMPLATE,
    build_rubric_block,
    load_rubrics,
    load_reviews,
)

API_KEY_PATH = os.environ.get("ANTHROPIC_API_KEY_FILE", "/Users/spangher/.anthropic-usc-key.txt")
POLL_INTERVAL = 30
MAX_WAIT = 60 * 60 * 6  # 6h cap


def get_api_key():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    return open(API_KEY_PATH).read().strip()


def build_user_msg(review):
    return (
        f"review_id: {review['review_id']}\n"
        f"paper_id: {review['paper_id']}\n"
        f"venue: {review['venue']}\n"
        f"decision: {review.get('decision')}\n"
        f"review_score: {review.get('review_score')}\n"
        f"is_meta_review: {review.get('is_meta_review')}\n"
        f"title: {review.get('title')}\n\n"
        f"---BEGIN REVIEW TEXT---\n{review['review_text']}\n---END REVIEW TEXT---"
    )


def main():
    api_key = get_api_key()
    client = anthropic.Anthropic(api_key=api_key)
    rubrics = load_rubrics()
    reviews = load_reviews()
    rubric_block = build_rubric_block(rubrics)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(n_rubrics=len(rubrics), rubric_block=rubric_block)
    print(f"Model={MODEL}; rubrics={len(rubrics)}; reviews={len(reviews)}; system_prompt_chars={len(system_prompt)}")

    requests = []
    for rev in reviews:
        custom_id = f"rev_{rev['review_id']}"
        requests.append({
            "custom_id": custom_id,
            "params": {
                "model": MODEL,
                "max_tokens": 12000,
                "system": system_prompt,
                "messages": [{"role": "user", "content": build_user_msg(rev)}],
            },
        })

    print(f"Submitting batch with {len(requests)} requests...")
    batch = client.messages.batches.create(requests=requests)
    print(f"Batch id: {batch.id} status: {batch.processing_status}")

    with open(os.path.join(SMOKE_DIR, "batch_id.txt"), "w") as f:
        f.write(batch.id + "\n")

    t0 = time.time()
    while True:
        b = client.messages.batches.retrieve(batch.id)
        elapsed = int(time.time() - t0)
        counts = b.request_counts
        print(f"  [{elapsed}s] status={b.processing_status} processing={counts.processing} succeeded={counts.succeeded} errored={counts.errored} canceled={counts.canceled} expired={counts.expired}", flush=True)
        if b.processing_status == "ended":
            break
        if elapsed > MAX_WAIT:
            print("MAX_WAIT exceeded")
            return
        time.sleep(POLL_INTERVAL)

    # Pull results
    print("Fetching results...")
    by_id = {}
    for r in client.messages.batches.results(batch.id):
        cid = r.custom_id
        rid = cid.replace("rev_", "")
        # Each result has result.type ('succeeded'|'errored'|'canceled'|'expired') and .message
        rtype = r.result.type
        if rtype == "succeeded":
            msg = r.result.message
            text = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
            if text.startswith("```"):
                lines = text.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines)
            try:
                obj = json.loads(text)
                parse_ok = True
            except json.JSONDecodeError as je:
                obj = {"_parse_error": str(je), "_raw": text[:2000]}
                parse_ok = False
            by_id[rid] = {
                "review_id": rid,
                "parse_ok": parse_ok,
                "obj": obj,
                "meta": {
                    "input_tokens": msg.usage.input_tokens,
                    "output_tokens": msg.usage.output_tokens,
                    "batch": True,
                },
            }
        else:
            err = getattr(r.result, "error", None)
            by_id[rid] = {
                "review_id": rid,
                "parse_ok": False,
                "obj": {"_batch_error": str(err), "_type": rtype},
                "meta": {"batch": True, "error_type": rtype},
            }

    # Write in input order, chunked into 4 parts of 5
    order = [rev["review_id"] for rev in reviews]
    chunks = [order[i:i + 5] for i in range(0, len(order), 5)]
    for i, ch in enumerate(chunks, 1):
        out_path = os.path.join(SMOKE_DIR, f"output_part_{i}.jsonl")
        with open(out_path, "w") as f:
            for rid in ch:
                if rid in by_id:
                    f.write(json.dumps(by_id[rid], ensure_ascii=False) + "\n")
        print(f"  wrote {out_path} ({len(ch)} records)")

    merged = os.path.join(SMOKE_DIR, "output_all.jsonl")
    with open(merged, "w") as f:
        for rid in order:
            if rid in by_id:
                f.write(json.dumps(by_id[rid], ensure_ascii=False) + "\n")
    print(f"Merged -> {merged}: {sum(1 for _ in open(merged))} lines")
    print(f"Done. parse_ok={sum(1 for r in by_id.values() if r['parse_ok'])} / {len(by_id)}")


if __name__ == "__main__":
    main()
