"""Extract evaluative norm passages from peer reviews."""

import json
import os
from pathlib import Path
from anthropic import Anthropic

INPUT_FILE = Path(__file__).parent / "inputs" / "batch_004.json"
OUTPUT_FILE = Path(__file__).parent / "outputs" / "batch_004.jsonl"

EXTRACTION_PROMPT = """Extract ALL evaluative passages from this peer review where the reviewer makes a judgment (praise or criticism) based on quality criteria.

For EACH distinct evaluative passage, extract:
1. "quote": The exact verbatim substring from the review text (max 300 chars - trim to evaluative core if needed)
2. "polarity": "pos" (praise), "neg" (criticism), or "mixed"
3. "aspect": A 2-6 word name for the quality criterion in the reviewer's own framing

Extract EVERY evaluative passage. Include passages about novelty, clarity, experimental quality, theoretical soundness, reproducibility, related work coverage, significance, presentation, motivation, dataset quality, etc.

Return a JSON object with a "passages" array. Each passage must have: quote (exact substring), polarity, aspect.

Review text:
{review_text}"""


def extract_passages_for_review(client, review_text, review_id):
    """Extract passages from a single review."""

    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4000,
        messages=[{
            "role": "user",
            "content": EXTRACTION_PROMPT.format(review_text=review_text)
        }]
    )

    # Parse response
    content = response.content[0].text

    # Extract JSON from response
    try:
        # Look for JSON in the response
        start_idx = content.find('{')
        end_idx = content.rfind('}') + 1
        if start_idx >= 0 and end_idx > start_idx:
            json_str = content[start_idx:end_idx]
            result = json.loads(json_str)
            passages = result.get('passages', [])
        else:
            passages = []
    except json.JSONDecodeError:
        print(f"Failed to parse JSON for review {review_id}")
        passages = []

    # Verify quotes are exact substrings
    verified_passages = []
    for p in passages:
        quote = p.get('quote', '')
        if quote in review_text:
            verified_passages.append(p)
        else:
            print(f"Warning: Quote not found as exact substring in {review_id}: {quote[:50]}...")

    return verified_passages


def main():
    # Initialize Anthropic client
    client = Anthropic()

    # Read input
    with open(INPUT_FILE, 'r') as f:
        reviews = json.load(f)

    print(f"Processing {len(reviews)} reviews...")

    # Process each review
    results = []
    for i, review in enumerate(reviews):
        print(f"Processing review {i+1}/{len(reviews)}: {review['review_id']}")

        passages = extract_passages_for_review(
            client,
            review['review_text'],
            review['review_id']
        )

        results.append({
            'review_id': review['review_id'],
            'paper_id': review['paper_id'],
            'passages': passages
        })

    # Write JSONL output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    # Print stats
    total_passages = sum(len(r['passages']) for r in results)
    avg_passages = total_passages / len(results) if results else 0

    print(f"\nExtraction complete:")
    print(f"  Reviews processed: {len(results)}")
    print(f"  Total passages: {total_passages}")
    print(f"  Avg passages/review: {avg_passages:.1f}")


if __name__ == '__main__':
    main()
