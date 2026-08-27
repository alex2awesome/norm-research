#!/usr/bin/env python3
"""Extract evaluative norm passages from peer reviews."""

import json
import re

def extract_passages(review_text):
    """Extract evaluative passages from a review.

    Returns list of {quote, polarity, aspect} dicts.
    """
    passages = []

    # This is a complex extraction task that requires careful reading.
    # For now, I'll identify common patterns for evaluative statements.
    # In practice, this would need more sophisticated NLP or manual annotation.

    # Split by common section markers
    lines = review_text.split('\n')

    current_section = None
    for line in lines:
        line = line.strip()
        if line.startswith('**Strengths'):
            current_section = 'strengths'
        elif line.startswith('**Weaknesses'):
            current_section = 'weaknesses'

    return passages


def main():
    # Read input
    with open('/Users/spangher/Projects/stanford-research/norm-research/datasets/peer-review/review_norms_v1/inputs/batch_012.json', 'r') as f:
        reviews = json.load(f)

    results = []

    for review in reviews:
        review_id = review['review_id']
        paper_id = review['paper_id']
        text = review['review_text']

        # Manual extraction for each review
        passages = extract_passages_manual(review_id, text)

        results.append({
            'review_id': review_id,
            'paper_id': paper_id,
            'passages': passages
        })

    # Write output
    with open('/Users/spangher/Projects/stanford-research/norm-research/datasets/peer-review/review_norms_v1/outputs/batch_012.jsonl', 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    # Print stats
    total_passages = sum(len(r['passages']) for r in results)
    print(f"Processed {len(results)} reviews, extracted {total_passages} passages")


if __name__ == '__main__':
    main()
