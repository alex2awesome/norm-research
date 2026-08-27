"""Extract evaluative norm passages from all peer reviews in batch_004.json"""

import json
import os
from pathlib import Path

# Read the first batch results
first_batch_file = Path(__file__).parent / "extracted_passages.json"
with open(first_batch_file, 'r') as f:
    first_batch = json.load(f)

print(f"First batch contained {len(first_batch)} reviews with {sum(len(r['passages']) for r in first_batch)} passages")

# Now I need to manually extract from the remaining 13 reviews
# Reviews 5-17 from batch_004.json

# For now, let's create the output structure for the remaining reviews
# This will be completed by processing reviews 5-17

remaining_reviews = [
    # Review 5-8: iclr_bE239PSGIGZ
    {"review_id": "iclr_bE239PSGIGZ_r0", "paper_id": "iclr_bE239PSGIGZ"},
    {"review_id": "iclr_bE239PSGIGZ_r1", "paper_id": "iclr_bE239PSGIGZ"},
    {"review_id": "iclr_bE239PSGIGZ_r2", "paper_id": "iclr_bE239PSGIGZ"},
    {"review_id": "iclr_bE239PSGIGZ_r3", "paper_id": "iclr_bE239PSGIGZ"},
    # Review 9-11: iclr_wfZGut6e09
    {"review_id": "iclr_wfZGut6e09_r0", "paper_id": "iclr_wfZGut6e09"},
    {"review_id": "iclr_wfZGut6e09_r1", "paper_id": "iclr_wfZGut6e09"},
    {"review_id": "iclr_wfZGut6e09_r2", "paper_id": "iclr_wfZGut6e09"},
    # Review 12-16: iclr_TW7d65uYu5M
    {"review_id": "iclr_TW7d65uYu5M_r0", "paper_id": "iclr_TW7d65uYu5M"},
    {"review_id": "iclr_TW7d65uYu5M_r1", "paper_id": "iclr_TW7d65uYu5M"},
    {"review_id": "iclr_TW7d65uYu5M_r2", "paper_id": "iclr_TW7d65uYu5M"},
    {"review_id": "iclr_TW7d65uYu5M_r3", "paper_id": "iclr_TW7d65uYu5M"},
    # Review 17-20: iclr_9wOQOgNe-w
    {"review_id": "iclr_9wOQOgNe-w_r0", "paper_id": "iclr_9wOQOgNe-w"},
    {"review_id": "iclr_9wOQOgNe-w_r1", "paper_id": "iclr_9wOQOgNe-w"},
    {"review_id": "iclr_9wOQOgNe-w_r2", "paper_id": "iclr_9wOQOgNe-w"},
    # Review 21-24: iclr_tvwNdOKhuF5
    {"review_id": "iclr_tvwNdOKhuF5_r0", "paper_id": "iclr_tvwNdOKhuF5"},
    {"review_id": "iclr_tvwNdOKhuF5_r1", "paper_id": "iclr_tvwNdOKhuF5"},
    {"review_id": "iclr_tvwNdOKhuF5_r2", "paper_id": "iclr_tvwNdOKhuF5"},
    {"review_id": "iclr_tvwNdOKhuF5_r3", "paper_id": "iclr_tvwNdOKhuF5"},
    # Review 25-27: iclr_Ab0o8YMJ8a
    {"review_id": "iclr_Ab0o8YMJ8a_r0", "paper_id": "iclr_Ab0o8YMJ8a"},
    {"review_id": "iclr_Ab0o8YMJ8a_r1", "paper_id": "iclr_Ab0o8YMJ8a"},
    {"review_id": "iclr_Ab0o8YMJ8a_r2", "paper_id": "iclr_Ab0o8YMJ8a"},
    # Review 28-31: iclr_SYB4WrJql1n
    {"review_id": "iclr_SYB4WrJql1n_r0", "paper_id": "iclr_SYB4WrJql1n"},
    {"review_id": "iclr_SYB4WrJql1n_r1", "paper_id": "iclr_SYB4WrJql1n"},
    {"review_id": "iclr_SYB4WrJql1n_r2", "paper_id": "iclr_SYB4WrJql1n"},
    {"review_id": "iclr_SYB4WrJql1n_r3", "paper_id": "iclr_SYB4WrJql1n"},
    # Review 32: iclr_hZftxQGJ4Re
    {"review_id": "iclr_hZftxQGJ4Re_r0", "paper_id": "iclr_hZftxQGJ4Re"},
    # Review 33-34: ANCH
    {"review_id": "ANCH-1", "paper_id": "ANCH-1"},
    {"review_id": "ANCH-2", "paper_id": "ANCH-2"},
]

print(f"Total reviews to process: {len(first_batch) + len(remaining_reviews)}")
