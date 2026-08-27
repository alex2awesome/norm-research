#!/bin/bash
# Top-up runner for join leftovers after OpenAlex's per-IP daily budget
# (10,000 credits, resets midnight UTC) cut the singleton passes short.
#
# Run from this directory on any host with a fresh OpenAlex budget
# (laptop after reset, or an skampere host):
#
#   bash finish_leftovers.sh
#
# It is safe to re-run: every stage resumes from its jsonl cache.
set -e
PY=${PY:-python3}

echo "== Task A: award join leftovers =="
# regenerate remainder (keys not yet in cache) and run the batched joiner
$PY - <<'EOF'
import pandas as pd, json, csv
df = pd.read_csv('../best_papers/best_papers_awards.csv')
done = set()
for line in open('../best_papers/join_cache.jsonl'):
    done.add(json.loads(line)['_key'])
with open('/tmp/awards_leftover.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['key', 'title', 'year'])
    n = 0
    for _, r in df.iterrows():
        key = f"{r.venue}|{r.year}|{r.title}"
        if key not in done:
            w.writerow([key, r.title, r.year]); n += 1
print(f"{n} award leftovers")
EOF
$PY batch_title_join.py --input /tmp/awards_leftover.csv \
    --out ../best_papers/join_cache.jsonl --batch 15 --threshold 0.85 \
    --fallback full
(cd ../best_papers && $PY join_openalex.py --no-query && \
 $PY finalize_labels.py)

echo "== Task B: citation join leftovers =="
$PY - <<'EOF'
import pandas as pd, json, csv, sys
sys.path.insert(0, '.')
from join_and_build import norm_title
dblp = pd.read_csv('dblp_papers.csv')
src = pd.read_parquet('openalex_source_works.parquet')
dblp['ntitle'] = dblp.title.map(norm_title)
src['ntitle'] = src.title.map(norm_title)
dblp = dblp.drop_duplicates(['venue', 'year', 'ntitle'])
src_keys = set(zip(src.venue, src.ntitle))
done = set()
for line in open('title_join_cache.jsonl'):
    done.add(json.loads(line)['_key'])
with open('/tmp/taskb_leftover.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['key', 'title', 'year'])
    n = 0
    for _, r in dblp.iterrows():
        if (r.venue, r.ntitle) in src_keys:
            continue
        key = f"{r.venue}|{r.year}|{r.title}"
        if key not in done:
            w.writerow([key, r.title, r.year]); n += 1
print(f"{n} task B leftovers")
EOF
$PY batch_title_join.py --input /tmp/taskb_leftover.csv \
    --out title_join_cache.jsonl --batch 20 --threshold 0.90 --fallback title
$PY join_and_build.py
echo "done -- rebuild complete. scp outputs to sk3 (see README)."
