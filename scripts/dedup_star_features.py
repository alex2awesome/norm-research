#!/usr/bin/env python3
"""Step 2a: Dedup 133K STaR features into ~13K canonical features via kmeans.

Embeds feature name+description, clusters, picks canonical representative per cluster.
Computes frequency and lift (% as z_{i,y_i}) per canonical feature.
"""
import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to star_features.jsonl')
    parser.add_argument('--n-clusters', type=int, default=13000)
    parser.add_argument('--embedding-model', default='all-MiniLM-L6-v2')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=512)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load features
    logger.info('Loading features from %s', args.input)
    records = []
    with open(args.input) as f:
        for line in f:
            records.append(json.loads(line))

    # Extract all features with metadata
    features = []  # list of dicts: name, description, verifiability, example_id, label, polarity
    for r in records:
        if not r.get('parse_success'):
            continue
        eid = r['example_id']
        label = r['label']
        for feat in r.get('features_for', []):
            features.append({
                'name': feat.get('name', ''),
                'description': feat.get('description', ''),
                'verifiability': feat.get('verifiability', ''),
                'inputs_needed': feat.get('inputs_needed', ''),
                'expected_output': feat.get('expected_output', ''),
                'example_id': eid,
                'label': label,
                'polarity': 1,  # z_{i,1}
                'is_right': label == 1,  # z_{i,y_i} if label matches polarity
            })
        for feat in r.get('features_against', []):
            features.append({
                'name': feat.get('name', ''),
                'description': feat.get('description', ''),
                'verifiability': feat.get('verifiability', ''),
                'inputs_needed': feat.get('inputs_needed', ''),
                'expected_output': feat.get('expected_output', ''),
                'example_id': eid,
                'label': label,
                'polarity': 0,  # z_{i,0}
                'is_right': label == 0,  # z_{i,y_i} if label matches polarity
            })

    logger.info('Total feature instances: %d', len(features))

    # Build texts for embedding: name + description
    texts = [f['name'] + ': ' + f['description'][:100] for f in features]

    # Embed
    logger.info('Embedding %d features with %s', len(texts), args.embedding_model)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.embedding_model)
    embeddings = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True)
    logger.info('Embeddings shape: %s', embeddings.shape)

    # KMeans
    n_clusters = min(args.n_clusters, len(features))
    logger.info('Running KMeans with %d clusters', n_clusters)
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=4096, random_state=42, n_init=3)
    labels = kmeans.fit_predict(embeddings)
    logger.info('KMeans done')

    # Build canonical features per cluster
    cluster_features = defaultdict(list)
    for i, feat in enumerate(features):
        cluster_features[labels[i]].append(feat)

    canonical = []
    for cluster_id in sorted(cluster_features.keys()):
        members = cluster_features[cluster_id]
        # Pick most common name as canonical
        name_counts = Counter(m['name'] for m in members)
        canonical_name = name_counts.most_common(1)[0][0]

        # Pick best description (from most common name instance)
        desc = ''
        for m in members:
            if m['name'] == canonical_name and m['description']:
                desc = m['description']
                break

        # Compute stats
        n_total = len(members)
        n_right = sum(1 for m in members if m['is_right'])
        n_wrong = n_total - n_right
        lift = n_right / max(n_total, 1)

        # Verifiability consensus
        v_counts = Counter(m['verifiability'].lower() for m in members)
        v_majority = v_counts.most_common(1)[0][0] if v_counts else 'unknown'

        # Collect unique names in cluster for dedup reference
        all_names = list(name_counts.most_common(10))

        # Sample inputs_needed and expected_output
        inputs_needed = ''
        expected_output = ''
        for m in members:
            if m.get('inputs_needed'):
                inputs_needed = m['inputs_needed']
                break
        for m in members:
            if m.get('expected_output'):
                expected_output = m['expected_output']
                break

        canonical.append({
            'cluster_id': int(cluster_id),
            'canonical_name': canonical_name,
            'description': desc[:300],
            'frequency': n_total,
            'n_right': n_right,
            'n_wrong': n_wrong,
            'lift': round(lift, 3),
            'verifiability': v_majority,
            'inputs_needed': inputs_needed[:200],
            'expected_output': expected_output[:200],
            'name_variants': all_names,
            'n_unique_names': len(name_counts),
        })

    # Sort by frequency
    canonical.sort(key=lambda x: x['frequency'], reverse=True)

    # Save
    canon_path = out_dir / 'canonical_features.jsonl'
    with open(canon_path, 'w') as f:
        for c in canonical:
            f.write(json.dumps(c) + '\n')

    # Summary stats
    thin = sum(1 for c in canonical if 'thin' in c['verifiability'])
    thick = sum(1 for c in canonical if 'thick' in c['verifiability'])
    total_freq = sum(c['frequency'] for c in canonical)

    logger.info('Saved %d canonical features to %s', len(canonical), canon_path)
    logger.info('Thin: %d (%.0f%%), Thick: %d (%.0f%%)', thin, 100*thin/len(canonical), thick, 100*thick/len(canonical))
    logger.info('Top 20 canonical features:')
    for c in canonical[:20]:
        logger.info('  %5dx [%s] lift=%.2f | %s', c['frequency'], c['verifiability'][:5], c['lift'], c['canonical_name'])


if __name__ == '__main__':
    main()
