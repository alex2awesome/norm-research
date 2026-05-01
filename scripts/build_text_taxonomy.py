#!/usr/bin/env python3
"""Build text taxonomy from canonical features via embedding-sorted incremental LLM hierarchy building.

1. Load canonical features (from dedup step)
2. Embed with text embedding model, sort by similarity
3. Feed batches of k features to LLM with existing hierarchy
4. LLM merges features into hierarchy, adding parent nodes as needed
5. Hierarchy grows incrementally
"""
import argparse
import ast
import json
import logging
import re
import sys
import numpy as np
from pathlib import Path
from copy import deepcopy

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


HIERARCHY_PROMPT = """You are building a taxonomy of evaluation criteria for a {text_type}.

{hierarchy_section}

Here are {n_features} new evaluation criteria to merge into the taxonomy.
Each has a name, frequency (how often it appeared), and lift (how predictive it is):

{features_section}

Merge these into the hierarchy:
- If a feature fits under an existing node, place it there.
- If features are siblings without a parent, create a parent node.
- If features are in very different parts of the taxonomy, add multiple parents.
- Rearrange existing nodes if the new features reveal better structure.
- Parent nodes should be ABSTRACT concepts (e.g. "Methodological Quality", "Writing Style").
- Leaf nodes are the specific criteria.
- Keep the taxonomy balanced — no node should have more than ~10 direct children.

Output the COMPLETE updated taxonomy as JSON:

```json
{{
  "name": "Root",
  "children": [
    {{
      "name": "Abstract Category",
      "children": [
        {{
          "name": "Specific Criterion",
          "frequency": 100,
          "lift": 0.7,
          "leaf": true
        }}
      ]
    }}
  ]
}}
```

Rules:
- Every original feature must appear as a leaf node somewhere.
- Internal nodes are categories you create — they have "children" but no "frequency"/"lift".
- Leaf nodes have "frequency", "lift", and "leaf": true.
- The taxonomy should be 2-4 levels deep (not too flat, not too deep).
- Use clear, descriptive names for internal nodes.

Return ONLY the JSON in ```json ... ``` markers."""


def build_hierarchy_section(hierarchy):
    if hierarchy is None:
        return "No existing taxonomy yet — you're building from scratch.\n"

    # Count nodes
    def count_nodes(node):
        n = 1
        for child in node.get('children', []):
            n += count_nodes(child)
        return n

    n = count_nodes(hierarchy)
    # For large hierarchies, show only the top 2 levels
    if n > 100:
        summary = summarize_hierarchy(hierarchy, max_depth=2)
        return f"Current taxonomy ({n} nodes, showing top 2 levels):\n```json\n{json.dumps(summary, indent=2)}\n```\n\nFull taxonomy has {n} nodes across multiple levels.\n"

    return f"Current taxonomy ({n} nodes):\n```json\n{json.dumps(hierarchy, indent=2)}\n```\n\n"


def summarize_hierarchy(node, max_depth=2, depth=0):
    """Truncate hierarchy to max_depth for prompt."""
    result = {"name": node["name"]}
    if node.get("leaf"):
        result["frequency"] = node.get("frequency", 0)
        result["lift"] = node.get("lift", 0)
        result["leaf"] = True
        return result
    children = node.get("children", [])
    if depth < max_depth:
        result["children"] = [summarize_hierarchy(c, max_depth, depth + 1) for c in children]
    else:
        n_leaves = count_leaves(node)
        result["children"] = f"[... {len(children)} children, {n_leaves} total leaves]"
    return result


def count_leaves(node):
    if node.get("leaf"):
        return 1
    return sum(count_leaves(c) for c in node.get("children", []))


def build_features_section(features):
    lines = []
    for f in features:
        name = f.get('canonical_name', '?')
        freq = f.get('frequency', 0)
        lift = f.get('lift', 0)
        v = f.get('verifiability', '?')
        lines.append(f'- "{name}" (freq={freq}, lift={lift:.2f}, {v})')
    return '\n'.join(lines)


def extract_json_from_response(response):
    """Extract JSON from response, handling various formats."""
    # Try ```json ... ```
    start = response.find('```json')
    if start >= 0:
        code_start = response.find('\n', start)
        if code_start >= 0:
            end = response.find('```', code_start + 1)
            if end >= 0:
                try:
                    return json.loads(response[code_start + 1:end].strip())
                except json.JSONDecodeError:
                    pass

    # Try raw JSON (find outermost {...})
    depth = 0
    json_start = -1
    for i, ch in enumerate(response):
        if ch == '{':
            if depth == 0:
                json_start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and json_start >= 0:
                try:
                    return json.loads(response[json_start:i + 1])
                except json.JSONDecodeError:
                    json_start = -1
    return None


def validate_taxonomy(taxonomy, expected_leaves):
    """Check that taxonomy contains all expected leaf features."""
    found_leaves = set()

    def walk(node):
        if node.get('leaf'):
            found_leaves.add(node['name'])
        for child in node.get('children', []):
            if isinstance(child, dict):
                walk(child)

    walk(taxonomy)
    missing = expected_leaves - found_leaves
    extra = found_leaves - expected_leaves
    return found_leaves, missing, extra


def sort_by_similarity(features, embedding_model_name):
    """Embed features and sort so similar ones are adjacent."""
    logger.info('Embedding %d features with %s', len(features), embedding_model_name)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(embedding_model_name, trust_remote_code=True)

    texts = [f.get('canonical_name', '') + ': ' + f.get('description', '')[:200] for f in features]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    # Greedy nearest neighbor sort
    n = len(features)
    visited = [False] * n
    order = []
    current = 0
    visited[current] = True
    order.append(current)

    for _ in range(n - 1):
        best_idx = -1
        best_sim = -1
        for j in range(n):
            if not visited[j]:
                sim = float(embeddings[current] @ embeddings[j])
                if sim > best_sim:
                    best_sim = sim
                    best_idx = j
        if best_idx >= 0:
            visited[best_idx] = True
            order.append(best_idx)
            current = best_idx

    return [features[i] for i in order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True, help='canonical_features.jsonl')
    parser.add_argument('--model', required=True, help='vLLM model path for hierarchy building')
    parser.add_argument('--embedding-model', default='all-MiniLM-L6-v2',
                        help='Text embedding model for similarity sorting')
    parser.add_argument('--text-type', default='scientific paper submitted to an academic venue')
    parser.add_argument('--k', type=int, default=30, help='Features per batch')
    parser.add_argument('--max-features', type=int, default=None, help='Limit total features processed')
    parser.add_argument('--max-model-len', type=int, default=32768)
    parser.add_argument('--max-tokens', type=int, default=8192)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.90)
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = out_dir / 'snapshots'
    snapshots_dir.mkdir(exist_ok=True)

    # Load features
    features = []
    with open(args.features) as f:
        for line in f:
            features.append(json.loads(line))

    # Optionally limit
    if args.max_features:
        # Sort by frequency first, take top N
        features.sort(key=lambda x: x.get('frequency', 0), reverse=True)
        features = features[:args.max_features]

    logger.info('Loaded %d features', len(features))

    # Sort by embedding similarity
    sorted_features = sort_by_similarity(features, args.embedding_model)

    # Initialize vLLM
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype='auto',
        kv_cache_dtype='fp8',
    )
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    # Iterative hierarchy building
    hierarchy = None
    all_leaf_names = set()
    n_batches = (len(sorted_features) + args.k - 1) // args.k

    logger.info('Building taxonomy: %d features, k=%d, %d batches', len(sorted_features), args.k, n_batches)

    for batch_idx in range(n_batches):
        start = batch_idx * args.k
        end = min(start + args.k, len(sorted_features))
        batch = sorted_features[start:end]

        batch_names = set(f.get('canonical_name', '') for f in batch)
        all_leaf_names.update(batch_names)

        prompt = HIERARCHY_PROMPT.format(
            text_type=args.text_type,
            hierarchy_section=build_hierarchy_section(hierarchy),
            n_features=len(batch),
            features_section=build_features_section(batch),
        )

        outputs = llm.generate([prompt], sampling, use_tqdm=False)
        response = outputs[0].outputs[0].text

        new_hierarchy = extract_json_from_response(response)
        if new_hierarchy is None:
            logger.warning('Batch %d/%d: could not parse hierarchy JSON', batch_idx + 1, n_batches)
            with open(out_dir / 'failed_responses.jsonl', 'a') as f:
                f.write(json.dumps({'batch': batch_idx, 'response': response[:3000]}) + '\n')
            continue

        # Validate
        found, missing, extra = validate_taxonomy(new_hierarchy, all_leaf_names)
        if missing:
            logger.warning('Batch %d/%d: %d leaves missing from taxonomy', batch_idx + 1, n_batches, len(missing))

        hierarchy = new_hierarchy
        n_nodes = count_leaves(hierarchy)

        logger.info('Batch %d/%d: taxonomy has %d leaves, %d total nodes',
                     batch_idx + 1, n_batches, n_nodes,
                     sum(1 for _ in _walk_all(hierarchy)))

        # Save snapshot
        snapshot_path = snapshots_dir / f'taxonomy_batch_{batch_idx:04d}.json'
        with open(snapshot_path, 'w') as f:
            json.dump(hierarchy, f, indent=2)

    # Save final taxonomy
    final_path = out_dir / 'final_taxonomy.json'
    with open(final_path, 'w') as f:
        json.dump(hierarchy, f, indent=2)

    logger.info('=== DONE ===')
    if hierarchy:
        logger.info('Final taxonomy: %d leaves, %d total nodes',
                     count_leaves(hierarchy),
                     sum(1 for _ in _walk_all(hierarchy)))
    logger.info('Saved to %s', final_path)


def _walk_all(node):
    """Walk all nodes in the taxonomy."""
    yield node
    for child in node.get('children', []):
        if isinstance(child, dict):
            yield from _walk_all(child)


if __name__ == '__main__':
    main()
