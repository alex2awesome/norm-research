#!/usr/bin/env python3
"""Build code library from coded features via embedding-sorted incremental refactoring.

1. Load successful programs from multipass coding
2. Embed all programs with code embedding model
3. Sort by embedding similarity (similar programs adjacent)
4. Feed batches of k programs to refactorer incrementally
5. Library grows via JSON diffs (add/modify/delete)
"""
import argparse
import ast
import json
import logging
import re
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ---- Refactoring prompt ----

REFACTOR_PROMPT = """You are maintaining a shared Python utility library for text analysis.

{library_section}
Here are {n_programs} programs that check evaluation criteria for a scientific paper.
Many share similar logic. Extract reusable utility functions.

{programs_section}

Output ONLY the changes as JSON — do NOT repeat unchanged functions:

```json
{{
  "add": [
    {{"name": "function_name", "code": "def function_name(text: str) -> int:\\n    \\"\\"\\"One-line docstring.\\"\\"\\"\\n    ..."}}
  ],
  "modify": [
    {{"name": "existing_func", "code": "def existing_func(...): ..."}}
  ],
  "delete": ["redundant_func"],
  "reasoning": "Brief explanation"
}}
```

Rules:
- Only stdlib (re, string, collections, statistics, math).
- Each function: SHORT (under 15 lines), 1-line docstring, type hints.
- Only ADD genuinely new patterns. Don't duplicate what's in the library.
- If no new patterns, return empty add/modify/delete.

Return ONLY JSON in ```json ... ``` markers."""


def build_library_section(library_code):
    if not library_code or library_code.strip() == '':
        return "No existing library yet — you're starting from scratch.\n"

    # Count functions
    try:
        tree = ast.parse(library_code)
        n_funcs = sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    except SyntaxError:
        n_funcs = '?'

    return f"Current library ({n_funcs} functions):\n```python\n{library_code}\n```\n\n"


def build_programs_section(programs):
    parts = []
    for i, prog in enumerate(programs):
        name = prog.get('canonical_name', f'program_{i}')
        code = prog['source_code']
        # Truncate very long programs
        if len(code) > 1500:
            code = code[:1500] + '\n    # ... truncated'
        parts.append(f"# Criterion: {name}\n{code}")
    return '\n\n---\n\n'.join(parts)


def extract_diff_json(response):
    # Find ```json ... ```
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
    # Try raw JSON
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass
    return None


def apply_diff(library_code, diff):
    """Apply add/modify/delete diff to library."""
    if not library_code:
        library_code = '"""Shared utility library for text analysis."""\nimport re\nimport string\nimport math\nfrom collections import Counter\nfrom statistics import mean\n'

    lines = library_code.split('\n')

    # Parse existing functions
    try:
        tree = ast.parse(library_code)
    except SyntaxError:
        # Can't parse — just append new functions
        for item in diff.get('add', []):
            library_code += '\n\n' + item['code']
        return library_code

    # Collect function source ranges
    func_ranges = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef):
            func_ranges[node.name] = (node.lineno - 1, node.end_lineno)

    delete_names = set(diff.get('delete', []))
    modify_map = {m['name']: m['code'] for m in diff.get('modify', [])}

    # Rebuild: header + kept/modified functions + new functions
    # Find header (everything before first function)
    header_end = 0
    for i, line in enumerate(lines):
        if line.startswith('def '):
            header_end = i
            break
    else:
        header_end = len(lines)

    parts = ['\n'.join(lines[:header_end])]
    seen = set()

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef):
            name = node.name
            if name in delete_names:
                continue
            if name in modify_map:
                parts.append('\n\n' + modify_map[name])
            else:
                start, end = func_ranges[name]
                parts.append('\n\n' + '\n'.join(lines[start:end]))
            seen.add(name)

    for item in diff.get('add', []):
        name = item['name']
        code = item['code']
        if name not in seen:
            try:
                ast.parse(code)
                parts.append('\n\n' + code)
                seen.add(name)
            except SyntaxError as e:
                logger.warning('Skipping %s (syntax error): %s', name, e)

    return '\n'.join(parts).strip() + '\n'


def sort_by_similarity(programs, embedding_model_name):
    """Embed programs and sort so similar ones are adjacent."""
    logger.info('Embedding %d programs with %s', len(programs), embedding_model_name)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(embedding_model_name, trust_remote_code=True)

    texts = [p['source_code'][:1000] for p in programs]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    # Sort by traveling-salesman-ish greedy nearest neighbor
    # Start from first program, always go to nearest unvisited
    n = len(programs)
    visited = [False] * n
    order = []

    # Start from program 0
    current = 0
    visited[current] = True
    order.append(current)

    for _ in range(n - 1):
        # Find nearest unvisited
        best_idx = -1
        best_sim = -1
        current_emb = embeddings[current]
        for j in range(n):
            if not visited[j]:
                sim = float(current_emb @ embeddings[j])
                if sim > best_sim:
                    best_sim = sim
                    best_idx = j
        if best_idx >= 0:
            visited[best_idx] = True
            order.append(best_idx)
            current = best_idx

    sorted_programs = [programs[i] for i in order]
    logger.info('Sorted programs by similarity')
    return sorted_programs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--programs', required=True, help='feature_programs_merged.jsonl or reparsed')
    parser.add_argument('--model', required=True, help='vLLM model path for refactoring')
    parser.add_argument('--embedding-model', default='jinaai/jina-code-embeddings-0.5b')
    parser.add_argument('--k', type=int, default=50, help='Programs per refactoring batch')
    parser.add_argument('--max-model-len', type=int, default=32768)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.90)
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = out_dir / 'snapshots'
    snapshots_dir.mkdir(exist_ok=True)

    # Load successful programs
    programs = []
    with open(args.programs) as f:
        for line in f:
            r = json.loads(line)
            if r.get('execution_success') and r.get('source_code'):
                programs.append(r)
    logger.info('Loaded %d successful programs', len(programs))

    if not programs:
        logger.error('No successful programs to refactor')
        return

    # Sort by embedding similarity
    sorted_programs = sort_by_similarity(programs, args.embedding_model)

    # Initialize vLLM for refactoring
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype='auto',
        kv_cache_dtype='fp8',
    )
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    # Incremental refactoring
    library_code = ''
    n_batches = (len(sorted_programs) + args.k - 1) // args.k
    metrics = []

    logger.info('Starting incremental refactoring: %d programs, k=%d, %d batches',
                len(sorted_programs), args.k, n_batches)

    for batch_idx in range(n_batches):
        start = batch_idx * args.k
        end = min(start + args.k, len(sorted_programs))
        batch = sorted_programs[start:end]

        # Build prompt
        # If library is large, show only relevant subset
        library_for_prompt = library_code
        if library_code and library_code.count('\n') > 300:
            # Similarity route: show top-8 functions
            # For now, just truncate to last 200 lines as simple approximation
            lib_lines = library_code.split('\n')
            # Keep header + last 200 lines
            header = []
            for line in lib_lines:
                if line.startswith('def '):
                    break
                header.append(line)
            library_for_prompt = '\n'.join(header) + '\n' + '\n'.join(lib_lines[-200:])
            library_for_prompt += f'\n# ... ({library_code.count(chr(10))} total lines, showing relevant subset)'

        prompt = REFACTOR_PROMPT.format(
            library_section=build_library_section(library_for_prompt),
            n_programs=len(batch),
            programs_section=build_programs_section(batch),
        )

        # Generate
        outputs = llm.generate([prompt], sampling, use_tqdm=False)
        response = outputs[0].outputs[0].text

        # Parse diff
        diff = extract_diff_json(response)
        if diff is None:
            logger.warning('Batch %d/%d: could not parse diff', batch_idx + 1, n_batches)
            # Save failed response for debugging
            with open(out_dir / 'failed_responses.jsonl', 'a') as f:
                f.write(json.dumps({'batch': batch_idx, 'response': response[:2000]}) + '\n')
            continue

        n_add = len(diff.get('add', []))
        n_mod = len(diff.get('modify', []))
        n_del = len(diff.get('delete', []))
        reasoning = diff.get('reasoning', '')

        if n_add == 0 and n_mod == 0 and n_del == 0:
            logger.info('Batch %d/%d: no changes', batch_idx + 1, n_batches)
        else:
            prev_library = library_code
            library_code = apply_diff(library_code, diff)

            # Count functions
            try:
                tree = ast.parse(library_code)
                n_funcs = sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
            except SyntaxError:
                n_funcs = '?'
                logger.warning('Library has syntax errors after batch %d', batch_idx + 1)

            logger.info('Batch %d/%d: +%d -%d ~%d → %s functions | %s',
                        batch_idx + 1, n_batches, n_add, n_del, n_mod,
                        n_funcs, reasoning[:80])

            # Save snapshot
            snapshot_path = snapshots_dir / f'library_batch_{batch_idx:04d}.py'
            snapshot_path.write_text(library_code)

            metrics.append({
                'batch': batch_idx,
                'n_add': n_add,
                'n_modify': n_mod,
                'n_delete': n_del,
                'n_functions': n_funcs if isinstance(n_funcs, int) else 0,
                'n_lines': library_code.count('\n'),
                'reasoning': reasoning[:200],
            })

    # Save final library
    final_path = out_dir / 'final_library.py'
    final_path.write_text(library_code)

    # Save metrics
    metrics_path = out_dir / 'refactoring_metrics.jsonl'
    with open(metrics_path, 'w') as f:
        for m in metrics:
            f.write(json.dumps(m) + '\n')

    # Summary
    try:
        tree = ast.parse(library_code)
        n_funcs = sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    except SyntaxError:
        n_funcs = '?'

    logger.info('=== DONE ===')
    logger.info('Final library: %s functions, %d lines', n_funcs, library_code.count('\n'))
    logger.info('Saved to %s', final_path)
    logger.info('Snapshots in %s', snapshots_dir)


if __name__ == '__main__':
    main()
