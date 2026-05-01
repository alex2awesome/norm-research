#!/usr/bin/env python3
"""Batch generate verification programs via vLLM offline inference.

Phase 1 only: generate programs for all examples. No refactoring.
Refactoring happens in a separate pass after embedding + clustering.
"""
import argparse
import ast
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are writing a Python function to predict whether a {text_type} is {positive_label} or {negative_label}.

def predict(text: str) -> float

Rules:
1. Return float between 0.0 and 1.0 (1.0 = {positive_label}).
2. ONLY Python standard library (re, string, collections, statistics, math, itertools, functools, textwrap, unicodedata, difflib, hashlib).
3. Write a NON-TRIVIAL program analyzing multiple structural properties.
4. Be SPECIFIC to this example — study the text and find concrete distinguishing patterns.
5. Include a docstring explaining your hypothesis.

After the function, list external tools that would help:
# TOOLS_NEEDED:
# - tool_name: description
# If none: # TOOLS_NEEDED: none

--- TEXT ---
{text}
--- END TEXT ---

Label: {label_word} ({label_value})

Write the function + TOOLS_NEEDED. Wrap in ```python ... ``` markers."""


def build_prompts(texts, labels, text_type, positive_label, negative_label):
    prompts = []
    for text, label in zip(texts, labels):
        label_word = positive_label if label == 1 else negative_label
        prompts.append(PROMPT_TEMPLATE.format(
            text_type=text_type,
            positive_label=positive_label,
            negative_label=negative_label,
            text=text[:3000],
            label_word=label_word,
            label_value=label,
        ))
    return prompts


def extract_code(response):
    # Try ```python ... ```
    pattern = r"```python\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try ``` ... ```
    pattern2 = r"```\s*\n(.*?)```"
    match2 = re.search(pattern2, response, re.DOTALL)
    if match2:
        code = match2.group(1).strip()
        if "def " in code:
            return code
    # Fallback: find raw def predict/check without backtick fences
    match3 = re.search(r"((?:import\s+\w+.*\n)*def\s+predict\s*\(.*?\n(?:.*\n)*)", response)
    if match3:
        code = match3.group(1).strip()
        # Trim anything after TOOLS_NEEDED or a non-code line
        lines = code.split('\n')
        trimmed = []
        for line in lines:
            if line.strip() and not line[0] in (' ', '\t', '#', 'd', 'i', 'f', 'e', 'r', 'w', 'c', 'n', 't', 'a', 's', 'p', 'l', 'm', 'g', '@'):
                break
            trimmed.append(line)
        if trimmed:
            return '\n'.join(trimmed).strip()
    # Last resort: find any def predict block
    match4 = re.search(r"(def\s+predict\s*\(text.*?\n(?:[ \t]+.*\n)*)", response)
    if match4:
        return match4.group(1).strip()
    return None


def extract_tools(code):
    if not code:
        return []
    tools = []
    in_block = False
    for line in code.split('\n'):
        s = line.strip()
        if s.startswith('# TOOLS_NEEDED:'):
            in_block = True
            rest = s[len('# TOOLS_NEEDED:'):].strip()
            if rest.lower() == 'none':
                return []
            if rest.startswith('- '):
                tools.append(rest[2:])
            continue
        if in_block and s.startswith('# - '):
            tools.append(s[4:])
        elif in_block and not s.startswith('#'):
            break
    return tools


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', default='peer-review')
    parser.add_argument('--n-examples', type=int, default=1000)
    parser.add_argument('--max-model-len', type=int, default=4096)
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.90)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--output-dir', default='outputs/verification_library/batch_1k')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_configs = {
        'peer-review': {
            'split_dir': REPO_ROOT / 'datasets' / 'peer-review' / 'splits',
            'text_col': 'text', 'label_col': 'judgement', 'id_col': 'paper_id',
            'text_type': 'scientific paper submitted to an academic venue',
            'positive_label': 'accepted', 'negative_label': 'rejected',
        },
    }
    ds = ds_configs[args.dataset]

    import pandas as pd
    import numpy as np
    train = pd.read_csv(ds['split_dir'] / 'train.csv.gz', low_memory=False)
    logger.info('Loaded %d training examples', len(train))

    np.random.seed(args.seed)
    sample = train.sample(min(args.n_examples, len(train)))
    texts = sample[ds['text_col']].astype(str).tolist()
    labels = sample[ds['label_col']].astype(int).tolist()
    ids = sample[ds['id_col']].astype(str).tolist()

    logger.info('Sampled %d examples (label dist: %d pos / %d neg)',
                len(texts), sum(labels), len(labels) - sum(labels))

    # Check cache
    programs_path = out_dir / 'programs.jsonl'
    existing_ids = set()
    if programs_path.exists():
        with open(programs_path) as f:
            for line in f:
                obj = json.loads(line)
                existing_ids.add(obj['example_id'])
        logger.info('Found %d cached programs', len(existing_ids))

    todo = [(t, l, eid) for t, l, eid in zip(texts, labels, ids) if eid not in existing_ids]
    if not todo:
        logger.info('All cached!')
        return

    todo_texts = [x[0] for x in todo]
    todo_labels = [x[1] for x in todo]
    todo_ids = [x[2] for x in todo]

    logger.info('Generating programs for %d examples', len(todo_texts))
    prompts = build_prompts(todo_texts, todo_labels, ds['text_type'], ds['positive_label'], ds['negative_label'])

    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype='auto',
        kv_cache_dtype='fp8',
    )
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    logger.info('Starting batch inference (%d prompts, batch_size=%d)', len(prompts), args.batch_size)

    all_outputs = []
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i:i + args.batch_size]
        outputs = llm.generate(batch, sampling, use_tqdm=True)
        all_outputs.extend(outputs)
        logger.info('Batch %d/%d done (%d total)',
                     i // args.batch_size + 1,
                     (len(prompts) + args.batch_size - 1) // args.batch_size,
                     len(all_outputs))

    # Save raw responses for debugging
    raw_path = out_dir / 'raw_responses.jsonl'
    with open(raw_path, 'a') as rf:
        for j, output in enumerate(all_outputs):
            rf.write(json.dumps({
                'example_id': todo_ids[j],
                'response': output.outputs[0].text,
            }) + '\n')
    logger.info('Saved raw responses to %s', raw_path)

    # Parse and save
    n_success = 0
    with open(programs_path, 'a') as f:
        for j, output in enumerate(all_outputs):
            response = output.outputs[0].text
            code = extract_code(response)
            tools = extract_tools(code)
            success = code is not None

            pid = hashlib.sha256(f'{todo_ids[j]}:{args.seed}'.encode()).hexdigest()[:12]
            record = {
                'program_id': pid,
                'example_id': todo_ids[j],
                'source_code': code or '',
                'hypothesis': '',
                'tools_needed': tools,
                'execution_success': success,
                'label': todo_labels[j],
                'generated_at': datetime.now().isoformat(),
            }

            if code:
                try:
                    tree = ast.parse(code)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            ds_ = ast.get_docstring(node)
                            if ds_:
                                record['hypothesis'] = ds_[:200]
                            break
                except SyntaxError:
                    record['execution_success'] = False
                    success = False

            if success:
                n_success += 1

            f.write(json.dumps(record) + '\n')

    logger.info('Done: %d/%d successful (%.0f%%)', n_success, len(all_outputs), 100 * n_success / len(all_outputs))
    logger.info('Saved to %s', programs_path)


if __name__ == '__main__':
    main()
