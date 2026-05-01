#!/usr/bin/env python3
"""Step 3: Code ALL canonical features (thin AND thick) with Qwen.

For each canonical feature, ask Qwen to write a Python program that checks it.
The measure of thin vs thick is empirical: does the program actually work?
"""
import argparse
import ast
import hashlib
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

CODE_FEATURE_PROMPT = """Write a Python function. Do NOT explain — just write code.

Criterion to check: "{name}"
Description: {description}
Frequency: {frequency} times across {n_examples} examples ({lift_pct}% lift).

```python
import re
import string
from collections import Counter

def check(text: str) -> float:
    \"\"\"THIN: Check for {name} using structural/lexical patterns.\"\"\"
```

Complete this function. Rules:
- Return float 0.0-1.0 (1.0 = criterion present).
- ONLY stdlib (re, string, collections, statistics, math).
- Make it GENERALIZABLE — use abstract patterns (e.g. regex for method-naming conventions, keyword lists for technical vocabulary) not specific strings from one paper.
- If truly impossible to approximate programmatically, change docstring to THICK: and return 0.5.
- Do NOT explain your reasoning. Do NOT write anything outside the code block.

Complete the function inside ```python ... ``` markers:"""


def build_prompts(features, text_type, n_examples):
    prompts = []
    for f in features:
        prompts.append(CODE_FEATURE_PROMPT.format(
            text_type=text_type,
            name=f['canonical_name'],
            description=f['description'][:500],
            frequency=f['frequency'],
            n_examples=n_examples,
            lift_pct=int(f['lift'] * 100),
            inputs_needed=f.get('inputs_needed', 'text content')[:200],
            expected_output=f.get('expected_output', 'float score')[:200],
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
    # Fallback: raw def check
    match3 = re.search(r"((?:import\s+\w+.*\n)*def\s+check\s*\(.*?\n(?:[ \t]+.*\n)*)", response)
    if match3:
        return match3.group(1).strip()
    # Also try def predict
    match4 = re.search(r"((?:import\s+\w+.*\n)*def\s+predict\s*\(.*?\n(?:[ \t]+.*\n)*)", response)
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


def classify_from_docstring(code):
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                ds = ast.get_docstring(node)
                if ds:
                    if ds.strip().upper().startswith('THICK:'):
                        return 'thick'
                    elif ds.strip().upper().startswith('THIN:'):
                        return 'thin'
    except SyntaxError:
        pass
    return 'unknown'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--features', required=True, help='Path to canonical_features.jsonl')
    parser.add_argument('--dataset', default='peer-review')
    parser.add_argument('--max-model-len', type=int, default=4096)
    parser.add_argument('--max-tokens', type=int, default=1536)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.90)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_configs = {
        'peer-review': {
            'text_type': 'scientific paper submitted to an academic venue',
            'n_examples': 20000,
        },
    }
    ds = ds_configs[args.dataset]

    # Load canonical features
    logger.info('Loading features from %s', args.features)
    features = []
    with open(args.features) as f:
        for line in f:
            features.append(json.loads(line))
    logger.info('Loaded %d canonical features', len(features))

    # Check cache
    programs_path = out_dir / 'feature_programs.jsonl'
    existing_ids = set()
    if programs_path.exists():
        with open(programs_path) as f:
            for line in f:
                obj = json.loads(line)
                existing_ids.add(obj['cluster_id'])
        logger.info('Found %d cached programs', len(existing_ids))

    todo = [f for f in features if f['cluster_id'] not in existing_ids]
    if not todo:
        logger.info('All cached!')
        return

    logger.info('Coding %d features', len(todo))
    prompts = build_prompts(todo, ds['text_type'], ds['n_examples'])

    # Save raw responses
    raw_path = out_dir / 'raw_responses.jsonl'

    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype='auto',
        kv_cache_dtype='fp8',
    )
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    logger.info('Starting batch inference (%d prompts)', len(prompts))

    all_outputs = []
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i:i + args.batch_size]
        outputs = llm.generate(batch, sampling, use_tqdm=True)
        all_outputs.extend(outputs)
        logger.info('Batch %d/%d done (%d total)',
                     i // args.batch_size + 1,
                     (len(prompts) + args.batch_size - 1) // args.batch_size,
                     len(all_outputs))

    # Save raw responses
    with open(raw_path, 'a') as rf:
        for j, output in enumerate(all_outputs):
            rf.write(json.dumps({
                'cluster_id': todo[j]['cluster_id'],
                'response': output.outputs[0].text,
            }) + '\n')

    # Parse and save
    n_success = 0
    n_thin = 0
    n_thick = 0
    with open(programs_path, 'a') as f:
        for j, output in enumerate(all_outputs):
            response = output.outputs[0].text
            code = extract_code(response)
            tools = extract_tools(code)
            classification = classify_from_docstring(code) if code else 'failed'
            success = code is not None

            # Validate syntax
            if code:
                try:
                    ast.parse(code)
                except SyntaxError:
                    success = False
                    classification = 'syntax_error'

            record = {
                'cluster_id': todo[j]['cluster_id'],
                'canonical_name': todo[j]['canonical_name'],
                'frequency': todo[j]['frequency'],
                'lift': todo[j]['lift'],
                'llm_verifiability': todo[j]['verifiability'],
                'code_classification': classification,
                'source_code': code or '',
                'tools_needed': tools,
                'execution_success': success,
            }

            if success:
                n_success += 1
                if classification == 'thin':
                    n_thin += 1
                elif classification == 'thick':
                    n_thick += 1

            f.write(json.dumps(record) + '\n')

    logger.info('Done: %d/%d successful (%.0f%%)', n_success, len(all_outputs), 100 * n_success / len(all_outputs))
    logger.info('Self-classified: %d thin, %d thick, %d unknown', n_thin, n_thick, n_success - n_thin - n_thick)
    logger.info('Saved to %s', programs_path)


if __name__ == '__main__':
    main()
