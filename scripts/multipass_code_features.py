#!/usr/bin/env python3
"""Multi-pass coding: run the same prompt N times, keep best result per feature.

Loads existing results, identifies failures, reruns only those,
merges best result (successful > syntax_error > failed) per cluster_id.
"""
import argparse
import ast
import json
import logging
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Same prompt as batch_code_features.py v2
CODE_FEATURE_PROMPT = """Write a Python function. Do NOT explain — just write code.

Criterion to check: "{name}"
Description: {desc}
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


def extract_code(response):
    response = response[:10000]
    start = response.find('```python')
    if start >= 0:
        code_start = response.find('\n', start)
        if code_start >= 0:
            end = response.find('```', code_start + 1)
            if end >= 0:
                return response[code_start + 1:end].strip()
    start = response.find('```\n')
    if start >= 0:
        end = response.find('```', start + 4)
        if end >= 0:
            code = response[start + 4:end].strip()
            if 'def ' in code:
                return code
    for func_name in ['check', 'predict']:
        pattern = f'def {func_name}(text'
        idx = response.find(pattern)
        if idx < 0:
            pattern = f'def {func_name}('
            idx = response.find(pattern)
        if idx >= 0:
            line_start = response.rfind('\n', 0, idx) + 1
            lines = response[line_start:].split('\n')
            result = [lines[0]]
            for line in lines[1:]:
                if line.startswith('    ') or line.startswith('\t') or line.strip() == '':
                    result.append(line)
                elif line.startswith('import ') or line.startswith('from ') or line.startswith('#'):
                    result.append(line)
                else:
                    break
            pre_lines = response[:line_start].split('\n')
            imports = []
            for pl in reversed(pre_lines):
                if pl.startswith('import ') or pl.startswith('from '):
                    imports.insert(0, pl)
                elif pl.strip() == '':
                    continue
                else:
                    break
            return '\n'.join(imports + result).strip()
    return None


def classify_code(code):
    if not code:
        return "failed", False
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                ds = ast.get_docstring(node)
                if ds:
                    up = ds.strip().upper()
                    if up.startswith("THICK:"):
                        return "thick", True
                    elif up.startswith("THIN:"):
                        return "thin", True
                    else:
                        return "unknown", True
                return "unknown", True
        return "unknown", True
    except SyntaxError:
        return "syntax_error", False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--features', required=True)
    parser.add_argument('--existing-results', required=True, help='Previous reparsed results')
    parser.add_argument('--n-passes', type=int, default=2)
    parser.add_argument('--dataset', default='peer-review')
    parser.add_argument('--max-model-len', type=int, default=4096)
    parser.add_argument('--max-tokens', type=int, default=1536)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.90)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_configs = {
        'peer-review': {'text_type': 'scientific paper', 'n_examples': 20000},
    }
    ds = ds_configs[args.dataset]

    # Load features
    features = {}
    with open(args.features) as f:
        for line in f:
            feat = json.loads(line)
            features[feat["cluster_id"]] = feat

    # Load existing results — keep best per cluster_id
    best = {}  # cluster_id -> record
    with open(args.existing_results) as f:
        for line in f:
            r = json.loads(line)
            cid = r["cluster_id"]
            best[cid] = r

    # Find failures to retry
    failures = [cid for cid, r in best.items() if not r["execution_success"]]
    logger.info('Loaded %d existing results, %d failures to retry', len(best), len(failures))

    if not failures:
        logger.info('No failures to retry!')
        return

    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype='auto',
        kv_cache_dtype='fp8',
    )
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    for pass_idx in range(args.n_passes):
        # Get current failures
        failures = [cid for cid, r in best.items() if not r["execution_success"]]
        if not failures:
            logger.info('Pass %d: no more failures!', pass_idx + 1)
            break

        logger.info('Pass %d: retrying %d failures', pass_idx + 1, len(failures))

        # Build prompts for failures
        prompts = []
        retry_cids = []
        for cid in failures:
            feat = features.get(cid, {})
            prompts.append(CODE_FEATURE_PROMPT.format(
                name=feat.get('canonical_name', ''),
                desc=feat.get('description', '')[:500],
                frequency=feat.get('frequency', 0),
                n_examples=ds['n_examples'],
                lift_pct=int(feat.get('lift', 0) * 100),
            ))
            retry_cids.append(cid)

        # Generate
        all_outputs = []
        for i in range(0, len(prompts), args.batch_size):
            batch = prompts[i:i + args.batch_size]
            outputs = llm.generate(batch, sampling, use_tqdm=True)
            all_outputs.extend(outputs)
            logger.info('Pass %d batch %d/%d done',
                        pass_idx + 1,
                        i // args.batch_size + 1,
                        (len(prompts) + args.batch_size - 1) // args.batch_size)

        # Save raw responses
        raw_path = out_dir / f'raw_responses_pass{pass_idx + 1}.jsonl'
        with open(raw_path, 'w') as rf:
            for j, output in enumerate(all_outputs):
                rf.write(json.dumps({
                    'cluster_id': retry_cids[j],
                    'response': output.outputs[0].text,
                }) + '\n')

        # Parse and update best
        n_fixed = 0
        for j, output in enumerate(all_outputs):
            cid = retry_cids[j]
            code = extract_code(output.outputs[0].text)
            classification, success = classify_code(code)

            if success and not best[cid]["execution_success"]:
                # This pass fixed a failure
                feat = features.get(cid, {})
                best[cid] = {
                    "cluster_id": cid,
                    "canonical_name": feat.get("canonical_name", ""),
                    "frequency": feat.get("frequency", 0),
                    "lift": feat.get("lift", 0),
                    "llm_verifiability": feat.get("verifiability", ""),
                    "code_classification": classification,
                    "source_code": code or "",
                    "execution_success": True,
                    "fixed_in_pass": pass_idx + 1,
                }
                n_fixed += 1

        remaining = sum(1 for r in best.values() if not r["execution_success"])
        logger.info('Pass %d: fixed %d, remaining failures: %d', pass_idx + 1, n_fixed, remaining)

    # Save merged results
    merged_path = out_dir / 'feature_programs_merged.jsonl'
    n_success = 0
    n_thin = 0
    n_thick = 0
    with open(merged_path, 'w') as f:
        for cid in sorted(best.keys()):
            r = best[cid]
            f.write(json.dumps(r) + '\n')
            if r["execution_success"]:
                n_success += 1
                if r["code_classification"] == "thin":
                    n_thin += 1
                elif r["code_classification"] == "thick":
                    n_thick += 1

    logger.info('Final: %d/%d successful (%.0f%%)', n_success, len(best), 100 * n_success / len(best))
    logger.info('Self-classified: %d thin, %d thick, %d unknown', n_thin, n_thick, n_success - n_thin - n_thick)
    logger.info('Saved to %s', merged_path)


if __name__ == '__main__':
    main()
