#!/usr/bin/env python3
"""Direction 2: STaR-like feature extraction with thin/thick classification.

For each example, generate features_for and features_against with:
- name, description
- verifiability: thin (programmable) or thick (needs judgment)
- inputs_needed: what info the check requires
- expected_output: what the check produces

Uses vLLM offline batch inference for throughput.
"""
import argparse
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

STAR_PROMPT = """You are analyzing a {text_type} to identify specific features that predict whether it will be {positive_label} or {negative_label}.

Study this text carefully:

--- TEXT ---
{text}
--- END TEXT ---

Identify 5-8 specific, concrete features of THIS text. For each feature:
- Argue whether it predicts {positive_label} (FOR) or {negative_label} (AGAINST)
- Classify it as THIN (could be checked by a Python program using regex, counting, structural analysis — no understanding needed) or THICK (requires reading comprehension, domain expertise, or subjective judgment to evaluate)
- Describe what inputs would be needed to check it and what the output would be

Output as JSON:

```json
{{
  "features_for": [
    {{
      "name": "short name",
      "description": "what this feature is and why it predicts {positive_label}",
      "verifiability": "thin or thick",
      "inputs_needed": "what information is needed to check this",
      "expected_output": "what the check produces (bool, count, score, etc.)"
    }}
  ],
  "features_against": [
    {{
      "name": "short name",
      "description": "what this feature is and why it predicts {negative_label}",
      "verifiability": "thin or thick",
      "inputs_needed": "what information is needed to check this",
      "expected_output": "what the check produces"
    }}
  ]
}}
```

Be specific to THIS text. Don't list generic criteria — identify concrete patterns you actually observe. Return ONLY the JSON."""


def build_prompts(texts, text_type, positive_label, negative_label):
    return [
        STAR_PROMPT.format(
            text_type=text_type,
            positive_label=positive_label,
            negative_label=negative_label,
            text=text[:3000],
        )
        for text in texts
    ]


def extract_json(response):
    match = re.search(r"```json\s*\n(.*?)```", response, re.DOTALL)
    text = match.group(1).strip() if match else response.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    # Try to find JSON object in response
    match2 = re.search(r"\{.*\}", response, re.DOTALL)
    if match2:
        try:
            return json.loads(match2.group(0))
        except json.JSONDecodeError:
            pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', default='peer-review')
    parser.add_argument('--n-examples', type=int, default=20000)
    parser.add_argument('--max-model-len', type=int, default=4096)
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.90)
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--output-dir', default='outputs/verification_library/star_features_20k')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_configs = {
        'math-stackexchange': {
            'split_dir': REPO_ROOT / 'datasets' / 'math-stackexchange' / 'splits',
            'text_col': 'text', 'label_col': 'label', 'id_col': 'id',
            'text_type': 'math question and answer from StackExchange',
            'positive_label': 'highly upvoted (good answer)',
            'negative_label': 'not highly upvoted (poor answer)',
        },
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
    if ds['id_col'] and ds['id_col'] in sample.columns:
        ids = sample[ds['id_col']].astype(str).tolist()
    else:
        ids = [str(i) for i in sample.index]

    logger.info('Sampled %d examples (label dist: %d pos / %d neg)',
                len(texts), sum(labels), len(labels) - sum(labels))

    # Check cache
    features_path = out_dir / 'star_features.jsonl'
    existing_ids = set()
    if features_path.exists():
        with open(features_path) as f:
            for line in f:
                obj = json.loads(line)
                existing_ids.add(obj['example_id'])
        logger.info('Found %d cached feature extractions', len(existing_ids))

    todo = [(t, l, eid) for t, l, eid in zip(texts, labels, ids) if eid not in existing_ids]
    if not todo:
        logger.info('All cached!')
        return

    todo_texts = [x[0] for x in todo]
    todo_labels = [x[1] for x in todo]
    todo_ids = [x[2] for x in todo]

    logger.info('Extracting features for %d examples', len(todo_texts))
    prompts = build_prompts(todo_texts, ds['text_type'], ds['positive_label'], ds['negative_label'])

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

    # Parse and save
    n_success = 0
    n_features_total = 0
    n_thin = 0
    n_thick = 0

    with open(features_path, 'a') as f:
        for j, output in enumerate(all_outputs):
            response = output.outputs[0].text
            parsed = extract_json(response)

            record = {
                'example_id': todo_ids[j],
                'label': todo_labels[j],
                'parse_success': parsed is not None,
                'features_for': [],
                'features_against': [],
                'generated_at': datetime.now().isoformat(),
            }

            if parsed:
                n_success += 1
                for key in ('features_for', 'features_against'):
                    features = parsed.get(key, [])
                    if isinstance(features, list):
                        record[key] = features
                        n_features_total += len(features)
                        for feat in features:
                            v = feat.get('verifiability', '').lower()
                            if 'thin' in v:
                                n_thin += 1
                            elif 'thick' in v:
                                n_thick += 1

            f.write(json.dumps(record) + '\n')

    logger.info('Done: %d/%d parsed successfully (%.0f%%)',
                n_success, len(all_outputs), 100 * n_success / len(all_outputs))
    logger.info('Total features: %d (%d thin, %d thick)',
                n_features_total, n_thin, n_thick)
    logger.info('Saved to %s', features_path)


if __name__ == '__main__':
    main()
