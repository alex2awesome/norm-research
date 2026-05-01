#!/usr/bin/env python3
"""Re-parse raw_responses.jsonl with improved code extraction."""
import argparse
import ast
import json
import re
from pathlib import Path
from collections import Counter


def extract_code(response):
    # Truncate very long responses to avoid regex catastrophic backtracking
    response = response[:10000]

    # 1. ```python ... ``` — find start and end separately to avoid backtracking
    start = response.find('```python')
    if start >= 0:
        code_start = response.find('\n', start)
        if code_start >= 0:
            end = response.find('```', code_start + 1)
            if end >= 0:
                return response[code_start + 1:end].strip()

    # 2. ``` ... ```
    start = response.find('```\n')
    if start >= 0:
        end = response.find('```', start + 4)
        if end >= 0:
            code = response[start + 4:end].strip()
            if 'def ' in code:
                return code

    # 3. Find def check/predict and grab indented block
    for func_name in ['check', 'predict']:
        pattern = f'def {func_name}(text'
        idx = response.find(pattern)
        if idx < 0:
            pattern = f'def {func_name}('
            idx = response.find(pattern)
        if idx >= 0:
            # Find start of line
            line_start = response.rfind('\n', 0, idx) + 1
            # Collect lines: the def line + all indented lines after
            lines = response[line_start:].split('\n')
            result = [lines[0]]
            for line in lines[1:]:
                if line.startswith('    ') or line.startswith('\t') or line.strip() == '':
                    result.append(line)
                elif line.startswith('import ') or line.startswith('from ') or line.startswith('#'):
                    result.append(line)
                else:
                    break
            # Also grab imports before the def
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', required=True)
    parser.add_argument('--features', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    features = {}
    with open(args.features) as f:
        for line in f:
            feat = json.loads(line)
            features[feat["cluster_id"]] = feat

    # Format distribution
    n_bp, n_bpl, n_raw, n_cont, n_nothing = 0, 0, 0, 0, 0
    with open(args.raw) as f:
        for line in f:
            r = json.loads(line)["response"]
            if re.search(r"```python", r):
                n_bp += 1
            elif re.search(r"```", r):
                n_bpl += 1
            elif re.search(r"def\s+(check|predict)\s*\(", r):
                n_raw += 1
            elif "return " in r:
                n_cont += 1
            else:
                n_nothing += 1

    total = n_bp + n_bpl + n_raw + n_cont + n_nothing
    print(f"Response formats ({total} total):")
    print(f"  ```python: {n_bp} ({100*n_bp//total}%)")
    print(f"  ``` plain: {n_bpl}")
    print(f"  raw def: {n_raw}")
    print(f"  continuation: {n_cont}")
    print(f"  nothing: {n_nothing} ({100*n_nothing//total}%)")

    # Parse
    n_success, n_thin, n_thick, n_syntax = 0, 0, 0, 0
    n_total = 0
    with open(args.raw) as f, open(args.output, 'w') as out:
        for line in f:
            obj = json.loads(line)
            cid = obj["cluster_id"]
            code = extract_code(obj["response"])
            n_total += 1

            classification = "failed"
            success = code is not None
            if code:
                try:
                    tree = ast.parse(code)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            ds = ast.get_docstring(node)
                            if ds:
                                up = ds.strip().upper()
                                if up.startswith("THICK:"):
                                    classification = "thick"
                                elif up.startswith("THIN:"):
                                    classification = "thin"
                                else:
                                    classification = "unknown"
                            break
                except SyntaxError:
                    success = False
                    classification = "syntax_error"
                    n_syntax += 1

            feat = features.get(cid, {})
            record = {
                "cluster_id": cid,
                "canonical_name": feat.get("canonical_name", ""),
                "frequency": feat.get("frequency", 0),
                "lift": feat.get("lift", 0),
                "llm_verifiability": feat.get("verifiability", ""),
                "code_classification": classification,
                "source_code": code or "",
                "execution_success": success,
            }
            out.write(json.dumps(record) + '\n')
            if success:
                n_success += 1
                if classification == "thin":
                    n_thin += 1
                elif classification == "thick":
                    n_thick += 1

    print(f"\nParse results:")
    print(f"  {n_success}/{n_total} successful ({100*n_success//max(n_total,1)}%)")
    print(f"  Syntax errors: {n_syntax}")
    print(f"  Self-classified: {n_thin} thin, {n_thick} thick, {n_success - n_thin - n_thick} unknown")

    # Compare Qwen vs Llama classification
    if n_success > 0:
        agree_thin = 0
        agree_thick = 0
        llama_thick_qwen_thin = 0
        llama_thin_qwen_thick = 0
        with open(args.output) as f:
            for line in f:
                r = json.loads(line)
                if not r["execution_success"]:
                    continue
                llama = r["llm_verifiability"]
                qwen = r["code_classification"]
                if "thin" in llama and qwen == "thin":
                    agree_thin += 1
                elif "thick" in llama and qwen == "thick":
                    agree_thick += 1
                elif "thick" in llama and qwen == "thin":
                    llama_thick_qwen_thin += 1
                elif "thin" in llama and qwen == "thick":
                    llama_thin_qwen_thick += 1

        print(f"\nLlama vs Qwen classification (on {n_success} successful):")
        print(f"  Both thin: {agree_thin}")
        print(f"  Both thick: {agree_thick}")
        print(f"  Llama=thick, Qwen=thin: {llama_thick_qwen_thin} (Qwen found a program!)")
        print(f"  Llama=thin, Qwen=thick: {llama_thin_qwen_thick} (Qwen couldn't code it)")


if __name__ == '__main__':
    main()
