"""S3: the decode hop. For each (metric, candidate): sample (probe-text-excerpt, score)
pairs from the encode pass, ask the DECODER (qwen-72b via OpenRouter — family-disjoint;
sees NO definitions, NO labels, NO candidate text) to reconstruct the scoring criterion.
Output: mo_reconstructions.json (metric, candidate form -> rubric_hat) + the re-execute
manifests (form_idx 800+j) for the sk1 scorer. Runs on sk3."""
import json
import re
import urllib.request
from pathlib import Path

import numpy as np

MD = Path("/lfs/skampere3/0/alexspan/mention_auc")
KEY = open("/lfs/skampere3/0/alexspan/.openrouter-api-key.txt").read().strip()

DEC = """You will see excerpts from {n} documents, each with a numeric score in [0,1]
assigned by an evaluator following ONE consistent hidden criterion. Infer the criterion.

{pairs}

Reply with ONLY your best reconstruction of the criterion as a single self-contained
instruction (1-3 sentences), concrete enough that another evaluator could apply it."""


def decode(pairs_txt, n):
    body = {"model": "qwen/qwen-2.5-72b-instruct", "max_tokens": 220, "temperature": 0.0,
            "messages": [{"role": "user", "content": DEC.format(n=n, pairs=pairs_txt)}]}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
                                 data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {KEY}",
                                          "Content-Type": "application/json"})
    for _ in range(5):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"].strip()
        except Exception:
            import time
            time.sleep(4)
    return None


texts = {}
for task, tf in (("humor", "humor_probe_texts.jsonl"), ("peer", "peer_probe_texts.jsonl"),
                 ("cw", "cw_probe_texts.jsonl")):
    texts[task] = {}
    for line in open(MD / tf):
        r = json.loads(line)
        texts[task][r["probe_id"]] = r["text"]

recon = []
for task in ("humor", "peer", "cw"):
    f = MD / f"mo_{task}_probes8b.json"
    if not f.exists():
        print(f"missing encode for {task}")
        continue
    d = json.load(open(f))
    ids, S = d["post_ids"], d["scores"]
    for key, vec in sorted(S.items()):
        v = np.asarray(vec, float)
        fin = np.where(np.isfinite(v))[0]
        if len(fin) < 60:
            continue
        # stratified sample of 14 (7 high / 7 low) for the decoder
        order = fin[np.argsort(v[fin])]
        pick = list(order[:7]) + list(order[-7:])
        pairs = "\n\n".join(
            f"--- doc {j+1} (score {v[i]:.2f}):\n{texts[task][ids[i]][:450]}"
            for j, i in enumerate(pick))
        hat = decode(pairs, len(pick))
        if hat and 20 < len(hat) < 1500:
            hat = re.sub(r"^```.*?\n|```$", "", hat, flags=re.S).strip()
            recon.append({"task": task, "key": key, "rubric_hat": hat})
            print(f"{task} {key} decoded ({len(hat)} ch)")
json.dump(recon, open(MD / "mo_reconstructions.json", "w"), indent=1)

per = {}
for r in recon:
    mid, fi = r["key"].split("__")
    per.setdefault(r["task"], []).append(
        {"metric_id": mid, "form_idx": 100 + int(fi), "rubric": r["rubric_hat"]})
for task, rows in per.items():
    json.dump(rows, open(MD / f"mo_{task}_hat_manifest.json", "w"), indent=0)
    print(task, len(rows), "reconstructions -> hat manifest")
