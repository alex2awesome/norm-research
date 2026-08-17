"""MP-1c k=3 decode hop (prereg 74d99a18c). Runs on sk3. Temp 0.7 + variant nonce."""
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

K = int(sys.argv[1])
MD = Path("/lfs/skampere3/0/alexspan/mention_auc")
KEY = open("/lfs/skampere3/0/alexspan/.openrouter-api-key.txt").read().strip()
DEC = """You will see excerpts from {n} documents, each with a numeric score in [0,1]
assigned by an evaluator following ONE consistent hidden criterion. Infer the criterion.
(Independent attempt variant {k}.)

{pairs}

Reply with ONLY your best reconstruction of the criterion as a single self-contained
instruction (1-3 sentences), concrete enough that another evaluator could apply it."""


def decode(pairs_txt, n):
    body = {"model": "qwen/qwen-2.5-72b-instruct", "max_tokens": 220,
            "temperature": 0.7,
            "messages": [{"role": "user",
                          "content": DEC.format(n=n, pairs=pairs_txt, k=K + 1)}]}
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
for line in open(MD / "peer_probe_texts.jsonl"):
    r = json.loads(line)
    texts[r["probe_id"]] = r["text"]
d = json.load(open(MD / "mp5_peer_probes8b.json"))
ids, S = d["post_ids"], d["scores"]
recon, per = [], []
for key, vec in sorted(S.items()):
    v = np.asarray(vec, float)
    fin = np.where(np.isfinite(v))[0]
    if len(fin) < 60:
        continue
    order = fin[np.argsort(v[fin])]
    pick = list(order[:7]) + list(order[-7:])
    pairs = "\n\n".join(f"--- doc {j+1} (score {v[i]:.2f}):\n{texts[ids[i]][:450]}"
                        for j, i in enumerate(pick))
    hat = decode(pairs, len(pick))
    if hat and 20 < len(hat) < 1500:
        hat = re.sub(r"^```.*?\n|```$", "", hat, flags=re.S).strip()
        recon.append({"key": key, "rubric_hat": hat})
        mid, fi = key.split("__")
        per.append({"metric_id": mid, "form_idx": 100 + int(fi), "rubric": hat})
json.dump(recon, open(MD / f"mp5_reconstructions_k{K}.json", "w"), indent=1)
json.dump(per, open(MD / f"mp5_hat_manifest_k{K}.json", "w"), indent=0)
print(f"k={K}: {len(recon)} reconstructions", flush=True)
