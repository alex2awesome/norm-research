"""Spot-test (Alex's ask): can SMALL LLMs replace Qwen3.5-122B for the V2
per-element disclosure check? 150 calibration pairs (75 gold / 75 mismatched),
same prompt, via OpenRouter. Also: element decomposition on 15 claims.

Run on laptop: python test_small_llm_element_judge_openrouter.py
"""
import json, os, re, time
from concurrent.futures import ThreadPoolExecutor

import requests

KEY = open(os.path.expanduser("~/.openrouter-api-key.txt")).read().strip()
URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS = [
    "meta-llama/llama-3.1-8b-instruct",
    "qwen/qwen3-14b",
    "google/gemma-3-12b-it",
    "qwen/qwen3-32b",
]
SYS = ("You are a USPTO patent examiner assistant. Judge whether a prior-art "
       "paragraph discloses a specific claim element under 35 USC 102 with "
       "broadest reasonable interpretation. The paragraph need not use the "
       "same words — it discloses the element if a person of ordinary skill "
       "would understand the element to be described.")
JSON_RE = re.compile(r'\{[^{}]*"disclosed"[^{}]*\}')


def call(model, messages, max_tokens=300, temperature=0.0, tries=3):
    for a in range(tries):
        try:
            r = requests.post(URL, timeout=120,
                              headers={"Authorization": f"Bearer {KEY}"},
                              json={"model": model, "messages": messages,
                                    "max_tokens": max_tokens,
                                    "temperature": temperature,
                                    "reasoning": {"enabled": False}})
            j = r.json()
            return j["choices"][0]["message"]["content"]
        except Exception as e:
            if a == tries - 1:
                return f"__ERROR__ {e} {str(j)[:200] if 'j' in dir() else ''}"
            time.sleep(2 * (a + 1))


def parse(txt):
    m = JSON_RE.findall(txt or "")
    if not m:
        return None
    try:
        j = json.loads(m[-1])
        return bool(j["disclosed"]), max(0, min(100, int(j.get("confidence", 50))))
    except Exception:
        return None


pairs = [json.loads(l) for l in open("/tmp/element_judge_sample150.jsonl")]
print(f"{len(pairs)} pairs; gold rate {sum(p['label'] for p in pairs)/len(pairs):.2f}")

from sklearn.metrics import roc_auc_score

# 122B reference on this same subset
ys = [p["label"] for p in pairs]
ref = [(p["qwen122b_conf"] if p["qwen122b_disclosed"] else 100 - p["qwen122b_conf"])
       for p in pairs]
print(f"REFERENCE qwen3.5-122B (local): conf-AUC = {roc_auc_score(ys, ref):.4f}")

results = {}
for model in MODELS:
    def judge(p):
        msg = [{"role": "system", "content": SYS},
               {"role": "user", "content":
                f"CLAIM ELEMENT:\n{p['element']}\n\nPRIOR-ART PARAGRAPH:\n{p['para']}"
                "\n\nDoes the paragraph disclose this claim element? Reason in at "
                "most 3 short sentences, then output exactly one final JSON line:\n"
                '{"disclosed": <true|false>, "confidence": <0-100>}'}]
        return parse(call(model, msg))
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        verdicts = list(ex.map(judge, pairs))
    ok = [(y, v) for y, v in zip(ys, verdicts) if v is not None]
    if len(ok) < 50:
        print(f"{model}: PARSE FAILURE ({len(ok)}/150 parsed)")
        continue
    y2 = [y for y, _ in ok]
    s2 = [(c if d else 100 - c) for _, (d, c) in ok]
    b2 = [int(d) for _, (d, _) in ok]
    agree = [int(d) == int(p["qwen122b_disclosed"])
             for (y, (d, c)), p in zip(zip(ys, verdicts), pairs) if (d, c) is not None
             ] if False else None
    results[model] = {
        "parsed": len(ok),
        "conf_auc": roc_auc_score(y2, s2),
        "bin_auc": roc_auc_score(y2, b2),
        "secs": round(time.time() - t0),
    }
    print(f"{model}: parsed {len(ok)}/150  conf-AUC={results[model]['conf_auc']:.4f}  "
          f"bin-AUC={results[model]['bin_auc']:.4f}  ({results[model]['secs']}s)")

# ---- decomposition comparison on 15 claims ----
print("\n--- decomposition (vs 122B element counts) ---")
dec = [json.loads(l) for l in open("/tmp/decomp_sample15.jsonl")]
for model in MODELS[:2] + [MODELS[3]]:
    def decompose(d):
        msg = [{"role": "system", "content":
                "You decompose US patent claims into their distinct limitations (elements)."},
               {"role": "user", "content":
                f"CLAIM:\n{d['claim_text']}\n\nList the claim's elements as a JSON "
                "array of short strings (one per distinct limitation, max 12). "
                "Output ONLY the JSON array."}]
        txt = call(model, msg, max_tokens=600)
        m = re.search(r"\[.*\]", txt or "", re.S)
        if not m:
            return None
        try:
            arr = json.loads(m.group(0))
            return [str(x) for x in arr] if isinstance(arr, list) and arr else None
        except Exception:
            return None
    with ThreadPoolExecutor(5) as ex:
        outs = list(ex.map(decompose, dec))
    ok = [(o, d) for o, d in zip(outs, dec) if o]
    diffs = [abs(len(o) - len(d["qwen122b_elements"])) for o, d in ok]
    print(f"{model}: decomposed {len(ok)}/15; |#els - 122B#els| mean="
          f"{sum(diffs)/max(1,len(diffs)):.1f}")
    if ok and model == MODELS[0]:
        o, d = ok[0]
        print("  example small-model elements:", json.dumps(o[:4], indent=1)[:400])
        print("  122B elements:               ", json.dumps(d["qwen122b_elements"][:4], indent=1)[:400])
print("DONE")
