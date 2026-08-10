"""4 tacitness assay, step 1 - make a trained executor articulate its own installed policy.

After an intervention (distillation / reward training) rescues a cell, ask the TRAINED
executor to state the policy it now executes, in the standard articulation registers
(definition / rules / dossier). Output is arm-bank-style JSON so the standard parts-1-2
pipeline can hand the articulation to a FRESH executor; the articulation-transfer gap
  rho(trained executor) - rho(fresh executor + self-articulation)
is the tacitness score: high gap = executable-but-inarticulable.

GPU script (offline vLLM via the shared backend; adapter routed by cfg.vllm_lora_path).

Usage:
  python -m methods.tacit_channels.channels.assay.elicit_articulation \
      --model Qwen/Qwen2.5-7B-Instruct --lora-adapter outputs/.../humor_x_n32 \
      --construct "Comedic escalation discipline" --domain humor \
      --k 4 --out outputs/tacit_channels/assay/humor_x_selfartic.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REGISTERS = {
    "self_definition": (
        "You apply a private evaluation criterion called \"{construct}\" when judging "
        "{domain} items. State, in 1-3 sentences, the definition of the criterion you "
        "actually apply - the distinction that drives your YES/NO judgments. Write only the "
        "definition."),
    "self_rules": (
        "You apply a private evaluation criterion called \"{construct}\" when judging "
        "{domain} items. Write the explicit decision rules you actually follow as a short "
        "numbered list (3-7 rules), concrete enough that another evaluator could apply them "
        "to a new item without asking you anything. Write only the rules."),
    "self_dossier": (
        "You apply a private evaluation criterion called \"{construct}\" when judging "
        "{domain} items. Write a compact dossier of the policy you actually apply: the core "
        "distinction, the boundary cases that flip your judgment, and the features you "
        "ignore even though others might weight them. Max 200 words."),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora-adapter", default=None)
    ap.add_argument("--construct", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--cell-id", default=None)
    ap.add_argument("--k", type=int, default=4, help="samples per register")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from methods.metric_implementer import config as cfgmod
    from methods.metric_implementer.vllm_backend import make_judge_backend
    cfg = cfgmod.ImplementerConfig()
    cfg.vllm_fake = args.fake
    cfg.vllm_tp_size = args.tp_size
    if args.lora_adapter:
        cfg.vllm_lora_path = str(args.lora_adapter)
    backend = make_judge_backend(args.model, cfg, temperature=args.temperature)

    prompts, keys = [], []
    for register, template in REGISTERS.items():
        for i in range(args.k):
            prompts.append(template.format(construct=args.construct, domain=args.domain))
            keys.append((register, i))
    # distinct seeds per sample -> k independent articulations at fixed temperature
    seeds = [20260722 + 7919 * i for i in range(len(prompts))]
    texts = backend.generate_batch(prompts, max_tokens=args.max_tokens, seed=seeds) \
        if "seed" in backend.generate_batch.__code__.co_varnames else \
        backend.generate_batch(prompts, max_tokens=args.max_tokens)

    arms = []
    for (register, i), text in zip(keys, texts):
        content = f"{args.construct}\n\n{text.strip()}"
        arms.append({
            "id": f"{register}_s{i}", "channel": "self_articulation",
            "provenance": f"trained_executor::{args.lora_adapter or 'base'}",
            "control_for": None,
            "forms": [{"id": "canonical", "prompt": content}],
        })
    payload = {
        "schema": "tacit_channels_self_articulation/v1",
        "construct": args.construct, "domain": args.domain, "cell_id": args.cell_id,
        "model": args.model, "lora_adapter": args.lora_adapter,
        "temperature": args.temperature, "k": args.k,
        "arms": arms,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {len(arms)} self-articulation arms -> {args.out}")


if __name__ == "__main__":
    main()
