#!/usr/bin/env python3
"""Distill the v2 silver-matcher decision stage (Sonnet) into Gemma + GEPA'd prompt (sk3, 1 GPU).

Question (user 2026-07-11): can GEPA+Gemma reach Sonnet-equivalent agreement on the choose-from-top10
+ ABSTAIN + NOISE decision, so the remaining 26-corpora re-match runs free on local batch vLLM?

Design (mirrors the certified Gemma-4+GEPA >= Sonnet precedent from peer-review silver extraction):
  data    silver_v2 decisions {norm, top10, label in top10|ABSTAIN|NOISE}, stable-hash split 8/1/1
          (md5(idx) -- never seeded-shuffle a growing list).
  arms    (a) always-top1 baseline (= v1 behavior); (b) VANILLA-prompt Gemma -- this number doubles as
          the INDEPENDENT cross-family convergence credential (pre-optimization, not distilled);
          (c) GEPA'd Gemma (reviser = GLM-4.7 via zai_anthropic, few rounds, dev-set agreement reward).
  eval    frozen TEST slice only: overall agreement w/ Sonnet, per-class recall (match/ABSTAIN/NOISE),
          Cohen kappa. Certify Gemma-matcher only if agreement within ~3pts of Sonnet self-consistency.
  scoring offline batch vLLM (envs/gemma4), generation + exact/fuzzy match to the candidate list;
          bad outputs -> retry different seed (never repetition_penalty).

Usage (sk3):
  HOME=/lfs/skampere3/0/alexspan CUDA_VISIBLE_DEVICES=<free> \
    envs/gemma4/bin/python distill_matcher_gemma.py --data silver_v2_decisions.jsonl \
    --rounds 3 --mutations 4 --n-dev 800 --n-test 2000
NOTE: data exporter = scripts/tools/export_v2_decisions.py (run after the full re-match completes).
"""
# -- implementation stub staged 2026-07-11; flesh out against sk3 gemma4 env when connectivity returns.
# The GEPA loop reuses run_gepa_for_plot.py's accept-if-better skeleton with agreement as R.
raise SystemExit("staged: complete against sk3 gemma4 env before running (see docstring)")
