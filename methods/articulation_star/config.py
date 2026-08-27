"""Config + task registry for the articulation-STaR loop."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass
class TaskSpec:
    name: str
    data_path: str            # relative to repo root, CSV(.gz) with `text` + `judgement`
    text_type: str            # e.g. "scientific paper abstract"
    positive_label: str       # e.g. "accepted"
    negative_label: str       # e.g. "rejected"


TASKS: Dict[str, TaskSpec] = {
    "peer_review": TaskSpec(
        name="peer_review",
        data_path="datasets/peer-review/splits/train.csv.gz",
        text_type="scientific paper abstract",
        positive_label="accepted",
        negative_label="rejected",
    ),
    "press_releases": TaskSpec(
        name="press_releases",
        data_path="datasets/press-releases/press_release_modeling_dataset.csv.gz",
        text_type="press release",
        positive_label="picked up by news outlets",
        negative_label="ignored by news outlets",
    ),
    "creative_writing": TaskSpec(
        name="creative_writing",
        data_path="datasets/creative-writing/litbench-to-train.csv.gz",
        text_type="short creative-writing story",
        positive_label="upvoted by readers",
        negative_label="ignored by readers",
    ),
}


@dataclass
class LoopConfig:
    # ── identity ─────────────────────────────────────────────
    task: str = "peer_review"
    run_name: str = "v0"

    # ── models ───────────────────────────────────────────────
    # Generator: small enough to LoRA-train on one shared GPU.
    generator_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    # Strong judge — different family from generator so steganographic
    # gaming is harder. Default = Qwen3.5-122B-A10B-FP8 cached on sk3 (see
    # [[reference_qwen35_vllm_sk3]]). Pass an HF id at runtime to override.
    judge_model: str = (
        "/lfs/skampere3/0/shared_hf_cache/models--Qwen--Qwen3.5-122B-A10B-FP8/"
        "snapshots/a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9"
    )
    # Weak judge — small enough that it can only pattern-match sentiment.
    # If both weak and strong can decode the label, the rationale is leaking
    # via cheap surface cues; drop it. If weak fails but strong succeeds,
    # the rationale carries information only a strong reader can recover —
    # our proxy for "substantive articulation."
    # Weak judge — dropped from 3B to 1B 2026-05-31 because 3B was hitting
    # nearly the same accuracy as the strong judge (55% vs 54%), defeating
    # the contrastive design. With 1B we expect a real strong-weak gap so
    # the contrastive filter actually isolates substantive articulations.
    # Weak judge stays FIXED across iters (no co-training): its purpose is
    # to be a cheap-reader probe of "is this rationale leaking via surface
    # cues?", and training it would defeat that interpretation.
    weak_judge_model: str = (
        "/lfs/skampere3/0/alexspan/.cache/huggingface/hub/"
        "models--meta-llama--Llama-3.2-1B-Instruct/snapshots/"
        "9213176726f574b556790deb65791e0c5aa438b6"
    )

    # ── data sizing ──────────────────────────────────────────
    # Per-iter, take a random subsample of train rows (full 56K is wasteful).
    n_train_subsample: int = 4_000
    # Number of rationale samples per artifact (STaR pass@k style).
    n_rationales_per_input: int = 4
    # Cap on artifact text characters in the prompt.
    max_text_chars: int = 4_000

    # ── generator decoding ──────────────────────────────────
    gen_temperature: float = 0.9
    gen_top_p: float = 0.95
    gen_max_tokens: int = 512

    # ── judge decoding ──────────────────────────────────────
    # Qwen3.5 is a reasoning model; even with /no_think it may emit short CoT.
    # Give it enough headroom to land on the ANSWER line; we parse the last
    # occurrence of either label phrase out of the full text.
    judge_max_tokens: int = 256
    judge_temperature: float = 0.0
    # Weak judge gets less headroom since it's not a reasoning model.
    weak_judge_max_tokens: int = 32

    # ── training ─────────────────────────────────────────────
    n_iters: int = 3
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 1e-4
    n_epochs: int = 1
    per_device_batch_size: int = 2
    grad_accum_steps: int = 8
    max_seq_len: int = 4_096

    # ── vLLM hardware ────────────────────────────────────────
    # Per [[feedback_gpu_usage]] / [[feedback_minimize_gpus]] — single GPU.
    vllm_gpu_mem_util: float = 0.90
    vllm_max_model_len: int = 8_192

    # ── output ───────────────────────────────────────────────
    output_root: str = "outputs/articulation_star"

    def iter_dir(self, iter_idx: int) -> Path:
        return Path(self.output_root) / self.task / self.run_name / f"iter_{iter_idx:02d}"
