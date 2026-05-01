"""Core explanation generation: per-example feature extraction via VLLM batch inference."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import LocalExplanationConfig, TaskMetadata
from .parsing import StarLocalResult, parse_feature_list_json, parse_star_local_output
from .prompts import render_rationalization_prompt, render_star_local_prompt

logger = logging.getLogger(__name__)


# =============================================================================
# Explanation Cache (JSONL-backed)
# =============================================================================

class ExplanationCache:
    """JSONL-backed cache for raw LLM explanation responses."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem: Dict[str, Dict[str, str]] = {}  # {explanation_type: {doc_id: response}}

    def _path(self, explanation_type: str) -> Path:
        return self.cache_dir / f"{explanation_type}.jsonl"

    def _load(self, explanation_type: str) -> Dict[str, str]:
        if explanation_type in self._mem:
            return self._mem[explanation_type]
        path = self._path(explanation_type)
        data = {}
        if path.exists():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        data[rec["id"]] = rec["response"]
                    except (json.JSONDecodeError, KeyError):
                        continue
        self._mem[explanation_type] = data
        return data

    def get(self, explanation_type: str, doc_id: str) -> Optional[str]:
        return self._load(explanation_type).get(doc_id)

    def get_available_ids(self, explanation_type: str) -> set:
        return set(self._load(explanation_type).keys())

    def store(self, explanation_type: str, doc_id: str, response: str):
        data = self._load(explanation_type)
        data[doc_id] = response
        with open(self._path(explanation_type), "a") as f:
            f.write(json.dumps({"id": doc_id, "response": response}) + "\n")

    def store_batch(self, explanation_type: str, doc_ids: List[str], responses: List[str]):
        data = self._load(explanation_type)
        with open(self._path(explanation_type), "a") as f:
            for doc_id, response in zip(doc_ids, responses):
                data[doc_id] = response
                f.write(json.dumps({"id": doc_id, "response": response}) + "\n")


# =============================================================================
# Batch text generation utility
# =============================================================================

def generate_text_batch(
    backend: Any,
    prompts: List[str],
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> List[str]:
    """Batch text generation via VLLM offline backend.

    Wraps the backend's LLM to generate responses for multiple prompts
    in a single batch call. Does NOT modify the backend object.
    """
    from vllm import SamplingParams

    # Format prompts using the tokenizer's chat template
    formatted = []
    tokenizer = backend._tokenizer
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        # enable_thinking=False disables Qwen3/3.5 reasoning blocks.
        # HF tokenizers pass unknown kwargs to the Jinja template context;
        # non-Qwen tokenizers silently ignore it.
        try:
            fmt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            try:
                fmt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                fmt = f"USER: {prompt}\n\nASSISTANT:"
        formatted.append(fmt)

    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    vllm_outputs = backend.llm.generate(formatted, params)
    return [
        vout.outputs[0].text.strip() if vout.outputs else ""
        for vout in vllm_outputs
    ]


# =============================================================================
# Local Explainer
# =============================================================================

class LocalExplainer:
    """Generate per-example explanations via VLLM batch inference."""

    def __init__(
        self,
        config: LocalExplanationConfig,
        scoring_backend: Any,
        task_description: str,
        task_metadata: TaskMetadata,
        cache_dir: str,
    ):
        self.config = config
        self.backend = scoring_backend
        self.task_description = task_description
        self.meta = task_metadata
        self.cache = ExplanationCache(cache_dir)
        # Chunk size for incremental generation + caching. Smaller = more
        # frequent cache writes (so a killed run loses less work and you can
        # peek at outputs mid-flight) at the cost of slightly more vLLM batch
        # boundary overhead. 1000 is a good trade-off.
        self.chunk_size = getattr(config, "explanation_chunk_size", 1000)

    def _generate_in_chunks(
        self,
        prompts: List[str],
        chunk_doc_ids: List[str],
        explanation_type: str,
    ):
        """Generate responses for `prompts` in chunks, caching after each chunk.

        Each chunk's outputs are written to the JSONL cache before the next
        chunk starts, so a killed/crashed run loses at most one chunk and the
        cache can be inspected mid-run.
        """
        n = len(prompts)
        if n == 0:
            return
        n_chunks = (n + self.chunk_size - 1) // self.chunk_size
        for chunk_idx in range(n_chunks):
            start = chunk_idx * self.chunk_size
            end = min(start + self.chunk_size, n)
            chunk_prompts = prompts[start:end]
            chunk_ids = chunk_doc_ids[start:end]
            logger.info(
                "%s chunk %d/%d: %d prompts (%d-%d of %d)",
                explanation_type, chunk_idx + 1, n_chunks,
                len(chunk_prompts), start + 1, end, n,
            )
            responses = generate_text_batch(
                self.backend, chunk_prompts,
                max_tokens=self.config.explanation_max_tokens,
                temperature=self.config.explanation_temperature,
            )
            self.cache.store_batch(explanation_type, chunk_ids, responses)
            logger.info("%s chunk %d/%d cached", explanation_type, chunk_idx + 1, n_chunks)

    def explain_rationalization_batch(
        self,
        texts: List[str],
        labels: List[int],
        doc_ids: List[str],
    ) -> Dict[str, List[str]]:
        """Approach A: For each (text, label), generate feature explanations.

        Returns: dict mapping doc_id -> list of extracted feature strings.
        """
        explanation_type = "rationalization_v1"
        cached_ids = self.cache.get_available_ids(explanation_type)

        # Identify uncached examples
        uncached_idx = [i for i, did in enumerate(doc_ids) if did not in cached_ids]
        if uncached_idx:
            logger.info(
                "Rationalization: %d cached, %d to generate",
                len(doc_ids) - len(uncached_idx), len(uncached_idx),
            )
            # Render prompts
            prompts = []
            for i in uncached_idx:
                label_word = self.meta.label_words.get(labels[i], str(labels[i]))
                prompts.append(render_rationalization_prompt(
                    task_description=self.task_description,
                    text_type=self.meta.text_type,
                    label_word=label_word,
                    text=texts[i],
                    n_features=self.config.features_per_example,
                ))

            # Chunked batch generate + cache (resumable, inspectable mid-flight)
            self._generate_in_chunks(
                prompts,
                [doc_ids[i] for i in uncached_idx],
                explanation_type,
            )

        # Parse all (cached + new)
        results = {}
        n_parse_failures = 0
        for doc_id in doc_ids:
            raw = self.cache.get(explanation_type, doc_id)
            if raw:
                features = parse_feature_list_json(raw)
                if not features:
                    n_parse_failures += 1
                results[doc_id] = features
            else:
                results[doc_id] = []
                n_parse_failures += 1

        if n_parse_failures:
            logger.warning(
                "Rationalization parse failures: %d / %d (%.1f%%)",
                n_parse_failures, len(doc_ids),
                100 * n_parse_failures / max(len(doc_ids), 1),
            )
        return results

    def explain_star_local_batch(
        self,
        texts: List[str],
        doc_ids: List[str],
    ) -> Dict[str, StarLocalResult]:
        """Approach B: Blind extraction + deliberation + prediction.

        Returns: dict mapping doc_id -> StarLocalResult.
        """
        explanation_type = "star_local_v1"
        cached_ids = self.cache.get_available_ids(explanation_type)

        uncached_idx = [i for i, did in enumerate(doc_ids) if did not in cached_ids]
        if uncached_idx:
            logger.info(
                "STaR-Local: %d cached, %d to generate",
                len(doc_ids) - len(uncached_idx), len(uncached_idx),
            )
            prompts = []
            # Use the registered task name so few-shot examples flow through.
            # Reverse-lookup the dataset_name from the task metadata registry.
            from .config import TASK_METADATA_REGISTRY
            dataset_name = ""
            for name, md in TASK_METADATA_REGISTRY.items():
                if md is self.meta:
                    dataset_name = name
                    break
            for i in uncached_idx:
                prompts.append(render_star_local_prompt(
                    task_description=self.task_description,
                    text=texts[i],
                    positive_label=self.meta.positive_label,
                    negative_label=self.meta.negative_label,
                    text_type=self.meta.text_type,
                    dataset_name=dataset_name,
                ))

            # Chunked batch generate + cache (resumable, inspectable mid-flight)
            self._generate_in_chunks(
                prompts,
                [doc_ids[i] for i in uncached_idx],
                explanation_type,
            )

        # Parse all
        results = {}
        n_parse_failures = 0
        for doc_id in doc_ids:
            raw = self.cache.get(explanation_type, doc_id)
            if raw:
                parsed = parse_star_local_output(
                    raw, self.meta.positive_label, self.meta.negative_label,
                )
                if not parsed.features_for and not parsed.features_against:
                    n_parse_failures += 1
                results[doc_id] = parsed
            else:
                results[doc_id] = StarLocalResult(
                    features_for=[], features_against=[],
                    prediction="", confidence="LOW",
                )
                n_parse_failures += 1

        if n_parse_failures:
            logger.warning(
                "STaR-Local parse failures: %d / %d (%.1f%%)",
                n_parse_failures, len(doc_ids),
                100 * n_parse_failures / max(len(doc_ids), 1),
            )
        return results
