"""Per-example program generation via Claude."""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import List, Optional

from .client import AnthropicClient
from .config import DatasetConfig, VerificationLibraryConfig
from .program_store import GeneratedProgram
from .prompts import (
    SYSTEM_PROMPT_GENERATION,
    render_approach1_generation,
    render_approach2_generation,
    render_program_generation,
    render_program_generation_v2,
)

logger = logging.getLogger(__name__)


def _extract_code(response: str) -> Optional[str]:
    """Parse Python code from ```python ... ``` blocks."""
    pattern = r"```python\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: try to find def predict
    pattern2 = r"(def predict\(.*?\n(?:.*\n)*)"
    match2 = re.search(pattern2, response)
    if match2:
        return match2.group(1).strip()
    return None


def _extract_hypothesis(source: str) -> str:
    """Extract docstring from predict function as hypothesis description."""
    try:
        import ast
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "predict":
                ds = ast.get_docstring(node)
                if ds:
                    return ds
    except Exception:
        pass
    return ""


def _extract_v2_response(response: str) -> Optional[dict]:
    """Parse V2 JSON response with inputs + predict_code + hypothesis."""
    pattern = r"```json\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    text = match.group(1).strip() if match else response.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "predict_code" in parsed:
            return parsed
        return None
    except (json.JSONDecodeError, TypeError):
        return None


def _make_program_id(example_id: str, round_idx: int, suffix: str = "") -> str:
    h = hashlib.sha256(f"{example_id}:{round_idx}:{suffix}".encode()).hexdigest()[:12]
    return h


def _extract_check_code(response: str) -> Optional[str]:
    """Parse check() function from response (Approach 2 uses `check` not `predict`)."""
    code = _extract_code(response)
    if code is None:
        return None
    # Normalize: if it defines `check`, keep as-is. If `predict`, rename.
    code = code.replace("def predict(", "def check(")
    return code


def _extract_tools_needed(source: str) -> List[str]:
    """Extract TOOLS_NEEDED declarations from program comments."""
    tools = []
    in_tools_block = False
    for line in source.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# TOOLS_NEEDED:"):
            in_tools_block = True
            rest = stripped[len("# TOOLS_NEEDED:"):].strip()
            if rest.lower() == "none":
                return []
            if rest.startswith("- "):
                tools.append(rest[2:].strip())
            elif rest:
                tools.append(rest)
            continue
        if in_tools_block and stripped.startswith("# - "):
            tools.append(stripped[4:].strip())
        elif in_tools_block and not stripped.startswith("#"):
            break  # end of tools block
    return tools


def _classify_from_docstring(source: str) -> str:
    """Extract THIN/THICK classification from docstring convention."""
    try:
        import ast
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                ds = ast.get_docstring(node)
                if ds:
                    if ds.strip().upper().startswith("THICK:"):
                        return "thick"
                    elif ds.strip().upper().startswith("THIN:"):
                        return "thin"
    except Exception:
        pass
    return "unknown"


class ProgramGenerator:
    def __init__(
        self,
        client: AnthropicClient,
        config: VerificationLibraryConfig,
        dataset_config: DatasetConfig,
    ):
        self.client = client
        self.config = config
        self.dataset_config = dataset_config

    async def generate_for_example(
        self,
        text: str,
        label: int,
        example_id: str,
        round_idx: int,
        library_code: Optional[str] = None,
    ) -> GeneratedProgram:
        prompt = render_program_generation(
            text=text,
            label_value=label,
            text_type=self.dataset_config.text_type,
            positive_label=self.dataset_config.positive_label,
            negative_label=self.dataset_config.negative_label,
            library_code=library_code,
        )

        response = await self.client.generate(
            prompt=prompt,
            max_tokens=self.config.max_tokens_program,
            temperature=self.config.temperature_program,
            system=SYSTEM_PROMPT_GENERATION,
        )

        source = _extract_code(response)
        if source is None:
            return GeneratedProgram(
                program_id=_make_program_id(example_id, round_idx),
                example_id=example_id,
                round_idx=round_idx,
                source_code="",
                hypothesis="",
                execution_success=False,
                error_message="Could not extract code from response",
            )

        hypothesis = _extract_hypothesis(source)

        return GeneratedProgram(
            program_id=_make_program_id(example_id, round_idx),
            example_id=example_id,
            round_idx=round_idx,
            source_code=source,
            hypothesis=hypothesis,
        )

    async def generate_batch(
        self,
        texts: List[str],
        labels: List[int],
        example_ids: List[str],
        round_idx: int,
        library_code: Optional[str] = None,
        cache_path: Optional[Path] = None,
    ) -> List[GeneratedProgram]:
        """Generate programs for a batch of examples with JSONL resumability."""
        prompts = [
            render_program_generation(
                text=text,
                label_value=label,
                text_type=self.dataset_config.text_type,
                positive_label=self.dataset_config.positive_label,
                negative_label=self.dataset_config.negative_label,
                library_code=library_code,
            )
            for text, label in zip(texts, labels)
        ]

        responses = await self.client.generate_batch(
            prompts=prompts,
            max_tokens=self.config.max_tokens_program,
            temperature=self.config.temperature_program,
            system=SYSTEM_PROMPT_GENERATION,
            cache_path=cache_path,
            cache_key_fn=lambda i: example_ids[i],
        )

        programs = []
        for i, response in enumerate(responses):
            if response is None:
                prog = GeneratedProgram(
                    program_id=_make_program_id(example_ids[i], round_idx),
                    example_id=example_ids[i],
                    round_idx=round_idx,
                    source_code="",
                    hypothesis="",
                    execution_success=False,
                    error_message="No response",
                )
            else:
                source = _extract_code(response)
                if source is None:
                    prog = GeneratedProgram(
                        program_id=_make_program_id(example_ids[i], round_idx),
                        example_id=example_ids[i],
                        round_idx=round_idx,
                        source_code="",
                        hypothesis="",
                        execution_success=False,
                        error_message="Could not extract code from response",
                    )
                else:
                    prog = GeneratedProgram(
                        program_id=_make_program_id(example_ids[i], round_idx),
                        example_id=example_ids[i],
                        round_idx=round_idx,
                        source_code=source,
                        hypothesis=_extract_hypothesis(source),
                    )
            programs.append(prog)

        return programs

    # =================================================================
    # V2: Extraction-aware generation
    # =================================================================

    async def generate_for_example_v2(
        self,
        text: str,
        label: int,
        example_id: str,
        round_idx: int,
        library_code: Optional[str] = None,
        extraction_schema: Optional[List[dict]] = None,
    ) -> GeneratedProgram:
        """V2: generate program that declares its input schema."""
        schema_str = json.dumps(extraction_schema, indent=2) if extraction_schema else None

        prompt = render_program_generation_v2(
            text=text,
            label_value=label,
            text_type=self.dataset_config.text_type,
            positive_label=self.dataset_config.positive_label,
            negative_label=self.dataset_config.negative_label,
            library_code=library_code,
            extraction_schema=schema_str,
        )

        response = await self.client.generate(
            prompt=prompt,
            max_tokens=self.config.max_tokens_program,
            temperature=self.config.temperature_program,
            system=SYSTEM_PROMPT_GENERATION,
        )

        parsed = _extract_v2_response(response)
        if parsed is None:
            return GeneratedProgram(
                program_id=_make_program_id(example_id, round_idx),
                example_id=example_id,
                round_idx=round_idx,
                source_code="",
                hypothesis="",
                version="v2",
                execution_success=False,
                error_message="Could not parse v2 JSON response",
            )

        return GeneratedProgram(
            program_id=_make_program_id(example_id, round_idx),
            example_id=example_id,
            round_idx=round_idx,
            source_code=parsed["predict_code"],
            hypothesis=parsed.get("hypothesis", ""),
            input_schema=parsed.get("inputs", []),
            version="v2",
        )

    async def generate_batch_v2(
        self,
        texts: List[str],
        labels: List[int],
        example_ids: List[str],
        round_idx: int,
        library_code: Optional[str] = None,
        extraction_schema: Optional[List[dict]] = None,
        cache_path: Optional[Path] = None,
    ) -> List[GeneratedProgram]:
        """V2 batch: programs declare input schemas."""
        schema_str = json.dumps(extraction_schema, indent=2) if extraction_schema else None

        prompts = [
            render_program_generation_v2(
                text=text,
                label_value=label,
                text_type=self.dataset_config.text_type,
                positive_label=self.dataset_config.positive_label,
                negative_label=self.dataset_config.negative_label,
                library_code=library_code,
                extraction_schema=schema_str,
            )
            for text, label in zip(texts, labels)
        ]

        responses = await self.client.generate_batch(
            prompts=prompts,
            max_tokens=self.config.max_tokens_program,
            temperature=self.config.temperature_program,
            system=SYSTEM_PROMPT_GENERATION,
            cache_path=cache_path,
            cache_key_fn=lambda i: example_ids[i],
        )

        programs = []
        for i, response in enumerate(responses):
            pid = _make_program_id(example_ids[i], round_idx)
            if response is None:
                programs.append(GeneratedProgram(
                    program_id=pid, example_id=example_ids[i],
                    round_idx=round_idx, source_code="", hypothesis="",
                    version="v2", execution_success=False,
                    error_message="No response",
                ))
                continue

            parsed = _extract_v2_response(response)
            if parsed is None:
                # Fallback: try v1 parsing
                source = _extract_code(response)
                if source:
                    programs.append(GeneratedProgram(
                        program_id=pid, example_id=example_ids[i],
                        round_idx=round_idx, source_code=source,
                        hypothesis=_extract_hypothesis(source),
                        version="v1",
                    ))
                else:
                    programs.append(GeneratedProgram(
                        program_id=pid, example_id=example_ids[i],
                        round_idx=round_idx, source_code="", hypothesis="",
                        version="v2", execution_success=False,
                        error_message="Could not parse response",
                    ))
            else:
                programs.append(GeneratedProgram(
                    program_id=pid, example_id=example_ids[i],
                    round_idx=round_idx,
                    source_code=parsed["predict_code"],
                    hypothesis=parsed.get("hypothesis", ""),
                    input_schema=parsed.get("inputs", []),
                    version="v2",
                ))

        return programs

    # =================================================================
    # Approach 1: Full-example agnostic programs (with optional z guidance)
    # =================================================================

    async def generate_approach1_batch(
        self,
        texts: List[str],
        labels: List[int],
        example_ids: List[str],
        round_idx: int,
        features_for_list: Optional[List[Optional[List[str]]]] = None,
        features_against_list: Optional[List[Optional[List[str]]]] = None,
        library_code: Optional[str] = None,
        cache_path: Optional[Path] = None,
    ) -> List[GeneratedProgram]:
        """Approach 1: One full program per example.

        Each program is a comprehensive structural analysis that tries to
        classify the example. Optionally guided by STaR z₊/z₋ but free
        to discover additional patterns.
        """
        prompts = []
        for i, (text, label) in enumerate(zip(texts, labels)):
            f_for = features_for_list[i] if features_for_list else None
            f_against = features_against_list[i] if features_against_list else None
            prompts.append(render_approach1_generation(
                text=text,
                label_value=label,
                text_type=self.dataset_config.text_type,
                positive_label=self.dataset_config.positive_label,
                negative_label=self.dataset_config.negative_label,
                features_for=f_for,
                features_against=f_against,
                library_code=library_code,
            ))

        responses = await self.client.generate_batch(
            prompts=prompts,
            max_tokens=self.config.max_tokens_program,
            temperature=self.config.temperature_program,
            system=SYSTEM_PROMPT_GENERATION,
            cache_path=cache_path,
            cache_key_fn=lambda i: example_ids[i],
        )

        programs = []
        for i, response in enumerate(responses):
            pid = _make_program_id(example_ids[i], round_idx, "a1")
            if response is None:
                programs.append(GeneratedProgram(
                    program_id=pid, example_id=example_ids[i],
                    round_idx=round_idx, source_code="", hypothesis="",
                    version="approach1", execution_success=False,
                    error_message="No response",
                ))
            else:
                source = _extract_code(response)
                if source:
                    programs.append(GeneratedProgram(
                        program_id=pid, example_id=example_ids[i],
                        round_idx=round_idx, source_code=source,
                        hypothesis=_extract_hypothesis(source),
                        tools_needed=_extract_tools_needed(source),
                        version="approach1",
                    ))
                else:
                    programs.append(GeneratedProgram(
                        program_id=pid, example_id=example_ids[i],
                        round_idx=round_idx, source_code="", hypothesis="",
                        version="approach1", execution_success=False,
                        error_message="Could not extract code",
                    ))
        return programs

    # =================================================================
    # Approach 2: Per-z program synthesis
    # =================================================================

    async def generate_approach2_batch(
        self,
        texts: List[str],
        labels: List[int],
        example_ids: List[str],
        rationales: List[str],
        polarities: List[str],
        round_idx: int,
        library_code: Optional[str] = None,
        cache_path: Optional[Path] = None,
    ) -> List[GeneratedProgram]:
        """Approach 2: One program per z₊/z₋ instance.

        Each (example, rationale) pair gets its own program. The program
        checks that ONE specific criterion, with access to the example
        text so the LLM can see what concrete patterns correspond to it.

        Args:
            texts: example texts (one per z instance)
            labels: example labels (one per z instance)
            example_ids: example IDs (one per z instance)
            rationales: z₊/z₋ texts (one per call)
            polarities: "for" or "against" (one per call)
        """
        prompts = []
        for text, label, rationale, polarity in zip(texts, labels, rationales, polarities):
            prompts.append(render_approach2_generation(
                text=text,
                label_value=label,
                rationale=rationale,
                polarity=polarity,
                text_type=self.dataset_config.text_type,
                positive_label=self.dataset_config.positive_label,
                negative_label=self.dataset_config.negative_label,
                library_code=library_code,
            ))

        responses = await self.client.generate_batch(
            prompts=prompts,
            max_tokens=self.config.max_tokens_program,
            temperature=self.config.temperature_program,
            system=SYSTEM_PROMPT_GENERATION,
            cache_path=cache_path,
            cache_key_fn=lambda i: f"{example_ids[i]}:{rationales[i][:50]}",
        )

        programs = []
        for i, response in enumerate(responses):
            pid = _make_program_id(example_ids[i], round_idx, rationales[i][:50])
            if response is None:
                programs.append(GeneratedProgram(
                    program_id=pid, example_id=example_ids[i],
                    round_idx=round_idx, source_code="", hypothesis="",
                    version="approach2", execution_success=False,
                    error_message="No response",
                ))
            else:
                source = _extract_check_code(response)
                if source:
                    classification = _classify_from_docstring(source)
                    programs.append(GeneratedProgram(
                        program_id=pid, example_id=example_ids[i],
                        round_idx=round_idx, source_code=source,
                        hypothesis=rationales[i],
                        version="approach2",
                        input_schema=[{
                            "rationale": rationales[i],
                            "polarity": polarities[i],
                            "self_classification": classification,
                        }],
                    ))
                else:
                    programs.append(GeneratedProgram(
                        program_id=pid, example_id=example_ids[i],
                        round_idx=round_idx, source_code="", hypothesis="",
                        version="approach2", execution_success=False,
                        error_message="Could not extract code",
                    ))
        return programs
