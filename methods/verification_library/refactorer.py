"""Periodic refactoring of accumulated programs into a shared utility library."""

import ast
import json
import logging
import re
from typing import List, Optional

from .client import AnthropicClient
from .config import DatasetConfig, VerificationLibraryConfig
from .program_store import GeneratedProgram
from .prompts import SYSTEM_PROMPT_REFACTORING, render_refactoring

logger = logging.getLogger(__name__)


def _extract_library_code(response: str) -> Optional[str]:
    """Extract Python code from refactoring response (legacy full-library mode)."""
    match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if "def " in code or "import " in code:
            return code
    return None


def _extract_diff_json(response: str) -> Optional[dict]:
    """Extract JSON diff from refactoring response (new diff-based mode)."""
    match = re.search(r"```json\s*\n(.*?)```", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Fallback: try parsing whole response as JSON
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass
    return None


def _apply_diff(existing_library: str, diff: dict) -> str:
    """Apply add/modify/delete diff to existing library code."""
    if not existing_library:
        existing_library = '"""Shared utility library for text analysis."""\n\nimport re\nimport string\nimport math\nfrom collections import Counter\nfrom statistics import mean\n'

    lines = existing_library.split("\n")

    # Parse existing function locations
    func_ranges = {}  # name -> (start_line, end_line)
    tree = ast.parse(existing_library)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Find end line (next function start or end of file)
            func_ranges[node.name] = node.lineno - 1  # 0-indexed start

    # Delete functions
    for name in diff.get("delete", []):
        if name in func_ranges:
            logger.info("Deleting function: %s", name)

    # Build new library: keep non-deleted functions, then add/modify
    # Simple approach: rebuild from AST
    kept_functions = set()
    delete_names = set(diff.get("delete", []))
    modify_map = {m["name"]: m["code"] for m in diff.get("modify", [])}

    # Collect all function source from existing library
    result_parts = []
    # Keep the imports/header
    header_end = 0
    for i, line in enumerate(lines):
        if line.startswith("def ") or (line.startswith("@") and i + 1 < len(lines) and lines[i + 1].startswith("def ")):
            header_end = i
            break
    else:
        header_end = len(lines)
    result_parts.append("\n".join(lines[:header_end]))

    # Process existing functions
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef):
            name = node.name
            if name in delete_names:
                continue
            if name in modify_map:
                result_parts.append("\n\n" + modify_map[name])
                kept_functions.add(name)
            else:
                # Extract original source
                start = node.lineno - 1
                end = node.end_lineno
                result_parts.append("\n\n" + "\n".join(lines[start:end]))
                kept_functions.add(name)

    # Add new functions
    for item in diff.get("add", []):
        name = item["name"]
        if name not in kept_functions:
            code = item["code"]
            # Validate syntax
            try:
                ast.parse(code)
                result_parts.append("\n\n" + code)
                logger.info("Added function: %s", name)
            except SyntaxError as e:
                logger.warning("Skipping function %s (syntax error): %s", name, e)

    return "\n".join(result_parts).strip() + "\n"


def validate_library(library_code: str) -> tuple[bool, str]:
    """Validate that the library is syntactically correct Python."""
    try:
        ast.parse(library_code)
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error in library: {e}"


def extract_function_names(library_code: str) -> set[str]:
    """Parse function names from library source using ast module."""
    try:
        tree = ast.parse(library_code)
        return {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
    except SyntaxError:
        return set()


def _extract_function_blocks(library_code: str) -> dict[str, str]:
    """Extract each function as a standalone code block, keyed by name."""
    if not library_code:
        return {}
    blocks = {}
    lines = library_code.split("\n")
    try:
        tree = ast.parse(library_code)
    except SyntaxError:
        return {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef):
            start = node.lineno - 1
            end = node.end_lineno
            blocks[node.name] = "\n".join(lines[start:end])
    return blocks


def _get_function_text_for_embedding(code_block: str) -> str:
    """Extract signature + docstring + body summary for embedding."""
    try:
        tree = ast.parse(code_block)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                sig = f"def {node.name}({', '.join(a.arg for a in node.args.args)})"
                doc = ast.get_docstring(node) or ""
                return f"{sig}\n{doc}\n{code_block}"
    except SyntaxError:
        pass
    return code_block[:500]


# Module-level cache for the embedding model (loaded once)
_embed_model = None
_embed_model_name = None


def _get_embed_model(model_name: str = "jinaai/jina-code-embeddings-0.5b"):
    """Lazy-load embedding model."""
    global _embed_model, _embed_model_name
    if _embed_model is None or _embed_model_name != model_name:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", model_name)
        _embed_model = SentenceTransformer(model_name, trust_remote_code=True)
        _embed_model_name = model_name
    return _embed_model


def _find_similar_functions(
    library_code: str,
    new_programs: list,
    top_k: int = 8,
    embedding_model: str = "jinaai/jina-code-embeddings-0.5b",
) -> str:
    """Find library functions most similar to new programs via code embeddings.

    Uses a code-specialized embedding model for semantic similarity.
    Falls back to token overlap if embedding model unavailable.
    """
    blocks = _extract_function_blocks(library_code)
    if not blocks or len(blocks) <= top_k:
        return library_code or ""

    # Collect texts to embed
    func_names = list(blocks.keys())
    func_texts = [_get_function_text_for_embedding(blocks[n]) for n in func_names]

    program_texts = []
    for prog in new_programs:
        src = prog.source_code if hasattr(prog, 'source_code') else str(prog)
        program_texts.append(src)
    combined_programs = "\n\n".join(program_texts)

    try:
        model = _get_embed_model(embedding_model)
        # Embed library functions + combined new programs
        all_texts = func_texts + [combined_programs]
        embeddings = model.encode(all_texts, normalize_embeddings=True)

        func_embeds = embeddings[:-1]  # (n_funcs, dim)
        prog_embed = embeddings[-1:]   # (1, dim)

        # Cosine similarity (already normalized)
        import numpy as np
        sims = (func_embeds @ prog_embed.T).flatten()

        # Select top-k
        top_indices = np.argsort(sims)[-top_k:][::-1]
        selected = [func_names[i] for i in top_indices]

        logger.info(
            "Similarity routing: selected %d/%d functions (sims: %.3f–%.3f)",
            len(selected), len(func_names),
            sims[top_indices[-1]], sims[top_indices[0]],
        )

    except Exception as e:
        logger.warning("Embedding failed (%s), falling back to token overlap", e)
        # Fallback: token overlap
        program_tokens = set()
        for prog in new_programs:
            src = prog.source_code if hasattr(prog, 'source_code') else str(prog)
            program_tokens.update(re.findall(r'\b\w+\b', src.lower()))

        scores = []
        for name, code in blocks.items():
            tokens = set(re.findall(r'\b\w+\b', code.lower()))
            total = len(tokens | program_tokens) or 1
            scores.append((name, len(tokens & program_tokens) / total))
        scores.sort(key=lambda x: x[1], reverse=True)
        selected = [name for name, _ in scores[:top_k]]

    # Build subset library string
    lines = library_code.split("\n")
    header_lines = []
    for line in lines:
        if line.startswith("def "):
            break
        header_lines.append(line)

    parts = ["\n".join(header_lines)]
    for name in selected:
        if name in blocks:
            parts.append("\n" + blocks[name])

    omitted = len(blocks) - len(selected)
    if omitted > 0:
        parts.append(f"\n# ... {omitted} other functions omitted (not relevant to current batch)")

    return "\n".join(parts)


class LibraryRefactorer:
    def __init__(
        self,
        client: AnthropicClient,
        config: VerificationLibraryConfig,
        dataset_config: DatasetConfig,
    ):
        self.client = client
        self.config = config
        self.dataset_config = dataset_config

    async def refactor(
        self,
        new_programs: List[GeneratedProgram],
        existing_library: Optional[str] = None,
        round_idx: int = 0,
    ) -> str:
        """Generate updated library by refactoring new programs + existing library.

        Shows only NEW programs since last refactor + the existing library.
        The library accumulates patterns incrementally.
        """
        program_sources = []
        for p in new_programs:
            if p.execution_success and p.source_code:
                # Strip TOOLS_NEEDED comments to keep refactoring input focused
                code = p.source_code
                tools_idx = code.find("# TOOLS_NEEDED:")
                if tools_idx > 0:
                    code = code[:tools_idx].rstrip()
                program_sources.append(code)

        if not program_sources:
            logger.warning("No successful programs to refactor in round %d", round_idx)
            return existing_library or ""

        # Similarity routing: if library is large, only show relevant functions
        library_for_prompt = existing_library
        if existing_library and existing_library.count("\n") > self.config.library_similarity_threshold:
            library_for_prompt = _find_similar_functions(
                existing_library,
                new_programs,
                top_k=self.config.library_similarity_top_k,
                embedding_model=self.config.embedding_model,
            )
            n_total = len(_extract_function_blocks(existing_library))
            n_shown = len(_extract_function_blocks(library_for_prompt))
            logger.info(
                "Library too large (%d lines) — showing %d/%d similar functions to refactorer",
                existing_library.count("\n"), n_shown, n_total,
            )

        prompt = render_refactoring(
            text_type=self.dataset_config.text_type,
            positive_label=self.dataset_config.positive_label,
            negative_label=self.dataset_config.negative_label,
            new_programs=program_sources,
            existing_library=library_for_prompt,
        )

        response = await self.client.generate(
            prompt=prompt,
            max_tokens=self.config.max_tokens_refactor,
            temperature=self.config.temperature_refactor,
            system=SYSTEM_PROMPT_REFACTORING,
        )

        # Try diff-based parsing first (new format)
        diff = _extract_diff_json(response)
        if diff is not None:
            n_add = len(diff.get("add", []))
            n_mod = len(diff.get("modify", []))
            n_del = len(diff.get("delete", []))
            reasoning = diff.get("reasoning", "")
            logger.info(
                "Library round %d: +%d -%d ~%d | %s",
                round_idx, n_add, n_del, n_mod, reasoning[:100],
            )
            if n_add == 0 and n_mod == 0 and n_del == 0:
                logger.info("No changes needed")
                return existing_library or ""
            try:
                updated = _apply_diff(existing_library or "", diff)
                ok, err = validate_library(updated)
                if ok:
                    return updated
                else:
                    logger.warning("Diff application produced invalid library: %s", err)
            except Exception as e:
                logger.warning("Diff application failed: %s", e)

        # Fallback: try legacy full-library extraction
        library_code = _extract_library_code(response)
        if library_code is not None:
            ok, err = validate_library(library_code)
            if ok:
                new_funcs = extract_function_names(library_code)
                old_funcs = extract_function_names(existing_library or "")
                added = new_funcs - old_funcs
                if added:
                    logger.info("Library round %d (legacy): added %s", round_idx, added)
                return library_code
            else:
                logger.error("Refactored library failed validation: %s", err)

        logger.error("Could not parse refactoring response (round %d)", round_idx)
        return existing_library or ""
