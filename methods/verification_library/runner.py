"""End-to-end pipeline orchestration for verification library discovery."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .client import LLMClient, MultiClient
from .combiner import ProgramCombiner
from .config import DatasetConfig, VerificationLibraryConfig
from .convergence import (
    ConvergenceMetrics,
    check_convergence,
    compute_convergence_metrics,
)
from .evaluator import EvaluationResult, ProgramEvaluator
from .extractor import compute_schema_thickness, extract_all_fields, merge_schemas
from .generator import ProgramGenerator
from .program_store import LibraryStore, ProgramStore
from .refactorer import LibraryRefactorer
from .sandbox import execute_program_single

logger = logging.getLogger(__name__)


def _build_clients(config: VerificationLibraryConfig) -> MultiClient:
    """Build generation + refactoring clients from config."""
    # Generation client (bulk, cheap)
    if config.generation_base_url:
        gen_client = LLMClient.from_openai_compatible(
            model=config.generation_model,
            base_url=config.generation_base_url,
            concurrency=config.generation_concurrency,
        )
    else:
        gen_client = LLMClient.from_anthropic(
            model=config.generation_model,
            concurrency=config.generation_concurrency,
        )

    # Refactoring client (quality, few calls)
    if config.refactoring_base_url:
        ref_client = LLMClient.from_openai_compatible(
            model=config.refactoring_model,
            base_url=config.refactoring_base_url,
            concurrency=config.refactoring_concurrency,
        )
    else:
        if config.refactoring_model.startswith("claude-"):
            ref_client = LLMClient.from_anthropic(
                model=config.refactoring_model,
                concurrency=config.refactoring_concurrency,
            )
        else:
            # Same endpoint as generation
            ref_client = LLMClient.from_openai_compatible(
                model=config.refactoring_model,
                base_url=config.generation_base_url,
                concurrency=config.refactoring_concurrency,
            )

    return MultiClient(generation=gen_client, refactoring=ref_client)


async def run_verification_library(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: VerificationLibraryConfig,
    dataset_config: DatasetConfig,
) -> Dict[str, Any]:
    """Run the full generate-refactor-evaluate loop.

    Returns summary dict with final metrics and convergence history.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize stores
    program_store = ProgramStore(output_dir)
    program_store.load()

    library_store = LibraryStore(output_dir)

    # Initialize clients
    clients = _build_clients(config)
    generator = ProgramGenerator(clients.generation, config, dataset_config)
    refactorer = LibraryRefactorer(clients.refactoring, config, dataset_config)

    # Prepare data
    np.random.seed(config.random_seed)

    text_col = dataset_config.text_column
    label_col = dataset_config.label_column
    id_col = dataset_config.id_column

    train_texts = train_df[text_col].astype(str).tolist()
    train_labels = train_df[label_col].values.astype(int)
    train_ids = train_df[id_col].astype(str).tolist() if id_col in train_df.columns else [str(i) for i in range(len(train_df))]

    test_texts = test_df[text_col].astype(str).tolist()[:config.test_set_size]
    test_labels = test_df[label_col].values.astype(int)[:config.test_set_size]

    # Sample training indices
    n_examples = min(config.max_examples, len(train_texts))
    sample_indices = np.random.choice(len(train_texts), size=n_examples, replace=False)

    evaluator = ProgramEvaluator(test_texts, test_labels, sandbox_timeout=config.sandbox_timeout)

    # Load existing library
    current_library = library_store.load_latest()
    metrics_history: List[ConvergenceMetrics] = []
    previous_accuracy = 0.5
    last_refactor_round = library_store.latest_round

    # Determine starting point (for resumability)
    start_idx = program_store.count
    if start_idx > 0:
        logger.info("Resuming from example %d (found %d existing programs)", start_idx, start_idx)

    # Main loop: generate programs in batches, refactor incrementally
    n_batches = (n_examples - start_idx + config.batch_size - 1) // config.batch_size
    logger.info(
        "Starting verification library: %d examples, %d batches of %d, "
        "refactor every %d programs, model=%s",
        n_examples - start_idx, n_batches, config.batch_size,
        config.refactor_size, config.generation_model,
    )

    refactor_round = 0
    programs_since_last_refactor: List[GeneratedProgram] = []

    for batch_idx in range(n_batches):
        batch_start = start_idx + batch_idx * config.batch_size
        batch_end = min(batch_start + config.batch_size, n_examples)
        batch_indices = sample_indices[batch_start:batch_end]

        batch_texts = [train_texts[i] for i in batch_indices]
        batch_labels = [int(train_labels[i]) for i in batch_indices]
        batch_ids = [train_ids[i] for i in batch_indices]

        logger.info(
            "=== Batch %d/%d: examples %d-%d ===",
            batch_idx + 1, n_batches, batch_start, batch_end - 1,
        )

        # 1. Generate programs (full batch at once — cheap, parallel)
        cache_path = output_dir / f"generation_cache_batch_{batch_idx:03d}.jsonl"
        if config.approach == "approach1":
            programs = await generator.generate_approach1_batch(
                texts=batch_texts,
                labels=batch_labels,
                example_ids=batch_ids,
                round_idx=batch_idx,
                library_code=current_library,
                cache_path=cache_path,
            )
        else:
            programs = await generator.generate_batch(
                texts=batch_texts,
                labels=batch_labels,
                example_ids=batch_ids,
                round_idx=batch_idx,
                library_code=current_library,
                cache_path=cache_path,
            )

        # 2. Execute each on its own example (sanity check)
        for prog, text, label in zip(programs, batch_texts, batch_labels):
            if prog.source_code:
                score, err = execute_program_single(
                    prog.source_code, text,
                    timeout=config.sandbox_timeout,
                    library_code=current_library,
                )
                if err:
                    prog.execution_success = False
                    prog.error_message = err
                else:
                    prog.self_correct = (score > 0.5) == (label == 1)

        # Store programs
        for prog in programs:
            program_store.add(prog)

        successful_in_batch = [p for p in programs if p.execution_success and p.source_code]
        programs_since_last_refactor.extend(successful_in_batch)
        n_ok = len(successful_in_batch)
        logger.info("Generated %d programs, %d successful", len(programs), n_ok)

        # 3. Incremental refactoring: process programs in small groups
        while len(programs_since_last_refactor) >= config.refactor_size:
            refactor_batch = programs_since_last_refactor[:config.refactor_size]
            programs_since_last_refactor = programs_since_last_refactor[config.refactor_size:]
            previous_library = current_library

            logger.info(
                "Refactor round %d: merging %d new programs into library",
                refactor_round, len(refactor_batch),
            )

            current_library = await refactorer.refactor(
                new_programs=refactor_batch,
                existing_library=current_library,
                round_idx=refactor_round,
            )
            refactor_round += 1

            # Save snapshot after each refactoring
            from .refactorer import extract_function_names
            n_funcs = len(extract_function_names(current_library)) if current_library else 0
            library_store.save_snapshot(refactor_round, current_library or "", {
                "n_functions": n_funcs,
                "n_programs_refactored": len(refactor_batch),
            })
            logger.info(
                "Library now has %d functions (%d lines)",
                n_funcs, (current_library or "").count("\n"),
            )

        # 4. Evaluate ensemble on test set (after each generation batch)
        all_successful = program_store.get_successful_programs()
        eval_result = evaluator.evaluate_all(
            programs=all_successful,
            library_code=current_library,
            round_idx=batch_idx,
            train_texts=batch_texts,
            train_labels=np.array(batch_labels),
            combination_method=config.combination_method,
        )

        logger.info(
            "Round %d eval: ensemble_acc=%.4f, ensemble_auc=%.4f, "
            "best_single=%.4f, n_programs=%d",
            round_idx,
            eval_result.ensemble_accuracy,
            eval_result.ensemble_auc,
            eval_result.best_single_accuracy,
            eval_result.n_programs_successful,
        )

        # 5. Compute convergence metrics
        conv_metrics = compute_convergence_metrics(
            round_idx=batch_idx,
            current_library=current_library or "",
            previous_library=previous_library if 'previous_library' in dir() else "",
            ensemble_accuracy=eval_result.ensemble_accuracy,
            ensemble_auc=eval_result.ensemble_auc,
            previous_accuracy=previous_accuracy,
            n_programs_total=program_store.count,
            n_programs_successful=len(all_successful),
        )
        metrics_history.append(conv_metrics)
        previous_accuracy = eval_result.ensemble_accuracy

        # 6. Check convergence
        if check_convergence(
            metrics_history,
            window=config.convergence_window,
            edit_threshold=config.convergence_edit_threshold,
            accuracy_threshold=config.convergence_accuracy_threshold,
        ):
            logger.info("Library converged at batch %d!", batch_idx)
            break

    # Refactor any remaining programs
    if programs_since_last_refactor:
        logger.info("Final refactor: %d remaining programs", len(programs_since_last_refactor))
        current_library = await refactorer.refactor(
            new_programs=programs_since_last_refactor,
            existing_library=current_library,
            round_idx=refactor_round,
        )

    # Final summary
    summary = {
        "config": {
            "model": config.generation_model,
            "max_examples": config.max_examples,
            "batch_size": config.batch_size,
            "dataset": dataset_config.name,
        },
        "total_programs": program_store.count,
        "successful_programs": len(program_store.get_successful_programs()),
        "total_rounds": len(metrics_history),
        "final_ensemble_accuracy": metrics_history[-1].ensemble_accuracy if metrics_history else 0,
        "final_ensemble_auc": metrics_history[-1].ensemble_auc if metrics_history else 0,
        "final_library_functions": metrics_history[-1].library_size_functions if metrics_history else 0,
        "converged": check_convergence(metrics_history) if metrics_history else False,
        "convergence_history": [m.to_dict() for m in metrics_history],
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary to %s", summary_path)

    if current_library:
        final_lib_path = output_dir / "final_library.py"
        final_lib_path.write_text(current_library)
        logger.info("Saved final library to %s", final_lib_path)

    return summary


async def run_verification_library_v2(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: VerificationLibraryConfig,
    dataset_config: DatasetConfig,
) -> Dict[str, Any]:
    """V2: extraction-aware pipeline.

    Programs declare input schemas. Extraction happens separately (thin
    fields via regex/program, thick fields via LLM). Schema gets refined
    and thinned during refactoring.
    """
    from .prompts import render_schema_refactoring

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    program_store = ProgramStore(output_dir)
    program_store.load()
    library_store = LibraryStore(output_dir)

    clients = _build_clients(config)
    generator = ProgramGenerator(clients.generation, config, dataset_config)
    refactorer = LibraryRefactorer(clients.refactoring, config, dataset_config)

    np.random.seed(config.random_seed)
    text_col = dataset_config.text_column
    label_col = dataset_config.label_column
    id_col = dataset_config.id_column

    train_texts = train_df[text_col].astype(str).tolist()
    train_labels = train_df[label_col].values.astype(int)
    train_ids = (
        train_df[id_col].astype(str).tolist()
        if id_col in train_df.columns
        else [str(i) for i in range(len(train_df))]
    )

    test_texts = test_df[text_col].astype(str).tolist()[:config.test_set_size]
    test_labels = test_df[label_col].values.astype(int)[:config.test_set_size]

    n_examples = min(config.max_examples, len(train_texts))
    sample_indices = np.random.choice(len(train_texts), size=n_examples, replace=False)

    # Synchronous LLM callable for extraction
    def generate_sync(prompt: str) -> str:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            clients.generation.generate(prompt, max_tokens=512, temperature=0.0)
        )

    # State
    current_library = library_store.load_latest()
    current_schema: List[dict] = []
    metrics_history: List[ConvergenceMetrics] = []
    previous_accuracy = 0.5
    last_refactor_round = library_store.latest_round
    start_idx = program_store.count

    n_batches = (n_examples - start_idx + config.batch_size - 1) // config.batch_size
    logger.info(
        "Starting V2 verification library: %d examples, %d batches, model=%s",
        n_examples - start_idx, n_batches, config.generation_model,
    )

    for batch_idx in range(n_batches):
        batch_start = start_idx + batch_idx * config.batch_size
        batch_end = min(batch_start + config.batch_size, n_examples)
        batch_indices = sample_indices[batch_start:batch_end]
        round_idx = batch_idx + last_refactor_round + 1

        batch_texts = [train_texts[i] for i in batch_indices]
        batch_labels = [int(train_labels[i]) for i in batch_indices]
        batch_ids = [train_ids[i] for i in batch_indices]

        logger.info("=== V2 Round %d: examples %d-%d ===", round_idx, batch_start, batch_end - 1)

        # 1. Generate v2 programs (with input schemas)
        cache_path = output_dir / f"generation_cache_v2_round_{round_idx:03d}.jsonl"
        programs = await generator.generate_batch_v2(
            texts=batch_texts,
            labels=batch_labels,
            example_ids=batch_ids,
            round_idx=round_idx,
            library_code=current_library,
            extraction_schema=current_schema if current_schema else None,
            cache_path=cache_path,
        )

        # 2. Collect new input schemas from this batch
        batch_new_inputs = []
        for prog in programs:
            if prog.input_schema:
                batch_new_inputs.extend(prog.input_schema)

        # 3. Merge new inputs into schema
        if batch_new_inputs:
            current_schema = merge_schemas(current_schema, batch_new_inputs)

        # 4. For each program, extract fields and test on own example
        for prog, text, label in zip(programs, batch_texts, batch_labels):
            if not prog.source_code:
                continue
            if prog.version == "v2" and prog.input_schema:
                extracted = extract_all_fields(text, prog.input_schema, generate_fn=generate_sync)
                from .sandbox import execute_programs_batch_v2
                results = execute_programs_batch_v2(
                    prog.source_code, [extracted],
                    timeout=config.sandbox_timeout,
                    library_code=current_library,
                )
                score, err = results[0]
            else:
                score, err = execute_program_single(
                    prog.source_code, text,
                    timeout=config.sandbox_timeout,
                    library_code=current_library,
                )
            if err:
                prog.execution_success = False
                prog.error_message = err
            else:
                prog.self_correct = (score > 0.5) == (label == 1)

        for prog in programs:
            program_store.add(prog)

        n_ok = sum(1 for p in programs if p.execution_success)
        logger.info("Generated %d programs (%d successful), schema has %d fields", len(programs), n_ok, len(current_schema))

        # 5. Refactor library + schema
        new_programs_since_last = program_store.get_programs_since_round(last_refactor_round + 1)
        previous_library = current_library

        current_library = await refactorer.refactor(
            new_programs=new_programs_since_last,
            existing_library=current_library,
            round_idx=round_idx,
        )

        # Refactor schema (thin LLM fields to regex/program where possible)
        if batch_new_inputs:
            schema_prompt = render_schema_refactoring(
                text_type=dataset_config.text_type,
                positive_label=dataset_config.positive_label,
                negative_label=dataset_config.negative_label,
                all_inputs=batch_new_inputs,
                existing_schema=current_schema,
            )
            schema_response = await clients.refactoring.generate(
                schema_prompt,
                max_tokens=config.max_tokens_refactor,
                temperature=config.temperature_refactor,
            )
            import re as _re
            _match = _re.search(r"```json\s*\n(.*?)```", schema_response, _re.DOTALL)
            if _match:
                try:
                    refined = json.loads(_match.group(1).strip())
                    if isinstance(refined, list):
                        current_schema = refined
                except json.JSONDecodeError:
                    pass

        last_refactor_round = round_idx

        # Save schema snapshot
        schema_path = output_dir / f"schema_round_{round_idx:03d}.json"
        with open(schema_path, "w") as f:
            json.dump(current_schema, f, indent=2)

        thickness = compute_schema_thickness(current_schema)
        logger.info(
            "Schema: %d fields, %.0f%% thin, %.0f%% thick (LLM)",
            thickness["total_fields"],
            thickness["thin_fraction"] * 100,
            thickness["thick_fraction"] * 100,
        )

        # 6. Evaluate (using v1 evaluator for now — programs still produce scores)
        all_successful = program_store.get_successful_programs()
        evaluator = ProgramEvaluator(test_texts, test_labels, sandbox_timeout=config.sandbox_timeout)
        eval_result = evaluator.evaluate_all(
            programs=[p for p in all_successful if p.version == "v1"],
            library_code=current_library,
            round_idx=round_idx,
            combination_method=config.combination_method,
        )

        logger.info(
            "Round %d eval: acc=%.4f, auc=%.4f, n_programs=%d",
            round_idx, eval_result.ensemble_accuracy, eval_result.ensemble_auc,
            eval_result.n_programs_successful,
        )

        # 7. Convergence
        conv_metrics = compute_convergence_metrics(
            round_idx=round_idx,
            current_library=current_library,
            previous_library=previous_library,
            ensemble_accuracy=eval_result.ensemble_accuracy,
            ensemble_auc=eval_result.ensemble_auc,
            previous_accuracy=previous_accuracy,
            n_programs_total=program_store.count,
            n_programs_successful=len(all_successful),
        )
        metrics_history.append(conv_metrics)
        library_store.save_snapshot(round_idx, current_library, {
            **conv_metrics.to_dict(),
            "schema_thickness": thickness,
        })
        previous_accuracy = eval_result.ensemble_accuracy

        if check_convergence(metrics_history):
            logger.info("Library converged at round %d!", round_idx)
            break

    # Final summary
    thickness = compute_schema_thickness(current_schema)
    summary = {
        "version": "v2",
        "config": {
            "model": config.generation_model,
            "max_examples": config.max_examples,
            "batch_size": config.batch_size,
            "dataset": dataset_config.name,
        },
        "total_programs": program_store.count,
        "successful_programs": len(program_store.get_successful_programs()),
        "total_rounds": len(metrics_history),
        "final_ensemble_accuracy": metrics_history[-1].ensemble_accuracy if metrics_history else 0,
        "final_ensemble_auc": metrics_history[-1].ensemble_auc if metrics_history else 0,
        "final_library_functions": metrics_history[-1].library_size_functions if metrics_history else 0,
        "converged": check_convergence(metrics_history) if metrics_history else False,
        "schema": {
            "total_fields": thickness["total_fields"],
            "thin_fraction": thickness["thin_fraction"],
            "thick_fraction": thickness["thick_fraction"],
            "fields": current_schema,
        },
        "convergence_history": [m.to_dict() for m in metrics_history],
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if current_library:
        (output_dir / "final_library.py").write_text(current_library)
    with open(output_dir / "final_schema.json", "w") as f:
        json.dump(current_schema, f, indent=2)

    logger.info("V2 done: %d programs, %d schema fields (%.0f%% thin)",
                summary["total_programs"], thickness["total_fields"],
                thickness["thin_fraction"] * 100)

    return summary
