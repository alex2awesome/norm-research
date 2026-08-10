# Patents — Paper Replications

## Purpose

This directory collects upstream paper releases (code + sample data) that we replicate, mine, or compare against for the **patents** task in this project. Each subdirectory is a vendored copy of a public research repo whose task formulation overlaps with one of our patent-side pipelines: §102 anticipation / novelty, first-draft approval prediction, and §112(b) indefiniteness. They are kept here so that data-loading code, baselines, and label definitions stay reproducible alongside the main codebase.

## Replication inventory

| Directory | Paper | Venue / Year | Task addressed | Link |
|---|---|---|---|---|
| `claim-compare/` | Parikh & Dori-Hacohen — *ClaimCompare: A Data Pipeline for Evaluation of Novelty Destroying Patent Pairs* | PatentSemTech @ SIGIR 2024 | §102 anticipation (novelty-destroying prior art pairs) | [PatentSemTech CFP](https://www.ifs.tuwien.ac.at/patentsemtech/cfp.html) |
| `FLAN-Graph/` | Gao, Yao, Zhao, He, Kumar, Krishnan, Shang — *Beyond Scaling: Predicting Patent Approval with Domain-specific Fine-grained Claim Dependency Graph* | arXiv 2404.14372 (2024) | First-draft patent approval prediction | [arXiv:2404.14372](https://arxiv.org/abs/2404.14372) |
| `pedantic-patentsemtech/` | *PEDANTIC: A Dataset for the Automatic Examination of Definiteness in Patent Claims* | arXiv 2505.21342 (2025) | §112(b) indefiniteness examination | [arXiv:2505.21342](https://arxiv.org/abs/2505.21342) |

## Per-replication notes

### claim-compare
Vendored as-is. We keep `pipeline.ipynb` (their data-construction pipeline), `evaluate_bge_m3_on_sample.py` (retrieval baseline), and `sample_dataset/` (1,045 electrochemical base patents, each with 25 candidate prior-art patents labeled novelty-destroying vs. related). Used as a labeled §102 anticipation source and as a sanity baseline for our claim-vs-prior-art retrieval setup.

### FLAN-Graph
Vendored with the original `FLAN-Graph/` and `Scaling_w_LLMs/` subdirectories plus a `download_patentap.py` helper for pulling the **PatentAP** Hugging Face dataset (`shangdatalab-ucsd/PatentAP`). PatentAP is the label source we treat as canonical for the first-draft approval task; their FLAN claim-dependency graph and LLM-prompting baselines are kept for direct comparison numbers.

### pedantic-patentsemtech
Vendored with their `src/` package, `data/`, `prompts/`, plus two thin wrappers (`replicate_llm_judge.py`, `replicate_lr_baseline.py`) we added to re-run the LR baseline and LLM-judge eval on the PEDANTIC release. The 14k §112(b)-annotated NLP-domain claims are our entry point into indefiniteness as a normative axis distinct from novelty/approval.

## Relation to current work

- **claim-compare → §102 anticipation pipeline.** Our patents §102 pipeline mirrors ClaimCompare's framing (base-claim vs. candidate prior-art claim, binary novelty-destroying label). Their electrochemical sample is one of our held-out evaluation sets, and their retrieval baseline (BGE-M3 over claims) is the floor we report against.
- **FLAN-Graph → first-draft approval prediction.** Our patent first-draft approval task (see `project_patents_first_draft_prediction.md`) reuses the PatentAP labels released with this paper. FLAN-Graph's LLM-prompting and embedding baselines are the reference numbers we cite; their fine-grained claim-dependency graph is a candidate feature source we have not yet integrated.
- **pedantic-patentsemtech → §112(b) indefiniteness.** PEDANTIC is the only public §112(b) corpus we know of. It supplies a separate normative dimension (claim *definiteness*, not novelty or overall allowability) that we use both as an auxiliary label and as a probe for whether our rubric-extraction pipeline rediscovers examiner-cited indefiniteness reasons. Their LLM-as-Judge protocol (free-form reason matching against examiner-cited reasons) is also the template for how we score articulated norms on this task.
