"""Build R1 (rule) families per task with Llama-3.3-70B-Instruct (vLLM).

For each task in --only (default: all 11):
  1. Load forms + locked L0 clusters + base-bge embeddings.
  2. Cluster representative text (mode of members) + centroid (mean of members).
  3. Greedy cover-once batching: anchor + top-(B-1) nearest unassigned NN.
  4. vLLM offline batched chat with the **task-specific** R1 prompt + few-shot.
  5. Parse JSON; assemble R1 family assignments per cluster.

Saves:
  match_out/r1/r1_families_<task>.json   {families:[...], cluster_to_family:{}}
  match_out/r1/r1_raw_<task>.jsonl       per-batch raw output (debugging)
"""
from __future__ import annotations

import os

os.environ["HOME"] = "/lfs/skampere3/0/alexspan"
os.environ["HF_HOME"] = "/lfs/skampere3/0/alexspan/hf_cache"
os.environ["HF_MODULES_CACHE"] = "/lfs/skampere3/0/alexspan/hf_cache/modules"
os.environ["XDG_CACHE_HOME"] = "/lfs/skampere3/0/alexspan/.cache"
os.environ["VLLM_CACHE_ROOT"] = "/lfs/skampere3/0/alexspan/vllm_cache"
os.environ["TRITON_CACHE_DIR"] = "/lfs/skampere3/0/alexspan/triton_cache"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

WORK = Path("/lfs/skampere3/0/alexspan/norm_embed")
FORMS = WORK / "canon_all_real_forms.jsonl"
EMB = WORK / "out"
MATCH_OUT = WORK / "match_out"
R1_OUT = MATCH_OUT / "r1"
MODEL_BASE = ("/lfs/skampere3/0/shared_hf_cache/"
              "models--meta-llama--Llama-3.3-70B-Instruct/snapshots")

TASKS = ["code-review", "creative-writing", "grant-funding", "humor",
         "legal-outcome-prediction", "math-stackexchange", "news-homepages",
         "notice-and-comment", "patents", "peer-review", "press-releases"]


SYSTEM_TEMPLATE = """You are organizing fine-grained rubric concepts into broader CRITERION FAMILIES (R1 rule-families) for the following evaluation domain.

DOMAIN: {background}

Each input is a deduplicated L0 cluster of rubric statements with a representative text. Many L0 clusters that look superficially different encode the SAME underlying RULE -- they prescribe the same target behaviour, only stated with different wording, examples, qualifications, or specificity. Group all variants of one rule into one R1 family.

A FAMILY = every L0 cluster that enforces one underlying rule. The TEST: "would a competent reviewer say these are checking the SAME thing, even if the way they phrase the check differs?" If yes, same family.

Specificity matters for the SCOPE of a rule, not its wording. Two concepts about the SAME scope of behaviour belong together even if one says it broadly and another adds a condition or example. But two concepts at clearly DIFFERENT scopes stay apart.

Two failure modes, BOTH equally wrong:
- UNDER-merging: splitting near-duplicate restatements of one rule into many singletons.
- OVER-merging: lumping genuinely distinct rules into one UMBRELLA family with a generic name. The more pernicious failure -- the next hierarchy level cannot undo it.

A family must encode ONE specific rule. The family NAME should be specific enough that, reading just the name, a reviewer knows what is being checked. In this domain, generic super-category names like {anti_umbrella} are red flags of over-merging -- DO NOT use them. Instead, name the SPECIFIC rule (e.g., {anti_umbrella_fix}).
{step_b_extra}
Sanity check before answering: if any family has >12 members AND a generic-sounding name, you have over-merged -- split it into specific sub-families. A tight batch of 30-40 related clusters should typically produce 8-20 SPECIFIC families, not 2-5 umbrella families.

OUTPUT VALID JSON ONLY, no commentary, no markdown fences:
{{
  "families": [
    {{"name": "<short, SPECIFIC noun phrase, 4-10 words>",
     "description": "<one sentence stating the underlying rule>",
     "members": ["<id>", "<id>", ...]}}
  ]
}}

Every input id must appear in exactly one family. Singleton families are allowed when a concept genuinely doesn't share a rule with any other input. Do not invent ids."""


# Per-task content. Each entry:
#   background       -- one-sentence domain description.
#   anti_umbrella    -- generic red-flag family names to AVOID for this domain.
#   anti_umbrella_fix-- "instead use names like ..." examples.
#   fewshot_user     -- ~10 task-specific rubric concepts.
#   fewshot_assistant-- expected family grouping showing BOTH correct merge of
#                       variants and correct split of distinct rules.
TASK_INFO = {
    "code-review": {
        "background": "Reviewing a proposed code change before merge. Rubrics come from code-review checklists, style guides, refactoring catalogs, engineering standards; each states something a reviewer checks about the code, tests, or design.",
        "anti_umbrella": '"Code clarity and conciseness", "Code organization and readability", "Avoid specific code constructs", "Code style and formatting", "Comprehensive testing"',
        "anti_umbrella_fix": '"Limit function return statements", "Use snake_case for variables", "Avoid deeply nested control flow"',
        "fewshot_user": [
            {"id": "X1", "rep": "The code should have consistent indentation"},
            {"id": "X2", "rep": "Use 4-space indentation in all files"},
            {"id": "X3", "rep": "Lines should be under 100 characters"},
            {"id": "X4", "rep": "Functions should not have too many return statements"},
            {"id": "X5", "rep": "Avoid scattered return statements"},
            {"id": "X6", "rep": "The function should have a single responsibility"},
            {"id": "X7", "rep": "Each module should have a single responsibility"},
            {"id": "X8", "rep": "Avoid using method chains (a.b().c())"},
            {"id": "X9", "rep": "Variables should use snake_case"},
            {"id": "X10", "rep": "Avoid deeply nested control flow"},
        ],
        "fewshot_assistant": {"families": [
            {"name": "Consistent indentation", "description": "Code should be indented consistently.", "members": ["X1", "X2"]},
            {"name": "Maximum line length", "description": "Lines should not exceed a maximum length.", "members": ["X3"]},
            {"name": "Limit function return statements", "description": "A function should not have excessive or scattered return statements.", "members": ["X4", "X5"]},
            {"name": "Single-responsibility principle", "description": "A code unit (function or module) should have one responsibility.", "members": ["X6", "X7"]},
            {"name": "Avoid method chains", "description": "Avoid long chains of method calls on returned objects.", "members": ["X8"]},
            {"name": "Variable naming convention", "description": "Variables should follow a defined naming convention.", "members": ["X9"]},
            {"name": "Limit nesting depth", "description": "Conditionals and loops should not be deeply nested.", "members": ["X10"]},
        ]},
    },
    "creative-writing": {
        "background": "Evaluating a piece of creative writing (fiction, story, manuscript). Rubrics come from craft books, style manuals, magazine submission guidelines, writing-contest criteria; each states something that distinguishes strong creative writing -- craft, structure, prose, character, voice.",
        "anti_umbrella": '"Authentic emotional expression", "Authentic representation", "Character development", "Narrative structure and pacing", "Worldbuilding and setting", "Creative freedom and balance"',
        "anti_umbrella_fix": "\"Show, don't tell\", \"Authentic representation of disability\", \"Pacing builds tension\"",
        "fewshot_user": [
            {"id": "X1", "rep": "Convey information through action and dialogue rather than direct statement (show, don't tell)"},
            {"id": "X2", "rep": "Avoid too much direct exposition; dramatize instead"},
            {"id": "X3", "rep": "The opening should immediately engage the reader"},
            {"id": "X4", "rep": "The first sentence should hook the reader"},
            {"id": "X5", "rep": "Characters should have clear motivations"},
            {"id": "X6", "rep": "Characters should have agency to drive the plot"},
            {"id": "X7", "rep": "The setting should be vivid and immersive"},
            {"id": "X8", "rep": "Use sensory detail to evoke setting"},
            {"id": "X9", "rep": "The story should have a satisfying resolution"},
            {"id": "X10", "rep": "Pacing should build and maintain tension"},
        ],
        "fewshot_assistant": {"families": [
            {"name": "Show, don't tell", "description": "Convey information through action, dialogue, and sensory detail rather than direct statement.", "members": ["X1", "X2"]},
            {"name": "Engaging opening hook", "description": "The opening (first sentence/paragraph) should hook the reader.", "members": ["X3", "X4"]},
            {"name": "Character motivation", "description": "Characters should have clear motivations that drive their actions.", "members": ["X5"]},
            {"name": "Character agency", "description": "Characters should have agency to drive the plot, not be passive.", "members": ["X6"]},
            {"name": "Vivid setting through sensory detail", "description": "The setting should be made vivid through sensory detail.", "members": ["X7", "X8"]},
            {"name": "Satisfying resolution", "description": "The story should have a satisfying conclusion.", "members": ["X9"]},
            {"name": "Pacing builds tension", "description": "Pacing should build and maintain narrative tension.", "members": ["X10"]},
        ]},
    },
    "grant-funding": {
        "background": "Evaluating a grant or research-funding proposal. Rubrics come from funder guidelines and grant-writing guides; each states something reviewers look for -- significance, approach, feasibility, budget, team.",
        "anti_umbrella": '"Strong proposal", "Effective grant writing", "Project planning and management", "Significance and impact"',
        "anti_umbrella_fix": '"Justified budget aligned with scope", "Specific Aim 1 has measurable outcomes", "Data management plan included"',
        "fewshot_user": [
            {"id": "X1", "rep": "The proposal should include a well-justified budget"},
            {"id": "X2", "rep": "The budget should align with the proposed scope"},
            {"id": "X3", "rep": "The proposal should address a significant problem"},
            {"id": "X4", "rep": "The research should fill a clear gap in the field"},
            {"id": "X5", "rep": "The proposed approach should be feasible"},
            {"id": "X6", "rep": "The PI should have relevant expertise"},
            {"id": "X7", "rep": "The team should be qualified for the proposed work"},
            {"id": "X8", "rep": "Specific Aim 1 should have measurable outcomes"},
            {"id": "X9", "rep": "The proposal should include a clear timeline"},
            {"id": "X10", "rep": "The proposal should include a data-management plan"},
        ],
        "fewshot_assistant": {"families": [
            {"name": "Justified budget aligned with scope", "description": "The budget should be justified and align with the scope of the work.", "members": ["X1", "X2"]},
            {"name": "Significance fills gap in field", "description": "The proposal addresses a significant problem that fills a clear gap in the field.", "members": ["X3", "X4"]},
            {"name": "Feasibility of approach", "description": "The proposed approach should be feasible.", "members": ["X5"]},
            {"name": "Qualified team and PI", "description": "The PI and team should have the relevant expertise for the work.", "members": ["X6", "X7"]},
            {"name": "Measurable Aim 1 outcomes", "description": "Specific Aim 1 should specify measurable outcomes.", "members": ["X8"]},
            {"name": "Clear project timeline", "description": "The proposal should include a clear timeline.", "members": ["X9"]},
            {"name": "Data management plan", "description": "The proposal should include a data-management plan.", "members": ["X10"]},
        ]},
    },
    "humor": {
        "background": "Evaluating humor -- a joke, comedy set, cartoon, or comic piece. Rubrics come from comedy-writing manuals and analyses of what makes things funny; each states something about joke construction, timing, target, originality.",
        "anti_umbrella": '"Effective humor delivery", "Cultural reference humor", "Social commentary through humor", "Visual comedy elements", "Setup and punchline structure", "Audience consideration", "Comedic tone and style", "Satire and social commentary"',
        "anti_umbrella_fix": '"Comedic timing on final word", "Punch up, not down", "Subverted audience expectations"',
        "fewshot_user": [
            {"id": "X1", "rep": "The humor should have effective comedic timing"},
            {"id": "X2", "rep": "The punchline should land at the right moment"},
            {"id": "X3", "rep": "The punchline should be on the final word of the sentence"},
            {"id": "X4", "rep": "The setup should establish a clear expectation"},
            {"id": "X5", "rep": "The humor should use incongruity (unexpected combinations)"},
            {"id": "X6", "rep": "The joke should subvert audience expectations"},
            {"id": "X7", "rep": "The humor should avoid targeting marginalized groups"},
            {"id": "X8", "rep": "The humor should punch up, not down"},
            {"id": "X9", "rep": "The humor should be original, not derivative"},
            {"id": "X10", "rep": "The humor should fit its intended audience"},
        ],
        "fewshot_assistant": {"families": [
            {"name": "Comedic timing", "description": "The humor should have effective timing for the punchline.", "members": ["X1", "X2"]},
            {"name": "Punchline on final word", "description": "The punchline should land on the final word of the sentence.", "members": ["X3"]},
            {"name": "Clear setup establishing expectation", "description": "The setup should establish a clear expectation that the punchline subverts.", "members": ["X4"]},
            {"name": "Incongruity and subverted expectations", "description": "Humor should use incongruity and unexpected combinations to subvert expectations.", "members": ["X5", "X6"]},
            {"name": "Punch up, not down", "description": "Humor should avoid targeting marginalized groups; punch up, not down.", "members": ["X7", "X8"]},
            {"name": "Originality of material", "description": "Humor should be original, not derivative.", "members": ["X9"]},
            {"name": "Audience fit", "description": "Humor should fit its intended audience.", "members": ["X10"]},
        ]},
    },
    "legal-outcome-prediction": {
        "background": "Evaluating a legal argument or case for how it is likely to fare. Rubrics come from legal writing and advocacy guides; each states something about the strength of a legal argument -- reasoning, use of precedent, framing, evidence.",
        "anti_umbrella": '"Strong legal argument", "Legal writing quality", "Effective advocacy"',
        "anti_umbrella_fix": '"Support with controlling precedent", "Address counter-arguments", "Clearly stated remedy"',
        "fewshot_user": [
            {"id": "X1", "rep": "The argument should be supported by relevant precedent"},
            {"id": "X2", "rep": "Cite controlling case law for each major point"},
            {"id": "X3", "rep": "The argument should address counter-arguments"},
            {"id": "X4", "rep": "The brief should anticipate the opposing position"},
            {"id": "X5", "rep": "Legal writing should be clear and plain"},
            {"id": "X6", "rep": "Avoid jargon and legalese where possible"},
            {"id": "X7", "rep": "The brief should follow proper citation form (e.g., Bluebook)"},
            {"id": "X8", "rep": "The facts should be stated accurately and concisely"},
            {"id": "X9", "rep": "The argument should establish a clear theory of the case"},
            {"id": "X10", "rep": "The remedy sought should be clearly stated"},
        ],
        "fewshot_assistant": {"families": [
            {"name": "Support with controlling precedent", "description": "Each major argument should be supported by relevant, controlling case law.", "members": ["X1", "X2"]},
            {"name": "Address counter-arguments", "description": "The argument should anticipate and address opposing positions.", "members": ["X3", "X4"]},
            {"name": "Plain, jargon-free writing", "description": "Legal writing should be clear, plain, and avoid unnecessary jargon.", "members": ["X5", "X6"]},
            {"name": "Proper citation form", "description": "The brief should follow the proper citation form (Bluebook).", "members": ["X7"]},
            {"name": "Accurate, concise fact statement", "description": "Facts should be stated accurately and concisely.", "members": ["X8"]},
            {"name": "Clear theory of the case", "description": "The argument should establish a clear theory of the case.", "members": ["X9"]},
            {"name": "Clearly stated remedy", "description": "The remedy sought should be clearly identified.", "members": ["X10"]},
        ]},
    },
    "math-stackexchange": {
        "background": "Evaluating a mathematics question or answer on a Q&A site. Rubrics come from guides on mathematical writing and on good questions/answers; each states something that distinguishes a strong, clear, correct, well-posed math post.",
        "anti_umbrella": '"Clear mathematical writing", "Effective math post", "Audience consideration", "Mathematical rigor"',
        "anti_umbrella_fix": '"LaTeX/MathJax for notation", "Show derivation steps", "Cite original sources"',
        "fewshot_user": [
            {"id": "X1", "rep": "The question should be clear and well-posed"},
            {"id": "X2", "rep": "State the question precisely with required context"},
            {"id": "X3", "rep": "Use LaTeX for mathematical notation"},
            {"id": "X4", "rep": "Mathematical expressions should be properly formatted with MathJax"},
            {"id": "X5", "rep": "The answer should show the steps of the derivation"},
            {"id": "X6", "rep": "Explanations should be didactic, not just the final result"},
            {"id": "X7", "rep": "Identify and consider the audience (student vs. researcher)"},
            {"id": "X8", "rep": "Avoid mathematical symbols at the start of a sentence"},
            {"id": "X9", "rep": "Provide a reference to original sources"},
            {"id": "X10", "rep": "The proof should be rigorous and complete"},
        ],
        "fewshot_assistant": {"families": [
            {"name": "Clear, well-posed question", "description": "The question should be clear and stated with required context.", "members": ["X1", "X2"]},
            {"name": "LaTeX/MathJax for notation", "description": "Mathematical notation should use LaTeX (MathJax) formatting.", "members": ["X3", "X4"]},
            {"name": "Show derivation steps", "description": "Answers should show the steps of the derivation, didactically.", "members": ["X5", "X6"]},
            {"name": "Audience awareness", "description": "Identify and consider the intended audience of the post.", "members": ["X7"]},
            {"name": "No symbols starting a sentence", "description": "Sentences should not begin with a mathematical symbol.", "members": ["X8"]},
            {"name": "Cite original sources", "description": "Cite original sources for results used.", "members": ["X9"]},
            {"name": "Rigorous, complete proof", "description": "The proof should be rigorous and complete.", "members": ["X10"]},
        ]},
    },
    "news-homepages": {
        "background": "Judging the newsworthiness of a story and whether it belongs on a news homepage. Rubrics come from journalism guides and newsroom practice; each states something that makes a story newsworthy or homepage-worthy -- timeliness, impact, audience relevance, prominence.",
        "anti_umbrella": '"Newsworthy story", "News quality", "Effective journalism"',
        "anti_umbrella_fix": '"Factual accuracy and verification", "Local relevance", "Inverted pyramid structure"',
        "fewshot_user": [
            {"id": "X1", "rep": "The story should be timely and recent"},
            {"id": "X2", "rep": "Lead with the most recent development"},
            {"id": "X3", "rep": "The story should be accurate"},
            {"id": "X4", "rep": "All facts should be verified before publication"},
            {"id": "X5", "rep": "The story should be balanced and free of bias"},
            {"id": "X6", "rep": "Sources should be properly attributed"},
            {"id": "X7", "rep": "The headline should accurately reflect the story"},
            {"id": "X8", "rep": "The story should be locally relevant"},
            {"id": "X9", "rep": "Use the inverted pyramid structure"},
            {"id": "X10", "rep": "The story should have human-interest appeal"},
        ],
        "fewshot_assistant": {"families": [
            {"name": "Timeliness and recency", "description": "The story should be timely; lead with the most recent development.", "members": ["X1", "X2"]},
            {"name": "Factual accuracy and verification", "description": "All facts should be accurate and verified.", "members": ["X3", "X4"]},
            {"name": "Balance and freedom from bias", "description": "The story should be balanced and free from bias.", "members": ["X5"]},
            {"name": "Source attribution", "description": "Sources should be properly attributed.", "members": ["X6"]},
            {"name": "Accurate headline reflects story", "description": "The headline should accurately reflect the story's content.", "members": ["X7"]},
            {"name": "Local relevance", "description": "The story should be locally relevant.", "members": ["X8"]},
            {"name": "Inverted pyramid structure", "description": "The story should use the inverted pyramid structure.", "members": ["X9"]},
            {"name": "Human-interest appeal", "description": "The story should have human-interest appeal.", "members": ["X10"]},
        ]},
    },
    "notice-and-comment": {
        "background": "Evaluating a public comment submitted on a proposed government regulation (notice-and-comment rulemaking). Rubrics come from guides on writing effective regulatory comments; each states something that makes a comment persuasive or useful to the agency -- specificity, evidence, legal grounding, constructiveness.",
        "anti_umbrella": '"Informative comments", "Cost-benefit analysis", "Compliance with regulations", "Comment specificity and detail", "Consider alternative approaches", "Evidence-based comment", "Regulatory impact analysis"',
        "anti_umbrella_fix": '"Cite empirical evidence", "Engage with specific NPRM provisions", "Comply with NEPA", "Quantified cost-benefit analysis"',
        "fewshot_user": [
            {"id": "X1", "rep": "The comment should support its claims with empirical evidence"},
            {"id": "X2", "rep": "Cite scientific evidence and data to back up arguments"},
            {"id": "X3", "rep": "The comment should engage with specific provisions of the proposed rule"},
            {"id": "X4", "rep": "Identify and discuss specific paragraphs or subsections of the NPRM"},
            {"id": "X5", "rep": "The comment should comply with the Paperwork Reduction Act"},
            {"id": "X6", "rep": "The comment should demonstrate compliance with NEPA"},
            {"id": "X7", "rep": "Suggest alternative regulatory approaches"},
            {"id": "X8", "rep": "Identify regulatory alternatives the agency has not considered"},
            {"id": "X9", "rep": "Provide a quantified cost-benefit analysis"},
            {"id": "X10", "rep": "The comment should be tailored to the specific proposal, not generic"},
        ],
        "fewshot_assistant": {"families": [
            {"name": "Empirical-evidence support", "description": "Comments should support claims with empirical evidence and data.", "members": ["X1", "X2"]},
            {"name": "Engage with specific NPRM provisions", "description": "The comment should engage with specific provisions or subsections of the proposed rule.", "members": ["X3", "X4"]},
            {"name": "Comply with Paperwork Reduction Act", "description": "The comment should comply with the Paperwork Reduction Act.", "members": ["X5"]},
            {"name": "Comply with NEPA", "description": "The comment should demonstrate compliance with the National Environmental Policy Act.", "members": ["X6"]},
            {"name": "Suggest alternative regulatory approaches", "description": "The comment should suggest alternative approaches the agency has not considered.", "members": ["X7", "X8"]},
            {"name": "Quantified cost-benefit analysis", "description": "The comment should provide a quantified cost-benefit analysis.", "members": ["X9"]},
            {"name": "Tailored to specific proposal", "description": "The comment should be tailored to the specific proposal, not generic.", "members": ["X10"]},
        ]},
    },
    "patents": {
        "background": "Evaluating a patent application or claim. Rubrics come from patent-drafting guides and patent-office standards; each states something that distinguishes a strong application -- claim clarity, scope, novelty support, specification quality.",
        "anti_umbrella": '"Clear patent claims", "Patent quality", "Patent application requirements"',
        "anti_umbrella_fix": '"Enablement by ordinary skill", "No new matter in amendments", "Unity of invention"',
        "fewshot_user": [
            {"id": "X1", "rep": "The specification should enable a person of ordinary skill in the art to make and use the invention"},
            {"id": "X2", "rep": "Provide enough detail for someone of ordinary skill to practice the invention"},
            {"id": "X3", "rep": "The claims should be clear and definite"},
            {"id": "X4", "rep": "Use precise claim language to avoid ambiguity"},
            {"id": "X5", "rep": "The amendment should not add new matter"},
            {"id": "X6", "rep": "The invention should be of patentable subject matter"},
            {"id": "X7", "rep": "The patent application should relate to a single invention"},
            {"id": "X8", "rep": "The independent claims should cover the broadest scope"},
            {"id": "X9", "rep": "The dependent claims should narrow the independent claim"},
            {"id": "X10", "rep": "The drawings should be consistent with the specification"},
        ],
        "fewshot_assistant": {"families": [
            {"name": "Enablement by ordinary skill", "description": "The specification should enable a person of ordinary skill in the art to make and use the invention.", "members": ["X1", "X2"]},
            {"name": "Clear and definite claims", "description": "Claims should be clear and definite, using precise language.", "members": ["X3", "X4"]},
            {"name": "No new matter in amendments", "description": "Amendments should not introduce new matter.", "members": ["X5"]},
            {"name": "Patentable subject matter", "description": "The invention should be of patentable subject matter.", "members": ["X6"]},
            {"name": "Unity of invention", "description": "The application should relate to a single invention.", "members": ["X7"]},
            {"name": "Independent claims at broadest scope", "description": "Independent claims should cover the broadest scope of the invention.", "members": ["X8"]},
            {"name": "Dependent claims narrow independent", "description": "Dependent claims should narrow the independent claim.", "members": ["X9"]},
            {"name": "Drawings consistent with specification", "description": "Drawings should be consistent with the specification.", "members": ["X10"]},
        ]},
    },
    "peer-review": {
        "background": "Evaluating an academic paper and the peer reviews of it. Rubrics come from journal/conference reviewer guidelines and peer-review guides; each states something reviewers assess -- novelty, soundness, clarity, evidence, reproducibility, related work.",
        "anti_umbrella": '"Transparent reporting", "Reproducibility and transparency", "Clear presentation", "Ethical considerations in research", "Evidence-supported claims"',
        "anti_umbrella_fix": '"Match claims to evidence strength", "Deposit data in public repository", "IRB approval obtained"',
        "fewshot_user": [
            {"id": "X1", "rep": "The claim should be supported by evidence"},
            {"id": "X2", "rep": "The conclusion should not exceed what the evidence supports"},
            {"id": "X3", "rep": "Strong evidence should be provided for the main claims"},
            {"id": "X4", "rep": "The submission should include data and analysis notebooks"},
            {"id": "X5", "rep": "Data should be deposited in a public repository"},
            {"id": "X6", "rep": "The paper should disclose all sources of funding"},
            {"id": "X7", "rep": "IRB or equivalent ethical approval should be obtained and reported"},
            {"id": "X8", "rep": "The work should discuss its limitations"},
            {"id": "X9", "rep": "The authors should clearly state the limitations of their work"},
            {"id": "X10", "rep": "The methods should be reported in sufficient detail to enable replication"},
        ],
        "fewshot_assistant": {"families": [
            {"name": "Match claims to evidence strength", "description": "Claims, conclusions, and language should not exceed the available evidence.", "members": ["X1", "X2", "X3"]},
            {"name": "Submission includes data and notebooks", "description": "Submission should include data, code, and analysis notebooks for reproducibility.", "members": ["X4"]},
            {"name": "Deposit data in public repository", "description": "Data should be deposited in a public repository.", "members": ["X5"]},
            {"name": "Disclose funding sources", "description": "All sources of funding should be disclosed.", "members": ["X6"]},
            {"name": "IRB ethical approval obtained", "description": "IRB or equivalent ethical approval should be obtained and reported.", "members": ["X7"]},
            {"name": "Discuss limitations explicitly", "description": "The work should explicitly discuss its limitations.", "members": ["X8", "X9"]},
            {"name": "Methods replicable in detail", "description": "Methods should be reported in sufficient detail to allow replication.", "members": ["X10"]},
        ]},
    },
    "press-releases": {
        "background": "Evaluating a press release. Rubrics come from PR and communications writing guides; each states something that distinguishes an effective press release -- newsworthiness, structure, clarity, quotes, headline, audience targeting.",
        "anti_umbrella": '"Effective press release", "Press release quality", "PR best practices"',
        "anti_umbrella_fix": "\"Target journalist's beat\", \"Quotable spokesperson quote\", \"Properly formatted dateline\"",
        "fewshot_user": [
            {"id": "X1", "rep": "The release should target a specific, well-understood audience"},
            {"id": "X2", "rep": "Tailor the release to the journalist's beat and outlet"},
            {"id": "X3", "rep": "The release should be structured as an inverted pyramid"},
            {"id": "X4", "rep": "Lead with the most important information first"},
            {"id": "X5", "rep": "The headline should be attention-grabbing yet accurate"},
            {"id": "X6", "rep": "Include a clear and quotable quote from a spokesperson"},
            {"id": "X7", "rep": "The release should be newsworthy"},
            {"id": "X8", "rep": "The story should answer the journalist's 'why now?' question"},
            {"id": "X9", "rep": "The dateline should be properly formatted"},
            {"id": "X10", "rep": "Include a boilerplate paragraph about the company"},
        ],
        "fewshot_assistant": {"families": [
            {"name": "Target specific audience and beat", "description": "Tailor the release to a specific audience (and the journalist's beat/outlet).", "members": ["X1", "X2"]},
            {"name": "Inverted pyramid structure", "description": "Use the inverted pyramid, with the most important information first.", "members": ["X3", "X4"]},
            {"name": "Attention-grabbing accurate headline", "description": "The headline should be attention-grabbing yet accurately reflect the story.", "members": ["X5"]},
            {"name": "Quotable spokesperson quote", "description": "Include a clear, quotable quote from a spokesperson.", "members": ["X6"]},
            {"name": "Newsworthiness and 'why now?'", "description": "The release should be newsworthy and answer 'why now?'.", "members": ["X7", "X8"]},
            {"name": "Properly formatted dateline", "description": "The dateline should be properly formatted.", "members": ["X9"]},
            {"name": "Boilerplate company paragraph", "description": "Include a boilerplate paragraph about the company.", "members": ["X10"]},
        ]},
    },
}


# --- Step A: re-name few-shot R1 family names to read as RULE STATEMENTS,
# not category labels. Goal: nudge LLM toward more specific family names so
# fewer L0 clusters get lumped under generic catch-all umbrellas.
_STEP_A_OVERRIDES = {
    "code-review": {
        "Consistent indentation": "Indent code consistently throughout the file",
        "Maximum line length": "Cap line length at the project's maximum",
        "Limit function return statements": "Limit return statements per function",
        "Single-responsibility principle": "Each function or module has exactly one responsibility",
        "Avoid method chains": "Avoid chaining method calls on returned objects",
        "Variable naming convention": "Use the project's variable naming convention",
        "Limit nesting depth": "Cap nesting depth of conditionals and loops",
    },
    "creative-writing": {
        "Show, don't tell": "Convey information through action and dialogue, not direct exposition",
        "Engaging opening hook": "Open with a sentence that immediately engages the reader",
        "Character motivation": "Each character has explicit motivations driving their actions",
        "Character agency": "Characters drive the plot through their own agency",
        "Vivid setting through sensory detail": "Establish setting through specific sensory detail",
        "Satisfying resolution": "Resolve the central conflict in a satisfying way",
        "Pacing builds tension": "Pacing builds and sustains narrative tension",
    },
    "grant-funding": {
        "Justified budget aligned with scope": "Budget is justified and aligned with proposed scope",
        "Significance fills gap in field": "The proposal addresses a clear gap in the field",
        "Feasibility of approach": "Proposed approach is feasible given resources and timeline",
        "Qualified team and PI": "PI and team have relevant expertise for the work",
        "Measurable Aim 1 outcomes": "Specific Aim 1 has measurable outcomes",
        "Clear project timeline": "Project timeline is clearly laid out",
        "Data management plan": "Data-management plan is included",
    },
    "humor": {
        "Comedic timing": "Punchline has effective comedic timing",
        "Punchline on final word": "Punchline lands on the final word of the sentence",
        "Clear setup establishing expectation": "Setup establishes a clear expectation the punchline subverts",
        "Incongruity and subverted expectations": "Use incongruity to subvert audience expectations",
        "Punch up, not down": "Punch up at power, not down at the marginalized",
        "Originality of material": "Material is original, not derivative",
        "Audience fit": "Humor fits its intended audience",
    },
    "legal-outcome-prediction": {
        "Support with controlling precedent": "Each major argument cites controlling precedent",
        "Address counter-arguments": "The brief directly addresses opposing counter-arguments",
        "Plain, jargon-free writing": "Use plain writing free of jargon and legalese",
        "Proper citation form": "Follow the proper citation form (Bluebook)",
        "Accurate, concise fact statement": "State facts accurately and concisely",
        "Clear theory of the case": "Establish a clear theory of the case",
        "Clearly stated remedy": "The remedy sought is clearly stated",
    },
    "math-stackexchange": {
        "Clear, well-posed question": "Question is clear and stated with required context",
        "LaTeX/MathJax for notation": "Mathematical notation uses LaTeX/MathJax formatting",
        "Show derivation steps": "Answers show derivation steps didactically",
        "Audience awareness": "Identify the intended audience explicitly",
        "No symbols starting a sentence": "Sentences do not begin with a mathematical symbol",
        "Cite original sources": "Cite original sources for results used",
        "Rigorous, complete proof": "Proof is rigorous and complete",
    },
    "news-homepages": {
        "Timeliness and recency": "Story leads with the most recent development",
        "Factual accuracy and verification": "All facts are verified before publication",
        "Balance and freedom from bias": "The story is balanced and free of bias",
        "Source attribution": "Sources are properly attributed in the story",
        "Accurate headline reflects story": "Headline accurately reflects the story's content",
        "Local relevance": "The story is locally relevant to the audience",
        "Inverted pyramid structure": "Body uses inverted-pyramid structure",
        "Human-interest appeal": "Story has clear human-interest appeal",
    },
    "notice-and-comment": {
        "Empirical-evidence support": "Comment cites empirical evidence for its claims",
        "Engage with specific NPRM provisions": "Comment engages with specific NPRM provisions",
        "Comply with Paperwork Reduction Act": "Demonstrate compliance with the Paperwork Reduction Act",
        "Comply with NEPA": "Demonstrate compliance with NEPA",
        "Suggest alternative regulatory approaches": "Suggest alternative regulatory approaches the agency hasn't considered",
        "Quantified cost-benefit analysis": "Provide a quantified cost-benefit analysis",
        "Tailored to specific proposal": "Comment is tailored to the specific proposal, not generic",
    },
    "patents": {
        "Enablement by ordinary skill": "Specification enables practice by ordinary skill in the art",
        "Clear and definite claims": "Claims are clear and definite, with precise language",
        "No new matter in amendments": "Amendments do not introduce new matter",
        "Patentable subject matter": "The invention is of patentable subject matter",
        "Unity of invention": "The application relates to a single invention",
        "Independent claims at broadest scope": "Independent claims cover the broadest scope",
        "Dependent claims narrow independent": "Dependent claims narrow the independent claim",
        "Drawings consistent with specification": "Drawings are consistent with the specification",
    },
    "peer-review": {
        "Match claims to evidence strength": "Match claims to the strength of the available evidence",
        "Submission includes data and notebooks": "Submission includes data, code, and analysis notebooks",
        "Deposit data in public repository": "Data is deposited in a public repository",
        "Disclose funding sources": "Disclose all sources of funding",
        "IRB ethical approval obtained": "IRB or equivalent ethical approval obtained",
        "Discuss limitations explicitly": "The work explicitly discusses its limitations",
        "Methods replicable in detail": "Methods are reported in detail sufficient to replicate",
    },
    "press-releases": {
        "Target specific audience and beat": "Release is tailored to a specific audience and journalist beat",
        "Inverted pyramid structure": "Body uses inverted-pyramid structure",
        "Attention-grabbing accurate headline": "Headline is attention-grabbing and accurately reflects the story",
        "Quotable spokesperson quote": "Include a quotable quote from a spokesperson",
        "Newsworthiness and 'why now?'": "Release is newsworthy and answers 'why now?'",
        "Properly formatted dateline": "Dateline is properly formatted",
        "Boilerplate company paragraph": "Include a boilerplate paragraph about the company",
    },
}
for _task, _overrides in _STEP_A_OVERRIDES.items():
    if _task in TASK_INFO:
        for _f in TASK_INFO[_task]["fewshot_assistant"]["families"]:
            if _f["name"] in _overrides:
                _f["name"] = _overrides[_f["name"]]


STEP_B_EXTRA = """
The family NAME should READ AS A SPECIFIC RULE STATEMENT (e.g., "Indent code consistently throughout the file", "Cite controlling precedent for each major claim"), NOT a category label (e.g., "Code formatting", "Citation practices"). If you find yourself reaching for a category-label name, the family is probably an over-merged umbrella -- split it.

Singleton families are NOT a failure mode. PREFER many specific singletons over bundling distinct rules under a generic category name. Do NOT avoid singletons to make families look fuller.
"""


def make_system(task, step_b=False):
    info = TASK_INFO[task]
    return SYSTEM_TEMPLATE.format(
        background=info["background"],
        anti_umbrella=info["anti_umbrella"],
        anti_umbrella_fix=info["anti_umbrella_fix"],
        step_b_extra=STEP_B_EXTRA if step_b else "")


def fewshot_messages(task):
    info = TASK_INFO[task]
    return [{"role": "user", "content": json.dumps(info["fewshot_user"], indent=1)},
            {"role": "assistant",
             "content": json.dumps(info["fewshot_assistant"], indent=1)}]


def load_task(task, forms):
    """Pool emb_bge in sorted-bucket+idx order; return rows + L2-normed emb."""
    by_bucket = defaultdict(list)
    for r in forms:
        by_bucket[r["bucket"]].append(r)
    rows, mats = [], []
    for bucket in sorted(by_bucket):
        rs = sorted(by_bucket[bucket], key=lambda r: r["idx"])
        p = EMB / f"emb_bge_{bucket}_{task}.npy"
        if not p.exists():
            return None, None
        e = np.load(p).astype(np.float32)
        if len(e) != len(rs):
            return None, None
        rows += rs
        mats.append(e)
    emb = np.vstack(mats)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    return rows, emb


def cluster_data(rows, emb, cl):
    """Per cluster: representative text, member rows, centroid embedding."""
    members = defaultdict(list)
    for i, r in enumerate(rows):
        members[cl[r["key"]]].append((i, r["canonical"] or ""))
    reps, centroids = {}, {}
    for cid, ms in members.items():
        texts = [m[1] for m in ms]
        reps[cid] = Counter(texts).most_common(1)[0][0]
        idxs = [m[0] for m in ms]
        c = emb[idxs].mean(0)
        c /= (np.linalg.norm(c) + 1e-9)
        centroids[cid] = c
    return reps, centroids, members


def make_batches(cids, centroids, batch_size, edge_thresh=None, knn=None):
    """Greedy cover-once: anchor + its top-(B-1) unassigned NN."""
    cids = list(cids)
    n = len(cids)
    cen = np.stack([centroids[c] for c in cids])
    cos = cen @ cen.T
    np.fill_diagonal(cos, -2.0)
    assigned = np.zeros(n, dtype=bool)
    batches = []
    while not assigned.all():
        un = np.where(~assigned)[0]
        if len(un) <= batch_size:
            batches.append([cids[i] for i in un])
            break
        sub = cos[np.ix_(un, un)]
        anchor_local = sub.mean(axis=1).argmax()
        anchor = un[anchor_local]
        scores = cos[anchor].copy()
        scores[assigned] = -2.0
        scores[anchor] = -2.0
        top = np.argpartition(-scores, batch_size - 1)[:batch_size - 1]
        batch_idx = np.concatenate([[anchor], top])
        batches.append([cids[i] for i in batch_idx])
        assigned[batch_idx] = True
    return batches, []


def make_sliding_batches(cids, centroids, batch_size, coverage=2.0):
    """Sliding-window batches: each cluster appears in ~`coverage` batches.

    Pick stride-spaced anchors from the density-sorted cluster list (highest
    mean-similarity-to-others first). Each anchor's batch is anchor + top-(B-1)
    NN (no exclusion). With stride = n / target_n_batches and target_n_batches
    = ceil(coverage * n / B), each cluster's top-B-NN set overlaps with about
    `coverage` anchor neighbourhoods on average -- so each cluster lands in
    ~coverage batches. Orphans (no anchor in their top-B NN) get appended in
    extra batches. This lets a same-rule concept with >B members span >1 batch
    while still keeping all of it together in each batch it appears in.
    """
    cids = list(cids)
    n = len(cids)
    cen = np.stack([centroids[c] for c in cids])
    cos = cen @ cen.T
    np.fill_diagonal(cos, -2.0)

    target_n_batches = max(1, int(np.ceil(coverage * n / batch_size)))
    density = cos.mean(axis=1)
    sorted_by_density = np.argsort(-density)
    stride = max(1, n // target_n_batches)
    anchors = sorted_by_density[::stride][:target_n_batches]

    batches = []
    appearances = np.zeros(n, dtype=int)
    for anchor in anchors:
        scores = cos[anchor].copy()
        scores[anchor] = -2.0
        k = min(batch_size - 1, n - 1)
        top = np.argpartition(-scores, k)[:k]
        batch_idx = np.concatenate([[int(anchor)], top.astype(int)])
        batches.append([cids[int(i)] for i in batch_idx])
        appearances[batch_idx] += 1

    missing = np.where(appearances == 0)[0]
    if len(missing) > 0:
        for s in range(0, len(missing), batch_size):
            batches.append([cids[int(i)] for i in missing[s:s + batch_size]])
    return batches, []


def build_messages(task, batch, reps, members, step_b=False):
    """Compose chat for this task's batch using task-specific system + few-shot."""
    cand = []
    for cid in batch:
        ms = members[cid]
        rep = reps[cid]
        alts = [t for _, t in ms if t != rep]
        seen, uniq = set(), []
        for a in alts:
            if a not in seen:
                seen.add(a)
                uniq.append(a)
            if len(uniq) >= 2:
                break
        cand.append({"id": f"C{cid}", "rep": rep[:200],
                     "alt": [a[:160] for a in uniq]})
    user_msg = json.dumps(cand, indent=1)
    return [{"role": "system", "content": make_system(task, step_b=step_b)},
            *fewshot_messages(task),
            {"role": "user", "content": user_msg}]


def parse_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


# ---------- merge pass ----------

MERGE_SYSTEM_TEMPLATE = """You are verifying whether pairs of R1 rule-families encode the SAME UNDERLYING RULE in this evaluation domain.

DOMAIN: {background}

The TIGHT test: SAME-RULE means a competent reviewer would say "these are checking THE SAME thing" -- you would put them under one one-sentence rule name. If you would have to write two distinct sentences to describe what each one checks, they are DIFFERENT rules.

CRITICAL: pairs that LOOK related because they share vocabulary are usually DIFFERENT rules. Reject these:
- "Function declarations should be properly formatted" vs "Function names should follow a convention" -> DIFFERENT (formatting vs naming, both about functions)
- "Limit function return statements" vs "Limit function length" -> DIFFERENT (return-count vs line-count)
- "Cite controlling precedent" vs "Address counter-arguments" -> DIFFERENT (both argument-strength)
- "Indent consistently" vs "Cap line length" -> DIFFERENT (both formatting)
- "Code documented consistently" vs "Code clearly named" -> DIFFERENT
- "Match claims to evidence strength" vs "Cite empirical evidence" -> DIFFERENT (one is calibration of language, other is sourcing)

Accept ONLY genuine same-rule pairs:
- "Indent code consistently throughout the file" vs "Use 4-space indentation throughout" -> SAME (same indentation rule, different specificity)
- "Each function has one responsibility" vs "Each module has one responsibility" -> SAME (same SRP, different scope unit)
- "Functions should be small" vs "Functions should be short" -> SAME (same length rule, paraphrase)

When in doubt -> answer FALSE. Over-merging is far worse than under-merging here.

For each pair below, output same_rule true or false.

OUTPUT VALID JSON ONLY, no commentary, no markdown fences:
{{"verdicts": [{{"pair": 1, "same_rule": true|false}}, {{"pair": 2, "same_rule": true|false}}, ...]}}"""


def assemble_per_task(items, reps, pre_singletons):
    """Turn per-batch LLM outputs into (all_families, cluster_appearances).

    all_families : list of {name, description, cluster_ids, batch_id}.
    cluster_appearances : cid -> list of indices into all_families.

    Each batch's LLM-assigned families are recorded; any cluster in a batch
    not assigned by the LLM gets a per-batch singleton. Pre-singletons
    (graph-isolated, no batch) are appended at the end.
    """
    all_fams = []
    appearances = defaultdict(list)
    for bi, batch, text in sorted(items):
        parsed = parse_json(text)
        in_batch_assigned = set()
        if parsed and "families" in parsed:
            for fam in parsed["families"]:
                fids = []
                for mid in fam.get("members", []):
                    try:
                        cid = int(str(mid).lstrip("C"))
                    except ValueError:
                        continue
                    if cid in batch and cid not in in_batch_assigned:
                        fids.append(cid)
                        in_batch_assigned.add(cid)
                if not fids:
                    continue
                idx = len(all_fams)
                all_fams.append({
                    "name": (fam.get("name") or "")[:120],
                    "description": (fam.get("description") or "")[:300],
                    "cluster_ids": fids,
                    "batch_id": bi,
                })
                for c in fids:
                    appearances[c].append(idx)
        # Per-batch fallback singletons
        for cid in batch:
            if cid not in in_batch_assigned:
                idx = len(all_fams)
                all_fams.append({
                    "name": (reps[cid] or "")[:120],
                    "description": reps[cid] or "",
                    "cluster_ids": [cid],
                    "batch_id": bi,
                })
                appearances[cid].append(idx)
    for cid in pre_singletons:
        idx = len(all_fams)
        all_fams.append({
            "name": (reps[cid] or "")[:120],
            "description": reps[cid] or "",
            "cluster_ids": [cid],
            "batch_id": -1,
        })
        appearances[cid].append(idx)
    return all_fams, appearances


def family_centroids(families, cluster_centroids):
    """Per-family centroid = L2-normed mean of constituent cluster centroids.

    cluster_centroids: {cluster_id: vector} -- already L2-normed per cluster.
    """
    if not cluster_centroids:
        return np.zeros((len(families), 1), dtype=np.float32)
    dim = next(iter(cluster_centroids.values())).shape[0]
    out = np.zeros((len(families), dim), dtype=np.float32)
    for i, fam in enumerate(families):
        vecs = [cluster_centroids[c] for c in fam["cluster_ids"]
                if c in cluster_centroids]
        if not vecs:
            continue
        c = np.stack(vecs).mean(0)
        n = np.linalg.norm(c)
        out[i] = c / n if n > 0 else c
    return out


def merge_candidates(all_fams, cluster_appearances, centroids,
                     centroid_thresh=0.85, include_overlap=False):
    """Return unique candidate pairs (i, j) ordered by confidence DESCENDING.

    Two sources unioned, but ordered: overlap pairs first (strong signal under
    sliding), then centroid pairs sorted by cos descending so high-confidence
    merges happen first under a constrained union-find (a max-family-size cap
    then rejects only the marginal late merges instead of arbitrary ones).
    """
    overlap = []
    seen = set()
    if include_overlap:
        for cid, idxs in cluster_appearances.items():
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    a, b = sorted((idxs[i], idxs[j]))
                    if a != b and (a, b) not in seen:
                        seen.add((a, b))
                        overlap.append((a, b))
    centroid_pairs = []
    if len(centroids) > 1 and centroid_thresh < 1.0:
        cos = centroids @ centroids.T
        np.fill_diagonal(cos, -2.0)
        ii, jj = np.where(cos >= centroid_thresh)
        for a, b in zip(ii.tolist(), jj.tolist()):
            if a < b:
                centroid_pairs.append((float(cos[a, b]), (a, b)))
        centroid_pairs.sort(key=lambda x: -x[0])
    out = list(overlap)
    for _, p in centroid_pairs:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def make_merge_message(task, pairs_batch, all_fams, reps):
    """Compose chat for a batch of merge candidate pairs."""
    info = TASK_INFO[task]
    sys_msg = MERGE_SYSTEM_TEMPLATE.format(background=info["background"])
    lines = []
    for k, (a, b) in enumerate(pairs_batch, 1):
        fa = all_fams[a]
        fb = all_fams[b]
        sa = [reps.get(c, "")[:80] for c in fa["cluster_ids"][:3]]
        sb = [reps.get(c, "")[:80] for c in fb["cluster_ids"][:3]]
        lines.append(
            f"Pair {k}:\n"
            f"  A: \"{fa['name']}\"  (n={len(fa['cluster_ids'])})\n"
            f"     desc: {fa['description'][:160]}\n"
            f"     members: {' | '.join(sa)}\n"
            f"  B: \"{fb['name']}\"  (n={len(fb['cluster_ids'])})\n"
            f"     desc: {fb['description'][:160]}\n"
            f"     members: {' | '.join(sb)}\n")
    user_msg = "\n".join(lines)
    return [{"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}]


def parse_merge_verdicts(text, n_pairs):
    parsed = parse_json(text)
    if not parsed or "verdicts" not in parsed:
        return {}
    out = {}
    for v in parsed.get("verdicts", []):
        try:
            p = int(v["pair"])
            if 1 <= p <= n_pairs:
                out[p] = bool(v.get("same_rule", False))
        except (KeyError, ValueError, TypeError):
            pass
    return out


def union_find_roots(n, yes_edges, fam_sizes=None, max_size=None):
    """Union-find with optional cap on merged family size.

    yes_edges should be ordered by descending confidence; early edges merge
    first, and a max_size cap then rejects only later (lower-confidence)
    edges that would push a merged group past the cap. This prevents the
    catastrophic transitive chain (A=B, B=C, ..., Z) from merging thousands
    of clusters into one mega-family.
    """
    parent = list(range(n))
    size = list(fam_sizes) if fam_sizes is not None else [1] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in yes_edges:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if max_size is not None and size[ra] + size[rb] > max_size:
            continue
        parent[rb] = ra
        size[ra] += size[rb]
    return [find(i) for i in range(n)]


def consolidate_with_merges(all_fams, cluster_appearances, roots):
    """Each cluster -> root family. If a cluster is in multiple roots,
    pick the root whose merged family has the most members (tie: smallest
    family-idx).
    """
    # First pass: per-root member counts (for tie-break).
    root_size = defaultdict(int)
    for i, fam in enumerate(all_fams):
        root_size[roots[i]] += len(fam["cluster_ids"])

    cluster_to_root = {}
    for cid, fam_idxs in cluster_appearances.items():
        if not fam_idxs:
            continue
        # roots of all families this cluster is in
        rs = set(roots[i] for i in fam_idxs)
        if len(rs) == 1:
            cluster_to_root[cid] = rs.pop()
        else:
            cluster_to_root[cid] = max(rs, key=lambda r: (root_size[r], -r))

    # Group clusters by root, build final families
    by_root = defaultdict(list)
    for cid, r in cluster_to_root.items():
        by_root[r].append(cid)

    # Choose name/description for each root: from the LARGEST contributing
    # sub-family in all_fams (most members).
    root_largest_subfam = {}
    for i, fam in enumerate(all_fams):
        r = roots[i]
        cur = root_largest_subfam.get(r)
        if cur is None or len(fam["cluster_ids"]) > len(cur["cluster_ids"]):
            root_largest_subfam[r] = fam

    final = []
    fid = 0
    for r, cids in sorted(by_root.items(), key=lambda x: -len(x[1])):
        fid += 1
        rep_fam = root_largest_subfam[r]
        final.append({
            "family_id": fid,
            "name": rep_fam["name"],
            "description": rep_fam["description"],
            "cluster_ids": sorted(set(cids)),
        })
    cluster_to_family = {c: f["family_id"]
                        for f in final for c in f["cluster_ids"]}
    return final, cluster_to_family


def consolidate_first_come(all_fams):
    """First family containing a cluster wins; later containers drop it."""
    final = []
    seen = set()
    cluster_to_family = {}
    fid = 0
    for fam in all_fams:
        keep = [c for c in fam["cluster_ids"] if c not in seen]
        if not keep:
            continue
        fid += 1
        final.append({"family_id": fid, "name": fam["name"],
                      "description": fam["description"], "cluster_ids": keep})
        for c in keep:
            cluster_to_family[c] = fid
        seen.update(keep)
    return final, cluster_to_family


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--batch-size", type=int, default=40)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--output-dir", default="",
                    help="Output subdir under match_out/ (default 'r1')")
    ap.add_argument("--step-b", action="store_true",
                    help="Add system-prompt extras (rule-statement naming + "
                         "singletons-preferred guidance)")
    ap.add_argument("--sliding", action="store_true",
                    help="Use sliding-window batching: each cluster appears in "
                         "~2 batches (coverage=2) to recover same-rule fragments")
    ap.add_argument("--coverage", type=float, default=2.0,
                    help="Per-cluster batch coverage for sliding (default 2.0)")
    ap.add_argument("--merge", action="store_true",
                    help="After R1 pass, run LLM cross-batch merge pass on "
                         "candidate same-rule family pairs")
    ap.add_argument("--centroid-thresh", type=float, default=0.92,
                    help="Family-centroid cosine threshold for merge candidates")
    ap.add_argument("--merge-iter", type=int, default=1,
                    help="Number of merge-pass iterations (default 1)")
    ap.add_argument("--max-merged-size", type=int, default=60,
                    help="Cap on size of a merged R1 family (rejects further "
                         "merges that would exceed this); prevents transitive "
                         "chains from accumulating thousands of clusters.")
    args = ap.parse_args()
    tasks = args.only.split(",") if args.only else TASKS
    r1_out = MATCH_OUT / args.output_dir if args.output_dir else R1_OUT
    r1_out.mkdir(exist_ok=True, parents=True)

    forms_by_task = defaultdict(list)
    for line in FORMS.open():
        r = json.loads(line)
        forms_by_task[r["task"]].append(r)

    task_data = {}
    all_jobs = []
    for task in tasks:
        rows, emb = load_task(task, forms_by_task[task])
        if rows is None:
            print(f"  {task}: emb missing -- skipped", flush=True)
            continue
        cl = json.loads((MATCH_OUT / f"clusters_{task}.json").read_text())
        reps, centroids, members = cluster_data(rows, emb, cl)
        cids = list(reps.keys())
        if args.sliding:
            batches, pre_singletons = make_sliding_batches(
                cids, centroids, args.batch_size, coverage=args.coverage)
        else:
            batches, pre_singletons = make_batches(
                cids, centroids, args.batch_size)
        print(f"  {task}: {len(cids)} clusters -> {len(batches)} batches "
              f"({'sliding' if args.sliding else 'cover-once'})", flush=True)
        task_data[task] = (reps, members, pre_singletons, centroids)
        for bi, batch in enumerate(batches):
            all_jobs.append((task, bi, batch,
                             build_messages(task, batch, reps, members,
                                            step_b=args.step_b)))

    print(f"\ntotal batches: {len(all_jobs)}; loading vLLM...", flush=True)

    from vllm import LLM, SamplingParams
    model_dir = MODEL_BASE + "/" + sorted(os.listdir(MODEL_BASE))[0]
    llm = LLM(model=model_dir, dtype="bfloat16", tensor_parallel_size=1,
              gpu_memory_utilization=0.9, max_model_len=args.max_model_len,
              enforce_eager=False)
    sampling = SamplingParams(temperature=0.0, max_tokens=3500)
    print("submitting...", flush=True)
    outputs = llm.chat([m for _, _, _, m in all_jobs], sampling, use_tqdm=True)

    by_task = defaultdict(list)
    for (task, bi, batch, _), out in zip(all_jobs, outputs):
        by_task[task].append((bi, batch, out.outputs[0].text))

    # --- Per-task assembly: write raw + build all_fams + cluster_appearances ---
    task_assembly = {}  # task -> (all_fams, cluster_appearances)
    for task in task_data:
        items = by_task.get(task, [])
        reps, members, pre_singletons, _ = task_data[task]
        with (r1_out / f"r1_raw_{task}.jsonl").open("w") as raw_f:
            for bi, batch, text in sorted(items):
                raw_f.write(json.dumps(
                    {"batch_idx": bi, "batch": batch, "raw": text}) + "\n")
        all_fams, appearances = assemble_per_task(items, reps, pre_singletons)
        task_assembly[task] = (all_fams, appearances)

    # --- Merge pass (optional) ---
    task_roots = {}
    if args.merge:
        print("\n=== merge pass ===", flush=True)
        all_merge_jobs = []  # (task, pair_batch, messages)
        for task, (all_fams, appearances) in task_assembly.items():
            reps, _, _, centroids = task_data[task]
            fam_cen = family_centroids(all_fams, centroids)
            cands = merge_candidates(all_fams, appearances, fam_cen,
                                     centroid_thresh=args.centroid_thresh,
                                     include_overlap=args.sliding)
            print(f"  {task}: {len(all_fams)} fams, {len(cands)} merge candidate pairs",
                  flush=True)
            for s in range(0, len(cands), 10):
                pair_batch = cands[s:s + 10]
                msgs = make_merge_message(task, pair_batch, all_fams, reps)
                all_merge_jobs.append((task, pair_batch, msgs))

        if all_merge_jobs:
            print(f"  submitting {len(all_merge_jobs)} merge prompts...",
                  flush=True)
            merge_outputs = llm.chat([m for _, _, m in all_merge_jobs],
                                      sampling, use_tqdm=True)
            task_yes_edges = defaultdict(list)
            task_total_pairs = defaultdict(int)
            for (task, pair_batch, _), out in zip(all_merge_jobs, merge_outputs):
                verdicts = parse_merge_verdicts(out.outputs[0].text,
                                                 len(pair_batch))
                task_total_pairs[task] += len(pair_batch)
                for k, (a, b) in enumerate(pair_batch, 1):
                    if verdicts.get(k, False):
                        task_yes_edges[task].append((a, b))
            for task, (all_fams, _) in task_assembly.items():
                yes = task_yes_edges.get(task, [])
                fam_sizes = [len(f["cluster_ids"]) for f in all_fams]
                roots = union_find_roots(len(all_fams), yes,
                                          fam_sizes=fam_sizes,
                                          max_size=args.max_merged_size)
                task_roots[task] = roots
                n_unique = len(set(roots))
                print(f"  {task}: {len(yes)}/{task_total_pairs.get(task, 0)} "
                      f"YES merges (cap={args.max_merged_size}) -> "
                      f"{len(all_fams)} -> {n_unique} families", flush=True)
        else:
            for task, (all_fams, _) in task_assembly.items():
                task_roots[task] = list(range(len(all_fams)))

    # --- Consolidate + write per task ---
    for task, (all_fams, appearances) in task_assembly.items():
        if args.merge:
            final, cluster_to_family = consolidate_with_merges(
                all_fams, appearances, task_roots[task])
        else:
            final, cluster_to_family = consolidate_first_come(all_fams)
        (r1_out / f"r1_families_{task}.json").write_text(json.dumps(
            {"families": final,
             "cluster_to_family": {str(k): v for k, v in cluster_to_family.items()}
             }, indent=1))
        n_multi = sum(1 for f in final if len(f["cluster_ids"]) > 1)
        print(f"  {task}: {len(final)} R1 families "
              f"({n_multi} multi, {len(final) - n_multi} singletons) "
              f"covering {len(cluster_to_family)} clusters", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
