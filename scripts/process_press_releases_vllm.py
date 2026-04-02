#!/usr/bin/env python3
"""Extract clean press-release body text from raw scraped pages using VLLM offline."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent

SYSTEM_PROMPT = (
    "You are a text extraction assistant. You will be given the raw scraped "
    "text of a web page that may contain a press release. "
    "Your task is to extract ONLY the main body text of the press release.\n\n"
    "Rules:\n"
    "- If the page contains a news release or press release, extract its "
    "full main body text faithfully. Do not summarize or paraphrase.\n"
    "- Remove all website chrome: navigation links, breadcrumbs, menu items, "
    "footer text, sidebar content, social media links, subscription prompts, "
    "cookie notices, and similar non-article elements.\n"
    "- If there is no identifiable main body text (e.g. the page is just "
    "navigation, an error page, or a landing page with no article), set "
    "news_release_found to false and return an empty string for result."
)

USER_TEMPLATE = (
    "Extract the press release body text from the following raw page content.\n\n"
    "Respond in JSON with two fields:\n"
    '  "news_release_found": true/false\n'
    '  "result": "<extracted text or empty string>"\n\n'
    "Raw page content:\n\n{text}"
)


def load_press_releases(input_path: Path) -> pd.DataFrame:
    """Load press releases, deduplicating by press_release_id."""
    print(f"Loading data from {input_path} ...")
    df = pd.read_csv(input_path, low_memory=False)
    unique = df.drop_duplicates(subset="press_release_id")[
        ["press_release_id", "text"]
    ].copy()
    unique = unique.dropna(subset=["text"])
    unique = unique.reset_index(drop=True)
    return unique


def build_conversations(texts: list[str], max_chars: int = 24000) -> list[list[dict]]:
    """Build chat conversations for VLLM, truncating very long pages."""
    convos = []
    for t in texts:
        truncated = t[:max_chars] if len(t) > max_chars else t
        convos.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(text=truncated)},
        ])
    return convos


def main():
    parser = argparse.ArgumentParser(description="Extract press release text via VLLM")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "datasets" / "press-releases" / "press_release_modeling_dataset.csv.gz",
        help="Input CSV or CSV.GZ with press_release_id and text columns",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL path")
    parser.add_argument("--batch-size", type=int, default=8192, help="Prompts per VLLM batch")
    parser.add_argument("--max-model-len", type=int, default=16384, help="VLLM max context length")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--max-chars", type=int, default=24000, help="Truncate input pages to N chars")
    parser.add_argument("--resume", action="store_true", help="Skip IDs already in output file")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.parent / "press_release_extracted.jsonl"

    # Load unique press releases
    unique_df = load_press_releases(args.input)
    print(f"  Total unique press releases: {len(unique_df)}")

    # Resume support: skip already-processed IDs
    done_ids = set()
    if args.resume and args.output.exists():
        with open(args.output) as f:
            for line in f:
                rec = json.loads(line)
                done_ids.add(rec["press_release_id"])
        print(f"  Resuming: {len(done_ids)} already processed, skipping them")
        unique_df = unique_df[~unique_df["press_release_id"].isin(done_ids)].reset_index(drop=True)
        print(f"  Remaining: {len(unique_df)}")

    if len(unique_df) == 0:
        print("Nothing to process.")
        return

    # Init VLLM
    from vllm import LLM, SamplingParams

    print(f"Initializing VLLM with {args.model} ...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.95,
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=4096)

    # Process in batches
    ids = unique_df["press_release_id"].tolist()
    texts = unique_df["text"].tolist()
    total = len(texts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    with open(args.output, mode) as fout:
        for start in tqdm(range(0, total, args.batch_size), desc="Batches"):
            end = min(start + args.batch_size, total)
            batch_ids = ids[start:end]
            batch_texts = texts[start:end]

            convos = build_conversations(batch_texts, max_chars=args.max_chars)
            try:
                outputs = llm.chat(convos, sampling_params, use_tqdm=True)
            except Exception as e:
                print(f"\nBatch {start}-{end} failed: {e}")
                print("Retrying individually...")
                outputs = []
                for i, convo in enumerate(convos):
                    try:
                        out = llm.chat([convo], sampling_params)
                        outputs.append(out[0])
                    except Exception as e2:
                        print(f"  Skipping PR {batch_ids[i]}: {e2}")
                        outputs.append(None)

            for pr_id, output in zip(batch_ids, outputs):
                if output is None:
                    rec = {
                        "press_release_id": int(pr_id),
                        "raw_response": "",
                        "news_release_found": False,
                        "extracted_text": "",
                        "error": True,
                    }
                else:
                    generated = output.outputs[0].text.strip()
                    rec = {
                        "press_release_id": int(pr_id),
                        "raw_response": generated,
                    }
                    try:
                        parsed = json.loads(generated)
                        rec["news_release_found"] = parsed.get("news_release_found", True)
                        rec["extracted_text"] = parsed.get("result", "")
                    except json.JSONDecodeError:
                        rec["news_release_found"] = True
                        rec["extracted_text"] = generated

                fout.write(json.dumps(rec) + "\n")
                fout.flush()

    print(f"\nDone. Output: {args.output}")
    print(f"  Processed {total} unique press releases")


if __name__ == "__main__":
    main()
