"""Batch evaluate ResearchQA experiments using Cloudsway GPT-4.1-mini."""
import json
import os
import sys
import argparse

# Set Cloudsway endpoint (read from environment)
os.environ.setdefault("OPENAI_API_KEY", os.environ.get("CLOUDSWAY_API_KEY", ""))
os.environ.setdefault("OPENAI_BASE_URL", os.environ.get("CLOUDSWAY_GPT4_MINI_URL", ""))

from compute_coverage import compute_coverage


def load_response_map(jsonl_path: str) -> dict:
    """Convert experiment jsonl to response_map format.

    Uses orig_id from original_data to match ResearchQA IDs.
    """
    response_map = {}
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            # Use orig_id (original ResearchQA ID) for matching
            orig_id = d.get("original_data", {}).get("orig_id", d["example_id"])
            response_map[orig_id] = {"answer": d["final_response"]}
    return response_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Experiment jsonl file")
    parser.add_argument("--data", type=str,
                        default="evaluation/research_qa_eval/researchqa_200.json",
                        help="ResearchQA data file (JSON)")
    parser.add_argument("--model", type=str, default="MaaS_4.1_mini")
    args = parser.parse_args()

    # Convert jsonl to response_map
    response_map = load_response_map(args.input)
    print(f"Loaded {len(response_map)} responses from {args.input}")

    # Output path
    basename = os.path.splitext(os.path.basename(args.input))[0]
    output_dir = os.path.dirname(args.input)
    output_path = os.path.join(output_dir, f"{basename}_coverage.json")

    # Run evaluation
    results, coverages = compute_coverage(
        data_path=args.data,
        response_map=response_map,
        output_path=output_path,
        model=args.model,
    )

    if coverages:
        avg = sum(coverages) / len(coverages)
        print(f"\n{'='*50}")
        print(f"File: {args.input}")
        print(f"Average Coverage: {avg:.4f}")
        print(f"Evaluated: {len(coverages)}/{len(response_map)}")
        print(f"Results saved to: {output_path}")
    else:
        print("No coverage scores computed.")


if __name__ == "__main__":
    main()
