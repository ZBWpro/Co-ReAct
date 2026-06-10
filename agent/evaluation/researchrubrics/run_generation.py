#!/usr/bin/env python3
"""
Generate agent responses for ResearchRubrics benchmark.

Reads 101 prompts from processed_data.jsonl, runs the Shenma agent pipeline,
and saves each response as {sample_id}.md in agent_responses/.

Usage:
    python run_generation.py --num 2           # test with 2 examples
    python run_generation.py --num 101         # run all
    python run_generation.py --config CONFIG   # custom config
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add agent root to path
AGENT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(AGENT_ROOT))
sys.path.insert(0, str(AGENT_ROOT / "workflows"))

from workflows.co_react_eval import CoReActEvalWorkflow


def load_researchrubrics_prompts(data_path: str, num: int = None):
    """Load prompts from processed_data.jsonl"""
    prompts = []
    with open(data_path) as f:
        for line in f:
            d = json.loads(line)
            prompts.append({
                "sample_id": d["sample_id"],
                "prompt": d["prompt"],
                "domain": d.get("domain", ""),
            })
    if num is not None:
        prompts = prompts[:num]
    return prompts


def extract_answer(response_text: str) -> str:
    """Extract content between <answer> and </answer> tags, or return full text."""
    if not response_text:
        return ""
    start = response_text.find("<answer>")
    end = response_text.find("</answer>")
    if start >= 0 and end >= 0:
        return response_text[start + len("<answer>"):end].strip()
    elif start >= 0:
        return response_text[start + len("<answer>"):].strip()
    return response_text.strip()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(AGENT_ROOT / "workflows" / "baseline_shenma_nothink_v2.yaml"),
                        help="Workflow config YAML")
    parser.add_argument("--num", type=int, default=None, help="Number of examples to run")
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "agent_responses"),
                        help="Output directory for .md files")
    parser.add_argument("--concurrent", type=int, default=5, help="Max concurrent requests")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    data_path = Path(__file__).parent / "data" / "researchrubrics" / "processed_data.jsonl"
    prompts = load_researchrubrics_prompts(str(data_path), args.num)
    print(f"Loaded {len(prompts)} prompts from ResearchRubrics")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip already completed
    done_ids = {p.stem for p in output_dir.glob("*.md")}
    remaining = [p for p in prompts if p["sample_id"] not in done_ids]
    print(f"Already done: {len(done_ids)}, remaining: {len(remaining)}")

    if not remaining:
        print("All done!")
        return

    # Initialize workflow
    workflow = CoReActEvalWorkflow(configuration=args.config, skip_service_check=True)
    print(f"Workflow initialized with config: {args.config}")

    # Process with concurrency limit
    sem = asyncio.Semaphore(args.concurrent)
    results = {"success": 0, "fail": 0}

    async def process_one(item):
        sample_id = item["sample_id"]
        prompt = item["prompt"]
        async with sem:
            t0 = time.time()
            try:
                print(f"[{results['success']+results['fail']+1}/{len(remaining)}] Processing {sample_id} ({item['domain']})")
                output = await workflow(
                    problem=prompt,
                    dataset_name="researchrubrics",
                    verbose=args.verbose,
                )
                response = output.get("final_response", "")
                answer = extract_answer(response)
                if not answer:
                    answer = response  # fallback to full text

                # Save as markdown
                md_path = output_dir / f"{sample_id}.md"
                md_path.write_text(answer, encoding="utf-8")
                elapsed = time.time() - t0
                print(f"  -> Saved {md_path.name} ({len(answer)} chars, {elapsed:.1f}s)")
                results["success"] += 1
            except Exception as e:
                elapsed = time.time() - t0
                print(f"  -> FAILED {sample_id}: {e} ({elapsed:.1f}s)")
                results["fail"] += 1

    tasks = [process_one(item) for item in remaining]
    await asyncio.gather(*tasks)

    print(f"\nDone! Success: {results['success']}, Failed: {results['fail']}")
    print(f"Responses saved to: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
