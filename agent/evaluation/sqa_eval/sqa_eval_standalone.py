#!/usr/bin/env python3
"""
Standalone SQA-CS-V2 evaluation script.
Reimplements astabench's 4 SQA metrics using Cloudsway Gemini proxy,
no inspect_ai or GOOGLE_API_KEY needed.

Metrics:
  1. ingredient_recall  — rubric-based content completeness
  2. answer_precision   — fraction of relevant paragraphs
  3. citation_recall    — claims supported by their citations
  4. citation_precision — supporting citations per claim

Usage:
  python sqa_eval_standalone.py --input_file <responses.jsonl> --rubric_file <rubrics.json> [--output_dir <dir>] [--limit N]
"""
import argparse
import asyncio
import json
import logging
import os
import re
import difflib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Cloudsway Gemini proxy (same as DRB eval) ──────────────────────────
CLOUDSWAY_URLS = {
    "gemini-2.5-flash": os.environ.get("CLOUDSWAY_GEMINI_FLASH_URL", ""),
    "gemini-2.5-pro":   os.environ.get("CLOUDSWAY_GEMINI_PRO_URL", ""),
}
CLOUDSWAY_API_KEY = os.environ.get("CLOUDSWAY_API_KEY", "")
SCORER_MODEL = "gemini-2.5-flash"  # use flash for speed, like astabench default

MAX_CONCURRENT = 5
SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT)


# ── LLM call helper ────────────────────────────────────────────────────
async def llm_call(
    messages: List[Dict[str, str]],
    model: str = SCORER_MODEL,
    temperature: float = 0.5,
    max_tokens: int = 4096,
    max_retries: int = 10,
    json_mode: bool = True,
) -> Optional[Dict]:
    """Call Cloudsway Gemini and parse JSON response."""
    url = CLOUDSWAY_URLS[model]
    headers = {
        "Authorization": f"Bearer {CLOUDSWAY_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    for attempt in range(max_retries + 1):
        try:
            async with SEMAPHORE:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                        if resp.status == 429:
                            delay = 2 * (1.5 ** attempt)
                            logger.warning(f"Rate limited, waiting {delay:.1f}s...")
                            await asyncio.sleep(delay)
                            continue
                        resp.raise_for_status()
                        data = await resp.json()

            content = data["choices"][0]["message"]["content"]
            # Parse JSON from response
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(content[json_start:json_end])
            else:
                raise ValueError(f"No JSON found in response: {content[:200]}")

        except Exception as e:
            logger.warning(f"Attempt {attempt+1}/{max_retries+1} failed: {e}")
            if attempt < max_retries:
                await asyncio.sleep(2 * (1.5 ** attempt))
            else:
                logger.error(f"All retries exhausted")
                return None
    return None


# ── 1. Ingredient Recall (rubric scoring) ──────────────────────────────
async def score_ingredient_recall(
    response_text: str, question: str, ingredients: List[Dict]
) -> float:
    """
    Jointly assess all rubric ingredients (criteria).
    Returns score between 0 and 1.
    """
    has_criterion = [ing for ing in ingredients if ing.get("criterion")]
    if not has_criterion:
        return 0.0

    criteria_str = "\n".join(
        f'<criterion>\n{i}. {x["criterion"]}\n<examples>\n{chr(10).join(x.get("examples", []))}\n</examples>\n</criterion>'
        for i, x in enumerate(has_criterion, 1)
    )

    system_prompt = """You will be given a question someone asked (in <question></question> tags) and the corresponding response (in <response></response> tags) given to them by an assistant.
You will then be given an enumerated list of criteria by which to evaluate the response. Each criterion specifies requirements that the answer must satisfy.
You will also be given a list of examples (in <examples></examples> tags, below each criterion) that illustrate the type of details that would satisfy the criterion.

For each criterion, return a score of 0, 1 or 2 indicating how appropriate the response is based on the given criterion. 0 means the response does not meet the criterion, 1 means the response somewhat meets the criterion, 2 means the response perfectly meets the criterion.

Return your result as a JSON object with a single key `scores` whose value is a list of objects, each having keys `criteria_idx`, `reasoning`, `score` and `evidence`."""

    user_prompt = f"<question>{question}</question>\n<response>{response_text}</response>\nCriteria:\n{criteria_str}"

    result = await llm_call([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])

    if not result or "scores" not in result:
        return 0.0

    # Compute weighted score
    weights = {x["name"]: x["weight"] for x in has_criterion}
    total_w = sum(weights.values())
    if total_w == 0:
        return 0.0
    weights = {k: v / total_w for k, v in weights.items()}

    score_map = {}
    for s in result["scores"]:
        idx = s.get("criteria_idx", 0) - 1
        if 0 <= idx < len(has_criterion):
            name = has_criterion[idx]["name"]
            score_map[name] = s.get("score", 0) / 2.0

    return sum(weights.get(k, 0) * score_map.get(k, 0) for k in weights)


# ── 2. Answer Precision ────────────────────────────────────────────────
PRECISION_PROMPT = """You are given a query and a corresponding long answer.

Goal: find irrelevant paragraphs in the answer. These are paragraphs that don't directly answer the query and shouldn't be in the answer.

Explicitly consider whether something may be indirectly relevant. Background that contextualizes the answer is relevant.

Go through the answer and output a list of irrelevant paragraphs. Every single paragraph needs to be considered.
Return your result as a JSON object with a single key `irrelevant_paragraphs` whose value is a list of objects, each having keys `reason` and `answer_text` (exact ENTIRE paragraph from the answer that is irrelevant).

Question: {query}
Answer: {answer}"""


async def score_answer_precision(question: str, response_text: str) -> float:
    prompt = PRECISION_PROMPT.format(query=question, answer=response_text)
    result = await llm_call([{"role": "user", "content": prompt}])

    if not result or "irrelevant_paragraphs" not in result:
        return 1.0  # assume all relevant if scorer fails

    irrelevant_texts = [p["answer_text"] for p in result["irrelevant_paragraphs"]]
    paragraphs = re.split(r"\n\s*\n", response_text.strip())
    if not paragraphs:
        return 1.0

    matching = set()
    for irr in irrelevant_texts:
        for idx, para in enumerate(paragraphs):
            if difflib.SequenceMatcher(None, irr.strip(), para).ratio() >= 0.85:
                matching.add(idx)
                break

    return 1 - (len(matching) / len(paragraphs))


# ── 3 & 4. Citation Recall & Precision ─────────────────────────────────
CITATION_GROUP_PROMPT = """You are a claim validator. For each claim made in the following text you will determine if it is supported by the quote from its corresponding inline citations. Consecutive sentences can share citations. Include claims in tables too. For references with only the title available, judge them as `supporting` if the title indicates relevance.

Return a JSON object with a single key `claims` which is a list of `claim` objects. Each has keys: `text`, `supporting` (list of inline citation IDs), `non_supporting` (list of inline citation IDs), and `is_fully_supported` (boolean). Each claim's citations should appear in either `supporting` or `non_supporting`, not both.

Text:
{text}

References:
{references}"""


async def score_citations(sections: List[Dict]) -> Dict[str, float]:
    """
    Evaluate citation recall and precision across all sections.
    Returns dict with citation_recall, citation_precision, f1.
    """
    n_attributable = 0
    n_extrapolatory = 0
    precisions = []

    for section in sections:
        text = section.get("text", "")
        citations = section.get("citations", [])
        if not text or not citations:
            continue

        # Format citations for the prompt
        ref_parts = []
        for c in citations:
            snippets = c.get("snippets", [])
            if isinstance(snippets, list):
                snippets_str = "... ".join(s for s in snippets if s.strip())
            else:
                snippets_str = str(snippets)
            ref_parts.append(f'{c["id"]}: {snippets_str}')

        references = "\n\n".join(ref_parts)
        prompt = CITATION_GROUP_PROMPT.format(text=text, references=references)

        result = await llm_call([{"role": "user", "content": prompt}])
        if not result or "claims" not in result:
            continue

        for claim in result["claims"]:
            supported = claim.get("is_fully_supported", False) and claim.get("supporting", [])
            n_attributable += 1 if supported else 0
            n_extrapolatory += 0 if supported else 1

            n_sup = len(claim.get("supporting", []))
            n_nsup = len(claim.get("non_supporting", []))
            if n_sup + n_nsup > 0:
                precisions.append(n_sup / (n_sup + n_nsup))

    total = n_attributable + n_extrapolatory
    recall = n_attributable / total if total > 0 else 0
    precision = sum(precisions) / len(precisions) if precisions else 0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0

    return {"citation_recall": recall, "citation_precision": precision, "f1": f1}


# ── Response parser ────────────────────────────────────────────────────
def extract_json_from_response(text: str) -> Optional[Dict]:
    """Extract JSON from model response, handling various formats."""
    # Try direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    if not text:
        return None

    # Find JSON in text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return None


def parse_response_sections(response_data) -> Tuple[str, List[Dict]]:
    """
    Parse response into plain text and sections with citations.
    Supports two formats:
      1. astabench JSON: {"sections": [{"text": ..., "citations": [...]}]}
      2. DR-Tulu markdown with <cite id="...">snippet</cite> tags
    """
    # If it's a dict with sections (astabench format)
    if isinstance(response_data, dict):
        sections = response_data.get("sections", [])
        if not sections:
            resp = response_data.get("response", {})
            if isinstance(resp, dict):
                sections = resp.get("sections", [])
        if sections:
            full_text = ""
            parsed_sections = []
            for sec in sections:
                text = sec.get("text", "")
                if text:
                    full_text += text + "\n\n"
                table = sec.get("table", {})
                if isinstance(table, dict) and table.get("text"):
                    full_text += table["text"] + "\n\n"
                parsed_sections.append({
                    "text": text + ("\n\n" + table["text"] if isinstance(table, dict) and table.get("text") else ""),
                    "citations": sec.get("citations", []),
                })
            return full_text.strip(), parsed_sections

    # DR-Tulu markdown format: split by headings or double newlines
    text = response_data if isinstance(response_data, str) else ""
    if not text:
        return "", []

    # Extract all <cite> tags and build a citation registry
    cite_pattern = re.compile(r'<cite\s+id="([^"]+)">(.*?)</cite>', re.DOTALL)
    all_cites = {m.group(1): m.group(2).strip() for m in cite_pattern.finditer(text)}

    # Split into sections by markdown headings
    heading_pattern = re.compile(r'^(#{1,4}\s+.+)$', re.MULTILINE)
    parts = heading_pattern.split(text)

    parsed_sections = []
    current_text = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if heading_pattern.match(part):
            # Save previous section
            if current_text.strip():
                sec_cites = _extract_section_citations(current_text, all_cites)
                parsed_sections.append({"text": current_text.strip(), "citations": sec_cites})
            current_text = part + "\n"
        else:
            current_text += part + "\n"

    # Last section
    if current_text.strip():
        sec_cites = _extract_section_citations(current_text, all_cites)
        parsed_sections.append({"text": current_text.strip(), "citations": sec_cites})

    # If no headings found, treat entire text as one section
    if not parsed_sections:
        sec_cites = _extract_section_citations(text, all_cites)
        parsed_sections = [{"text": text.strip(), "citations": sec_cites}]

    # Clean cite tags from full text for ingredient_recall / precision scoring
    clean_text = re.sub(r'<cite\s+id="[^"]+">|</cite>', '', text)
    return clean_text.strip(), parsed_sections


def _extract_section_citations(section_text: str, all_cites: Dict[str, str]) -> List[Dict]:
    """Extract citations that appear in a section."""
    cite_pattern = re.compile(r'<cite\s+id="([^"]+)">(.*?)</cite>', re.DOTALL)
    citations = []
    seen_ids = set()
    for m in cite_pattern.finditer(section_text):
        cid = m.group(1)
        snippet = m.group(2).strip()
        if cid not in seen_ids:
            seen_ids.add(cid)
            citations.append({"id": cid, "snippets": [snippet] if snippet else []})
    return citations


# ── Main evaluation ────────────────────────────────────────────────────
async def evaluate_single(
    response_raw: str,
    rubric: Dict,
    example_id: str,
) -> Dict[str, Any]:
    """Evaluate a single example across all 4 metrics."""
    question = rubric["question"]
    ingredients = rubric.get("ingredients", [])

    if not response_raw or not response_raw.strip():
        logger.warning(f"[{example_id}] Empty response")
        return {
            "example_id": example_id,
            "ingredient_recall": 0.0,
            "answer_precision": 0.0,
            "citation_recall": 0.0,
            "citation_precision": 0.0,
            "global_avg": 0.0,
            "error": "empty_response",
        }

    # Try JSON parse first (astabench format), fall back to raw markdown
    parsed = extract_json_from_response(response_raw)
    if parsed and (parsed.get("sections") or (isinstance(parsed.get("response"), dict) and parsed["response"].get("sections"))):
        full_text, sections = parse_response_sections(parsed)
    else:
        # DR-Tulu markdown format
        full_text, sections = parse_response_sections(response_raw)

    if not full_text.strip():
        logger.warning(f"[{example_id}] Empty response text after parsing")
        return {
            "example_id": example_id,
            "ingredient_recall": 0.0,
            "answer_precision": 0.0,
            "citation_recall": 0.0,
            "citation_precision": 0.0,
            "global_avg": 0.0,
            "error": "empty_text",
        }

    # Run all 3 scorers concurrently
    ing_task = score_ingredient_recall(full_text, question, ingredients)
    prec_task = score_answer_precision(question, full_text)
    cit_task = score_citations(sections)

    ing_score, prec_score, cit_scores = await asyncio.gather(
        ing_task, prec_task, cit_task
    )

    result = {
        "example_id": example_id,
        "ingredient_recall": ing_score,
        "answer_precision": prec_score,
        "citation_recall": cit_scores["citation_recall"],
        "citation_precision": cit_scores["citation_precision"],
    }
    result["global_avg"] = sum(result[k] for k in ["ingredient_recall", "answer_precision", "citation_recall", "citation_precision"]) / 4
    return result


async def run_eval(
    input_file: str,
    rubric_file: str,
    output_dir: str,
    limit: Optional[int] = None,
    max_concurrent: int = 3,
):
    global SEMAPHORE
    SEMAPHORE = asyncio.Semaphore(max_concurrent)

    os.makedirs(output_dir, exist_ok=True)

    # Load rubrics
    with open(rubric_file) as f:
        rubrics = json.load(f)
    rubric_map = {r["case_id"]: r for r in rubrics}
    logger.info(f"Loaded {len(rubrics)} rubrics from {rubric_file}")

    # Load responses
    responses = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                responses.append(json.loads(line))
    logger.info(f"Loaded {len(responses)} responses from {input_file}")

    if limit:
        responses = responses[:limit]

    # Match responses to rubrics
    results = []
    tasks = []

    for resp in responses:
        # Get example_id and response text
        example_id = resp.get("example_id") or resp.get("id") or resp.get("case_id", "unknown")
        raw_output = resp.get("final_response") or resp.get("pred_answer") or resp.get("response", "")

        rubric = rubric_map.get(str(example_id))
        if not rubric:
            # Try matching by question
            q = resp.get("problem") or resp.get("question", "")
            for r in rubrics:
                if r["question"].strip() == q.strip():
                    rubric = r
                    break

        if not rubric:
            logger.warning(f"[{example_id}] No matching rubric found, skipping")
            continue

        tasks.append(evaluate_single(raw_output, rubric, str(example_id)))

    logger.info(f"Evaluating {len(tasks)} examples...")

    # Run with controlled concurrency
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    valid_results = []
    for r in results:
        if isinstance(r, Exception):
            logger.error(f"Evaluation error: {r}")
            continue
        valid_results.append(r)

    # Compute aggregates
    metrics = ["ingredient_recall", "answer_precision", "citation_recall", "citation_precision", "global_avg"]
    agg = {}
    for m in metrics:
        vals = [r[m] for r in valid_results if m in r]
        agg[m] = sum(vals) / len(vals) if vals else 0.0

    # Write outputs
    results_file = os.path.join(output_dir, "sqa_results.jsonl")
    with open(results_file, "w") as f:
        for r in valid_results:
            f.write(json.dumps(r) + "\n")

    summary_file = os.path.join(output_dir, "sqa_summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"SQA-CS-V2 Evaluation Results (n={len(valid_results)})\n")
        f.write("=" * 50 + "\n\n")
        for m in metrics:
            f.write(f"{m}: {agg[m]:.4f}\n")
        f.write(f"\nErrors: {len(tasks) - len(valid_results)} / {len(tasks)}\n")

    # Print summary
    print("\n" + "=" * 50)
    print(f"SQA-CS-V2 Evaluation Results (n={len(valid_results)})")
    print("=" * 50)
    for m in metrics:
        print(f"  {m}: {agg[m]:.4f}")
    print(f"\n  Errors: {len(tasks) - len(valid_results)} / {len(tasks)}")
    print(f"  Results: {results_file}")
    print(f"  Summary: {summary_file}")

    return agg


def main():
    parser = argparse.ArgumentParser(description="Standalone SQA-CS-V2 Evaluation")
    parser.add_argument("--input_file", required=True, help="JSONL file with model responses")
    parser.add_argument("--rubric_file", default=None, help="JSON file with rubrics (default: data/sqa/rubrics_v2_recomputed.json)")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--max_concurrent", type=int, default=3, help="Max concurrent API calls")
    args = parser.parse_args()

    if args.rubric_file is None:
        args.rubric_file = str(Path(__file__).parent.parent.parent / "data" / "sqa" / "rubrics_v2_recomputed.json")

    if args.output_dir is None:
        args.output_dir = str(Path(args.input_file).parent / f"sqa_eval_{Path(args.input_file).stem}")

    asyncio.run(run_eval(
        input_file=args.input_file,
        rubric_file=args.rubric_file,
        output_dir=args.output_dir,
        limit=args.limit,
        max_concurrent=args.max_concurrent,
    ))


if __name__ == "__main__":
    main()
