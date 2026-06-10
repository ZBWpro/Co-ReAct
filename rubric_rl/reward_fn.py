"""
Reward function for GRPO training of the rubric generator.

verl calls compute_score() for each generated response with:
  - data_source: str ("rubric_gen")
  - solution_str: str (generated rubric text from Qwen3-8B)
  - ground_truth: str (JSON with actions and borda scores)
  - extra_info: dict (prompt_id, step_index, etc.)

Four reward components (quality 0.65, format 0.35):
  1. Spearman correlation (0.65): rubric-guided ranking vs expert Borda ranking
  2. Atomicity reward (0.15): judge evaluates whether criteria are atomic/verifiable
  3. Format reward (0.10): 3-5 criteria, weights sum to ~1.0
  4. Think format reward (0.10): properly closed <think>...</think> tags
  + Hard gate: if 4-gram repetition rate > 40%, entire reward = 0
"""

import json
import os
import random
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from scipy.stats import spearmanr

# Add rubric_rl dir to path so rubric_judge can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rubric_judge import rank_actions_by_rubric


# ============================================================================
# Rollout sample logging
# ============================================================================

ROLLOUT_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rollout_logs")
os.makedirs(ROLLOUT_LOG_DIR, exist_ok=True)
_sample_counter = 0
_SAMPLE_RATE = 50  # log 1 in every N calls


# ============================================================================
# Rubric parsing
# ============================================================================

RUBRIC_LINE_PATTERN = re.compile(r'\d+\.\s*\[([-\d.]+)\]\s*(.+)')


def parse_rubric(text: str) -> List[Tuple[float, str]]:
    """Parse rubric text into list of (weight, criterion_text)."""
    items = []
    for match in RUBRIC_LINE_PATTERN.finditer(text):
        try:
            weight = float(match.group(1))
        except ValueError:
            continue
        criterion = match.group(2).strip()
        items.append((weight, criterion))
    return items


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from generated rubric."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "<think>" in text:
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
    return text


# ============================================================================
# Reward components
# ============================================================================

def compute_format_reward(rubric_text: str) -> float:
    """Check rubric format: 3-5 items, positive weights sum ≈ 1.0.

    Returns 0.0, 0.5, or 1.0.
    """
    items = parse_rubric(rubric_text)
    if not items:
        return 0.0

    score = 0.0

    # Check item count: 3-5
    if 3 <= len(items) <= 5:
        score += 0.5

    # Check positive weights sum ≈ 1.0
    positive_sum = sum(w for w, _ in items if w > 0)
    if abs(positive_sum - 1.0) < 0.15:
        score += 0.5

    return score


def compute_think_reward(raw_response: str) -> float:
    """Check that <think>...</think> tags are properly closed.

    Returns 1.0 if think tags are properly paired, 0.0 otherwise.
    A response with no think tags also gets 1.0 (no-think is fine).
    """
    has_open = '<think>' in raw_response
    has_close = '</think>' in raw_response

    if not has_open and not has_close:
        # No think tags at all — acceptable
        return 1.0
    if has_open and has_close:
        # Check proper nesting: <think> before </think>
        open_pos = raw_response.find('<think>')
        close_pos = raw_response.find('</think>')
        if open_pos < close_pos:
            return 1.0
    # Unclosed <think> or stray </think> or wrong order
    return 0.0


def is_repetitive(text: str, n: int = 4, threshold: float = 0.40) -> bool:
    """Detect if text has excessive n-gram repetition.

    Returns True if >threshold of 4-grams are repeated (triggers hard zero reward).
    """
    words = re.findall(r'\w+', text.lower())
    if len(words) < n + 1:
        return False
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    total = len(ngrams)
    unique = len(set(ngrams))
    repeat_rate = 1.0 - unique / total
    return repeat_rate > threshold


def compute_atomicity_reward(atomic_count: Optional[int], total_criteria: int) -> float:
    """Compute atomicity reward from judge's evaluation.

    Returns atomic_count / total_criteria (proportion of atomic criteria).
    If atomic_count is unavailable, returns 0.0 (conservative fallback).
    """
    if atomic_count is None or total_criteria <= 0:
        return 0.0
    # Clamp atomic_count to valid range
    atomic_count = max(0, min(atomic_count, total_criteria))
    return atomic_count / total_criteria


def compute_focus_penalty(rubric_text: str) -> float:
    """Penalize rubrics that recommend multiple different tools.

    A good rubric focuses on evaluating ONE action (one tool call).
    If criteria mention multiple different tools, apply a penalty.

    Returns:
        1.0 if rubric focuses on one tool (no penalty)
        0.5 if rubric mentions 2 tools
        0.0 if rubric mentions 3+ tools
    """
    tool_patterns = {
        "google_search": re.compile(r'google_search', re.IGNORECASE),
        "snippet_search": re.compile(r'snippet_search', re.IGNORECASE),
        "browse_webpage": re.compile(r'browse_webpage', re.IGNORECASE),
        "answer": re.compile(r'<answer>|write_report|生成.*报告|generate.*answer|结束研究', re.IGNORECASE),
    }
    found_tools = set()
    for tool_name, pattern in tool_patterns.items():
        if pattern.search(rubric_text):
            found_tools.add(tool_name)

    n_tools = len(found_tools)
    if n_tools <= 1:
        return 1.0
    elif n_tools == 2:
        return 0.5
    else:
        return 0.0


def borda_to_ranking(borda_scores: Dict[str, float]) -> List[float]:
    """Convert Borda scores to ranking list with tie support.

    Borda: lower = better. Tied scores get average rank.
    Returns [rank_A, rank_B, rank_C, rank_D] (float, 1-based).
    Example: {"A": 4, "B": 7.5, "C": 10, "D": 7.5} → [1.0, 2.5, 4.0, 2.5]
    """
    labels = sorted(borda_scores.keys())
    scores = [(borda_scores[l], l) for l in labels]
    scores.sort(key=lambda x: x[0])

    # Group by score to detect ties
    ranks = {}
    i = 0
    while i < len(scores):
        j = i
        while j < len(scores) and scores[j][0] == scores[i][0]:
            j += 1
        # Positions i..j-1 are tied, assign average rank (1-based)
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[scores[k][1]] = avg_rank
        i = j

    return [ranks[l] for l in labels]


# ============================================================================
# Weights — quality 0.70, format 0.30
# ============================================================================

SPEARMAN_WEIGHT = 0.75
ATOMICITY_WEIGHT = 0.15
FORMAT_WEIGHT = 0.10
THINK_WEIGHT = 0.0
FOCUS_WEIGHT = 0.0


# ============================================================================
# verl entry point
# ============================================================================

async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    **kwargs,
) -> dict:
    """verl-compatible async reward function.

    Args:
        data_source: "rubric_gen"
        solution_str: Generated rubric text from Qwen3-8B
        ground_truth: JSON string with {question, actions: {A: {text, borda_score}, ...}, trajectory}
        extra_info: {prompt_id, step_index, ...}

    Returns:
        {"score": float, "spearman": float, "atomicity": float, "format": float, "think": float, "focus": float}
    """
    # Hard gate: repetition kills entire reward
    if is_repetitive(solution_str):
        return {
            "score": 0.0,
            "spearman": 0.0,
            "atomicity": 0.0,
            "format": 0.0,
            "think": 0.0,
            "focus": 0.0,
        }

    # Think format reward (on raw response, before stripping)
    think_r = compute_think_reward(solution_str)

    # Clean generated rubric (strip think tags + stray </think>)
    rubric_text = strip_think_tags(solution_str)
    rubric_text = re.sub(r'</think>', '', rubric_text).strip()

    # Format reward (no API call)
    format_r = compute_format_reward(rubric_text)

    # Focus penalty (no API call) — penalize multi-tool rubrics
    focus_r = compute_focus_penalty(rubric_text)

    # Parse rubric for total criteria count
    items = parse_rubric(rubric_text)
    total_criteria = len(items)

    # Spearman + Atomicity (both from single Gemini judge call)
    spearman_r = 0.0
    atomicity_r = 0.0

    if total_criteria >= 2:
        try:
            meta = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
            question = meta["question"]
            actions = {label: data["text"] for label, data in meta["actions"].items()}
            trajectory = meta.get("trajectory", "")

            result = await rank_actions_by_rubric(
                rubric=rubric_text,
                question=question,
                actions=actions,
                trajectory=trajectory,
            )

            if result is not None:
                rubric_ranking = result["ranking"]
                atomic_count = result.get("atomic_count")

                # Spearman correlation
                borda_scores = {
                    label: data["borda_score"]
                    for label, data in meta["actions"].items()
                }
                expert_ranking = borda_to_ranking(borda_scores)
                rho, _ = spearmanr(rubric_ranking, expert_ranking)
                # Handle NaN
                if rho == rho:  # not NaN
                    spearman_r = (rho + 1.0) / 2.0
                else:
                    spearman_r = 0.5

                # Atomicity from judge
                atomicity_r = compute_atomicity_reward(atomic_count, total_criteria)

        except Exception as e:
            print(f"[RewardFn] Reward computation failed: {e}")
            spearman_r = 0.0
            atomicity_r = 0.0

    total = (
        SPEARMAN_WEIGHT * spearman_r
        + ATOMICITY_WEIGHT * atomicity_r
        + FORMAT_WEIGHT * format_r
        + THINK_WEIGHT * think_r
        + FOCUS_WEIGHT * focus_r
    )

    # Sample logging for rollout inspection
    global _sample_counter
    _sample_counter += 1
    if _sample_counter % _SAMPLE_RATE == 1:
        try:
            step_idx = extra_info.get("step_index", "?") if extra_info else "?"
            prompt_id = extra_info.get("prompt_id", "?") if extra_info else "?"
            log_file = os.path.join(ROLLOUT_LOG_DIR, "samples.jsonl")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "prompt_id": prompt_id,
                    "step_index": step_idx,
                    "response": solution_str[:2000],
                    "rubric_text": rubric_text[:1000],
                    "scores": {"total": round(total, 4), "spearman": round(spearman_r, 4),
                               "atomicity": round(atomicity_r, 4), "format": round(format_r, 4),
                               "think": round(think_r, 4), "focus": round(focus_r, 4)},
                    "n_criteria": total_criteria,
                    "repetitive": is_repetitive(solution_str),
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass

    return {
        "score": total,
        "spearman": spearman_r,
        "atomicity": atomicity_r,
        "format": format_r,
        "think": think_r,
        "focus": focus_r,
    }
