"""
Rubric-based action ranking via Gemini 2.5 Pro (Cloudsway proxy).

Given a rubric and 4 candidate actions, calls Gemini once to rank
all actions simultaneously based on rubric criteria.

Uses Cloudsway proxy API (same pattern as judges.py and collect_rl_data.yaml).
Set CLOUDSWAY_GEMINI_PRO_URL and CLOUDSWAY_API_KEY in your environment.
"""

import asyncio
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import aiohttp


RUBRIC_RANKING_PROMPT = """你是深度研究任务的评审专家。你有两个任务：

**任务一：排序**
根据以下评判准则(Rubric)，对 {n_actions} 个候选 Action 从最优到最差排序。

**任务二：原子性评估**
逐条判断 Rubric 中每条准则是否是**原子性的**——即仅通过查看 Action 的工具名称(tool name)、查询参数(query)、URL 等即可判断 yes/no 的**具体可验证条件**。
原子性正例："应使用 snippet_search 搜索学术论文"、"查询应包含 'quantum computing'"、"不应重复搜索 'xxx'"
非原子性反例："信息应准确可靠"、"搜索应全面覆盖"、"来源应权威"

评判准则 (Rubric):
{rubric}

研究问题:
{question}

Agent 已执行的轨迹（当前状态上下文）:
{trajectory}

候选 Actions（已去除思考过程，仅保留实际工具调用和行为）:
{actions_text}

排序要求：
- 严格按照 Rubric 中的每条准则逐一判断每个 Action 的符合程度
- 结合 Agent 已执行的轨迹来理解当前研究状态，判断每个 Action 是否合理
- 仅根据 Action 的实际工具调用（tool name、query、URL）和行为来评判
- 不要考虑任何思考过程，只看最终执行的动作
- 按 Rubric 加权得分从高到低排序
- 如果多个 Action 质量确实相同，允许并列（用数组表示）

输出格式（严格 JSON，不要输出其他内容）:
{{"ranking": [{action_labels}], "atomic_count": <原子性准则的条数>}}
示例: {{"ranking": [{action_labels}], "atomic_count": 3}}
示例（有并列）: {{"ranking": [[{example_tie}], {remaining_labels}], "atomic_count": 2}}"""


ANSWER_PLACEHOLDER = '<call_tool name="write_report">开始撰写最终报告</call_tool>'


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from text."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "<think>" in text:
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
    return text


def _truncate_to_first_tool_call(text: str) -> str:
    """Keep only up to the first complete <call_tool>...</call_tool>."""
    close_pos = text.find("</call_tool>")
    if close_pos >= 0:
        return text[:close_pos + len("</call_tool>")]
    return text


def _abstract_answer(text: str) -> str:
    """Replace action with write_report placeholder if it contains <answer>."""
    if "<answer>" in text:
        return ANSWER_PLACEHOLDER
    return text


def clean_for_judge(text: str) -> str:
    """Clean action text for judge: strip think tags, hallucinated tool_output, abstract answer.

    Ported from collect_rl_data.py _clean_for_judge().
    """
    cleaned = strip_think_tags(text)
    cleaned = re.sub(r'</think>', '', cleaned)
    if '<think>' in cleaned:
        tool_match = re.search(r'(<call_tool[\s\S]*)', cleaned)
        answer_match = re.search(r'(<answer>[\s\S]*)', cleaned)
        if tool_match:
            cleaned = tool_match.group(1)
        elif answer_match:
            cleaned = answer_match.group(1)
        else:
            cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'<tool_output>[\s\S]*', '', cleaned)
    cleaned = _truncate_to_first_tool_call(cleaned)
    cleaned = cleaned.strip()
    cleaned = _abstract_answer(cleaned)
    return cleaned


def truncate_action(action: str, max_len: int = 4000) -> str:
    """Truncate action text, keeping tool call portions."""
    if len(action) <= max_len:
        return action

    # Keep tool call blocks
    tool_calls = list(re.finditer(
        r"<call_tool.*?</call_tool>", action, flags=re.DOTALL
    ))
    if tool_calls:
        last_call = tool_calls[-1]
        call_text = last_call.group()
        remaining = max_len - len(call_text) - 50
        if remaining > 0:
            return action[:remaining] + "\n...\n" + call_text
        return call_text[:max_len]

    return action[:max_len]


def parse_ranking(response_text: str, n: int) -> Optional[Dict]:
    """Parse judge response into ranks list + atomic_count.

    Returns {"ranking": [rank_A, rank_B, ...], "atomic_count": int|None} or None on failure.

    Ranking supports ties — tied items share the average rank, e.g. [1.5, 1.5, 3, 4].

    Format examples:
      {"ranking": ["A", "B", "C", "D"], "atomic_count": 3}
      {"ranking": [["A", "B"], "C", "D"], "atomic_count": 2}
    """
    # Try to extract JSON object
    parsed = None
    json_start = response_text.find('{')
    json_end = response_text.rfind('}')
    if json_start >= 0 and json_end > json_start:
        try:
            parsed = json.loads(response_text[json_start:json_end + 1])
        except json.JSONDecodeError:
            pass

    if parsed and "ranking" in parsed:
        items = parsed["ranking"]
        atomic_count = parsed.get("atomic_count")
    else:
        # Fallback: find array (simple case only, no nested)
        arr_match = re.search(r'\[([^\[\]]+)\]', response_text)
        if not arr_match:
            return None
        items = [s.strip().strip('"').strip("'") for s in arr_match.group(1).split(",")]
        atomic_count = None

    valid_labels = set(chr(ord("A") + i) for i in range(n))
    ranks = [0.0] * n
    seen_labels = set()
    current_rank = 1

    for item in items:
        if isinstance(item, list):
            group_labels = [str(l).strip().upper() for l in item]
        else:
            group_labels = [str(item).strip().upper()]

        for label in group_labels:
            if label not in valid_labels or label in seen_labels:
                return None
            seen_labels.add(label)

        group_size = len(group_labels)
        avg_rank = current_rank + (group_size - 1) / 2.0
        for label in group_labels:
            action_idx = ord(label) - ord("A")
            ranks[action_idx] = avg_rank
        current_rank += group_size

    if seen_labels != valid_labels:
        return None

    # Validate atomic_count if present
    if atomic_count is not None:
        try:
            atomic_count = int(atomic_count)
        except (TypeError, ValueError):
            atomic_count = None

    return {"ranking": ranks, "atomic_count": atomic_count}


# Cloudsway defaults (same as collect_rl_data.yaml gemini judge config)
DEFAULT_CLOUDSWAY_URL = os.environ.get("CLOUDSWAY_GEMINI_PRO_URL", "")
DEFAULT_CLOUDSWAY_API_KEY = os.environ.get("CLOUDSWAY_API_KEY", "")


async def rank_actions_by_rubric(
    rubric: str,
    question: str,
    actions: Dict[str, str],
    trajectory: str = "",
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    timeout: int = 300,
    max_action_chars: int = 4000,
) -> Optional[Dict]:
    """Rank actions by rubric using Gemini 2.5 Pro via Cloudsway.

    Args:
        rubric: Generated rubric text (e.g. "1. [0.4] condition...")
        question: Research question
        actions: {"A": action_text, "B": ..., "C": ..., "D": ...}
        trajectory: Agent execution trajectory (think tags should be pre-stripped)
        api_url: Cloudsway endpoint (defaults to Gemini 2.5 Pro endpoint)
        api_key: Cloudsway API key
        temperature: Sampling temperature
        timeout: Request timeout in seconds
        max_action_chars: Max chars per action before truncation

    Returns:
        [rank_A, rank_B, rank_C, rank_D] (float, 1-based, ties share avg rank), or None on failure.
        Also includes atomic_count from judge's atomicity evaluation.
    """
    api_url = api_url or os.environ.get("CLOUDSWAY_GEMINI_URL", DEFAULT_CLOUDSWAY_URL)
    api_key = api_key or os.environ.get("CLOUDSWAY_API_KEY", DEFAULT_CLOUDSWAY_API_KEY)

    # Build action text
    labels = sorted(actions.keys())
    n = len(labels)
    action_lines = []
    for label in labels:
        cleaned = clean_for_judge(actions[label])
        cleaned = truncate_action(cleaned, max_action_chars)
        action_lines.append(f"[{label}] {cleaned}")

    actions_text = "\n\n".join(action_lines)
    # Build label examples for the prompt
    action_labels_str = ", ".join(f'"{l}"' for l in labels)
    if n >= 2:
        example_tie = f'"{labels[0]}", "{labels[1]}"'
        remaining_labels = ", ".join(f'"{l}"' for l in labels[2:])
    else:
        example_tie = action_labels_str
        remaining_labels = ""

    prompt = RUBRIC_RANKING_PROMPT.format(
        n_actions=n,
        rubric=rubric,
        question=question,
        trajectory=trajectory if trajectory else "(no prior trajectory)",
        actions_text=actions_text,
        action_labels=action_labels_str,
        example_tie=example_tie,
        remaining_labels=remaining_labels,
    )

    messages = [{"role": "user", "content": prompt}]

    # Cloudsway format: same as judges.py _call_cloudsway()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {"messages": messages}
    if temperature is not None:
        data["temperature"] = temperature

    # Retry with exponential backoff (3 attempts)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    api_url, headers=headers, json=data,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status == 429 or resp.status >= 500:
                        # Rate limit or server error — retry
                        text = await resp.text()
                        wait = 2 ** attempt * 5  # 5s, 10s, 20s
                        print(f"[RubricJudge] {resp.status} error, retry {attempt+1}/{max_retries} in {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    if resp.status != 200:
                        text = await resp.text()
                        print(f"[RubricJudge] Cloudsway error {resp.status}: {text[:200]}")
                        return None
                    result = await resp.json()
                    response_text = result["choices"][0]["message"]["content"]
                    ranking = parse_ranking(response_text, n)
                    if ranking is None and attempt < max_retries - 1:
                        # Parse failed — retry with fresh call
                        print(f"[RubricJudge] Parse failed, retry {attempt+1}/{max_retries}")
                        await asyncio.sleep(2)
                        continue
                    return ranking
        except asyncio.TimeoutError:
            wait = 2 ** attempt * 5
            print(f"[RubricJudge] Timeout, retry {attempt+1}/{max_retries} in {wait}s")
            if attempt < max_retries - 1:
                await asyncio.sleep(wait)
                continue
            return None
        except Exception as e:
            print(f"[RubricJudge] Error: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            return None

    return None


PAIRWISE_COMPARISON_PROMPT = """你是深度研究任务的评审专家。请根据以下评判准则(Rubric)，判断两个候选 Action 哪个更优。

评判准则 (Rubric):
{rubric}

研究问题:
{question}

Agent 已执行的轨迹（当前状态上下文）:
{trajectory}

候选 Actions（已去除思考过程，仅保留实际工具调用和行为）:
[A] {action_a}

[B] {action_b}

判断要求：
- 严格按照 Rubric 中的每条准则逐一判断每个 Action 的符合程度
- 结合 Agent 已执行的轨迹来理解当前研究状态
- 仅根据 Action 的实际工具调用（tool name、query、URL）和行为来评判
- 按 Rubric 加权得分判断哪个 Action 更优

输出格式（严格 JSON，不要输出其他内容）:
{{"winner": "A"}} 或 {{"winner": "B"}} 或 {{"winner": "tie"}}"""


async def compare_actions_pairwise(
    rubric: str,
    question: str,
    action_a: str,
    action_b: str,
    trajectory: str = "",
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    timeout: int = 300,
    max_action_chars: int = 4000,
) -> Optional[str]:
    """Compare two actions pairwise using Gemini judge.

    Returns "A", "B", or "tie". Returns None on failure.
    """
    api_url = api_url or os.environ.get("CLOUDSWAY_GEMINI_URL", DEFAULT_CLOUDSWAY_URL)
    api_key = api_key or os.environ.get("CLOUDSWAY_API_KEY", DEFAULT_CLOUDSWAY_API_KEY)

    cleaned_a = clean_for_judge(action_a)
    cleaned_a = truncate_action(cleaned_a, max_action_chars)
    cleaned_b = clean_for_judge(action_b)
    cleaned_b = truncate_action(cleaned_b, max_action_chars)

    prompt = PAIRWISE_COMPARISON_PROMPT.format(
        rubric=rubric,
        question=question,
        trajectory=trajectory if trajectory else "(no prior trajectory)",
        action_a=cleaned_a,
        action_b=cleaned_b,
    )

    messages = [{"role": "user", "content": prompt}]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {"messages": messages}
    if temperature is not None:
        data["temperature"] = temperature

    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    api_url, headers=headers, json=data,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status == 429 or resp.status >= 500:
                        text = await resp.text()
                        wait = 2 ** attempt * 5
                        print(f"[PairwiseJudge] {resp.status} error, retry {attempt+1}/{max_retries} in {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    if resp.status != 200:
                        text = await resp.text()
                        print(f"[PairwiseJudge] Cloudsway error {resp.status}: {text[:200]}")
                        return None
                    result = await resp.json()
                    response_text = result["choices"][0]["message"]["content"]
                    # Parse winner
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}')
                    if json_start >= 0 and json_end > json_start:
                        try:
                            parsed = json.loads(response_text[json_start:json_end + 1])
                            winner = parsed.get("winner", "").strip().upper()
                            if winner in ("A", "B", "TIE"):
                                return winner
                        except json.JSONDecodeError:
                            pass
                    # Fallback: look for A or B in response
                    if '"A"' in response_text or "'A'" in response_text:
                        return "A"
                    if '"B"' in response_text or "'B'" in response_text:
                        return "B"
                    if attempt < max_retries - 1:
                        print(f"[PairwiseJudge] Parse failed, retry {attempt+1}/{max_retries}")
                        await asyncio.sleep(2)
                        continue
                    return None
        except asyncio.TimeoutError:
            wait = 2 ** attempt * 5
            print(f"[PairwiseJudge] Timeout, retry {attempt+1}/{max_retries} in {wait}s")
            if attempt < max_retries - 1:
                await asyncio.sleep(wait)
                continue
            return None
        except Exception as e:
            print(f"[PairwiseJudge] Error: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            return None

    return None


async def rank_actions_batch(
    rubrics: List[str],
    questions: List[str],
    actions_list: List[Dict[str, str]],
    max_concurrent: int = 16,
    **kwargs,
) -> List[Optional[Dict]]:
    """Rank multiple (rubric, actions) pairs concurrently.

    Args:
        rubrics: List of rubric texts
        questions: List of research questions
        actions_list: List of action dicts
        max_concurrent: Max concurrent API calls
        **kwargs: Passed to rank_actions_by_rubric

    Returns:
        List of ranking results (None for failures)
    """
    sem = asyncio.Semaphore(max_concurrent)

    async def _rank_one(rubric, question, actions):
        async with sem:
            return await rank_actions_by_rubric(
                rubric=rubric, question=question, actions=actions, **kwargs,
            )

    tasks = [
        _rank_one(r, q, a)
        for r, q, a in zip(rubrics, questions, actions_list)
    ]
    return await asyncio.gather(*tasks)
