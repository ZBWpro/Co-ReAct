"""
Council of Judges for ranking candidate actions in Co-ReAct RL data collection.

Supports two API backends:
- OpenAI-compatible (Dashscope): GLM-5, Qwen3-235B
- Cloudsway proxy: GPT-5, Gemini
"""

import asyncio
import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp

RANKING_PROMPT_TEMPLATE = """你是深度研究任务的评审专家。下面是一个研究 Agent 正在执行的完整轨迹，包含它每一步的工具调用（<call_tool>）和工具返回结果（<tool_output>）。

Agent 现在需要决定下一步行动，有 {n_actions} 个候选 Action。请根据完整的研究上下文，按质量从高到低排序。质量相同的 Action 可以并列。

候选 Action 有两种类型:
- 工具调用（<call_tool>）: 继续搜索/浏览获取更多信息
- write_report: 停止信息收集，直接开始撰写最终研究报告

注意: write_report 意味着 Agent 认为已收集到足够的信息来完成这个深度研究任务。只有当轨迹中已积累了充分、多角度的信息，足以撰写一份全面的研究报告时，才应该优先选择 write_report。如果当前信息仍有明显缺口或研究问题的关键方面尚未覆盖，应优先选择继续搜索。

评估维度：
1. 信息增量：该 Action 是否能带来新的、尚未获取的关键信息（避免重复搜索）
2. 工具选择：选择的工具是否适合当前需求（搜索/浏览/撰写报告）
3. 查询质量：查询词是否精准、是否能找到有价值的信息
4. 策略合理性：结合已有信息，该步骤是否推进了整体研究目标
5. 信息充分性（仅当候选中有 write_report 时）：轨迹中已收集的信息是否足以完成一份高质量的深度研究报告

研究问题:
{question}

Agent 执行轨迹:
{trajectory}

现在 Agent 需要选择下一步行动，以下是 {n_actions} 个候选 Action:
{actions_text}

请严格按以下 JSON 格式输出排序（从最优到最差），质量相同的用数组并列，不要输出其他内容:
示例: {{"ranking": [{example_ranking}]}}"""


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


@dataclass
class JudgeClient:
    """Unified client for calling judge models via different API backends."""

    name: str
    api_type: str  # "openai" or "cloudsway"
    api_key: str
    temperature: Optional[float] = 0.7
    # OpenAI-compatible fields
    base_url: Optional[str] = None
    model_name: Optional[str] = None
    # Cloudsway fields
    url: Optional[str] = None
    timeout: int = 120

    async def call(self, messages: List[Dict[str, str]]) -> str:
        """Call the judge model and return the response content."""
        if self.api_type == "openai":
            return await self._call_openai(messages)
        elif self.api_type == "cloudsway":
            return await self._call_cloudsway(messages)
        else:
            raise ValueError(f"Unknown api_type: {self.api_type}")

    async def _call_openai(self, messages: List[Dict[str, str]]) -> str:
        """Call OpenAI-compatible API (Dashscope for GLM-5, Qwen3-235B)."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 256,
        }
        if self.temperature is not None:
            data["temperature"] = self.temperature

        url = f"{self.base_url}/chat/completions"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise Exception(f"Judge {self.name} API error {resp.status}: {text}")
                result = await resp.json()
                return result["choices"][0]["message"]["content"]

    async def _call_cloudsway(self, messages: List[Dict[str, str]]) -> str:
        """Call Cloudsway proxy API (GPT-5, Gemini)."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"messages": messages}
        if self.temperature is not None:
            data["temperature"] = self.temperature

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.url, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise Exception(f"Judge {self.name} API error {resp.status}: {text}")
                result = await resp.json()
                return result["choices"][0]["message"]["content"]

    async def rank(
        self, question: str, trajectory: str, actions: List[str]
    ) -> Optional[List[int]]:
        """Rank actions and return list of ranks (1-based) for each action index.

        Shuffles action order and uses non-sequential labels (X, Y, Z, W)
        to mitigate position bias in LLM judges.
        """
        n = len(actions)
        LABELS = ["X", "Y", "Z", "W", "V", "U"][:n]

        # Strip <think>...</think> from actions — judges evaluate tool calls only
        cleaned_actions = [strip_think_tags(a) for a in actions]
        # Strip <think>...</think> from trajectory as well
        cleaned_trajectory = strip_think_tags(trajectory)

        # Shuffle action order to eliminate position bias
        indices = list(range(n))
        random.shuffle(indices)
        shuffled_actions = [cleaned_actions[i] for i in indices]
        # Map: shuffled position -> original index
        # Map: label -> original index
        label_to_orig = {}
        action_lines = []
        for pos, (label, orig_idx) in enumerate(zip(LABELS, indices)):
            label_to_orig[label] = orig_idx
            action_lines.append(f"[{label}] {shuffled_actions[pos]}")

        actions_text = "\n".join(action_lines)
        # Build example ranking: e.g. '["X", "Y"], "Z", "W"' for 4 actions
        # Show one tie example to hint the format
        if n >= 4:
            example_ranking = f'["{LABELS[0]}", "{LABELS[1]}"], "{LABELS[2]}", "{LABELS[3]}"'
        elif n == 3:
            example_ranking = f'["{LABELS[0]}", "{LABELS[1]}"], "{LABELS[2]}"'
        else:
            example_ranking = ", ".join(f'"{l}"' for l in LABELS)

        prompt = RANKING_PROMPT_TEMPLATE.format(
            n_actions=n,
            question=question,
            trajectory=cleaned_trajectory,
            actions_text=actions_text,
            example_ranking=example_ranking,
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            response_text = await self.call(messages)
            return self._parse_ranking(response_text, n, LABELS, label_to_orig)
        except asyncio.TimeoutError:
            print(f"[Judge {self.name}] Error: Timeout after {self.timeout}s")
            return None
        except Exception as e:
            print(f"[Judge {self.name}] Error: {type(e).__name__}: {e}")
            return None

    @staticmethod
    def _parse_ranking(
        response_text: str, n: int,
        labels: List[str] = None,
        label_to_orig: Dict[str, int] = None,
    ) -> Optional[List[float]]:
        """Parse judge response into ranks list (supports ties).

        Returns [rank_for_action_0, rank_for_action_1, ...] where tied items
        share the average rank, e.g. [1.5, 1.5, 3, 4].

        Format examples:
          {"ranking": ["X", "Y", "Z", "W"]}           → no ties
          {"ranking": [["X", "Y"], "Z", "W"]}          → X=Y tied at rank 1
          {"ranking": ["X", ["Y", "Z"], "W"]}          → Y=Z tied at rank 2
        """
        if labels is None:
            labels = [chr(ord("A") + i) for i in range(n)]
        if label_to_orig is None:
            label_to_orig = {l: i for i, l in enumerate(labels)}

        valid_labels = set(labels)

        # Try to find JSON in response
        json_match = re.search(r'\{[^{}]*"ranking"\s*:\s*\[.*?\]\s*\}', response_text, re.DOTALL)
        if not json_match:
            # Fallback: find any array
            arr_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if not arr_match:
                print(f"[Judge] Failed to parse ranking from: {response_text[:200]}")
                return None
            try:
                items = json.loads(arr_match.group())
            except json.JSONDecodeError:
                print(f"[Judge] JSON parse error: {arr_match.group()[:200]}")
                return None
        else:
            try:
                parsed = json.loads(json_match.group())
                items = parsed.get("ranking", [])
            except json.JSONDecodeError:
                print(f"[Judge] JSON parse error: {json_match.group()[:200]}")
                return None

        # Parse items: each item is either a string label or a list of tied labels
        ranks = [0.0] * n
        seen_labels = set()
        current_rank = 1

        for item in items:
            if isinstance(item, list):
                # Tied group
                group_labels = [str(l).strip().upper() for l in item]
            else:
                group_labels = [str(item).strip().upper()]

            # Validate all labels
            for label in group_labels:
                if label not in valid_labels:
                    print(f"[Judge] Invalid label '{label}' in ranking (expected {labels})")
                    return None
                if label in seen_labels:
                    print(f"[Judge] Duplicate label '{label}' in ranking")
                    return None
                seen_labels.add(label)

            # Assign average rank for tied group
            group_size = len(group_labels)
            avg_rank = current_rank + (group_size - 1) / 2.0
            for label in group_labels:
                orig_idx = label_to_orig[label]
                ranks[orig_idx] = avg_rank
            current_rank += group_size

        # Verify all labels are assigned
        if seen_labels != valid_labels:
            missing = valid_labels - seen_labels
            print(f"[Judge] Missing labels in ranking: {missing}")
            return None

        return ranks


@dataclass
class JudgeRanker:
    """Orchestrates multiple judges for action ranking with Borda count aggregation."""

    judges: List[JudgeClient] = field(default_factory=list)

    async def rank_actions(
        self, question: str, trajectory: str, actions: List[str]
    ) -> Dict[str, Any]:
        """Call all judges in parallel, return rankings and Borda scores."""
        import time as _time

        async def _timed_rank(judge):
            t0 = _time.time()
            result = await judge.rank(question, trajectory, actions)
            elapsed = _time.time() - t0
            return judge.name, result, elapsed

        tasks = [_timed_rank(judge) for judge in self.judges]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        rankings = {}
        timings = []
        for result in results:
            if isinstance(result, Exception):
                print(f"[JudgeRanker] judge failed: {result}")
                continue
            name, ranking, elapsed = result
            timings.append(f"{name}={elapsed:.1f}s")
            if ranking is not None:
                rankings[name] = ranking
        if timings:
            print(f"[JudgeRanker] Per-judge time: {', '.join(timings)}")

        borda = self.borda_count(rankings, len(actions))
        best_idx = borda.index(min(borda)) if borda else 0

        return {
            "expert_rankings": rankings,
            "borda_scores": borda,
            "best_action_idx": best_idx,
            "num_valid_judges": len(rankings),
        }

    @staticmethod
    def borda_count(
        rankings: Dict[str, List[float]], n_actions: int
    ) -> List[float]:
        """Aggregate rankings via Borda count. Lower score = better.
        Supports float ranks from tied rankings (e.g., 1.5, 1.5, 3, 4)."""
        if not rankings:
            return list(range(1, n_actions + 1))

        scores = [0.0] * n_actions
        for judge_ranking in rankings.values():
            for i, rank in enumerate(judge_ranking):
                scores[i] += rank
        # Round to avoid floating point noise
        return [round(s, 2) for s in scores]

    @classmethod
    def from_config(cls, judges_config: Dict[str, Dict]) -> "JudgeRanker":
        """Create JudgeRanker from YAML config dict.

        Supports both direct values and env var references:
        - url/api_key: used directly
        - url_env/api_key_env: resolved from environment variables
        """
        judges = []
        for name, cfg in judges_config.items():
            # Resolve URL: prefer url_env (env var name) over url (direct value)
            url = cfg.get("url")
            if "url_env" in cfg:
                url = os.environ.get(cfg["url_env"], url or "")
            # Resolve API key: prefer api_key_env over api_key
            api_key = cfg.get("api_key", "")
            if "api_key_env" in cfg:
                api_key = os.environ.get(cfg["api_key_env"], api_key or "")

            judge = JudgeClient(
                name=name,
                api_type=cfg["api_type"],
                api_key=api_key,
                temperature=cfg.get("temperature"),
                base_url=cfg.get("base_url"),
                model_name=cfg.get("model"),
                url=url,
                timeout=cfg.get("timeout", 120),
            )
            judges.append(judge)
        return cls(judges=judges)
