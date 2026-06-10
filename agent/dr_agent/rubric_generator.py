"""
Rubric Generator for Co-ReAct: generates evaluation rubrics and verifies agent actions.

Supports two backends:
- openai: vLLM / OpenAI-compatible
- cloudsway: Cloudsway proxy (GPT-5, Gemini, etc.)
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import aiohttp

AutoTokenizer = None

# Rubric prompt templates (14B, 2-message format with embedded few-shot)
import os as _os
import sys as _sys
_rubric_rl_dir = _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "rubric_rl")
if _os.path.isdir(_rubric_rl_dir):
    _sys.path.insert(0, _os.path.abspath(_rubric_rl_dir))
from rubric_prompt import (
    RUBRIC_V2_SYSTEM_EN as SM14B_SYSTEM_EN,
    RUBRIC_V2_SYSTEM_ZH as SM14B_SYSTEM_ZH,
    RUBRIC_V2_USER_EN as SM14B_USER_EN,
    RUBRIC_V2_USER_ZH as SM14B_USER_ZH,
)


# ============================================================================
# Verify prompt
# ============================================================================

RUBRIC_VERIFY_PROMPT = """根据以下评判准则，判断 Agent 的行动是否合理。

评判准则:
{rubric}

Agent 行动:
{action}

请逐条判断，计算加权得分（满足条目得该权重，不满足得 0；负权重条目被触发则扣分，未触发得 0）。
总分 >= 0.6 即通过。

输出格式（严格 JSON，不要输出其他内容）:
{{"pass": true, "weighted_score": 0.7, "feedback": "简短反馈"}}"""


# ============================================================================
# Phase hints
# ============================================================================

PHASE_HINT_SM_EARLY_ZH = "当前处于研究早期（工具调用 {n} 次），应优先拓宽搜索范围、获取关键信息。此阶段不应建议生成 <answer>。"
PHASE_HINT_SM_EARLY_EN = "Currently in early research phase ({n} tool calls), prioritize broadening search and gathering key information. Do not suggest <answer> at this stage."
PHASE_HINT_SM_MID_ZH = "当前已调用工具 {n} 次，研究正在推进。请关注搜索策略的质量和信息覆盖度。如果核心信息已基本收集完毕，可以建议生成 <answer>。"
PHASE_HINT_SM_MID_EN = "Currently {n} tool calls made, research is progressing. Focus on search strategy quality and information coverage. If core information has been largely gathered, consider suggesting <answer> generation."
PHASE_HINT_LATE_ZH = "当前已调用工具 {n} 次（上限 {max}），信息可能已充分。如果信息足够回答问题，应建议 agent 生成 <answer>；如果仍有关键缺口，应精准补充。"
PHASE_HINT_LATE_EN = "Already {n} tool calls (limit {max}). If enough information has been gathered, suggest the agent generate <answer>; if critical gaps remain, target them precisely."


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks and unclosed <think> from text."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "<think>" in text:
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
    return text


def clean_rubric_output(text: str) -> str:
    """Clean markdown formatting from rubric model output.

    Some GRPO-trained models add **bold**, ### headers, or preamble text.
    This strips those to recover the bare rubric lines.
    """
    match = re.search(r"(?m)^\s*1\s*[\.\)]\s*", text)
    if match:
        text = text[match.start():]
    text = text.replace("**", "")
    text = re.sub(r"(?m)^#+\s+.*$\n?", "", text)
    text = re.sub(r"(?m)^\s+[-•]\s+.*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _filter_hallucinated_urls(rubric: str) -> str:
    """Remove rubric lines containing hallucinated URLs (e.g. xxx.org/paper.pdf)."""
    lines = rubric.split("\n")
    filtered = []
    for line in lines:
        if "xxx.org" in line or "example.com/paper" in line:
            continue
        filtered.append(line)
    result = "\n".join(filtered).strip()
    counter = 0
    renumbered_lines = []
    for line in result.split("\n"):
        m = re.match(r"^\s*\d+\s*[\.\)]\s*", line)
        if m:
            counter += 1
            line = re.sub(r"^\s*\d+\s*[\.\)]", f"{counter}.", line, count=1)
        renumbered_lines.append(line)
    return "\n".join(renumbered_lines).strip()


def smart_truncate_action(action: str, max_len: int = 8000) -> str:
    """Truncate action intelligently: keep tool calls, strip think blocks."""
    if len(action) <= max_len:
        return action

    stripped = strip_think_tags(action)
    if len(stripped) <= max_len:
        return stripped

    tool_calls = list(re.finditer(
        r"<call_tool.*?</call_tool>", stripped, flags=re.DOTALL
    ))
    if tool_calls:
        last_call = tool_calls[-1]
        call_text = last_call.group()
        remaining = max_len - len(call_text) - 100
        if remaining > 0:
            prefix = stripped[:remaining]
            return prefix + "\n...(truncated)...\n" + call_text
        return call_text[:max_len]

    return stripped[:max_len]


def detect_language(text: str) -> str:
    """Detect if text is primarily Chinese or English."""
    chinese_chars = len(re.findall(r'[一-鿿]', text))
    total_chars = len(text.strip())
    if total_chars == 0:
        return "en"
    return "zh" if chinese_chars / total_chars > 0.15 else "en"


@dataclass
class RubricGenerator:
    """Generate evaluation rubrics and verify actions.

    Supports two API backends:
    - api_type="openai": vLLM / OpenAI-compatible (default)
    - api_type="cloudsway": Cloudsway proxy (for GPT-5, etc.)
    """

    # OpenAI-compatible fields
    base_url: str = "http://localhost:30002/v1"
    model_name: str = "Qwen/Qwen3-14B"
    api_key: str = "dummy-key"
    api_type: str = "openai"  # "openai" or "cloudsway"

    # Cloudsway fields (used when api_type="cloudsway")
    url: Optional[str] = None

    temperature: Optional[float] = 0.7
    max_tokens: int = 1024
    timeout: int = 120
    repetition_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    max_think_tokens: Optional[int] = None

    strip_think: bool = False
    prompt_version: str = "strongmodel_qwen14b"
    skip_few_shot: bool = False
    max_trajectory_tokens: int = 0

    _tokenizer: Optional[object] = field(default=None, repr=False)

    def _get_tokenizer(self):
        if self._tokenizer is None:
            global AutoTokenizer
            if AutoTokenizer is None:
                from transformers import AutoTokenizer as _AT
                AutoTokenizer = _AT
            print(f"[RubricGen] Loading tokenizer from {self.model_name}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            print(f"[RubricGen] Tokenizer loaded (vocab_size={self._tokenizer.vocab_size})")
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        return len(self._get_tokenizer().encode(text, add_special_tokens=False))

    async def _call(self, messages: List[Dict[str, str]]) -> str:
        if self.api_type == "cloudsway":
            return await self._call_cloudsway(messages)
        return await self._call_openai(messages)

    async def _call_with_temp0(self, messages: List[Dict[str, str]]) -> str:
        orig_temp = self.temperature
        self.temperature = 0
        try:
            return await self._call(messages)
        finally:
            self.temperature = orig_temp

    async def _call_openai(self, messages: List[Dict[str, str]]) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            data["temperature"] = self.temperature
        if self.repetition_penalty is not None:
            data["repetition_penalty"] = self.repetition_penalty
        if self.frequency_penalty is not None:
            data["frequency_penalty"] = self.frequency_penalty
        if self.max_think_tokens is not None:
            data["max_think_tokens"] = self.max_think_tokens

        if "dashscope" in self.base_url:
            data["enable_thinking"] = False

        api_url = f"{self.base_url}/chat/completions"

        for attempt in range(2):
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    api_url, headers=headers, json=data,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise Exception(f"RubricGenerator API error {resp.status}: {text}")
                    result = await resp.json()
                    if getattr(self, "usage_sink", None) and isinstance(result, dict):
                        u = result.get("usage") or {}
                        try:
                            self.usage_sink["prompt_tokens"] += int(u.get("prompt_tokens", 0) or 0)
                            self.usage_sink["completion_tokens"] += int(u.get("completion_tokens", 0) or 0)
                            self.usage_sink["total_tokens"] += int(u.get("total_tokens", 0) or 0)
                            self.usage_sink["calls"] += 1
                        except Exception:
                            pass
                    content = result["choices"][0]["message"]["content"]
                    finish_reason = result["choices"][0].get("finish_reason", "unknown")

            stripped = strip_think_tags(content)

            is_degenerate = (
                "<think>" in content
                and "</think>" not in content
                and finish_reason == "length"
            )

            if is_degenerate and attempt == 0:
                print(f"[RubricGen] Degenerate think loop detected (len={len(content)}), retrying with temp=0...")
                data["temperature"] = 0
                continue

            if is_degenerate:
                print(f"[RubricGen] Degenerate on retry too, returning empty")
                return ""

            return stripped

        return ""

    async def _call_cloudsway(self, messages: List[Dict[str, str]]) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"messages": messages}
        if self.temperature is not None:
            data["temperature"] = self.temperature

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.url, headers=headers, json=data,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise Exception(f"RubricGenerator Cloudsway error {resp.status}: {text}")
                result = await resp.json()
                if getattr(self, "usage_sink", None) and isinstance(result, dict):
                    u = result.get("usage") or {}
                    try:
                        self.usage_sink["prompt_tokens"] += int(u.get("prompt_tokens", 0) or 0)
                        self.usage_sink["completion_tokens"] += int(u.get("completion_tokens", 0) or 0)
                        self.usage_sink["total_tokens"] += int(u.get("total_tokens", 0) or 0)
                        self.usage_sink["calls"] += 1
                    except Exception:
                        pass
                return result["choices"][0]["message"]["content"]

    async def generate(
        self, question: str, trajectory: str,
        tool_call_count: int = 0, max_tool_calls: int = 10,
        reflection: str = None,
    ) -> Optional[str]:
        """Generate a rubric for the next action.

        Returns None if trajectory is empty (first step — no rubric needed).
        """
        if not trajectory or not trajectory.strip():
            return None

        if self.strip_think:
            trajectory = strip_think_tags(trajectory)

        if self.max_trajectory_tokens > 0 and self.count_tokens(trajectory) > self.max_trajectory_tokens:
            tokenizer = self._get_tokenizer()
            token_ids = tokenizer.encode(trajectory, add_special_tokens=False)
            tail_text = tokenizer.decode(token_ids[-self.max_trajectory_tokens:])
            boundary = tail_text.find("</tool_output>")
            if boundary != -1:
                tail_text = tail_text[boundary + len("</tool_output>"):].lstrip()
            trajectory = "...(earlier steps truncated)...\n" + tail_text

        lang = detect_language(question)

        if tool_call_count >= max_tool_calls - 3:
            phase_hint = (
                PHASE_HINT_LATE_ZH.format(n=tool_call_count, max=max_tool_calls) if lang == "zh"
                else PHASE_HINT_LATE_EN.format(n=tool_call_count, max=max_tool_calls)
            )
        elif tool_call_count >= 3:
            phase_hint = (
                PHASE_HINT_SM_MID_ZH.format(n=tool_call_count) if lang == "zh"
                else PHASE_HINT_SM_MID_EN.format(n=tool_call_count)
            )
        else:
            phase_hint = (
                PHASE_HINT_SM_EARLY_ZH.format(n=tool_call_count) if lang == "zh"
                else PHASE_HINT_SM_EARLY_EN.format(n=tool_call_count)
            )

        if lang == "zh":
            system_prompt = SM14B_SYSTEM_ZH
            user_prompt = SM14B_USER_ZH.format(
                question=question, trajectory=trajectory, phase_hint=phase_hint,
            )
        else:
            system_prompt = SM14B_SYSTEM_EN
            user_prompt = SM14B_USER_EN.format(
                question=question, trajectory=trajectory, phase_hint=phase_hint,
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            rubric = await self._call(messages)
            rubric = clean_rubric_output(rubric)
            is_valid = bool(re.search(r"\d+\.\s*\[[-\d.]+\]", rubric))
            if not is_valid:
                print(f"[RubricGen] Format invalid, retrying with temp=0: {rubric[:100]}")
                rubric2 = await self._call_with_temp0(list(messages))
                rubric2 = clean_rubric_output(rubric2)
                if bool(re.search(r"\d+\.\s*\[[-\d.]+\]", rubric2)):
                    rubric = rubric2
            return rubric
        except Exception as e:
            print(f"[RubricGen] Error generating rubric: {e}")
            return ""

    async def verify(self, rubric: str, action: str, question: str = None, verify_version: str = "v1", trajectory: str = None) -> Tuple[bool, str]:
        """Verify whether an action satisfies the rubric."""
        action_truncated = smart_truncate_action(action, max_len=8000)
        prompt = RUBRIC_VERIFY_PROMPT.format(rubric=rubric, action=action_truncated)
        messages = [{"role": "user", "content": prompt}]

        try:
            response = await self._call(messages)
            passed, feedback = self._parse_verify_response(response)
            return passed, feedback
        except Exception as e:
            print(f"[RubricGen] Error verifying action: {e}")
            return True, "verification skipped due to error"

    @staticmethod
    def _parse_verify_response(response: str) -> Tuple[bool, str]:
        brace_depth = 0
        start = -1
        for i, c in enumerate(response):
            if c == '{':
                if brace_depth == 0:
                    start = i
                brace_depth += 1
            elif c == '}':
                brace_depth -= 1
                if brace_depth == 0 and start >= 0:
                    try:
                        parsed = json.loads(response[start:i+1])
                        passed = bool(parsed.get("pass", True))
                        feedback = str(parsed.get("feedback", ""))
                        return passed, feedback
                    except json.JSONDecodeError:
                        start = -1
                        continue

        lower = response.lower()
        if '"pass": false' in lower or '"pass":false' in lower:
            return False, response[:200]
        return True, ""

    @staticmethod
    def format_rubric_suffix(rubric: str, tool_call_count: int = 0, min_tool_calls: int = 3) -> str:
        suffix = (
            "\n\n[Rubric for next action]\n"
            f"{rubric}\n"
            "[End Rubric — follow these criteria for your next action]\n"
        )
        return suffix
