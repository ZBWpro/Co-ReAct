"""
Convert dr_tulu_trajectory_preferences.jsonl → verl-compatible Parquet.

Each record becomes a rubric-generation prompt with reward metadata
containing the 4 candidate actions and their expert Borda rankings.

Usage:
    python prepare_data.py
    python prepare_data.py --max-trajectory-tokens 6000 --tokenizer-path /path/to/model --eval-ratio 0.1
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# ============================================================================
# Prompts — copied from agent/dr_agent/rubric_generator.py to avoid import deps
# ============================================================================

RUBRIC_SYSTEM_PROMPT_ZH = """你是深度研究 Agent 的策略规划专家。你只输出 rubric 条目，不输出任何分析、解释或前言。"""
RUBRIC_SYSTEM_PROMPT_EN = """You are a strategy planning expert for a deep research agent. Output ONLY rubric criteria lines, no analysis, explanation, or preamble."""

RUBRIC_USER_PROMPT_ZH = """根据研究问题和 Agent 已执行的轨迹，为下一步 action 生成 3-5 条评判准则。

研究问题: {question}

Agent 执行轨迹:
{trajectory}

Agent 可用工具:
- google_search(query): 网页搜索，返回网页标题和摘要片段。适合获取概览信息、新闻、非学术来源、一般网页内容。可选参数: gl(地区), hl(语言)。
- snippet_search(query): 学术论文片段搜索（Semantic Scholar），返回论文中的相关片段和引用信息。适合查找研究论文、实验数据、具体数字、学术引用。可选参数: limit(数量), year(年份范围如"2020-2025"), fieldsOfStudy(学科领域如"Chemistry, Medicine")。
- browse_webpage(url): 打开指定 URL 并提取页面完整内容。适合阅读搜索结果 URL 的详细内容。
- <answer>: 结束研究并生成最终报告。当收集到足够信息时使用——不要过度搜索。

{phase_hint}

规则:
- 每条只说一件事，对照 agent 输出的工具名和参数就能判断 yes/no
- 不要写抽象维度（如"信息全面性""来源权威性"），要写具体的、可验证的条件
- 如果推荐 browse_webpage，必须引用前面搜索结果中实际出现过的 URL，绝不能编造虚假 URL
- 可以用负权重表示应避免的错误，正向权重之和为 1.0
- 用与研究问题相同的语言写 criterion

严格按照上面示例的格式输出 3-5 条条目，不要输出其他任何内容。"""

RUBRIC_USER_PROMPT_EN = """Based on the research question and the agent's execution trajectory, generate 3-5 evaluation criteria for the next action.

Research question: {question}

Agent execution trajectory:
{trajectory}

Available tools:
- google_search(query): Web search, returns page titles and snippets. Best for overviews, news, non-academic sources, general web content. Optional params: gl (region), hl (language).
- snippet_search(query): Academic paper snippet search (Semantic Scholar), returns relevant passages from papers with citation info. Best for finding research papers, experimental data, specific numbers, academic citations. Optional params: limit (count), year (range e.g. "2020-2025"), fieldsOfStudy (e.g. "Chemistry, Medicine").
- browse_webpage(url): Opens a specific URL and extracts the full page content. Best for reading detailed content from a search result URL.
- <answer>: Conclude research and generate the final report. Should be used when sufficient information has been gathered — do not over-search.

{phase_hint}

Rules:
- Each criterion states ONE verifiable condition about the tool name, query terms, or URL
- Do NOT write abstract dimensions (e.g. "source quality", "information diversity") — write concrete, checkable conditions
- If recommending browse_webpage, you MUST reference a specific URL that appeared in previous search results — NEVER fabricate URLs
- Use negative weights for errors to avoid; positive weights must sum to 1.0
- Write criteria in the same language as the research question

Output 3-5 criteria lines strictly following the format shown in the examples above. Do not output anything else."""

RUBRIC_FEW_SHOT_ZH = """示例 A（研究初期，推荐 google_search）:
1. [0.4] 应使用 google_search 搜索后量子密码学的最新进展（尚无相关搜索结果，需要广泛概览）
2. [0.3] 查询词应包含"NIST post-quantum standardization"等具体术语，避免过于宽泛
3. [0.2] 查询应使用英文（该领域英文资料远多于中文）
4. [0.1] 应设置 gl="us" 以获取更多英文搜索结果

示例 B（已有搜索结果，推荐 snippet_search）:
1. [0.4] 应使用 snippet_search 搜索 CRYSTALS-Kyber 的性能基准实验数据（google_search 已返回综述页，需要具体实验数据）
2. [0.3] 查询词应针对具体指标（如密钥生成时间、密文大小），而非笼统的"post-quantum cryptography"
3. [0.2] 应设置 year="2022-2025" 以聚焦近期实现成果
4. [0.1] 查询应使用英文以获取更多学术论文

示例 C（搜索摘要不足，推荐 browse_webpage）:
1. [0.4] 应使用 browse_webpage 打开搜索结果中 NIST 官方页面的 URL（搜索摘要中关于标准化时间线的信息不足，需要阅读全文）
2. [0.3] 该 URL 在之前的搜索结果中出现过且尚未被访问
3. [0.2] 该页面可能包含搜索摘要中未完整展示的对比表格
4. [0.1] 优先选择官方来源（如 .gov 域名）

示例 D（信息充足，推荐 <answer>）:
1. [0.4] 应生成 <answer>——已从多个来源收集到性能基准、安全证明和标准化进展
2. [0.3] 学术论文和官方文档中的关键数据点已齐备，可以回答研究问题
3. [0.2] 继续搜索可能收益递减，应避免过度搜索
4. [0.1] 应在回答中引用已收集的来源"""

RUBRIC_FEW_SHOT_EN = """Example A (early research, recommend google_search):
1. [0.4] Should use google_search to find recent developments on post-quantum cryptography (no prior search results, need broad overview)
2. [0.3] Query should include specific terms like "NIST post-quantum standardization", not overly broad keywords
3. [0.2] Query should be in English for broader source coverage in this field
4. [0.1] Should set gl="us" to get more English-language search results

Example B (has search results, recommend snippet_search):
1. [0.4] Should use snippet_search to find experimental data on CRYSTALS-Kyber performance benchmarks (google_search already returned overview pages, need specific data)
2. [0.3] Query should target specific metrics (e.g. key generation time, ciphertext size), not generic "post-quantum cryptography"
3. [0.2] Should set year="2022-2025" to focus on recent implementations
4. [0.1] Query should be in English to access more academic papers

Example C (search snippets insufficient, recommend browse_webpage):
1. [0.4] Should use browse_webpage to open the NIST official page URL from search results (search snippets lack detailed standardization timeline, need full content)
2. [0.3] The URL appeared in previous search results and has not been visited yet
3. [0.2] The page likely contains comparison tables not fully shown in search snippets
4. [0.1] Prefer official sources (e.g. .gov domains)

Example D (sufficient information, recommend <answer>):
1. [0.4] Should generate <answer> — performance benchmarks, security proofs, and standardization status have all been collected
2. [0.3] Key data points from both academic papers and official documents are available to answer the question
3. [0.2] Continuing to search would likely yield diminishing returns, avoid over-searching
4. [0.1] Should cite the collected sources in the answer"""

# Phase hints (V6: original V1 phase hints)
PHASE_HINT_EARLY_ZH = "当前处于研究早期（工具调用 {n} 次），应优先拓宽搜索范围、获取关键信息。rubric 应包含一条负权重条目禁止此时生成 <answer>。"
PHASE_HINT_EARLY_EN = "Currently in early research phase ({n} tool calls), prioritize broadening search and gathering key information. Rubric should include a negative-weight criterion forbidding <answer> generation at this stage."
PHASE_HINT_MID_ZH = "当前已调用工具 {n} 次，研究正在推进。请关注搜索策略的质量和信息覆盖度。"
PHASE_HINT_MID_EN = "Currently {n} tool calls made, research is progressing. Focus on search strategy quality and information coverage."
PHASE_HINT_LATE_ZH = "当前已调用工具 {n} 次（上限 {max}），信息可能已充分。如果信息足够回答问题，应建议 agent 生成 <answer>；如果仍有关键缺口，应精准补充。"
PHASE_HINT_LATE_EN = "Already {n} tool calls (limit {max}). If enough information has been gathered, suggest the agent generate <answer>; if critical gaps remain, target them precisely."


def detect_language(text: str) -> str:
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    total_chars = len(text.strip())
    if total_chars == 0:
        return "en"
    return "zh" if chinese_chars / total_chars > 0.15 else "en"


def truncate_trajectory(trajectory: str, max_tokens: int = 6000, tokenizer=None) -> str:
    """Truncate trajectory by token count, keeping the structure intact.

    Strategy: keep the tail (most recent actions) since they are most relevant
    for generating the next rubric. If truncation happens, add a note.
    """
    if tokenizer is None:
        # Fallback: rough estimate 1 token ≈ 3.5 chars
        max_chars = int(max_tokens * 3.5)
        if len(trajectory) <= max_chars:
            return trajectory
        tail = trajectory[-max_chars:]
    else:
        token_ids = tokenizer.encode(trajectory, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return trajectory
        # Decode only the last max_tokens tokens
        tail = tokenizer.decode(token_ids[-max_tokens:])

    # Try to start at a clean boundary
    for tag in ["<think>", "<call_tool", "<tool_output>"]:
        idx = tail.find(tag)
        if idx != -1 and idx < len(tail) // 10:  # within first 10%
            tail = tail[idx:]
            break

    return "...(earlier trajectory omitted)...\n" + tail


def get_phase_hint(step_index: int, lang: str, max_tool_calls: int = 10) -> str:
    """Get phase-aware hint based on step index."""
    if step_index >= max_tool_calls - 3:  # 7+
        if lang == "zh":
            return PHASE_HINT_LATE_ZH.format(n=step_index, max=max_tool_calls)
        return PHASE_HINT_LATE_EN.format(n=step_index, max=max_tool_calls)
    elif step_index >= 3:
        if lang == "zh":
            return PHASE_HINT_MID_ZH.format(n=step_index)
        return PHASE_HINT_MID_EN.format(n=step_index)
    else:
        if lang == "zh":
            return PHASE_HINT_EARLY_ZH.format(n=step_index)
        return PHASE_HINT_EARLY_EN.format(n=step_index)


def build_prompt_messages(
    question: str,
    trajectory: str,
    step_index: int,
    max_trajectory_tokens: int = 6000,
    prompt_style: str = "v4",
    tokenizer=None,
) -> List[Dict[str, str]]:
    """Build chat messages for rubric generation.

    prompt_style:
      - "v4": 4-message format (system + fake few-shot turn + user task)
      - "llama": 2-message format (system with embedded examples + user task)
    """
    lang = detect_language(question)
    # Strip think tags — rubric should focus on collected info, not reasoning process
    trajectory = strip_think_tags(trajectory)
    trajectory = re.sub(r'</think>', '', trajectory).strip()
    phase_hint = get_phase_hint(step_index, lang)

    # Token-based truncation
    trajectory = truncate_trajectory(trajectory, max_tokens=max_trajectory_tokens, tokenizer=tokenizer)

    if prompt_style in ("llama", "qwen14b"):
        if prompt_style == "qwen14b":
            from rubric_prompt_strongmodel_qwen14b import (
                RUBRIC_V2_SYSTEM_EN, RUBRIC_V2_SYSTEM_ZH,
                RUBRIC_V2_USER_EN, RUBRIC_V2_USER_ZH,
            )
        else:
            from rubric_prompt_v2 import (
                RUBRIC_V2_SYSTEM_EN, RUBRIC_V2_SYSTEM_ZH,
                RUBRIC_V2_USER_EN, RUBRIC_V2_USER_ZH,
            )
        if lang == "zh":
            system_prompt = RUBRIC_V2_SYSTEM_ZH
            user_prompt = RUBRIC_V2_USER_ZH.format(
                question=question, trajectory=trajectory, phase_hint=phase_hint,
            )
        else:
            system_prompt = RUBRIC_V2_SYSTEM_EN
            user_prompt = RUBRIC_V2_USER_EN.format(
                question=question, trajectory=trajectory, phase_hint=phase_hint,
            )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    else:
        # v4 style: 4-message with fake few-shot turn
        if lang == "zh":
            system_prompt = RUBRIC_SYSTEM_PROMPT_ZH
            user_prompt = RUBRIC_USER_PROMPT_ZH.format(
                question=question, trajectory=trajectory, phase_hint=phase_hint,
            )
            few_shot = RUBRIC_FEW_SHOT_ZH
        else:
            system_prompt = RUBRIC_SYSTEM_PROMPT_EN
            user_prompt = RUBRIC_USER_PROMPT_EN.format(
                question=question, trajectory=trajectory, phase_hint=phase_hint,
            )
            few_shot = RUBRIC_FEW_SHOT_EN

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Example output:"},
            {"role": "assistant", "content": few_shot},
            {"role": "user", "content": user_prompt},
        ]


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
    # Strip paired <think>...</think>
    cleaned = strip_think_tags(text)
    # Strip stray </think>
    cleaned = re.sub(r'</think>', '', cleaned)
    # Unclosed <think>: keep content after <call_tool> or <answer> if present
    if '<think>' in cleaned:
        tool_match = re.search(r'(<call_tool[\s\S]*)', cleaned)
        answer_match = re.search(r'(<answer>[\s\S]*)', cleaned)
        if tool_match:
            cleaned = tool_match.group(1)
        elif answer_match:
            cleaned = answer_match.group(1)
        else:
            cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL)
    # Strip hallucinated <tool_output>
    cleaned = re.sub(r'<tool_output>[\s\S]*', '', cleaned)
    # Truncate to first tool call
    cleaned = _truncate_to_first_tool_call(cleaned)
    cleaned = cleaned.strip()
    # Abstract answer → write_report placeholder
    cleaned = _abstract_answer(cleaned)
    return cleaned


def build_reward_meta(record: Dict) -> str:
    """Extract reward metadata: actions with cleaned text and borda scores, plus question and trajectory."""
    actions = {}
    for label, action_data in record["actions"].items():
        actions[label] = {
            "text": clean_for_judge(action_data["text"]),
            "borda_score": action_data["borda_score"],
        }
    # Include trajectory (with think tags stripped) for judge context
    trajectory = strip_think_tags(record.get("trajectory", ""))
    # Also strip stray </think> from trajectory
    trajectory = re.sub(r'</think>', '', trajectory).strip()
    return json.dumps({
        "question": record["question"],
        "actions": actions,
        "trajectory": trajectory,
    }, ensure_ascii=False)


# Per-step sample caps: front-heavy (matches real usage) but enough coverage for all steps
STEP_CAPS = {
    1: 2500,
    2: 2500,
    3: 2500,
    4: 2500,
    5: 1861,
    6: 1662,
    7: 1485,
    8: 1345,
    9: 1225,
    10: 1112,
}

# Uniform cap matching data_v4 distribution (~1500 per step)
STEP_CAPS_UNIFORM = {i: 1500 for i in range(1, 11)}


def balanced_sample(records: List[Dict], step_caps: Dict[int, int], seed: int = 42) -> List[Dict]:
    """Downsample per step_index to avoid early-step dominance.

    Groups records by step_index, caps each group, and samples by prompt_id
    so that all steps from a selected prompt stay together within each group.
    """
    rng = random.Random(seed)

    # Group by step_index
    by_step = defaultdict(list)
    for r in records:
        by_step[r["extra_info"]["step_index"]].append(r)

    sampled = []
    print("\n  Step balancing:")
    for step in sorted(by_step.keys()):
        group = by_step[step]
        cap = step_caps.get(step, 1500)
        if len(group) <= cap:
            sampled.extend(group)
            print(f"    step {step:2d}: {len(group):5d} → {len(group):5d} (keep all)")
        else:
            # Sample by prompt_id to keep coherence
            pid_to_records = defaultdict(list)
            for r in group:
                pid_to_records[r["extra_info"]["prompt_id"]].append(r)
            pids = list(pid_to_records.keys())
            rng.shuffle(pids)
            selected = []
            for pid in pids:
                selected.extend(pid_to_records[pid])
                if len(selected) >= cap:
                    break
            sampled.extend(selected)
            print(f"    step {step:2d}: {len(group):5d} → {len(selected):5d} (capped at {cap})")

    print(f"    total: {len(records)} → {len(sampled)}")
    return sampled


def process_dataset(
    input_path: str,
    output_dir: str,
    eval_ratio: float = 0.1,
    max_trajectory_tokens: int = 6000,
    balance_steps: bool = True,
    prompt_style: str = "v4",
    step_cap_style: str = "default",
    tokenizer_path: str = None,
):
    """Main processing pipeline."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer for token-based truncation
    tokenizer = None
    if tokenizer_path:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        print(f"Loaded tokenizer from {tokenizer_path} (vocab_size={tokenizer.vocab_size})")
    else:
        print("Warning: no --tokenizer-path provided, using char-based estimate for truncation")

    step_caps = STEP_CAPS_UNIFORM if step_cap_style == "uniform" else STEP_CAPS

    records = []
    skipped_step0 = 0
    skipped_no_trajectory = 0

    print(f"Reading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)

            # Filter step_index == 0 (empty trajectory, rubric not needed)
            if record["step_index"] == 0:
                skipped_step0 += 1
                continue

            # Double-check trajectory is non-empty
            if not record.get("trajectory", "").strip():
                skipped_no_trajectory += 1
                continue

            # Build prompt
            prompt = build_prompt_messages(
                question=record["question"],
                trajectory=record["trajectory"],
                step_index=record["step_index"],
                max_trajectory_tokens=max_trajectory_tokens,
                prompt_style=prompt_style,
                tokenizer=tokenizer,
            )

            # Build reward metadata (actions + borda scores for reward computation)
            reward_meta_str = build_reward_meta(record)

            # verl schema: prompt=list[dict], reward_model=dict, extra_info=dict
            records.append({
                "data_source": "rubric_gen",
                "prompt": prompt,  # list[dict] — verl tokenizes it via apply_chat_template
                "reward_model": {
                    "style": "rule",
                    "ground_truth": reward_meta_str,  # JSON string passed to compute_score
                },
                "extra_info": {
                    "prompt_id": record["prompt_id"],
                    "step_index": record["step_index"],
                },
            })

            if line_num % 5000 == 0:
                print(f"  Processed {line_num} lines, kept {len(records)} records...")

    print(f"\nTotal lines read: {line_num}")
    print(f"Skipped step_index=0: {skipped_step0}")
    print(f"Skipped empty trajectory: {skipped_no_trajectory}")
    print(f"Valid records: {len(records)}")

    # Step-balanced sampling (train only, eval keeps all for fair comparison)
    if balance_steps:
        records = balanced_sample(records, step_caps)

    # Split by prompt_id (so all steps from one prompt stay together)
    unique_prompt_ids = sorted(set(r["extra_info"]["prompt_id"] for r in records))
    split_idx = int(len(unique_prompt_ids) * (1 - eval_ratio))
    train_ids = set(unique_prompt_ids[:split_idx])
    eval_ids = set(unique_prompt_ids[split_idx:])

    train_records = [r for r in records if r["extra_info"]["prompt_id"] in train_ids]
    eval_records = [r for r in records if r["extra_info"]["prompt_id"] in eval_ids]

    # Print final step distribution
    from collections import Counter
    train_steps = Counter(r["extra_info"]["step_index"] for r in train_records)
    print("\n  Final train step distribution:")
    for s in sorted(train_steps.keys()):
        pct = train_steps[s] / len(train_records) * 100
        print(f"    step {s:2d}: {train_steps[s]:5d} ({pct:5.1f}%)")

    # Save as Parquet
    train_path = output_dir / "train.parquet"
    eval_path = output_dir / "eval.parquet"

    pd.DataFrame(train_records).to_parquet(train_path, index=False)
    pd.DataFrame(eval_records).to_parquet(eval_path, index=False)

    print(f"\nTrain: {len(train_records)} records ({len(train_ids)} prompts) → {train_path}")
    print(f"Eval:  {len(eval_records)} records ({len(eval_ids)} prompts) → {eval_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare rubric RL training data")
    parser.add_argument(
        "--input", default="/root/storage/kjz/dr-tulu/rl_data/rl_data_merged_v4.jsonl",
        help="Input JSONL path",
    )
    parser.add_argument(
        "--output-dir", default="/root/storage/kjz/dr-tulu/rubric_rl/data_v5",
        help="Output directory for Parquet files",
    )
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--max-trajectory-tokens", type=int, default=6000)
    parser.add_argument("--prompt-style", choices=["v4", "llama", "qwen14b"], default="v4",
                        help="Prompt style: v4 (4-msg with fake few-shot) or llama (2-msg) or qwen14b (2-msg, 4 few-shot, single-tool constraint)")
    parser.add_argument("--step-cap-style", choices=["default", "uniform"], default="default",
                        help="Step cap: default (front-heavy) or uniform (1500 per step, matches data_v4)")
    parser.add_argument("--tokenizer-path", type=str, default=None,
                        help="Path to tokenizer for token-based trajectory truncation")
    args = parser.parse_args()

    process_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        eval_ratio=args.eval_ratio,
        max_trajectory_tokens=args.max_trajectory_tokens,
        prompt_style=args.prompt_style,
        step_cap_style=args.step_cap_style,
        tokenizer_path=args.tokenizer_path,
    )
