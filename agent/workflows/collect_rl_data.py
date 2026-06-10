"""
Co-ReAct RL Data Collection Workflow (Dual-Model Text Completion).

At each tool-call step in the DR-Tulu ReAct loop:
1. Sample 4 actions from Qwen3-8B (force_think) + 4 from Qwen3-14B + 4 from Qwen3-32B (shenma API),
   each at different temperatures [0.1, 0.4, 0.7, 1.0], all in parallel
2. Dedup by cleaned text, then BM25 greedy diversity selection: 12 → 4 most diverse candidates
3. Council of Judges ranks the 4 candidates (ties allowed)
4. Borda count aggregation, pick Rank-1 action to continue
5. Save (state, actions, expert_rankings) per step as JSONL

Usage:
    python -u workflows/collect_rl_data.py \
        --config workflows/collect_rl_data.yaml \
        deep_research_bench \
        -o /root/storage/kjz/dr-tulu/rl_data/drb_rl_data.jsonl \
        -n 2 -v
"""

import asyncio
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import sys

import aiohttp
import dotenv

# Ensure workflows/ dir is on path for sibling imports
_workflows_dir = os.path.dirname(os.path.abspath(__file__))
if _workflows_dir not in sys.path:
    sys.path.insert(0, _workflows_dir)

from dr_agent.client import LLMToolClient
from dr_agent.judges import JudgeClient, JudgeRanker, strip_think_tags
from dr_agent.workflow import BaseWorkflow, BaseWorkflowConfiguration

from auto_search_sft import (
    AutoReasonSearchWorkflow,
    SearchAgent,
)

dotenv.load_dotenv(Path(__file__).parent.parent.parent / ".env")

# ── Temperature schedule shared by both models ──
TEMPERATURE_SCHEDULE = [0.1, 0.4, 0.7, 1.0]

ANSWER_PLACEHOLDER = '<call_tool name="write_report">开始撰写最终报告</call_tool>'


# ── Shenma API client for Qwen3-14B ──

async def _shenma_completion(
    prompt: str,
    model: str,
    temperature: float,
    max_new_tokens: int = 4096,
    env: str = "预发",
    timeout: int = 120,
) -> str:
    """Call shenma streaming completions API and return full response text."""
    if env == "预发":
        url = os.environ.get("SHENMA_URL", "")
    else:
        url = os.environ.get("SHENMA_URL_PROD", "")

    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {os.environ.get('SHENMA_API_KEY', '')}",
    }
    payload = {
        "session_id": f"sess_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}",
        "request_id": f"req_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}",
        "model": model,
        "prompt": prompt,
        "stream": True,
        "extra_args": {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": 0.9,
        },
    }

    latest_content = ""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url, headers=headers, json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Shenma API error {resp.status}: {text[:200]}")
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                json_str = line[len("data:"):].strip()
                if json_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(json_str)
                    latest_content = chunk["choices"][0]["message"]["content"]
                    # Early stop: if <answer>, <tool_output>, or second <call_tool> detected
                    if "<answer>" in latest_content or "<tool_output>" in latest_content:
                        break
                    # Stop after first complete </call_tool> to prevent multi-tool generation
                    if "</call_tool>" in latest_content:
                        break
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
    # Truncate at <tool_output> if present (hallucinated tool output)
    if "<tool_output>" in latest_content:
        latest_content = latest_content[:latest_content.index("<tool_output>")]
    # Truncate to first complete <call_tool>...</call_tool> to prevent multi-tool
    first_close = latest_content.find("</call_tool>")
    if first_close >= 0:
        latest_content = latest_content[:first_close + len("</call_tool>")]
    return latest_content.strip()


# ── BM25 greedy diversity selection ──

def _bm25_select_diverse(texts: List[str], k: int = 4) -> List[int]:
    """Greedily select k most diverse texts using BM25 similarity.

    Returns list of indices into the original texts list.
    """
    from rank_bm25 import BM25Okapi

    if len(texts) <= k:
        return list(range(len(texts)))

    # Tokenize (simple whitespace + punctuation split)
    tokenized = [re.split(r'\s+|(?<=[^\w])|(?=[^\w])', t) for t in texts]
    tokenized = [[w for w in tokens if w.strip()] for tokens in tokenized]

    bm25 = BM25Okapi(tokenized)

    # Greedy selection: start with longest text, then pick most different
    selected = [max(range(len(texts)), key=lambda i: len(texts[i]))]
    remaining = set(range(len(texts))) - set(selected)

    while len(selected) < k and remaining:
        best_idx = -1
        best_min_sim = float("inf")
        for cand in remaining:
            # Compute max similarity to any already-selected text
            scores = bm25.get_scores(tokenized[cand])
            max_sim = max(scores[s] for s in selected)
            if max_sim < best_min_sim:
                best_min_sim = max_sim
                best_idx = cand
        if best_idx >= 0:
            selected.append(best_idx)
            remaining.discard(best_idx)
        else:
            break

    return selected


def _abstract_answer(text: str) -> str:
    """Replace entire action with write_report if it contains <answer>.

    Some models hallucinate fake tool_output before <answer>; we replace
    the whole thing so judges only see the write_report placeholder.
    Also handles unclosed <answer> tags.
    """
    if "<answer>" in text:
        return ANSWER_PLACEHOLDER
    return text


def _is_invalid_action(text: str) -> bool:
    """Check if action is invalid and should be discarded.

    Invalid actions:
    1. Contains <tool_output> without <call_tool> (hallucinated tool output)
    2. Contains neither <call_tool> nor <answer> (raw text response, not a valid action)
    """
    if "<tool_output>" in text and "<call_tool" not in text:
        return True
    if "<call_tool" not in text and "<answer>" not in text:
        return True
    return False


def _clean_for_judge(text: str) -> str:
    """Clean action text for judge/BM25: strip think tags, hallucinated tool_output, abstract answer."""
    # Strip paired <think>...</think>
    cleaned = strip_think_tags(text)
    # Strip stray </think>
    cleaned = re.sub(r'</think>', '', cleaned)
    # Unclosed <think>: keep content after <call_tool> or <answer> if present
    if '<think>' in cleaned:
        # Try to salvage the tool call / answer after unclosed think
        tool_match = re.search(r'(<call_tool[\s\S]*)', cleaned)
        answer_match = re.search(r'(<answer>[\s\S]*)', cleaned)
        if tool_match:
            cleaned = tool_match.group(1)
        elif answer_match:
            cleaned = answer_match.group(1)
        else:
            # Truly no action content, remove everything
            cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL)
    # Strip hallucinated <tool_output> — judge only needs to see the <call_tool> part
    cleaned = re.sub(r'<tool_output>[\s\S]*', '', cleaned)
    # Truncate to first tool call — multi-tool actions should be judged by first call only
    cleaned = _truncate_to_first_tool_call(cleaned)
    cleaned = cleaned.strip()
    # Abstract answer → write_report placeholder
    cleaned = _abstract_answer(cleaned)
    return cleaned


def _truncate_to_first_tool_call(text: str) -> str:
    """Truncate action text to keep only up to the first complete <call_tool>...</call_tool>.

    Preserves <think>...</think> before the tool call. Strips any subsequent
    tool calls that the model hallucinated (shenma models especially).
    """
    # Find first </call_tool>
    close_pos = text.find("</call_tool>")
    if close_pos >= 0:
        return text[:close_pos + len("</call_tool>")]
    return text


def _normalize_for_dedup(text: str) -> str:
    """Normalize cleaned text for dedup: strip punctuation/whitespace differences.

    Actions that differ only in trailing punctuation (e.g. '?' vs '。') are
    essentially identical and should be deduplicated.
    """
    # Remove common Chinese/English punctuation
    text = re.sub(r'[，。？！、；：""''（）\?\!\.\,\;\:\'\"\(\)]', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class RLDataCollectionWorkflow(AutoReasonSearchWorkflow):
    """Collect (state, N_actions, expert_rankings) data at every tool-call node."""

    _default_configuration_path = os.path.join(
        os.path.dirname(__file__), "collect_rl_data.yaml"
    )

    class Configuration(AutoReasonSearchWorkflow.Configuration):
        # Judges — loaded from YAML dict
        judges: Optional[Dict[str, Dict]] = None

        # Output
        rl_output_dir: str = "/root/storage/kjz/dr-tulu/rl_data"

        # Force thinking for Qwen3-8B (text completion mode)
        force_think: bool = True

        # Shenma API models
        shenma_models: List[str] = None  # e.g. ["qwen3-14b-kkk", "qwen3-32b-kkk"]
        shenma_env: str = "预发"
        shenma_timeout: int = 120

        # BM25 diversity selection: from 15 candidates pick top-k
        diversity_top_k: int = 4

    def setup_components(self) -> None:
        """Reuse parent's tool/agent setup, then build judge ranker."""
        super().setup_components()

        cfg = self.configuration
        if cfg.judges:
            self.judge_ranker = JudgeRanker.from_config(cfg.judges)
            print(f"[RLData] Loaded {len(self.judge_ranker.judges)} judges: "
                  f"{[j.name for j in self.judge_ranker.judges]}")
        else:
            self.judge_ranker = JudgeRanker(judges=[])
            print("[RLData] WARNING: No judges configured!")

    async def __call__(
        self,
        problem: str,
        dataset_name: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        verbose: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run the ReAct loop with dual-model sampling, BM25 diversity selection,
        and judge ranking (with ties) at each step.
        """
        cfg = self.configuration

        # --- Build initial prompt via SearchAgent (same as co_react_eval.py) ---
        client: LLMToolClient = self.search_agent.client
        prompt_messages = self.search_agent.prompt(
            question=problem, dataset_name=dataset_name, history=messages or [],
        )
        initial_prompt = client._messages_to_prompt(prompt_messages)
        # Inject minimum tool call constraint before assistant turn
        min_calls_hint = "\nIMPORTANT: You must call at least 3 tools before generating <answer>. Do thorough research first.\n"
        initial_prompt = initial_prompt.replace(
            "<|im_start|>assistant\n",
            min_calls_hint + "<|im_start|>assistant\n",
        )
        current_context = initial_prompt
        stop_sequences = client._get_all_stop_sequences()

        # Set up browse tool query
        if hasattr(self, "composed_browse_tool"):
            from dr_agent.tool_interface.chained_tool import ChainedTool
            if isinstance(self.composed_browse_tool, ChainedTool):
                self.composed_browse_tool.tools[0].bm25_query = problem
                self.composed_browse_tool.tools[-1].agent.question = problem
            else:
                self.composed_browse_tool.bm25_query = problem

        # --- Core RL data collection loop ---
        base_max_tokens = cfg.search_agent_max_tokens
        max_tool_calls = cfg.search_agent_max_tool_calls

        step_data_list = []
        tool_call_count = 0
        iteration = 0

        while True:
            iteration += 1

            # Token budget check
            ctx_tokens = client._count_tokens(current_context)

            if verbose:
                print(f"\n{'='*60}")
                print(f"[RLData] Step {iteration} | Tool calls: {tool_call_count}/{max_tool_calls}")
                print(f"[RLData] Context tokens: {ctx_tokens}/{base_max_tokens}")

            dynamic_max_tokens = max(base_max_tokens - ctx_tokens, 0)
            if dynamic_max_tokens <= 100:
                if verbose:
                    print("[RLData] Token budget exhausted, stopping.")
                break

            # --- 1. Dual-model sampling (5 temps × 2 models = 10, all parallel) ---
            t0 = time.time()
            all_raw, source_tags = await self._sample_dual_model(
                current_context, dynamic_max_tokens, stop_sequences, client, cfg, verbose,
                step_index=iteration,
            )
            sample_time = time.time() - t0

            if not all_raw:
                print(f"[RLData] No valid actions at step {iteration}, stopping.")
                break

            if verbose:
                print(f"[RLData] Sampled {len(all_raw)} actions in {sample_time:.1f}s")

            # --- 2. Dedup by normalized text, then BM25 diversity selection: 12 → top_k ---
            # Actions differing only in punctuation (e.g. '?' vs '。') are near-duplicates
            bm25_texts = [_clean_for_judge(t) for t in all_raw]
            seen_texts = {}
            deduped_indices = []
            for idx, text in enumerate(bm25_texts):
                norm = _normalize_for_dedup(text)
                if norm not in seen_texts:
                    seen_texts[norm] = idx
                    deduped_indices.append(idx)
            deduped_raw = [all_raw[i] for i in deduped_indices]
            deduped_tags = [source_tags[i] for i in deduped_indices]
            deduped_bm25 = [bm25_texts[i] for i in deduped_indices]

            if verbose and len(deduped_indices) < len(all_raw):
                print(f"[RLData] Dedup: {len(all_raw)} → {len(deduped_indices)} unique")

            selected_indices = _bm25_select_diverse(deduped_bm25, k=cfg.diversity_top_k)
            actions = [deduped_raw[i] for i in selected_indices]
            action_tags = [deduped_tags[i] for i in selected_indices]

            if verbose:
                print(f"[RLData] BM25 diversity: {len(all_raw)} → {len(actions)}")
                for i, (a, tag) in enumerate(zip(actions, action_tags)):
                    preview = _clean_for_judge(a)[:120].replace("\n", " ")
                    print(f"  [{chr(ord('A')+i)}] ({tag}) {preview}...")

            # --- 3.5. Skip saving if only 1 unique candidate (no preference signal) ---
            if len(actions) <= 1:
                if verbose:
                    print(f"[RLData] Only {len(actions)} unique action(s), skipping judge & save (no preference signal)")
                # Still use the single action to continue trajectory
                best_action = actions[0]
                best_action_for_ctx = re.sub(r'<tool_output>[\s\S]*', '', best_action).rstrip()
                best_action_for_ctx = _truncate_to_first_tool_call(best_action_for_ctx)
                current_context += best_action_for_ctx
                tool_match = client._find_first_tool_call(best_action)
                if not tool_match:
                    if verbose:
                        print("[RLData] No tool call in single action, finishing.")
                    break
                if tool_call_count >= max_tool_calls:
                    if verbose:
                        print(f"[RLData] Exceeded max tool calls ({max_tool_calls}), stopping.")
                    break
                # Execute tool (same logic as step 8 below)
                tool = tool_match
                t0 = time.time()
                tool_output = await tool(best_action)
                tool_time = time.time() - t0
                tool_call_count += 1
                if verbose:
                    print(f"[RLData] Executed {tool.name} in {tool_time:.1f}s")
                if tool_output.called:
                    result_formatted = tool.format_result(tool_output)
                    current_context += result_formatted
                continue

            # --- 4. Call judges (strip think + abstract answer for presentation) ---
            trajectory = current_context[len(initial_prompt):]
            t0 = time.time()
            # Prepare actions for judges: clean think + abstract answer
            judge_actions = [_clean_for_judge(a) for a in actions]
            ranking_result = await self.judge_ranker.rank_actions(
                question=problem,
                trajectory=strip_think_tags(trajectory),
                actions=judge_actions,
            )
            judge_time = time.time() - t0

            if verbose:
                print(f"[RLData] Judge ranking in {judge_time:.1f}s: "
                      f"valid={ranking_result['num_valid_judges']}, "
                      f"borda={ranking_result['borda_scores']}")

            best_idx = ranking_result["best_action_idx"]

            # --- 5. Save step data ---
            actions_dict = {}
            for ai, action_text in enumerate(actions):
                label = chr(ord("A") + ai)
                per_judge_ranks = {}
                for judge_name, ranks in ranking_result["expert_rankings"].items():
                    per_judge_ranks[judge_name] = ranks[ai]
                actions_dict[label] = {
                    "text": action_text,
                    "source": action_tags[ai],
                    "borda_score": ranking_result["borda_scores"][ai],
                    "rankings": per_judge_ranks,
                    "is_best": ai == best_idx,
                }

            step_record = {
                "step_index": iteration - 1,
                "state_tokens": ctx_tokens,
                "trajectory": trajectory,
                "actions": actions_dict,
                "best_action": chr(ord("A") + best_idx),
                "num_valid_judges": ranking_result["num_valid_judges"],
                "sample_time": round(sample_time, 2),
                "judge_time": round(judge_time, 2),
            }
            step_data_list.append(step_record)

            # --- 6. Continue context with best action ---
            best_action = actions[best_idx]
            # Strip hallucinated <tool_output> and multi-tool before appending to context
            # to prevent trajectory pollution (real tool output will be appended after execution)
            best_action_for_ctx = re.sub(r'<tool_output>[\s\S]*', '', best_action).rstrip()
            best_action_for_ctx = _truncate_to_first_tool_call(best_action_for_ctx)
            current_context += best_action_for_ctx

            if verbose:
                print(f"[RLData] Selected action [{chr(ord('A')+best_idx)}] "
                      f"(borda={ranking_result['borda_scores'][best_idx]}, "
                      f"source={action_tags[best_idx]})")

            # --- 7. Check for tool call in best action ---
            tool_match = client._find_first_tool_call(best_action)
            if not tool_match:
                if verbose:
                    print("[RLData] No tool call in best action (likely <answer>), finishing.")
                break

            # Check tool call limit
            if tool_call_count >= max_tool_calls:
                if verbose:
                    print(f"[RLData] Exceeded max tool calls ({max_tool_calls}), stopping.")
                error_output = tool_match._create_error_output(
                    error_msg="exceed allowed tool call requests",
                    call_id="",
                    runtime=0,
                    output="Exceed allowed tool call requests.",
                )
                result_formatted = tool_match.format_result(error_output)
                current_context += "\n" + result_formatted
                break

            # --- 8. Execute tool ---
            tool = tool_match
            t0 = time.time()
            tool_output = await tool(best_action)
            tool_time = time.time() - t0
            tool_call_count += 1

            if verbose:
                print(f"[RLData] Executed {tool.name} in {tool_time:.1f}s")

            # Append tool result to context
            if tool_output.called:
                result_formatted = tool.format_result(tool_output)
                current_context += result_formatted

            # Check token budget after tool execution
            if client._count_tokens(current_context) >= base_max_tokens:
                if verbose:
                    print("[RLData] Token budget exhausted after tool execution, stopping.")
                break

        # --- Build return value ---
        generated_text = current_context[len(initial_prompt):]

        # Extract final answer if present
        final_response = ""
        if "<answer>" in generated_text:
            parts = generated_text.split("<answer>")
            if len(parts) > 1:
                final_response = parts[1].split("</answer>")[0].strip()
        else:
            final_response = generated_text

        # Clean think tags
        if "</think>" in final_response:
            final_response = "".join(final_response.split("</think>")[1:]).strip()

        return {
            "final_response": final_response,
            "full_traces": generated_text,
            "rl_step_data": step_data_list,
            "total_steps": len(step_data_list),
            "total_tool_calls": tool_call_count,
        }

    async def _sample_dual_model(
        self,
        current_context: str,
        dynamic_max_tokens: int,
        stop_sequences: List[str],
        client: "LLMToolClient",
        cfg: "Configuration",
        verbose: bool,
        step_index: int = 0,
    ) -> Tuple[List[str], List[str]]:
        """Sample from local vLLM + shenma models at 5 temperatures each, all in parallel.

        Returns (action_texts, source_tags) where source_tag is 'model@T=x'.
        Uses <answer> as early stop to avoid generating full reports.
        At step 0, write_report/answer actions are discarded (no info collected yet).
        """
        max_tokens = min(dynamic_max_tokens, 4096)
        tasks = []
        tags = []

        # Add <answer> to stop sequences so models stop immediately on answer
        answer_stop = list(stop_sequences) + ["<answer>"]

        # -- Qwen3-8B (local vLLM): force_think, 5 temperatures --
        gen_prompt_8b = current_context + "\n<think>\n"
        for temp in TEMPERATURE_SCHEDULE:
            tasks.append(
                client._generate_n_responses_vllm(
                    prompt=gen_prompt_8b, n=1, temperature=temp,
                    max_tokens=max_tokens, stop_sequences=answer_stop,
                )
            )
            tags.append(f"qwen3-8b@T={temp}")

        # -- Shenma API models: no think, 5 temperatures each --
        gen_prompt_shenma = current_context + "\n<think></think>\n"
        shenma_models = cfg.shenma_models or ["qwen3-14b-kkk"]
        for model in shenma_models:
            # Extract short name for tag (e.g. "qwen3-14b-kkk" -> "qwen3-14b")
            short_name = model.replace("-kkk", "")
            for temp in TEMPERATURE_SCHEDULE:
                tasks.append(
                    _shenma_completion(
                        prompt=gen_prompt_shenma,
                        model=model,
                        temperature=temp,
                        max_new_tokens=max_tokens,
                        env=cfg.shenma_env,
                        timeout=cfg.shenma_timeout,
                    )
                )
                tags.append(f"{short_name}@T={temp}")

        # Fire all requests in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_actions = []
        all_tags = []
        for i, (res, tag) in enumerate(zip(results, tags)):
            if isinstance(res, Exception):
                if verbose:
                    print(f"[RLData] {tag} failed: {res}")
                continue
            if isinstance(res, list):
                # vLLM returns list (n=1 → single item)
                for text in res:
                    action = "<think>\n" + text
                    # If stopped at <answer>, mark as answer action
                    if action.rstrip().endswith("<answer>"):
                        action = ANSWER_PLACEHOLDER
                    # Step 0: discard write_report — no info collected yet
                    if step_index == 0 and ("write_report" in action or "<answer>" in action):
                        if verbose:
                            print(f"[RLData] {tag} discarded: write_report at step 0")
                        continue
                    # Discard invalid actions (check on cleaned version to handle think edge cases)
                    if _is_invalid_action(_clean_for_judge(action)):
                        if verbose:
                            print(f"[RLData] {tag} discarded: invalid action (no <call_tool> or <answer>)")
                        continue
                    all_actions.append(action)
                    all_tags.append(tag)
            elif isinstance(res, str):
                # Shenma returns string — save raw output, cleaning happens in _clean_for_judge()
                if not res.strip():
                    continue
                action = res
                # If contains <answer>, treat as answer action
                if "<answer>" in action:
                    action = ANSWER_PLACEHOLDER
                # Step 0: discard write_report — no info collected yet
                if step_index == 0 and ("write_report" in action or "<answer>" in action):
                    if verbose:
                        print(f"[RLData] {tag} discarded: write_report at step 0")
                    continue
                # Discard invalid actions (check on cleaned version to handle think edge cases)
                if _is_invalid_action(_clean_for_judge(action)):
                    if verbose:
                        print(f"[RLData] {tag} discarded: invalid action (no <call_tool> or <answer>)")
                    continue
                all_actions.append(action)
                all_tags.append(tag)
            else:
                if verbose:
                    print(f"[RLData] {tag} unexpected result type: {type(res)}")

        return all_actions, all_tags

    @staticmethod
    def _load_local_jsonl(path: str) -> List[Dict[str, Any]]:
        """Load dataset from a local JSONL file. Each line must have 'id' and 'problem' fields."""
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    async def collect_and_save(
        self,
        dataset_config: Dict[str, Any],
        output_file: str,
        max_concurrent_tasks: int = 3,
        verbose: bool = True,
        num_examples: Optional[int] = None,
        data_file: Optional[str] = None,
        shard_id: Optional[int] = None,
        num_shards: Optional[int] = None,
    ):
        """
        Run RL data collection on a dataset and save step-level JSONL.

        Each line in the output file is one step from one prompt:
        {prompt_id, step_index, state_tail, actions, expert_rankings, borda_scores, ...}

        Uses asyncio.Semaphore for internal concurrency: while one prompt waits
        for judge APIs (~20-30s), another prompt can do vLLM sampling on the GPU.

        Args:
            data_file: Path to local JSONL file with {id, problem, ...} per line.
                       If provided, dataset_config is ignored for loading.
            max_concurrent_tasks: Number of prompts to process concurrently within
                                  this worker. Default 3 overlaps judge wait time.
        """
        if data_file:
            print(f"[RLData] Loading from local file: {data_file}")
            dataset = self._load_local_jsonl(data_file)
        else:
            from dr_agent.dataset_utils.load_dataset import load_dataset
            dataset = load_dataset(dataset_config)

        # Sort: rl-data first, then sft-data (by 'origin' field)
        origin_order = {"rl-data": 0, "sft-data": 1}
        dataset.sort(key=lambda d: origin_order.get(d.get("origin", ""), 2))
        rl_count = sum(1 for d in dataset if d.get("origin") == "rl-data")
        sft_count = sum(1 for d in dataset if d.get("origin") == "sft-data")
        print(f"[RLData] Sorted by origin: {rl_count} rl-data first, then {sft_count} sft-data")

        # Apply sharding first (each worker gets its slice of the full dataset)
        if shard_id is not None and num_shards is not None:
            dataset = dataset[shard_id::num_shards]
            print(f"[RLData] Shard {shard_id}/{num_shards}: {len(dataset)} examples")

        # Then limit per-worker count (for testing)
        if num_examples is not None:
            dataset = dataset[:num_examples]

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check existing results for resumption
        existing_ids = set()
        if output_path.exists():
            with open(output_path, "r") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        existing_ids.add(record.get("prompt_id"))

        remaining = [d for d in dataset if d["id"] not in existing_ids]
        total_remaining = len(remaining)
        print(f"[RLData] {len(existing_ids)} already done, {total_remaining} remaining")
        print(f"[RLData] Internal concurrency: {max_concurrent_tasks}")

        # Concurrency control
        sem = asyncio.Semaphore(max_concurrent_tasks)
        file_lock = asyncio.Lock()
        completed_count = [0]  # mutable counter for closure

        async def process_one(idx: int, example: Dict[str, Any]):
            prompt_id = example["id"]
            problem = example["problem"]
            dataset_name = dataset_config.get("name", "unknown")
            origin = example.get("origin", "")

            async with sem:
                print(f"\n{'#'*70}")
                print(f"[RLData] Example {idx+1}/{total_remaining} | ID: {prompt_id} | origin: {origin}")
                print(f"[RLData] Problem: {problem[:100]}...")
                print(f"{'#'*70}")

                try:
                    result = await self(
                        problem=problem,
                        dataset_name=dataset_name,
                        verbose=verbose,
                    )

                    # Write step-level records (protected by lock)
                    async with file_lock:
                        with open(output_path, "a", encoding="utf-8") as f:
                            for step in result.get("rl_step_data", []):
                                record = {
                                    "prompt_id": prompt_id,
                                    "question": problem,
                                    "origin": origin,
                                    **step,
                                }
                                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    completed_count[0] += 1
                    n_steps = len(result.get("rl_step_data", []))
                    print(f"[RLData] Saved {n_steps} steps for {prompt_id} "
                          f"({completed_count[0]}/{total_remaining} done)")

                except Exception as e:
                    print(f"[RLData] ERROR processing {prompt_id}: {e}")
                    import traceback
                    traceback.print_exc()

        # Launch all tasks, semaphore controls actual concurrency
        tasks = [process_one(i, ex) for i, ex in enumerate(remaining)]
        await asyncio.gather(*tasks)

        print(f"\n[RLData] Done! {completed_count[0]}/{total_remaining} completed. Output: {output_file}")


if __name__ == "__main__":
    import argparse
    import logging

    import litellm

    parser = argparse.ArgumentParser(description="Co-ReAct RL Data Collection")
    parser.add_argument("dataset_name", type=str, nargs="?", default="long_form",
                        help="Dataset name (e.g., deep_research_bench). Ignored if --data-file is set.")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--data-file", type=str, default=None,
                        help="Local JSONL file with {id, problem, ...} per line")
    parser.add_argument("--num-examples", "-n", type=int, default=None, help="Number of examples")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    # Parallel / sharding args
    parser.add_argument("--port-override", type=int, default=None,
                        help="Override vLLM port in search_agent_base_url")
    parser.add_argument("--shard-id", type=int, default=None,
                        help="Worker shard ID (0-based). Use with --num-shards.")
    parser.add_argument("--num-shards", type=int, default=None,
                        help="Total number of shards/workers.")
    parser.add_argument("--max-concurrent", type=int, default=3,
                        help="Number of prompts to process concurrently within this worker (default: 3)")
    args = parser.parse_args()

    if not args.verbose:
        logging.getLogger("mcp.client.streamable_http").setLevel(logging.WARNING)
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        litellm.turn_off_message_logging = True

    config_file = args.config or os.path.join(os.path.dirname(__file__), "collect_rl_data.yaml")

    # Build config overrides dict — must be applied BEFORE workflow construction
    # because setup_components() creates the LLMToolClient with base_url in __init__
    config_overrides = {}
    if args.port_override is not None:
        # Read the YAML to get the original URL, compute the new one
        import yaml
        with open(config_file, "r") as f:
            raw_cfg = yaml.safe_load(f)
        old_url = raw_cfg.get("search_agent_base_url", "http://localhost:30001/v1")
        new_url = re.sub(r':\d+/', f':{args.port_override}/', old_url)
        config_overrides["search_agent_base_url"] = new_url
        print(f"[RLData] Port override: {old_url} -> {new_url}")

    workflow = RLDataCollectionWorkflow(
        configuration=config_file,
        skip_service_check=True,
        **config_overrides,
    )

    dataset_config = {
        "name": args.dataset_name,
        "num_examples": args.num_examples,
    }

    asyncio.run(
        workflow.collect_and_save(
            dataset_config=dataset_config,
            output_file=args.output,
            verbose=args.verbose,
            num_examples=args.num_examples,
            data_file=args.data_file,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            max_concurrent_tasks=args.max_concurrent,
        )
    )
