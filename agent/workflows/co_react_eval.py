"""
Co-ReAct DRB Evaluation Workflow (Text Completion Mode).

At each tool-call step in the DR-Tulu ReAct loop:
1. Qwen3-8B generates a rubric (3-5 weighted criteria) based on trajectory
2. Rubric is injected as a shadow suffix into DR-Tulu's prompt
3. DR-Tulu generates an action (text completion with force_think)
4. Qwen3-8B verifies the action against the rubric
5. If verification fails, feedback is appended and DR-Tulu retries once

Usage:
    python -u workflows/co_react_eval.py generate-dataset deep_research_bench \
        --config workflows/co_react_eval.yaml \
        --output eval_output/co_react/deep_research_bench.jsonl \
        --verbose --max-concurrent 1 --use-cache --batch-size 5
"""

import asyncio
import json as json_module
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import aiohttp
import dotenv

# Ensure workflows/ dir is on path for sibling imports
_workflows_dir = os.path.dirname(os.path.abspath(__file__))
if _workflows_dir not in sys.path:
    sys.path.insert(0, _workflows_dir)

from dr_agent.client import GenerateWithToolsOutput, LLMToolClient, ToolOutput
from dr_agent.rubric_generator import RubricGenerator
from dr_agent.workflow import BaseWorkflow, BaseWorkflowConfiguration

from auto_search_sft import (
    AutoReasonSearchWorkflow,
    AnswerAgent,
    SearchAgent,
)

dotenv.load_dotenv(Path(__file__).parent.parent.parent / ".env")


class CoReActEvalWorkflow(AutoReasonSearchWorkflow):
    """Co-ReAct evaluation: ReAct loop with rubric generation, injection, and verification.

    Uses text completion mode with force_think (inject <think> after tool outputs).
    """

    _default_configuration_path = os.path.join(
        os.path.dirname(__file__), "co_react_eval.yaml"
    )

    class Configuration(AutoReasonSearchWorkflow.Configuration):
        # Rubric Generator (Qwen3-8B, shared with browse agent on port 30002)
        rubric_base_url: str = "http://localhost:30002/v1"
        rubric_model_name: str = "/root/storage/kjz/dr-tulu/models/Qwen3-8B"
        rubric_api_key: str = "dummy-key"
        rubric_api_type: str = "openai"  # "openai" or "cloudsway"
        rubric_url: Optional[str] = None  # Cloudsway direct URL (used when api_type="cloudsway")
        rubric_temperature: Optional[float] = 0.7
        rubric_max_tokens: int = 1024
        rubric_timeout: int = 120
        rubric_max_retries: int = 1  # max retries when verification fails
        rubric_strip_think: bool = False  # strip <think> from trajectory before rubric generation
        rubric_skip_verify: bool = False  # skip verification, only inject rubric into prompt
        rubric_disabled: bool = False  # disable rubric entirely (for baseline)
        rubric_repetition_penalty: Optional[float] = None
        rubric_frequency_penalty: Optional[float] = None
        rubric_max_think_tokens: Optional[int] = None
        rubric_prompt_version: str = "strongmodel_qwen14b"
        rubric_skip_few_shot: bool = False  # skip few-shot example in rubric generation
        rubric_max_trajectory_tokens: int = 0  # 0 = no limit; >0 = truncate trajectory to ~N tokens (keep tail)
        verify_version: str = "v1"
        # Separate verify model (defaults to None = use rubric generator for verify)
        verify_base_url: Optional[str] = None
        verify_model_name: Optional[str] = None
        verify_api_key: Optional[str] = None
        verify_api_type: Optional[str] = None
        verify_url: Optional[str] = None  # Cloudsway URL for verify model
        verify_temperature: Optional[float] = None
        verify_max_tokens: Optional[int] = None
        verify_timeout: Optional[int] = None
        force_think: bool = True  # inject <think> after each tool output (always true in text completion mode)
        use_chat_mode: bool = False  # use vLLM chat completion API instead of text completion
        use_reflection: bool = False  # enable feedforward reflection (replaces verify retry)
        skip_answer_agent: bool = False  # skip answer_agent, use search agent's inline <answer> directly
        no_think_instruction: str = ""  # if set, append to initial prompt to suppress <think>
        answer_think_instruction: str = ""  # if set, prepend to answer prompt to allow <think> during answer
        answer_additional_prompt: str = ""  # if set, append to answer instruction (e.g. "Please write as detailed and long as possible")
        # Shenma API mode (replaces local vLLM for search agent)
        use_shenma: bool = False
        shenma_url: str = os.environ.get("SHENMA_URL", "")
        shenma_model: str = "qwen3-8b-kkk-amd3"
        shenma_api_key: str = os.environ.get("SHENMA_API_KEY", "")
        shenma_timeout: int = 180
        # Cloudsway API mode (replaces local vLLM for search agent)
        use_cloudsway: bool = False
        cloudsway_url: str = os.environ.get("CLOUDSWAY_URL", "")  # Cloudsway chat completions endpoint
        cloudsway_api_key: str = os.environ.get("CLOUDSWAY_API_KEY", "")
        cloudsway_temperature: float = 0.0
        cloudsway_max_tokens: int = 32000
        cloudsway_timeout: int = 180

    def setup_components(self) -> None:
        """Reuse parent's tool/agent setup, then build rubric generator."""
        super().setup_components()

        cfg = self.configuration
        print(f"[CoReAct] force_think={cfg.force_think}, rubric_disabled={cfg.rubric_disabled}")
        if cfg.use_reflection:
            print(f"[CoReAct] Feedforward reflection mode enabled")
        if cfg.use_shenma:
            print(f"[CoReAct] Shenma API mode: model={cfg.shenma_model}")
        if cfg.use_cloudsway:
            print(f"[CoReAct] Cloudsway API mode: url={cfg.cloudsway_url}")

        if cfg.rubric_disabled:
            self.rubric_gen = None
            print("[CoReAct] Rubric disabled — running as baseline")
        else:
            self.rubric_gen = RubricGenerator(
                base_url=cfg.rubric_base_url,
                model_name=cfg.rubric_model_name,
                api_key=cfg.rubric_api_key,
                api_type=cfg.rubric_api_type,
                url=cfg.rubric_url,
                temperature=cfg.rubric_temperature,
                max_tokens=cfg.rubric_max_tokens,
                timeout=cfg.rubric_timeout,
                strip_think=cfg.rubric_strip_think,
                repetition_penalty=cfg.rubric_repetition_penalty,
                frequency_penalty=cfg.rubric_frequency_penalty,
                max_think_tokens=cfg.rubric_max_think_tokens,
                prompt_version=cfg.rubric_prompt_version,
                skip_few_shot=cfg.rubric_skip_few_shot,
                max_trajectory_tokens=cfg.rubric_max_trajectory_tokens,
            )
            backend = cfg.rubric_api_type
            target = cfg.rubric_url if backend == "cloudsway" else f"{cfg.rubric_base_url} / {cfg.rubric_model_name}"
            print(f"[CoReAct] RubricGenerator initialized: backend={backend}, target={target}, strip_think={cfg.rubric_strip_think}")

        # Separate verify model (if configured)
        if cfg.verify_base_url or cfg.verify_url:
            self.verify_gen = RubricGenerator(
                base_url=cfg.verify_base_url or cfg.rubric_base_url,
                model_name=cfg.verify_model_name or cfg.rubric_model_name,
                api_key=cfg.verify_api_key or cfg.rubric_api_key,
                api_type=cfg.verify_api_type or cfg.rubric_api_type,
                url=cfg.verify_url,
                temperature=cfg.verify_temperature if cfg.verify_temperature is not None else 0.3,
                max_tokens=cfg.verify_max_tokens or cfg.rubric_max_tokens,
                timeout=cfg.verify_timeout or cfg.rubric_timeout,
            )
            v_backend = cfg.verify_api_type or cfg.rubric_api_type
            v_target = cfg.verify_url if v_backend == "cloudsway" else f"{cfg.verify_base_url} / {cfg.verify_model_name}"
            print(f"[CoReAct] Separate verify model: backend={v_backend}, target={v_target}")
        else:
            self.verify_gen = None

        # Initialize aux-token accumulator (reset per example in _run_single_pass).
        # Tracks tokens spent on auxiliary helpers: rubric generation, verify, self-refine
        # critique, best-of-N scoring, step-back abstraction, CRITIC tool critique, reflection.
        self._aux_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}

        # Install usage sink onto rubric_gen and verify_gen so they accumulate into self._aux_usage.
        if self.rubric_gen is not None:
            self.rubric_gen.usage_sink = self._aux_usage
        if self.verify_gen is not None:
            self.verify_gen.usage_sink = self._aux_usage

    def _aux_add(self, usage: Optional[Dict[str, Any]]) -> None:
        """Accumulate an OpenAI-format usage dict into self._aux_usage."""
        if not usage:
            return
        self._aux_usage["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        self._aux_usage["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
        self._aux_usage["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
        self._aux_usage["calls"] += 1

    async def _run_single_pass(
        self,
        problem: str,
        dataset_name: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        verbose: bool = True,
        search_callback: Optional[Any] = None,
        step_callback: Optional[Any] = None,
        reflection: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a single pass of the ReAct loop. Extracted from __call__ to support Reflexion (2-pass)."""
        cfg = self.configuration

        # Reset per-example aux-token accumulator. The sink is shared with rubric_gen/verify_gen
        # and the 5 helper methods via self._aux_add(), so all of them feed into this dict.
        self._aux_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
        if getattr(self, "rubric_gen", None) is not None:
            self.rubric_gen.usage_sink = self._aux_usage
        if getattr(self, "verify_gen", None) is not None:
            self.verify_gen.usage_sink = self._aux_usage

        # --- Build initial prompt (same as SearchAgent) ---
        client: LLMToolClient = self.search_agent.client
        agent_messages = self.search_agent.prompt(
            question=problem, dataset_name=dataset_name, history=messages,
        )

        # ===== Chat completion mode: keep messages as list =====
        if cfg.use_chat_mode:
            chat_messages = [dict(m) for m in agent_messages]  # deep copy
            if cfg.no_think_instruction:
                chat_messages.append({"role": "user", "content": cfg.no_think_instruction})
                if verbose:
                    print(f"[ChatMode] Injected no-think instruction as user message")
            if reflection:
                reflection_block = (
                    f"[Reflection from previous attempt]\n{reflection}\n"
                    "[Use this reflection to improve your research strategy.]"
                )
                chat_messages.append({"role": "user", "content": reflection_block})
                if verbose:
                    print(f"[ChatMode] Injected reflection ({len(reflection)} chars)")
            # We still build text prompt for token counting reference
            prompt = client._messages_to_prompt(agent_messages)
            current_context = prompt  # kept for compatibility in return value reconstruction
            if verbose:
                print(f"[ChatMode] Chat completion mode enabled, {len(chat_messages)} initial messages")
        else:
            chat_messages = None  # not used in text mode
            if cfg.use_cloudsway or client.is_commercial_api_model:
                # For Cloudsway / commercial APIs, use fallback text prompt
                prompt = client._fallback_messages_to_prompt(agent_messages)
            else:
                prompt = client._messages_to_prompt(agent_messages)

            # Inject no-think instruction if configured
            if cfg.no_think_instruction:
                prompt += f"\n{cfg.no_think_instruction}\n"
                if verbose:
                    print(f"[CoReAct] Injected no-think instruction")

            # Inject reflection into prompt if provided (Reflexion pass 2)
            if reflection:
                reflection_block = (
                    f"\n[Reflection from previous attempt]\n{reflection}\n"
                    "[Use this reflection to improve your research strategy.]\n"
                )
                prompt += reflection_block
                if verbose:
                    print(f"[Reflexion] Injected reflection ({len(reflection)} chars) into prompt")

            current_context = prompt

        # Set up browse tool
        if hasattr(self, "composed_browse_tool"):
            from dr_agent.tool_interface.chained_tool import ChainedTool
            if isinstance(self.composed_browse_tool, ChainedTool):
                self.composed_browse_tool.tools[0].bm25_query = problem
                self.composed_browse_tool.tools[-1].agent.question = problem
            else:
                self.composed_browse_tool.bm25_query = problem

        # --- Core loop (text completion mode) ---
        base_max_tokens = cfg.search_agent_max_tokens
        max_tool_calls = cfg.search_agent_max_tool_calls
        stop_sequences = client._get_all_stop_sequences()
        max_retries = cfg.rubric_max_retries
        skip_verify = cfg.rubric_skip_verify
        rubric_disabled = cfg.rubric_disabled
        force_think = cfg.force_think

        rubric_log = []
        tool_calls_list: List[ToolOutput] = []
        tool_call_count = 0
        iteration = 0
        last_reflection = None  # feedforward reflection from previous step

        while True:
            iteration += 1
            if chat_messages is not None:
                ctx_tokens = sum(client._count_tokens(m["content"]) for m in chat_messages)
            else:
                ctx_tokens = client._count_tokens(current_context)
            if verbose:
                mode_tag = "ChatMode" if chat_messages is not None else "CoReAct"
                print(f"\n{'='*60}")
                print(f"[{mode_tag}] Step {iteration} | Tool calls: {tool_call_count}/{max_tool_calls}")
                print(f"[{mode_tag}] Context tokens: ~{ctx_tokens}/{base_max_tokens}")
            dynamic_max_tokens = max(base_max_tokens - ctx_tokens, 0)
            if dynamic_max_tokens <= 100:
                if verbose:
                    print(f"[{mode_tag}] Token budget exhausted, stopping.")
                break

            trajectory = current_context[len(prompt):] if chat_messages is None else ""

            if rubric_disabled:
                # Plain generation (no rubric)
                gen_prompt = current_context
                if force_think:
                    gen_prompt += "\n<think>\n"
                response = await self._generate_response(
                    gen_prompt,
                    stop_sequences=stop_sequences,
                    temperature=cfg.search_agent_temperature,
                    max_tokens=min(dynamic_max_tokens, 4096),
                )
                if force_think:
                    response = "<think>\n" + response
                rubric_log.append({"step": iteration, "mode": "baseline"})

            elif chat_messages is not None:
                # ===== Chat Mode Co-ReAct: rubric as user message, chat completion =====
                if rubric_disabled:
                    rubric = None
                    rubric_time = 0.0
                else:
                    # Build trajectory from assistant messages for rubric generation
                    from dr_agent.rubric_generator import strip_think_tags
                    assistant_texts = [m["content"] for m in chat_messages if m["role"] == "assistant"]
                    trajectory_for_rubric = "\n".join(assistant_texts)
                    trajectory_clean = strip_think_tags(trajectory_for_rubric)
                    t0 = time.time()
                    rubric = await self.rubric_gen.generate(
                        problem, trajectory_clean,
                        tool_call_count=tool_call_count, max_tool_calls=max_tool_calls,
                        reflection=last_reflection if cfg.use_reflection else None,
                    )
                    rubric_time = time.time() - t0

                if rubric is None:
                    if verbose:
                        print(f"[ChatMode] No rubric (first step or disabled)")
                else:
                    if verbose:
                        print(f"[ChatMode] Rubric generated in {rubric_time:.1f}s:")
                        for line in rubric.split("\n")[:5]:
                            print(f"  {line}")
                    # Inject rubric as a user message
                    rubric_text = RubricGenerator.format_rubric_suffix(
                        rubric, tool_call_count=tool_call_count
                    )
                    chat_messages.append({"role": "user", "content": rubric_text})

                response = await self._generate_response_chat(
                    chat_messages, stop_sequences=stop_sequences,
                    temperature=cfg.search_agent_temperature,
                    max_tokens=min(dynamic_max_tokens, 4096),
                )
                # Remove rubric message after generation — only this step needs it
                if rubric is not None:
                    chat_messages.pop()  # remove rubric user message

                passed, feedback, verify_time = True, "", 0.0
                retry_used = False
                if not skip_verify and not cfg.use_reflection and rubric is not None:
                    t0 = time.time()
                    verifier = self.verify_gen if self.verify_gen else self.rubric_gen
                    # v4: pass trajectory for context-aware verification
                    from dr_agent.rubric_generator import strip_think_tags
                    traj_for_verify = strip_think_tags(trajectory) if trajectory else None
                    passed, feedback = await verifier.verify(rubric, response, question=problem, verify_version=cfg.verify_version, trajectory=traj_for_verify)
                    verify_time = time.time() - t0
                    if verbose:
                        status = "PASS" if passed else "FAIL"
                        print(f"[ChatMode] Verification: {status} ({verify_time:.1f}s)")
                        if feedback:
                            print(f"[ChatMode] Feedback: {feedback[:150]}")
                    if not passed and max_retries > 0:
                        retry_used = True
                        if verbose:
                            print(f"[ChatMode] Retrying with feedback...")
                        # Temporarily add failed response + feedback for retry generation
                        chat_messages.append({"role": "assistant", "content": response})
                        feedback_text = f"[Rubric Feedback: {feedback}. Please reconsider your action.]"
                        if rubric is not None:
                            rubric_text_for_retry = RubricGenerator.format_rubric_suffix(
                                rubric, tool_call_count=tool_call_count
                            )
                            feedback_text += "\n" + rubric_text_for_retry
                        chat_messages.append({"role": "user", "content": feedback_text})
                        response = await self._generate_response_chat(
                            chat_messages, stop_sequences=stop_sequences,
                            temperature=cfg.search_agent_temperature,
                            max_tokens=min(dynamic_max_tokens, 4096),
                        )
                        # Remove failed response and feedback — only keep retry result
                        chat_messages.pop()  # remove feedback
                        chat_messages.pop()  # remove failed response
                elif verbose and rubric is not None:
                    if cfg.use_reflection:
                        print(f"[ChatMode] Verification skipped (reflection mode)")
                    else:
                        print(f"[ChatMode] Verification skipped (inject-only mode)")
                rubric_log.append({
                    "step": iteration,
                    "rubric": rubric if rubric is not None else "(skipped — first step)",
                    "passed": passed, "feedback": feedback, "retry_used": retry_used,
                    "rubric_time": round(rubric_time, 2) if rubric else 0.0,
                    "verify_time": round(verify_time, 2),
                    "reflection": last_reflection if cfg.use_reflection else None,
                })

            else:
                # Default Co-ReAct: rubric → generate → verify (TEXT completion mode)
                if rubric_disabled:
                    rubric = None
                    rubric_time = 0.0
                else:
                    from dr_agent.rubric_generator import strip_think_tags
                    trajectory_clean = strip_think_tags(trajectory)
                    t0 = time.time()
                    rubric = await self.rubric_gen.generate(
                        problem, trajectory_clean,
                        tool_call_count=tool_call_count, max_tool_calls=max_tool_calls,
                    )
                    rubric_time = time.time() - t0
                if rubric is None:
                    if verbose:
                        print(f"[CoReAct] No rubric (first step or disabled)")
                    rubric_suffix = ""
                else:
                    if verbose:
                        print(f"[CoReAct] Rubric generated in {rubric_time:.1f}s:")
                        for line in rubric.split("\n")[:5]:
                            print(f"  {line}")
                    rubric_suffix = RubricGenerator.format_rubric_suffix(
                        rubric, tool_call_count=tool_call_count
                    )
                gen_prompt = current_context
                if rubric_suffix:
                    gen_prompt += rubric_suffix
                if force_think:
                    gen_prompt += "\n<think>\n"
                response = await self._generate_response(
                    gen_prompt, stop_sequences=stop_sequences,
                    temperature=cfg.search_agent_temperature,
                    max_tokens=min(dynamic_max_tokens, 4096),
                )
                if force_think:
                    response = "<think>\n" + response
                passed, feedback, verify_time = True, "", 0.0
                retry_used = False
                if not skip_verify and rubric is not None:
                    t0 = time.time()
                    verifier = self.verify_gen if self.verify_gen else self.rubric_gen
                    # v4: pass trajectory for context-aware verification
                    traj_for_verify = strip_think_tags(trajectory) if trajectory else None
                    passed, feedback = await verifier.verify(rubric, response, question=problem, verify_version=cfg.verify_version, trajectory=traj_for_verify)
                    verify_time = time.time() - t0
                    if verbose:
                        status = "PASS" if passed else "FAIL"
                        print(f"[CoReAct] Verification: {status} ({verify_time:.1f}s)")
                        if feedback:
                            print(f"[CoReAct] Feedback: {feedback[:150]}")
                    if not passed and max_retries > 0:
                        retry_used = True
                        if verbose:
                            print(f"[CoReAct] Retrying with feedback...")
                        retry_prompt = (
                            current_context + response
                            + f"\n[Rubric Feedback: {feedback}. Please reconsider your action.]\n" + rubric_suffix
                        )
                        if force_think:
                            retry_prompt += "\n<think>\n"
                        response = await self._generate_response(
                            retry_prompt, stop_sequences=stop_sequences,
                            temperature=cfg.search_agent_temperature,
                            max_tokens=min(dynamic_max_tokens, 4096),
                        )
                        if force_think:
                            response = "<think>\n" + response
                elif verbose and rubric is not None:
                    print(f"[CoReAct] Verification skipped (inject-only mode)")
                rubric_log.append({
                    "step": iteration,
                    "rubric": rubric if rubric is not None else "(skipped — first step)",
                    "passed": passed, "feedback": feedback, "retry_used": retry_used,
                    "rubric_time": round(rubric_time, 2), "verify_time": round(verify_time, 2),
                })

            # ⑥ Append response to context
            if chat_messages is not None:
                chat_messages.append({"role": "assistant", "content": response})
            else:
                current_context += response

            if step_callback:
                if asyncio.iscoroutinefunction(step_callback):
                    await step_callback(response, [])
                else:
                    step_callback(response, [])

            if chat_messages is not None:
                post_gen_tokens = sum(client._count_tokens(m["content"]) for m in chat_messages)
            else:
                post_gen_tokens = client._count_tokens(current_context)
            if post_gen_tokens >= base_max_tokens:
                if verbose:
                    print("[CoReAct] Token budget exhausted after generation, stopping.")
                break

            tool_match = client._find_first_tool_call(response)
            if not tool_match:
                if verbose:
                    print("[CoReAct] No tool call found (likely <answer> or natural end), finishing.")
                break

            if tool_call_count >= max_tool_calls:
                if verbose:
                    print(f"[CoReAct] Exceeded max tool calls ({max_tool_calls}), stopping.")
                error_output = tool_match._create_error_output(
                    error_msg="exceed allowed tool call requests", call_id="", runtime=0,
                    output="Exceed allowed tool call requests.",
                )
                tool_calls_list.append(error_output)
                result_formatted = tool_match.format_result(error_output)
                if chat_messages is not None:
                    chat_messages.append({"role": "user", "content": result_formatted})
                else:
                    current_context += "\n" + result_formatted
                break

            tool = tool_match
            max_tool_retries = 3
            tool_output = None
            for attempt in range(1, max_tool_retries + 1):
                try:
                    t0 = time.time()
                    tool_output = await tool(response)
                    tool_time = time.time() - t0
                    if not (tool_output.timeout or (tool_output.error and "timed out" in str(tool_output.error))):
                        break
                    if attempt < max_tool_retries:
                        if verbose:
                            print(f"[CoReAct] Tool {tool.name} timed out ({tool_time:.0f}s), retrying ({attempt}/{max_tool_retries})...")
                        await asyncio.sleep(2)
                    else:
                        if verbose:
                            print(f"[CoReAct] Tool {tool.name} timed out after {max_tool_retries} attempts")
                except Exception as e:
                    tool_time = time.time() - t0
                    if verbose:
                        print(f"[CoReAct] Tool {tool.name} error: {e}")
                    tool_output = tool_match._create_error_output(
                        error_msg=str(e), call_id="", runtime=tool_time, output=f"Tool error: {e}",
                    )
                    break

            tool_call_count += 1
            tool_calls_list.append(tool_output)
            if verbose:
                print(f"[CoReAct] Executed {tool.name} in {tool_time:.1f}s")

            if step_callback:
                if asyncio.iscoroutinefunction(step_callback):
                    await step_callback("", [tool_output])
                else:
                    step_callback("", [tool_output])

            if tool_output.called:
                result_formatted = tool.format_result(tool_output)
                if chat_messages is not None:
                    chat_messages.append({"role": "user", "content": result_formatted})
                else:
                    current_context += result_formatted

                # Feedforward reflection: reflect on action vs rubric after tool execution
                if cfg.use_reflection and chat_messages is not None and rubric is not None:
                    from dr_agent.rubric_generator import strip_think_tags
                    action_clean = strip_think_tags(response)
                    t0 = time.time()
                    last_reflection = await self.rubric_gen.reflect(
                        problem, rubric, action_clean, result_formatted,
                        tool_call_count=tool_call_count, max_tool_calls=max_tool_calls,
                    )
                    reflect_time = time.time() - t0
                    if verbose:
                        print(f"[Reflection] ({reflect_time:.1f}s): {last_reflection[:150]}")

            if chat_messages is not None:
                post_tool_tokens = sum(client._count_tokens(m["content"]) for m in chat_messages)
            else:
                post_tool_tokens = client._count_tokens(current_context)
            if post_tool_tokens >= base_max_tokens:
                if verbose:
                    print("[CoReAct] Token budget exhausted after tool execution, stopping.")
                break

        # --- Build return value ---
        if chat_messages is not None:
            # Reconstruct generated_text from non-initial messages
            # Skip rubric user messages (they contain [Rubric for next action])
            initial_count = len(agent_messages) + (1 if cfg.no_think_instruction else 0) + (1 if reflection else 0)
            text_parts = []
            for m in chat_messages[initial_count:]:
                content = m["content"]
                # Skip rubric injection messages
                if m["role"] == "user" and "[Rubric for next action]" in content:
                    continue
                # Skip verification feedback messages
                if m["role"] == "user" and ("[Verification Failed]" in content or "[Rubric Feedback:" in content):
                    continue
                text_parts.append(content)
            generated_text = "\n".join(text_parts)
        else:
            generated_text = current_context[len(prompt):]

        browsed_links = []
        searched_links = []
        for tool_output in tool_calls_list:
            if hasattr(tool_output, "tool_name"):
                if tool_output.tool_name in ["snippet_search", "google_search"]:
                    if hasattr(tool_output, "documents"):
                        searched_links.extend(
                            [doc.url for doc in tool_output.documents if hasattr(doc, "url")]
                        )
                elif tool_output.tool_name == "browse_webpage":
                    if hasattr(tool_output, "documents"):
                        browsed_links.extend(
                            [doc.url for doc in tool_output.documents if hasattr(doc, "url") and doc.url]
                        )
        browsed_links = list(set(browsed_links))
        searched_links = list(set(searched_links))

        traces_output = GenerateWithToolsOutput(
            generated_text=generated_text,
            total_tokens=client._count_tokens(generated_text),
            tool_call_count=tool_call_count,
            stopped_reason="natural",
            tool_calls=tool_calls_list,
        )

        answer_max_tokens = cfg.search_agent_max_tokens
        if cfg.skip_answer_agent and "<answer>" in generated_text:
            if verbose:
                print("[CoReAct] skip_answer_agent=True, using inline <answer> directly")
            final_response = self.search_agent.postprocess_output(traces_output)
        elif chat_messages is not None:
            # Chat mode answer generation
            if "<answer>" in generated_text:
                if verbose:
                    print("[ChatMode] Inline <answer> found, using directly")
                final_response = self.search_agent.postprocess_output(traces_output)
            else:
                if verbose:
                    print("[ChatMode] Generating answer via chat completion...")
                answer_instruction = "Now please generate an answer based on the search results by far:"
                if cfg.answer_additional_prompt:
                    answer_instruction += f" {cfg.answer_additional_prompt}"
                chat_messages.append({"role": "user", "content": f"{answer_instruction}\n<answer>"})
                answer_text = await self._generate_response_chat(
                    chat_messages, stop_sequences=["</answer>"],
                    temperature=cfg.search_agent_temperature,
                    max_tokens=min(answer_max_tokens, 4096),
                )
                if not answer_text.startswith("<answer>"):
                    answer_text = "<answer>\n" + answer_text
                if "</answer>" not in answer_text:
                    answer_text += "\n</answer>"
                generated_text += "\n" + answer_text
                traces_output = GenerateWithToolsOutput(
                    generated_text=generated_text,
                    total_tokens=client._count_tokens(generated_text),
                    tool_call_count=tool_call_count,
                    stopped_reason="natural",
                    tool_calls=tool_calls_list,
                )
                final_response = self.search_agent.postprocess_output(traces_output)
        elif cfg.use_cloudsway:
            # Use Cloudsway API for answer generation
            if verbose:
                print("[CoReAct] Generating answer via Cloudsway API...")
            answer_instruction = "Now please generate an answer based on the search results by far:"
            if cfg.answer_additional_prompt:
                answer_instruction += f" {cfg.answer_additional_prompt}"
            answer_prompt = current_context + f"\n{answer_instruction}\n<answer>\n"
            answer_text = await self._cloudsway_completion(answer_prompt, stop_sequences=["</answer>"], temperature=cfg.cloudsway_temperature, max_tokens=cfg.cloudsway_max_tokens)
            if not answer_text.startswith("<answer>"):
                answer_text = "<answer>\n" + answer_text
            if "</answer>" not in answer_text:
                answer_text += "\n</answer>"
            generated_text += "\n" + answer_text
            traces_output = GenerateWithToolsOutput(
                generated_text=generated_text,
                total_tokens=client._count_tokens(generated_text),
                tool_call_count=tool_call_count,
                stopped_reason="natural",
                tool_calls=tool_calls_list,
            )
            final_response = self.search_agent.postprocess_output(traces_output)
        elif cfg.use_shenma:
            # Use Shenma streaming for answer generation too
            if verbose:
                print("[CoReAct] Generating answer via Shenma API...")
            think_prefix = ""
            if cfg.answer_think_instruction:
                think_prefix = f"\n{cfg.answer_think_instruction}\n"
                if verbose:
                    print("[CoReAct] Injected answer-think instruction (thinking enabled for answer)")
            answer_instruction = "Now please generate an answer based on the search results by far:"
            if cfg.answer_additional_prompt:
                answer_instruction += f" {cfg.answer_additional_prompt}"
            answer_prompt = current_context + think_prefix + f"\n{answer_instruction}\n<answer>\n"
            answer_text = await self._shenma_completion(answer_prompt, temperature=cfg.search_agent_temperature, max_new_tokens=4096)
            if not answer_text.startswith("<answer>"):
                answer_text = "<answer>\n" + answer_text
            if "</answer>" not in answer_text:
                answer_text += "\n</answer>"
            generated_text += "\n" + answer_text
            traces_output = GenerateWithToolsOutput(
                generated_text=generated_text,
                total_tokens=client._count_tokens(generated_text),
                tool_call_count=tool_call_count,
                stopped_reason="natural",
                tool_calls=tool_calls_list,
            )
            final_response = self.search_agent.postprocess_output(traces_output)
        else:
            if verbose:
                has_inline = "<answer>" in generated_text
                print(f"[CoReAct] Using answer_agent for final answer (inline <answer> found: {has_inline}, max_tokens={answer_max_tokens})")
            answer_instruction = "Now please generate an answer based on the search results by far:"
            if cfg.answer_additional_prompt:
                answer_instruction += f" {cfg.answer_additional_prompt}"
            answer_result = await self.answer_agent(
                question=problem, history=generated_text, dataset_name=dataset_name,
                additional_instructions=answer_instruction,
                generation_prefix="<answer>",
                max_tokens=answer_max_tokens, temperature=cfg.search_agent_temperature,
                verbose=verbose,
            )
            final_response = self.answer_agent.postprocess_output(answer_result)
            answer_result.tool_calls = [traces_output.model_dump()]
            traces_output = answer_result

        return {
            "final_response": final_response,
            "full_traces": traces_output,
            "browsed_links": browsed_links,
            "searched_links": searched_links,
            "rubric_log": rubric_log,
            "total_tool_calls": tool_call_count,
            "generated_text": generated_text,
            "aux_token_usage": dict(self._aux_usage),
        }

    async def _shenma_completion(self, prompt: str, temperature: float = 0.0, max_new_tokens: int = 4096) -> str:
        """Call Shenma streaming completions API with early stop on tool call / answer."""
        cfg = self.configuration
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {cfg.shenma_api_key}",
        }
        payload = {
            "session_id": f"sess_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}",
            "request_id": f"req_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}",
            "model": cfg.shenma_model,
            "prompt": prompt,
            "stream": True,
            "extra_args": {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "temperature": max(temperature, 0.01),
                "top_p": 0.9,
            },
        }
        latest_content = ""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                cfg.shenma_url, headers=headers, json=payload,
                timeout=aiohttp.ClientTimeout(total=cfg.shenma_timeout),
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
                        chunk = json_module.loads(json_str)
                        latest_content = chunk["choices"][0]["message"]["content"]
                        if "<tool_output>" in latest_content:
                            break
                        if "</call_tool>" in latest_content:
                            break
                    except (json_module.JSONDecodeError, KeyError, TypeError):
                        continue
        # Truncate hallucinated tool output
        if "<tool_output>" in latest_content:
            latest_content = latest_content[:latest_content.index("<tool_output>")]
        # Truncate to first complete call_tool
        first_close = latest_content.find("</call_tool>")
        if first_close >= 0:
            latest_content = latest_content[:first_close + len("</call_tool>")]
        # Truncate after </answer> if present
        answer_close = latest_content.find("</answer>")
        if answer_close >= 0:
            latest_content = latest_content[:answer_close + len("</answer>")]
        return latest_content.strip()

    async def _cloudsway_completion(self, prompt: str, stop_sequences=None, temperature=0.0, max_tokens=4096) -> str:
        """Call Cloudsway chat completion API directly via aiohttp."""
        cfg = self.configuration
        headers = {
            "Authorization": f"Bearer {cfg.cloudsway_api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop_sequences:
            data["stop"] = stop_sequences

        async with aiohttp.ClientSession() as session:
            async with session.post(
                cfg.cloudsway_url, headers=headers, json=data,
                timeout=aiohttp.ClientTimeout(total=cfg.cloudsway_timeout),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise Exception(f"Cloudsway API error {resp.status}: {text[:500]}")
                result = await resp.json()
                content = result["choices"][0]["message"]["content"] or ""

        # Re-append stop token if the API stripped it (same logic as tool call detection)
        if stop_sequences and content:
            for stop_tok in stop_sequences:
                if stop_tok in prompt and not content.endswith(stop_tok):
                    # Check if adding this stop token creates a valid tool call
                    test_content = content + stop_tok
                    if "</call_tool>" in test_content or "</answer>" in test_content:
                        content = test_content
                        break
        return content

    async def _generate_response_chat(self, messages: list, stop_sequences=None, temperature=0.0, max_tokens=4096) -> str:
        """Generate response using vLLM chat completion API (message-based)."""
        import litellm
        client: LLMToolClient = self.search_agent.client
        model_name = client.model_name
        if not model_name.startswith("hosted_vllm/"):
            model_name = f"hosted_vllm/{model_name}"
        params = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": stop_sequences,
            "api_key": client.api_key,
            "api_base": client.base_url,
        }
        response = await litellm.acompletion(**params)
        content = response.choices[0].message.content or ""
        # Re-append stop token if stripped by the API
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "stop" and stop_sequences:
            for stop_tok in stop_sequences:
                test_content = content + stop_tok
                if "</call_tool>" in test_content or "</answer>" in test_content:
                    content = test_content
                    break
        return content

    async def _generate_response(self, prompt: str, stop_sequences=None, temperature=0.0, max_tokens=4096) -> str:
        """Route generation to Cloudsway, Shenma, commercial API, or local vLLM."""
        cfg = self.configuration
        if cfg.use_cloudsway:
            return await self._cloudsway_completion(prompt, stop_sequences=stop_sequences,
                                                    temperature=temperature, max_tokens=max_tokens)
        elif cfg.use_shenma:
            return await self._shenma_completion(prompt, temperature=temperature, max_new_tokens=max_tokens)
        else:
            client: LLMToolClient = self.search_agent.client
            if client.is_commercial_api_model:
                # For commercial APIs, wrap text prompt as a single user message
                messages = [{"role": "user", "content": prompt}]
                return await client._generate_single_response_commercial_api(
                    messages, stop_sequences=stop_sequences,
                    temperature=temperature, max_tokens=max_tokens,
                )
            else:
                return await client._generate_single_response_vllm(
                    prompt, stop_sequences=stop_sequences,
                    temperature=temperature, max_tokens=max_tokens,
                )

    async def __call__(
        self,
        problem: str,
        dataset_name: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        verbose: bool = True,
        search_callback: Optional[Any] = None,
        step_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Entry point. Delegates to _run_single_pass."""
        return await self._run_single_pass(
            problem, dataset_name, messages, verbose, search_callback, step_callback,
        )


if __name__ == "__main__":
    CoReActEvalWorkflow.app()
