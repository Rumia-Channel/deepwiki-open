"""Agent loop with thinking mode and function calling.

Supports both DeepSeek V4 and OpenAI GPT-5 series.

DeepSeek specifics:
  - thinking via extra_body: {"thinking": {"type": "enabled"}}
  - reasoning_effort inside thinking dict or as top-level
  - reasoning_content MUST be passed back for tool-call turns (400 error otherwise)

OpenAI GPT-5 specifics:
  - reasoning_effort as top-level parameter (medium/high/xhigh)
  - GPT-5 also returns reasoning_content in streaming deltas
  - reasoning items should be passed back for tool-call turns
  - Prompt caching is automatic (transparent)

Shared behavior:
  - Streaming deltas contain: reasoning_content (thinking tokens), content (final answer), tool_calls
  - Tool calls accumulate across chunk deltas (index-based merging)
  - Loop continues while tool_calls are present, stops on final content
"""

import logging
from typing import AsyncGenerator, List, Dict, Any, Optional

from adalflow.core.types import ModelType

log = logging.getLogger(__name__)


async def run_agent_loop(
    model_client,
    api_kwargs: Dict[str, Any],
    tool_executor,
    max_tool_rounds: int = 15,
    yield_thinking: bool = False,
) -> AsyncGenerator[str, None]:
    """Execute the agent loop with tool calling (DeepSeek + OpenAI compatible).

    Works with any OpenAI-compatible API that supports:
      - Streaming chat completions with tool_calls
      - reasoning_content in delta chunks (optional, skipped if absent)
      - Standard tool role messages

    For DeepSeek specifically, reasoning_content is tracked and included in
    assistant messages for tool-call turns (required by DeepSeek API).

    Args:
        model_client: ModelClient with .acall(api_kwargs, model_type) method.
        api_kwargs: Base API kwargs including messages. Mutated in-place to
                    track conversation history.
        tool_executor: ToolExecutor instance for running tool calls.
        max_tool_rounds: Maximum tool-calling rounds before forcing a stop.
        yield_thinking: If True, yield reasoning_content chunks as they stream.

    Yields:
        str chunks of the final assistant response content.
    """
    messages = api_kwargs.setdefault("messages", [])

    for round_idx in range(max_tool_rounds):
        log.debug(f"Agent round {round_idx + 1}/{max_tool_rounds}")

        round_kwargs = {**api_kwargs, "stream": True}
        from api.tools.agent_tools import AGENT_TOOLS
        round_kwargs["tools"] = AGENT_TOOLS

        accumulated_content = ""
        accumulated_reasoning = ""
        tool_calls_acc: List[Dict[str, Any]] = []
        finish_reason = None

        response = await model_client.acall(
            api_kwargs=round_kwargs, model_type=ModelType.LLM
        )

        async for chunk in response:
            choices = getattr(chunk, "choices", [])
            if not choices:
                continue

            choice = choices[0]
            finish_reason = getattr(choice, "finish_reason", None) or finish_reason
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            # Collect reasoning_content (DeepSeek + GPT-5)
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                accumulated_reasoning += reasoning
                if yield_thinking:
                    yield f"<!--thinking: {reasoning}-->"

            # Collect content (final answer text)
            content_delta = getattr(delta, "content", None)
            if content_delta:
                accumulated_content += content_delta

            # Collect tool calls (may arrive in fragments across deltas)
            tool_deltas = getattr(delta, "tool_calls", None)
            if tool_deltas:
                for tc_delta in tool_deltas:
                    idx = tc_delta.index
                    while len(tool_calls_acc) <= idx:
                        tool_calls_acc.append({
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""}
                        })
                    if tc_delta.id:
                        tool_calls_acc[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_acc[idx]["function"]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls_acc[idx]["function"]["arguments"] += tc_delta.function.arguments

        log.debug(
            f"Round {round_idx + 1} complete: "
            f"content={len(accumulated_content)} chars, "
            f"reasoning={len(accumulated_reasoning)} chars, "
            f"tool_calls={len(tool_calls_acc)}, "
            f"finish={finish_reason}"
        )

        if finish_reason == "insufficient_system_resource":
            yield "\n\n[Inference interrupted due to system resource shortage. Response may be incomplete.]"
            if accumulated_content:
                yield accumulated_content
            return

        # If model made tool calls, execute and continue
        if tool_calls_acc:
            assistant_msg: Dict[str, Any] = {"role": "assistant"}
            if accumulated_content:
                assistant_msg["content"] = accumulated_content
            else:
                assistant_msg["content"] = None

            # Include reasoning_content (required for DeepSeek tool-call turns, best-effort for OpenAI)
            if accumulated_reasoning:
                assistant_msg["reasoning_content"] = accumulated_reasoning

            assistant_msg["tool_calls"] = tool_calls_acc
            messages.append(assistant_msg)

            tool_results = []
            for tc in tool_calls_acc:
                log.info(
                    f"Executing tool: {tc['function']['name']}"
                    f"({tc['function']['arguments'][:100]})"
                )
                result = tool_executor.execute(_make_tool_call_obj(tc))
                tool_results.append(result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

            tool_names = [tc["function"]["name"] for tc in tool_calls_acc]
            yield f"\n\n*[Used tools: {', '.join(tool_names)}]*\n\n"
            continue

        # Final answer
        if accumulated_content:
            yield accumulated_content
        return

    yield "\n\n[Agent loop reached maximum tool-calling rounds. The analysis may be incomplete.]"
    if accumulated_content:
        yield accumulated_content


class _FakeToolCall:
    """Minimal wrapper so tool executor can access function.name/arguments."""
    def __init__(self, tc_dict: Dict[str, Any]):
        self.id = tc_dict.get("id", "")
        self.function = _FakeFunction(tc_dict.get("function", {}))


class _FakeFunction:
    def __init__(self, fn_dict: Dict[str, Any]):
        self.name = fn_dict.get("name", "")
        self.arguments = fn_dict.get("arguments", "{}")


def _make_tool_call_obj(tc_dict: Dict[str, Any]) -> _FakeToolCall:
    return _FakeToolCall(tc_dict)
