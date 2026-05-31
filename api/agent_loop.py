"""DeepSeek V4 Agent Loop with thinking mode and function calling.

Handles the full agent interaction pattern:
  1. Send messages + tools to DeepSeek API (streaming)
  2. Model returns either: final content, or tool_calls with reasoning
  3. If tool_calls: execute tools, append results + reasoning_content to history, loop
  4. If final content: yield to caller, done

CRITICAL DeepSeek behavior:
  - For tool-call turns, reasoning_content MUST be passed back in subsequent requests.
    The API returns 400 if reasoning_content is missing from tool-call assistant messages.
  - For non-tool turns, reasoning_content is optional and ignored by the API.
  - The safest approach: always include reasoning_content when it exists.
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
    """Execute the DeepSeek agent loop with tool calling.

    Manages streaming chat with automatic tool execution. Each time the model
    requests tool calls, they are executed and results fed back. The loop
    continues until the model produces a final answer or the round limit is hit.

    Args:
        model_client: The DeepSeekClient instance (must have .acall method).
        api_kwargs: Base API kwargs including messages, model, stream, etc.
                    This dict is mutated to track message history during the loop.
        tool_executor: ToolExecutor instance for running tool calls.
        max_tool_rounds: Maximum tool-calling rounds before forcing a stop.
        yield_thinking: If True, yield reasoning_content chunks as they stream.
                        Otherwise only yield final content.

    Yields:
        str chunks of the final assistant response content. If yield_thinking
        is True, also yields reasoning chunks prefixed with a marker.
    """
    messages = api_kwargs.setdefault("messages", [])

    for round_idx in range(max_tool_rounds):
        log.debug(f"Agent round {round_idx + 1}/{max_tool_rounds}")

        # Clone kwargs for this round (stream=True, tools included)
        round_kwargs = {**api_kwargs, "stream": True}
        from api.tools.agent_tools import AGENT_TOOLS
        round_kwargs["tools"] = AGENT_TOOLS

        # Accumulate content from streaming response
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

            # Collect reasoning_content
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                accumulated_reasoning += reasoning
                if yield_thinking:
                    yield f"<!--thinking: {reasoning}-->"

            # Collect content
            content_delta = getattr(delta, "content", None)
            if content_delta:
                accumulated_content += content_delta

            # Collect tool calls (arrive in deltas, may be fragmented)
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

        # Handle finish reasons
        if finish_reason == "insufficient_system_resource":
            yield "\n\n[DeepSeek: inference interrupted due to system resource shortage. The response may be incomplete.]"
            if accumulated_content:
                yield accumulated_content
            return

        # If model has tool calls, execute them and continue the loop
        if tool_calls_acc:
            # Build assistant message with reasoning_content (CRITICAL for DeepSeek)
            assistant_msg: Dict[str, Any] = {"role": "assistant"}
            if accumulated_content:
                assistant_msg["content"] = accumulated_content
            else:
                assistant_msg["content"] = None

            if accumulated_reasoning:
                assistant_msg["reasoning_content"] = accumulated_reasoning

            # Add tool calls
            assistant_msg["tool_calls"] = tool_calls_acc
            messages.append(assistant_msg)

            # Execute each tool and append results
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

            # Stream tool execution summary to user
            tool_names = [tc["function"]["name"] for tc in tool_calls_acc]
            yield f"\n\n*[Used tools: {', '.join(tool_names)}]*\n\n"

            # Continue to next round
            continue

        # No tool calls - this is the final answer
        if accumulated_content:
            yield accumulated_content
        return

    # Max rounds reached
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
