"""DeepSeek ModelClient integration with full optimization support.

DeepSeek API features beyond OpenAI compatibility:
  - Thinking/reasoning mode via extra_body: thinking.type=enabled/disabled
  - reasoning_effort control (high/max), defaults to "max" for agent-grade reasoning
  - Automatic disk-based KV cache (always-on, transparent to caller)
  - reasoning_content exposed in streaming deltas alongside content
  - user_id for KV cache isolation + scheduling isolation
  - Cache monitoring: prompt_cache_hit_tokens / prompt_cache_miss_tokens
  - Special finish_reason: insufficient_system_resource
  - 1M token context window / 384K max output (both flash and pro)
  - frequency_penalty and presence_penalty are deprecated (no effect)
  - stream_options.include_usage for per-request token tracking

reasoning_content handling (DeepSeek-specific):
  The thinking mode returns a reasoning_content field at the same level as content
  in each delta. The requirements for passing it back to the API differ by context:

  1. NORMAL CHAT (no tool calls):
     - reasoning_content from previous assistant turns is NOT needed.
     - The API will ignore it if passed.
     - When building message history, only content needs to be preserved.

  2. TOOL CALLS (agent mode):
     - If a turn involves function/tool calls, the assistant message's
       reasoning_content MUST be passed back in ALL subsequent requests.
     - Failing to do so results in a 400 error from the API.
     - Use build_assistant_message() to construct a proper message dict that
       includes both content and reasoning_content.

  Streaming note:
    In streaming mode, reasoning_content and content arrive in separate chunks
    within the same stream. The reasoning_content chunks come first, followed by
    content chunks. Both are on delta.reasoning_content and delta.content respectively.
    For display purposes, only delta.content should be yielded to the user.

API docs: https://api-docs.deepseek.com/
"""

import os
from typing import (
    Dict,
    Sequence,
    Optional,
    List,
    Any,
    TypeVar,
    Callable,
    Generator,
    Union,
    Literal,
)
import re
import logging
import backoff

from adalflow.utils.lazy_import import safe_import, OptionalPackages

openai = safe_import(OptionalPackages.OPENAI.value[0], OptionalPackages.OPENAI.value[1])

from openai import OpenAI, AsyncOpenAI, Stream
from openai import (
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    UnprocessableEntityError,
    BadRequestError,
)
from openai.types.chat import ChatCompletionChunk, ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from adalflow.core.model_client import ModelClient
from adalflow.core.types import (
    ModelType,
    EmbedderOutput,
    CompletionUsage,
    GeneratorOutput,
)

log = logging.getLogger(__name__)
T = TypeVar("T")

# DeepSeek-specific constants
DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_CACHE_HIT_PRICE_RATIO = 0.02  # ~50x-120x cheaper on cache hit


def estimate_token_count(text: str) -> int:
    tokens = text.split()
    return len(tokens)


def parse_stream_response_for_deepseek(completion: ChatCompletionChunk) -> str:
    """Parse streaming response, filtering out reasoning_content from display output.

    In thinking mode, DeepSeek streams reasoning_content chunks first, then
    content chunks. This function only returns content (the final answer),
    skipping the chain-of-thought tokens.
    """
    delta = completion.choices[0].delta
    if delta is None:
        return None
    content = getattr(delta, "content", None)
    return content


def build_assistant_message(
    content: str,
    reasoning_content: Optional[str] = None,
    tool_calls: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Build a proper assistant message dict for DeepSeek's thinking mode.

    When tool calls are involved, reasoning_content MUST be included in the
    message dict passed back to the API in subsequent turns. The API enforces
    this strictly and returns 400 if reasoning_content is missing for
    tool-call turns.

    For normal (non-tool-call) turns, reasoning_content is optional and will
    be ignored by the API if passed.

    Args:
        content: The assistant's text response.
        reasoning_content: The chain-of-thought from thinking mode (optional for non-tool turns).
        tool_calls: Tool call definitions if the assistant invoked tools.

    Returns:
        A message dict suitable for appending to the messages list.
    """
    message: Dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


class DeepSeekClient(ModelClient):
    __doc__ = r"""DeepSeek-optimized ModelClient supporting all DeepSeek-specific features.

    Supports both chat completion and embedding APIs via OpenAI-compatible interface.
    Automatically handles:
      - Thinking/reasoning mode (thinking.type, reasoning_effort)
      - reasoning_content filtering in streaming responses
      - Disk-based KV cache monitoring
      - user_id for cache isolation
      - Proper model defaults and parameter filtering

    Usage:
        >>> client = DeepSeekClient()
        >>> # Non-thinking mode (standard chat):
        >>> response = client.call(api_kwargs={"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "Hello"}]}, model_type=ModelType.LLM)
        >>> # Thinking mode:
        >>> response = client.call(api_kwargs={"model": "deepseek-v4-pro", "messages": [...], "extra_body": {"thinking": {"type": "enabled"}}}, model_type=ModelType.LLM)

    Environment Variables:
        DEEPSEEK_API_KEY: API key for DeepSeek API (required)
        DEEPSEEK_BASE_URL: Custom base URL (default: https://api.deepseek.com)

    Args:
        api_key: DeepSeek API key. Defaults to DEEPSEEK_API_KEY env var.
        base_url: API base URL. Defaults to https://api.deepseek.com.
        user_id: Optional user ID for KV cache isolation (max 512 chars, [a-zA-Z0-9\-_]).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        user_id: Optional[str] = None,
        input_type: Literal["text", "messages"] = "text",
    ):
        super().__init__()
        self._api_key = api_key
        self._env_api_key_name = "DEEPSEEK_API_KEY"
        self._env_base_url_name = "DEEPSEEK_BASE_URL"
        self.base_url = base_url or os.getenv(
            self._env_base_url_name, DEEPSEEK_DEFAULT_BASE_URL
        )
        self._user_id = user_id or os.getenv("DEEPSEEK_USER_ID")
        self.sync_client = self.init_sync_client()
        self.async_client = None
        self._input_type = input_type
        self._api_kwargs = {}

    def init_sync_client(self):
        api_key = self._api_key or os.getenv(self._env_api_key_name)
        if not api_key:
            raise ValueError(
                f"Environment variable {self._env_api_key_name} must be set"
            )
        return OpenAI(api_key=api_key, base_url=self.base_url)

    def init_async_client(self):
        api_key = self._api_key or os.getenv(self._env_api_key_name)
        if not api_key:
            raise ValueError(
                f"Environment variable {self._env_api_key_name} must be set"
            )
        return AsyncOpenAI(api_key=api_key, base_url=self.base_url)

    def parse_chat_completion(
        self,
        completion: Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]],
    ) -> "GeneratorOutput":
        log.debug(f"completion: {completion}")
        try:
            data = completion.choices[0].message.content
        except Exception as e:
            log.error(f"Error parsing the completion: {e}")
            return GeneratorOutput(data=None, error=str(e), raw_response=completion)

        try:
            usage = self._track_deepseek_usage(completion)
            return GeneratorOutput(
                data=None, error=None, raw_response=data, usage=usage
            )
        except Exception as e:
            log.error(f"Error tracking the completion usage: {e}")
            return GeneratorOutput(data=None, error=str(e), raw_response=data)

    def _track_deepseek_usage(
        self,
        completion: Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]],
    ) -> CompletionUsage:
        """Track usage including DeepSeek-specific cache hit/miss tokens."""
        try:
            usage = completion.usage
            if usage:
                cache_hit = getattr(usage, "prompt_cache_hit_tokens", None)
                cache_miss = getattr(usage, "prompt_cache_miss_tokens", None)
                log.debug(
                    f"DeepSeek cache: hit={cache_hit}, miss={cache_miss}, "
                    f"prompt={usage.prompt_tokens}, completion={usage.completion_tokens}"
                )
                return CompletionUsage(
                    completion_tokens=usage.completion_tokens,
                    prompt_tokens=usage.prompt_tokens,
                    total_tokens=usage.total_tokens,
                )
            return CompletionUsage(
                completion_tokens=None, prompt_tokens=None, total_tokens=None
            )
        except Exception as e:
            log.error(f"Error tracking usage: {e}")
            return CompletionUsage(
                completion_tokens=None, prompt_tokens=None, total_tokens=None
            )

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        """Convert AdalFlow inputs to DeepSeek API format.

        Handles DeepSeek-specific parameters:
          - thinking: extra_body parameter for reasoning mode
          - reasoning_effort: included in model_kwargs or extracted from conversation
          - user_id: KV cache isolation

        When thinking mode is enabled, temperature/top_p/presence_penalty/frequency_penalty
        have no effect and are filtered out.
        """
        final_model_kwargs = model_kwargs.copy()

        if model_type == ModelType.EMBEDDER:
            if isinstance(input, str):
                input = [input]
            if not isinstance(input, Sequence):
                raise TypeError("input must be a sequence of text")
            final_model_kwargs["input"] = input
            return final_model_kwargs

        elif model_type == ModelType.LLM:
            messages: List[Dict[str, str]] = []
            images = final_model_kwargs.pop("images", None)
            detail = final_model_kwargs.pop("detail", "auto")

            if self._input_type == "messages":
                system_start_tag = "<START_OF_SYSTEM_PROMPT>"
                system_end_tag = "<END_OF_SYSTEM_PROMPT>"
                user_start_tag = "<START_OF_USER_PROMPT>"
                user_end_tag = "<END_OF_USER_PROMPT>"

                pattern = (
                    rf"{system_start_tag}\s*(.*?)\s*{system_end_tag}\s*"
                    rf"{user_start_tag}\s*(.*?)\s*{user_end_tag}"
                )
                regex = re.compile(pattern, re.DOTALL)
                match = regex.match(input)
                system_prompt, input_str = None, None

                if match:
                    system_prompt = match.group(1)
                    input_str = match.group(2)

                if system_prompt and input_str:
                    messages.append({"role": "system", "content": system_prompt})
                    if images:
                        content = [{"type": "text", "text": input_str}]
                        if isinstance(images, (str, dict)):
                            images = [images]
                        for img in images:
                            content.append(
                                {"type": "image_url", "image_url": {"url": img, "detail": detail}}
                            )
                        messages.append({"role": "user", "content": content})
                    else:
                        messages.append({"role": "user", "content": input_str})

            if len(messages) == 0:
                if images:
                    content = [{"type": "text", "text": input}]
                    if isinstance(images, (str, dict)):
                        images = [images]
                    for img in images:
                        content.append(
                            {"type": "image_url", "image_url": {"url": img, "detail": detail}}
                        )
                    messages.append({"role": "user", "content": content})
                else:
                    messages.append({"role": "user", "content": input})

            final_model_kwargs["messages"] = messages

            # DeepSeek-specific: filter out deprecated parameters
            for deprecated in ["frequency_penalty", "presence_penalty"]:
                final_model_kwargs.pop(deprecated, None)

            # Handle thinking/reasoning mode
            thinking_config = final_model_kwargs.pop("thinking", None)
            reasoning_effort = final_model_kwargs.pop("reasoning_effort", None)

            # Build extra_body for thinking mode
            extra_body = final_model_kwargs.pop("extra_body", {}) or {}

            if thinking_config:
                extra_body["thinking"] = thinking_config
                if reasoning_effort:
                    extra_body["thinking"]["reasoning_effort"] = reasoning_effort

                # In thinking mode, temperature/top_p have no effect
                final_model_kwargs.pop("temperature", None)
                final_model_kwargs.pop("top_p", None)

            # Add user_id for KV cache isolation (both top-level and extra_body for SDK compat)
            if self._user_id:
                final_model_kwargs["user_id"] = self._user_id
                extra_body["user_id"] = self._user_id

            if extra_body:
                final_model_kwargs["extra_body"] = extra_body

            # Add stream_options for usage tracking in streaming mode
            if final_model_kwargs.get("stream"):
                final_model_kwargs["stream_options"] = {"include_usage": True}

            return final_model_kwargs

        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @backoff.on_exception(
        backoff.expo,
        (
            APITimeoutError,
            InternalServerError,
            RateLimitError,
            UnprocessableEntityError,
            BadRequestError,
        ),
        max_time=5,
    )
    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        log.info(f"api_kwargs: {api_kwargs}")
        self._api_kwargs = api_kwargs

        if model_type == ModelType.EMBEDDER:
            return self.sync_client.embeddings.create(**api_kwargs)
        elif model_type == ModelType.LLM:
            if "stream" in api_kwargs and api_kwargs.get("stream", False):
                log.debug("streaming call")
                return self.sync_client.chat.completions.create(**api_kwargs)
            else:
                log.debug("non-streaming call converted to streaming")
                streaming_kwargs = api_kwargs.copy()
                streaming_kwargs["stream"] = True

                stream_response = self.sync_client.chat.completions.create(
                    **streaming_kwargs
                )

                accumulated_content = ""
                id_val = ""
                model_val = ""
                created_val = 0
                finish_reason = "stop"

                for chunk in stream_response:
                    id_val = getattr(chunk, "id", None) or id_val
                    model_val = getattr(chunk, "model", None) or model_val
                    created_val = getattr(chunk, "created", 0) or created_val
                    choices = getattr(chunk, "choices", [])
                    if len(choices) > 0:
                        finish_reason = getattr(
                            choices[0], "finish_reason", finish_reason
                        ) or finish_reason
                        delta = getattr(choices[0], "delta", None)
                        if delta is not None:
                            text = getattr(delta, "content", None)
                            if text is not None:
                                accumulated_content += text or ""

                return ChatCompletion(
                    id=id_val,
                    model=model_val,
                    created=created_val,
                    object="chat.completion",
                    choices=[
                        Choice(
                            index=0,
                            finish_reason=finish_reason,
                            message=ChatCompletionMessage(
                                content=accumulated_content, role="assistant"
                            ),
                        )
                    ],
                )
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @backoff.on_exception(
        backoff.expo,
        (
            APITimeoutError,
            InternalServerError,
            RateLimitError,
            UnprocessableEntityError,
            BadRequestError,
        ),
        max_time=5,
    )
    async def acall(
        self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED
    ):
        self._api_kwargs = api_kwargs
        if self.async_client is None:
            self.async_client = self.init_async_client()
        if model_type == ModelType.EMBEDDER:
            return await self.async_client.embeddings.create(**api_kwargs)
        elif model_type == ModelType.LLM:
            return await self.async_client.chat.completions.create(**api_kwargs)
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @classmethod
    def from_dict(cls: type[T], data: Dict[str, Any]) -> T:
        obj = super().from_dict(data)
        obj.sync_client = obj.init_sync_client()
        obj.async_client = obj.init_async_client()
        return obj

    def to_dict(self) -> Dict[str, Any]:
        exclude = ["sync_client", "async_client"]
        output = super().to_dict(exclude=exclude)
        return output
