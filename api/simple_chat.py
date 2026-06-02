import asyncio
import json
import logging
import os
from typing import List, Optional
from urllib.parse import unquote

from google import genai as google_genai
from adalflow.components.model_client.ollama_client import OllamaClient
from adalflow.core.types import ModelType
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.config import get_model_config, configs, OPENROUTER_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, WIKI_AUTH_MODE, WIKI_AUTH_CODE, GOOGLE_API_KEY
from api.data_pipeline import count_tokens, get_file_content
from api.openai_client import OpenAIClient
from api.openrouter_client import OpenRouterClient
from api.bedrock_client import BedrockClient
from api.azureai_client import AzureAIClient
from api.dashscope_client import DashscopeClient
from api.deepseek_client import DeepSeekClient
from api.agent_loop import run_agent_loop
from api.tools.agent_tools import ToolExecutor
from api.cag import cag_context
from api.prompts import (
    DEEP_RESEARCH_FIRST_ITERATION_PROMPT,
    DEEP_RESEARCH_FINAL_ITERATION_PROMPT,
    DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT,
    SIMPLE_CHAT_SYSTEM_PROMPT
)

# Configure logging
from api.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# Initialize FastAPI app
app = FastAPI(
    title="Simple Chat API",
    description="Simplified API for streaming chat completions"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=False,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Models for the API
class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class ChatCompletionRequest(BaseModel):
    """
    Model for requesting a chat completion.
    """
    repo_url: str = Field(..., description="URL of the repository to query")
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    filePath: Optional[str] = Field(None, description="Optional path to a file in the repository to include in the prompt")
    token: Optional[str] = Field(None, description="Personal access token for private repositories")
    type: Optional[str] = Field("github", description="Type of repository (e.g., 'github', 'gitlab', 'bitbucket')")

    # model parameters
    provider: str = Field("google", description="Model provider (google, openai, openrouter, ollama, bedrock, azure, dashscope, deepseek)")
    model: Optional[str] = Field(None, description="Model name for the specified provider")

    # thinking / reasoning parameters
    thinking_enabled: Optional[bool] = Field(None, description="Override thinking mode on/off")
    reasoning_effort: Optional[str] = Field(None, description="Override reasoning effort level")

    language: Optional[str] = Field("en", description="Language for content generation (e.g., 'en', 'ja', 'zh', 'es', 'kr', 'vi')")
    relevant_files: Optional[List[str]] = Field(None, description="List of relevant file paths from the wiki structure to include as CAG context")
    force_reclone: Optional[bool] = Field(False, description="Force re-clone of the repository before generating")
    excluded_dirs: Optional[str] = Field(None, description="Comma-separated list of directories to exclude from processing")
    excluded_files: Optional[str] = Field(None, description="Comma-separated list of file patterns to exclude from processing")
    included_dirs: Optional[str] = Field(None, description="Comma-separated list of directories to include exclusively")
    included_files: Optional[str] = Field(None, description="Comma-separated list of file patterns to include exclusively")
    authorization_code: Optional[str] = Field(None, description="Authorization code when auth mode is enabled")

@app.post("/chat/completions/stream")
async def chat_completions_stream(request: ChatCompletionRequest):
    """Stream a chat completion response directly using Google Generative AI"""
    try:
        # Validate auth if enabled
        if WIKI_AUTH_MODE and (not request.authorization_code or request.authorization_code != WIKI_AUTH_CODE):
            raise HTTPException(status_code=401, detail="Authorization required. Please provide a valid authorization code.")

        # Check if request contains very large input
        input_too_large = False
        if request.messages and len(request.messages) > 0:
            last_message = request.messages[-1]
            if hasattr(last_message, 'content') and last_message.content:
                tokens = count_tokens(last_message.content, request.provider == "ollama")
                logger.info(f"Request size: {tokens} tokens")
                if tokens > 8000:
                    logger.warning(f"Request exceeds recommended token limit ({tokens} > 7500)")
                    input_too_large = True

        # Validate request
        if not request.messages or len(request.messages) == 0:
            raise HTTPException(status_code=400, detail="No messages provided")

        last_message = request.messages[-1]
        if last_message.role != "user":
            raise HTTPException(status_code=400, detail="Last message must be from the user")

        # CAG: simple in-memory conversation history
        conversation_turns = []
        # Process previous messages to build conversation history
        for i in range(0, len(request.messages) - 1, 2):
            if i + 1 < len(request.messages):
                user_msg = request.messages[i]
                assistant_msg = request.messages[i + 1]
                if user_msg.role == "user" and assistant_msg.role == "assistant":
                    conversation_turns.append((user_msg.content, assistant_msg.content))

        # Check if this is a Deep Research request
        is_deep_research = False
        research_iteration = 1

        # Process messages to detect Deep Research requests
        for msg in request.messages:
            if hasattr(msg, 'content') and msg.content and "[DEEP RESEARCH]" in msg.content:
                is_deep_research = True
                # Only remove the tag from the last message
                if msg == request.messages[-1]:
                    # Remove the Deep Research tag
                    msg.content = msg.content.replace("[DEEP RESEARCH]", "").strip()

        # Count research iterations if this is a Deep Research request
        if is_deep_research:
            research_iteration = sum(1 for msg in request.messages if msg.role == 'assistant') + 1
            logger.info(f"Deep Research request detected - iteration {research_iteration}")

            # Check if this is a continuation request
            if "continue" in last_message.content.lower() and "research" in last_message.content.lower():
                # Find the original topic from the first user message
                original_topic = None
                for msg in request.messages:
                    if msg.role == "user" and "continue" not in msg.content.lower():
                        original_topic = msg.content.replace("[DEEP RESEARCH]", "").strip()
                        logger.info(f"Found original research topic: {original_topic}")
                        break

                if original_topic:
                    # Replace the continuation message with the original topic
                    last_message.content = original_topic
                    logger.info(f"Using original topic for research: {original_topic}")

        # Get the query from the last message
        query = last_message.content

        # CAG: Use shared full-repo context block (KV-cacheable across pages)
        context_text = ""

        if not input_too_large:
            try:
                context_text = cag_context.get_context_block(
                    request.repo_url, request.type, request.token,
                    force_reclone=request.force_reclone or False
                )
                if not context_text:
                    logger.warning("CAG: context block is empty")
            except Exception as e:
                logger.error(f"CAG: error building context: {str(e)}")
                context_text = ""

        # Get repository information
        repo_url = request.repo_url
        repo_name = repo_url.split("/")[-1] if "/" in repo_url else repo_url

        # Determine repository type
        repo_type = request.type

        # Get language information
        language_code = request.language or configs["lang_config"]["default"]
        supported_langs = configs["lang_config"]["supported_languages"]
        language_name = supported_langs.get(language_code, "English")

        # Create system prompt
        if is_deep_research:
            # Check if this is the first iteration
            is_first_iteration = research_iteration == 1

            # Check if this is the final iteration
            is_final_iteration = research_iteration >= 5

            if is_first_iteration:
                system_prompt = DEEP_RESEARCH_FIRST_ITERATION_PROMPT.format(
                    repo_type=repo_type,
                    repo_url=repo_url,
                    repo_name=repo_name,
                    language_name=language_name
                )
            elif is_final_iteration:
                system_prompt = DEEP_RESEARCH_FINAL_ITERATION_PROMPT.format(
                    repo_type=repo_type,
                    repo_url=repo_url,
                    repo_name=repo_name,
                    research_iteration=research_iteration,
                    language_name=language_name
                )
            else:
                system_prompt = DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT.format(
                    repo_type=repo_type,
                    repo_url=repo_url,
                    repo_name=repo_name,
                    research_iteration=research_iteration,
                    language_name=language_name
                )
        else:
            system_prompt = SIMPLE_CHAT_SYSTEM_PROMPT.format(
                repo_type=repo_type,
                repo_url=repo_url,
                repo_name=repo_name,
                language_name=language_name
            )

        # Fetch file content if provided
        file_content = ""
        if request.filePath:
            try:
                file_content = get_file_content(request.repo_url, request.filePath, request.type, request.token)
                logger.info(f"Successfully retrieved content for file: {request.filePath}")
            except Exception as e:
                logger.error(f"Error retrieving file content: {str(e)}")
                # Continue without file content if there's an error

        # Format conversation history
        conversation_history = ""
        for user_query, assistant_response in conversation_turns:
            conversation_history += f"<turn>\n<user>{user_query}</user>\n<assistant>{assistant_response}</assistant>\n</turn>\n"

        # Create the prompt with CAG context BEFORE conversation history
        # (so DeepSeek KV cache shares the system_prompt + context prefix across pages)
        CONTEXT_START = "<START_OF_CONTEXT>"
        CONTEXT_END = "<END_OF_CONTEXT>"

        prompt = f"/no_think {system_prompt}\n\n"

        # CAG context block: placed immediately after system prompt for KV cache sharing
        if context_text.strip():
            prompt += f"{CONTEXT_START}\n{context_text}\n{CONTEXT_END}\n\n"
        else:
            logger.info("No CAG context available")
            prompt += "<note>Generating without file context.</note>\n\n"

        # Conversation history comes AFTER the cached prefix
        if conversation_history:
            prompt += f"<conversation_history>\n{conversation_history}</conversation_history>\n\n"

        # Check if filePath is provided and fetch file content if it exists
        if file_content:
            prompt += f"<currentFileContent path=\"{request.filePath}\">\n{file_content}\n</currentFileContent>\n\n"

        prompt += f"<query>\n{query}\n</query>\n\nAssistant: "

        model_config = get_model_config(request.provider, request.model)["model_kwargs"]
        # Apply runtime thinking overrides from request
        if request.thinking_enabled is not None:
            model_config["thinking"] = request.thinking_enabled
        elif request.thinking_enabled is False:
            model_config.pop("thinking", None)
        if request.reasoning_effort:
            model_config["reasoning_effort"] = request.reasoning_effort
        tool_executor = None  # for DeepSeek agent loop
        openai_tool_executor = None  # for OpenAI GPT-5 agent loop
        repo_cache_path = cag_context._repos.get(request.repo_url)

        if request.provider == "ollama":
            prompt += " /no_think"

            model = OllamaClient()
            model_kwargs = {
                "model": model_config["model"],
                "stream": True,
                "options": {
                    "temperature": model_config["temperature"],
                    "top_p": model_config["top_p"],
                    "num_ctx": model_config["num_ctx"]
                }
            }

            api_kwargs = model.convert_inputs_to_api_kwargs(
                input=prompt,
                model_kwargs=model_kwargs,
                model_type=ModelType.LLM
            )
        elif request.provider == "openrouter":
            logger.info(f"Using OpenRouter with model: {request.model}")

            # Check if OpenRouter API key is set
            if not OPENROUTER_API_KEY:
                logger.warning("OPENROUTER_API_KEY not configured, but continuing with request")
                # We'll let the OpenRouterClient handle this and return a friendly error message

            model = OpenRouterClient()
            model_kwargs = {
                "model": request.model,
                "stream": True,
                "temperature": model_config["temperature"]
            }
            # Only add top_p if it exists in the model config
            if "top_p" in model_config:
                model_kwargs["top_p"] = model_config["top_p"]

            api_kwargs = model.convert_inputs_to_api_kwargs(
                input=prompt,
                model_kwargs=model_kwargs,
                model_type=ModelType.LLM
            )
        elif request.provider == "openai":
            logger.info(f"Using Openai protocol with model: {request.model}")

            # Check if an API key is set for Openai
            if not OPENAI_API_KEY:
                logger.warning("OPENAI_API_KEY not configured, but continuing with request")
                # We'll let the OpenAIClient handle this and return an error message

            # Initialize Openai client
            model = OpenAIClient()
            model_kwargs = {
                "model": request.model,
                "stream": True,
                "temperature": model_config.get("temperature", 0.7),
            }
            for key in ["top_p", "max_tokens", "reasoning_effort"]:
                if key in model_config:
                    model_kwargs[key] = model_config[key]

            api_kwargs = model.convert_inputs_to_api_kwargs(
                input=prompt,
                model_kwargs=model_kwargs,
                model_type=ModelType.LLM
            )
            # Create tool executor for agent loop (GPT-5 reasoning models)
            openai_tool_executor = ToolExecutor(
                rag_instance=None,
                repo_cache_path=repo_cache_path
            ) if "reasoning_effort" in model_config else None
        elif request.provider == "bedrock":
            logger.info(f"Using AWS Bedrock with model: {request.model}")

            # Check if AWS credentials are set
            if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
                logger.warning("AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY not configured, but continuing with request")
                # We'll let the BedrockClient handle this and return an error message

            # Initialize Bedrock client
            model = BedrockClient()
            model_kwargs = {
                "model": request.model,
            }
            for key in ["temperature", "top_p", "max_tokens"]:
                if key in model_config:
                    model_kwargs[key] = model_config[key]
            # Claude extended thinking
            if "thinking" in model_config:
                model_kwargs["thinking"] = model_config["thinking"]

            api_kwargs = model.convert_inputs_to_api_kwargs(
                input=prompt,
                model_kwargs=model_kwargs,
                model_type=ModelType.LLM
            )
        elif request.provider == "azure":
            logger.info(f"Using Azure AI with model: {request.model}")

            # Initialize Azure AI client
            model = AzureAIClient()
            model_kwargs = {
                "model": request.model,
                "stream": True,
                "temperature": model_config["temperature"],
                "top_p": model_config["top_p"]
            }

            api_kwargs = model.convert_inputs_to_api_kwargs(
                input=prompt,
                model_kwargs=model_kwargs,
                model_type=ModelType.LLM
            )
        elif request.provider == "dashscope":
            logger.info(f"Using Dashscope with model: {request.model}")

            model = DashscopeClient()
            model_kwargs = {
                "model": request.model,
                "stream": True,
                "temperature": model_config["temperature"],
                "top_p": model_config["top_p"],
            }

            api_kwargs = model.convert_inputs_to_api_kwargs(
                input=prompt,
                model_kwargs=model_kwargs,
                model_type=ModelType.LLM,
            )
        elif request.provider == "deepseek":
            logger.info(f"Using DeepSeek with model: {request.model}")

            if not DEEPSEEK_API_KEY:
                logger.warning("DEEPSEEK_API_KEY not configured")

            model = DeepSeekClient()
            model_kwargs = {
                "model": request.model,
                "stream": True,
                "temperature": model_config.get("temperature", 0.7),
            }
            if "top_p" in model_config and "thinking" not in model_config:
                model_kwargs["top_p"] = model_config["top_p"]
            if "max_tokens" in model_config:
                model_kwargs["max_tokens"] = model_config["max_tokens"]

            # DeepSeek thinking mode
            if model_config.get("thinking"):
                thinking_body = {"type": "enabled"}
                if model_config.get("reasoning_effort"):
                    thinking_body["reasoning_effort"] = model_config["reasoning_effort"]
                model_kwargs["thinking"] = thinking_body
                model_kwargs.pop("temperature", None)
                model_kwargs.pop("top_p", None)

            api_kwargs = model.convert_inputs_to_api_kwargs(
                input=prompt,
                model_kwargs=model_kwargs,
                model_type=ModelType.LLM,
            )
            # Create tool executor for agent loop
            tool_executor = ToolExecutor(
                rag_instance=None,
                repo_cache_path=repo_cache_path
            )
        else:
            # Google provider — use new google-genai SDK
            google_client = google_genai.Client(api_key=GOOGLE_API_KEY)

        # Create a streaming response
        async def response_stream():
            try:
                if request.provider == "ollama":
                    # Get the response and handle it properly using the previously created api_kwargs
                    response = await model.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
                    # Handle streaming response from Ollama
                    async for chunk in response:
                        text = getattr(chunk, 'response', None) or getattr(chunk, 'text', None) or str(chunk)
                        if text and not text.startswith('model=') and not text.startswith('created_at='):
                            text = text.replace('<think>', '').replace('</think>', '')
                            yield text
                elif request.provider == "openrouter":
                    try:
                        # Get the response and handle it properly using the previously created api_kwargs
                        logger.info("Making OpenRouter API call")
                        response = await model.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
                        # Handle streaming response from OpenRouter
                        async for chunk in response:
                            yield chunk
                    except Exception as e_openrouter:
                        logger.error(f"Error with OpenRouter API: {str(e_openrouter)}")
                        yield f"\nError with OpenRouter API: {str(e_openrouter)}\n\nPlease check that you have set the OPENROUTER_API_KEY environment variable with a valid API key."
                elif request.provider == "openai":
                    try:
                        if openai_tool_executor and "reasoning_effort" in model_kwargs:
                            logger.info("Starting OpenAI GPT-5 agent loop")
                            async for chunk in run_agent_loop(model, api_kwargs, openai_tool_executor):
                                yield chunk
                        else:
                            logger.info("Making OpenAI API call")
                            response = await model.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
                            async for chunk in response:
                               choices = getattr(chunk, "choices", [])
                               if len(choices) > 0:
                                   delta = getattr(choices[0], "delta", None)
                                   if delta is not None:
                                        text = getattr(delta, "content", None)
                                        if text is not None:
                                            yield text
                    except Exception as e_openai:
                        logger.error(f"Error with OpenAI API: {str(e_openai)}")
                        yield f"\nError with OpenAI API: {str(e_openai)}\n\nPlease check that you have set the OPENAI_API_KEY environment variable with a valid API key."
                elif request.provider == "bedrock":
                    try:
                        # Get the response and handle it properly using the previously created api_kwargs
                        logger.info("Making AWS Bedrock API call")
                        response = await model.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
                        # Handle response from Bedrock (not streaming yet)
                        if isinstance(response, str):
                            yield response
                        else:
                            # Try to extract text from the response
                            yield str(response)
                    except Exception as e_bedrock:
                        logger.error(f"Error with AWS Bedrock API: {str(e_bedrock)}")
                        yield f"\nError with AWS Bedrock API: {str(e_bedrock)}\n\nPlease check that you have set the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables with valid credentials."
                elif request.provider == "azure":
                    try:
                        # Get the response and handle it properly using the previously created api_kwargs
                        logger.info("Making Azure AI API call")
                        response = await model.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
                        # Handle streaming response from Azure AI
                        async for chunk in response:
                            choices = getattr(chunk, "choices", [])
                            if len(choices) > 0:
                                delta = getattr(choices[0], "delta", None)
                                if delta is not None:
                                    text = getattr(delta, "content", None)
                                    if text is not None:
                                        yield text
                    except Exception as e_azure:
                        logger.error(f"Error with Azure AI API: {str(e_azure)}")
                        yield f"\nError with Azure AI API: {str(e_azure)}\n\nPlease check that you have set the AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_VERSION environment variables with valid values."
                elif request.provider == "dashscope":
                    try:
                        logger.info("Making Dashscope API call")
                        response = await model.acall(
                            api_kwargs=api_kwargs, model_type=ModelType.LLM
                        )
                        async for text in response:
                            if text:
                                yield text
                    except Exception as e_dashscope:
                        logger.error(f"Error with Dashscope API: {str(e_dashscope)}")
                        yield (
                            f"\nError with Dashscope API: {str(e_dashscope)}\n\n"
                            "Please check that you have set the DASHSCOPE_API_KEY (and optionally "
                            "DASHSCOPE_WORKSPACE_ID) environment variables with valid values."
                        )
                elif request.provider == "deepseek":
                    try:
                        logger.info("Streaming DeepSeek response (no agent loop)")
                        from api.deepseek_client import parse_stream_response_for_deepseek
                        response = await model.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
                        async for chunk in response:
                            text = parse_stream_response_for_deepseek(chunk)
                            if text:
                                yield text
                    except Exception as e_deepseek:
                        logger.error(f"Error with DeepSeek API: {str(e_deepseek)}")
                        yield f"\nError with DeepSeek API: {str(e_deepseek)}\n\nPlease check that you have set the DEEPSEEK_API_KEY environment variable with a valid API key."
                else:
                    # Google provider — use new google-genai SDK
                    response = google_client.models.generate_content_stream(
                        model=model_config["model"],
                        contents=prompt,
                        config=google_genai.types.GenerateContentConfig(
                            temperature=model_config.get("temperature", 1.0),
                            top_p=model_config.get("top_p", 0.8),
                            top_k=model_config.get("top_k", 20),
                        ),
                    )
                    for chunk in response:
                        if chunk.text:
                            yield chunk.text

            except Exception as e_outer:
                logger.error(f"Error in streaming response: {str(e_outer)}")
                error_message = str(e_outer)

                # Check for token limit errors
                if "maximum context length" in error_message or "token limit" in error_message or "too many tokens" in error_message:
                    # If we hit a token limit error, try again without context
                    logger.warning("Token limit exceeded, retrying without context")
                    try:
                        # Create a simplified prompt without context
                        simplified_prompt = f"/no_think {system_prompt}\n\n"
                        if conversation_history:
                            simplified_prompt += f"<conversation_history>\n{conversation_history}</conversation_history>\n\n"

                        # Include file content in the fallback prompt if it was retrieved
                        if request.filePath and file_content:
                            simplified_prompt += f"<currentFileContent path=\"{request.filePath}\">\n{file_content}\n</currentFileContent>\n\n"

                        simplified_prompt += "<note>Generating without file context due to input size constraints.</note>\n\n"
                        simplified_prompt += f"<query>\n{query}\n</query>\n\nAssistant: "

                        if request.provider == "ollama":
                            simplified_prompt += " /no_think"

                            # Create new api_kwargs with the simplified prompt
                            fallback_api_kwargs = model.convert_inputs_to_api_kwargs(
                                input=simplified_prompt,
                                model_kwargs=model_kwargs,
                                model_type=ModelType.LLM
                            )

                            # Get the response using the simplified prompt
                            fallback_response = await model.acall(api_kwargs=fallback_api_kwargs, model_type=ModelType.LLM)

                            # Handle streaming fallback_response from Ollama
                            async for chunk in fallback_response:
                                text = getattr(chunk, 'response', None) or getattr(chunk, 'text', None) or str(chunk)
                                if text and not text.startswith('model=') and not text.startswith('created_at='):
                                    text = text.replace('<think>', '').replace('</think>', '')
                                    yield text
                        elif request.provider == "openrouter":
                            try:
                                # Create new api_kwargs with the simplified prompt
                                fallback_api_kwargs = model.convert_inputs_to_api_kwargs(
                                    input=simplified_prompt,
                                    model_kwargs=model_kwargs,
                                    model_type=ModelType.LLM
                                )

                                # Get the response using the simplified prompt
                                logger.info("Making fallback OpenRouter API call")
                                fallback_response = await model.acall(api_kwargs=fallback_api_kwargs, model_type=ModelType.LLM)

                                # Handle streaming fallback_response from OpenRouter
                                async for chunk in fallback_response:
                                    yield chunk
                            except Exception as e_fallback:
                                logger.error(f"Error with OpenRouter API fallback: {str(e_fallback)}")
                                yield f"\nError with OpenRouter API fallback: {str(e_fallback)}\n\nPlease check that you have set the OPENROUTER_API_KEY environment variable with a valid API key."
                        elif request.provider == "openai":
                            try:
                                fallback_api_kwargs = model.convert_inputs_to_api_kwargs(
                                    input=simplified_prompt,
                                    model_kwargs=model_kwargs,
                                    model_type=ModelType.LLM
                                )

                                if openai_tool_executor and "reasoning_effort" in model_kwargs:
                                    logger.info("Making fallback OpenAI agent call")
                                    async for chunk in run_agent_loop(model, fallback_api_kwargs, openai_tool_executor):
                                        yield chunk
                                else:
                                    logger.info("Making fallback OpenAI API call")
                                    fallback_response = await model.acall(api_kwargs=fallback_api_kwargs, model_type=ModelType.LLM)
                                    async for chunk in fallback_response:
                                        text = chunk if isinstance(chunk, str) else getattr(chunk, 'text', str(chunk))
                                        yield text
                            except Exception as e_fallback:
                                logger.error(f"Error with Openai API fallback: {str(e_fallback)}")
                                yield f"\nError with Openai API fallback: {str(e_fallback)}\n\nPlease check that you have set the OPENAI_API_KEY environment variable with a valid API key."
                        elif request.provider == "bedrock":
                            try:
                                # Create new api_kwargs with the simplified prompt
                                fallback_api_kwargs = model.convert_inputs_to_api_kwargs(
                                    input=simplified_prompt,
                                    model_kwargs=model_kwargs,
                                    model_type=ModelType.LLM
                                )

                                # Get the response using the simplified prompt
                                logger.info("Making fallback AWS Bedrock API call")
                                fallback_response = await model.acall(api_kwargs=fallback_api_kwargs, model_type=ModelType.LLM)

                                # Handle response from Bedrock
                                if isinstance(fallback_response, str):
                                    yield fallback_response
                                else:
                                    # Try to extract text from the response
                                    yield str(fallback_response)
                            except Exception as e_fallback:
                                logger.error(f"Error with AWS Bedrock API fallback: {str(e_fallback)}")
                                yield f"\nError with AWS Bedrock API fallback: {str(e_fallback)}\n\nPlease check that you have set the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables with valid credentials."
                        elif request.provider == "azure":
                            try:
                                # Create new api_kwargs with the simplified prompt
                                fallback_api_kwargs = model.convert_inputs_to_api_kwargs(
                                    input=simplified_prompt,
                                    model_kwargs=model_kwargs,
                                    model_type=ModelType.LLM
                                )

                                # Get the response using the simplified prompt
                                logger.info("Making fallback Azure AI API call")
                                fallback_response = await model.acall(api_kwargs=fallback_api_kwargs, model_type=ModelType.LLM)

                                # Handle streaming fallback response from Azure AI
                                async for chunk in fallback_response:
                                    choices = getattr(chunk, "choices", [])
                                    if len(choices) > 0:
                                        delta = getattr(choices[0], "delta", None)
                                        if delta is not None:
                                            text = getattr(delta, "content", None)
                                            if text is not None:
                                                yield text
                            except Exception as e_fallback:
                                logger.error(f"Error with Azure AI API fallback: {str(e_fallback)}")
                                yield f"\nError with Azure AI API fallback: {str(e_fallback)}\n\nPlease check that you have set the AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_VERSION environment variables with valid values."
                        elif request.provider == "dashscope":
                            try:
                                # Create new api_kwargs with the simplified prompt
                                fallback_api_kwargs = model.convert_inputs_to_api_kwargs(
                                    input=simplified_prompt,
                                    model_kwargs=model_kwargs,
                                    model_type=ModelType.LLM,
                                )

                                logger.info("Making fallback Dashscope API call")
                                fallback_response = await model.acall(
                                    api_kwargs=fallback_api_kwargs, model_type=ModelType.LLM
                                )

                                # DashscopeClient.acall (stream=True) returns an async
                                # generator of text chunks
                                async for text in fallback_response:
                                    if text:
                                        yield text
                            except Exception as e_fallback:
                                logger.error(
                                    f"Error with Dashscope API fallback: {str(e_fallback)}"
                                )
                                yield (
                                    f"\nError with Dashscope API fallback: {str(e_fallback)}\n\n"
                                    "Please check that you have set the DASHSCOPE_API_KEY (and optionally "
                                    "DASHSCOPE_WORKSPACE_ID) environment variables with valid values."
                                )
                        elif request.provider == "deepseek":
                            try:
                                fallback_api_kwargs = model.convert_inputs_to_api_kwargs(
                                    input=simplified_prompt,
                                    model_kwargs=model_kwargs,
                                    model_type=ModelType.LLM,
                                )

                                logger.info("Making fallback DeepSeek agent call")
                                async for chunk in run_agent_loop(model, fallback_api_kwargs, tool_executor):
                                    yield chunk
                            except Exception as e_fallback:
                                logger.error(f"Error with DeepSeek API fallback: {str(e_fallback)}")
                                yield f"\nError with DeepSeek API fallback: {str(e_fallback)}\n\nPlease check that you have set the DEEPSEEK_API_KEY environment variable with a valid API key."
                        else:
                            # Google provider fallback — use new google-genai SDK
                            fallback_config = get_model_config(request.provider, request.model)["model_kwargs"]
                            fallback_response = google_client.models.generate_content_stream(
                                model=fallback_config["model"],
                                contents=simplified_prompt,
                                config=google_genai.types.GenerateContentConfig(
                                    temperature=fallback_config.get("temperature", 0.7),
                                    top_p=fallback_config.get("top_p", 0.8),
                                    top_k=fallback_config.get("top_k", 40),
                                ),
                            )
                            for chunk in fallback_response:
                                if chunk.text:
                                    yield chunk.text
                    except Exception as e2:
                        logger.error(f"Error in fallback streaming response: {str(e2)}")
                        yield f"\nI apologize, but your request is too large for me to process. Please try a shorter query or break it into smaller parts."
                else:
                    # For other errors, return the error message
                    yield f"\nError: {error_message}"

        # Return streaming response
        return StreamingResponse(response_stream(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e_handler:
        error_msg = f"Error in streaming chat completion: {str(e_handler)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

# --- Batch page generation endpoint (CAG + SSE streaming) ---

class BatchPageItem(BaseModel):
    page_id: str = Field(..., description="Unique page identifier")
    prompt_content: str = Field(..., description="The complete prompt content for this page")

class BatchPageRequest(BaseModel):
    """Request model for batch wiki page generation."""
    repo_url: str = Field(..., description="URL of the repository")
    type: Optional[str] = Field("github", description="Repository type")
    token: Optional[str] = Field(None, description="Access token for private repos")
    provider: str = Field("deepseek", description="Model provider")
    model: Optional[str] = Field(None, description="Model name")
    thinking_enabled: Optional[bool] = Field(None)
    reasoning_effort: Optional[str] = Field(None)
    force_reclone: Optional[bool] = Field(False)
    pages: List[BatchPageItem] = Field(..., description="List of pages to generate")

@app.post("/chat/batch")
async def chat_batch(request: BatchPageRequest):
    """Generate multiple wiki pages in parallel via SSE streaming.

    Each page is generated independently with the shared CAG context.
    Results are streamed back as SSE events tagged with page_id.
    """
    if not request.pages:
        raise HTTPException(status_code=400, detail="No pages provided")

    semaphore = asyncio.Semaphore(min(len(request.pages), _BATCH_MAX_CONCURRENT))

    # Pre-warm CAG context (clone/build once for all pages)
    context_text = ""
    try:
        context_text = cag_context.get_context_block(
            request.repo_url, request.type, request.token,
            force_reclone=request.force_reclone or False
        )
    except Exception as e:
        logger.error(f"Batch CAG error: {e}")

    model_config = get_model_config(request.provider, request.model)

    async def generate_page(page: BatchPageItem) -> List[str]:
        """Generate one page, returning SSE-formatted chunks."""
        async with semaphore:
            chunks_to_send: List[str] = []
            try:
                system_prompt = (
                    "You are an expert technical writer and software architect.\n"
                )
                CONTEXT_START = "<START_OF_CONTEXT>"
                CONTEXT_END = "<END_OF_CONTEXT>"

                prompt = f"/no_think {system_prompt}\n\n"
                if context_text.strip():
                    prompt += f"{CONTEXT_START}\n{context_text}\n{CONTEXT_END}\n\n"
                prompt += f"<query>\n{page.prompt_content}\n</query>\n\nAssistant: "

                model_kwargs = {**model_config["model_kwargs"]}
                if request.thinking_enabled is not None:
                    model_kwargs["thinking"] = request.thinking_enabled
                if request.reasoning_effort:
                    model_kwargs["reasoning_effort"] = request.reasoning_effort
                model_kwargs["stream"] = True
                model_kwargs["model"] = request.model

                if request.provider == "deepseek":
                    from api.deepseek_client import DeepSeekClient, parse_stream_response_for_deepseek
                    if model_kwargs.get("thinking"):
                        thinking_body = {"type": "enabled"}
                        if model_kwargs.get("reasoning_effort"):
                            thinking_body["reasoning_effort"] = model_kwargs["reasoning_effort"]
                        model_kwargs["thinking"] = thinking_body
                        model_kwargs.pop("temperature", None)
                        model_kwargs.pop("top_p", None)
                    client = DeepSeekClient()
                elif request.provider == "openai":
                    from api.openai_client import OpenAIClient
                    client = OpenAIClient()
                else:
                    # Fallback to SimpleChat-style streaming
                    from api.openai_client import OpenAIClient
                    client = OpenAIClient()

                api_kwargs = client.convert_inputs_to_api_kwargs(
                    input=prompt,
                    model_kwargs=model_kwargs,
                    model_type=ModelType.LLM
                )

                response = await client.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)

                if request.provider == "deepseek":
                    from api.deepseek_client import parse_stream_response_for_deepseek
                    async for chunk in response:
                        text = parse_stream_response_for_deepseek(chunk)
                        if text:
                            chunks_to_send.append(
                                f"data: {json.dumps({'page_id': page.page_id, 'chunk': text})}\n\n"
                            )
                else:
                    async for chunk in response:
                        choices = getattr(chunk, "choices", [])
                        if len(choices) > 0:
                            delta = getattr(choices[0], "delta", None)
                            if delta is not None:
                                text = getattr(delta, "content", None)
                                if text is not None:
                                    chunks_to_send.append(
                                        f"data: {json.dumps({'page_id': page.page_id, 'chunk': text})}\n\n"
                                    )

            except Exception as e:
                logger.error(f"Batch page {page.page_id} error: {e}")
                chunks_to_send.append(
                    f"data: {json.dumps({'page_id': page.page_id, 'error': str(e)})}\n\n"
                )

            # Signal page completion
            chunks_to_send.append(
                f"data: {json.dumps({'page_id': page.page_id, 'done': True})}\n\n"
            )
            return chunks_to_send

    async def event_stream():
        tasks = [generate_page(page) for page in request.pages]
        # Process results as they complete
        for coro in asyncio.as_completed(tasks):
            chunks = await coro
            for c in chunks:
                yield c

    response_headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=response_headers,
    )

@app.get("/")
async def root():
    """Root endpoint to check if the API is running"""
    return {"status": "API is running", "message": "Navigate to /docs for API documentation"}
