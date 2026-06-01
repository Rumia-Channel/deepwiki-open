# DeepWiki API

Backend API for DeepWiki-Open, providing AI-powered code analysis and wiki generation.

## Features

- **8 LLM Providers**: Google Gemini, OpenAI, OpenRouter, Ollama (local), AWS Bedrock, Azure AI, DashScope, DeepSeek
- **4 Embedding Backends**: OpenAI, Google, Ollama, AWS Bedrock
- **Streaming Chat**: Real-time responses via HTTP SSE (`/chat/completions/stream`) and WebSocket (`/ws/chat`)
- **Deep Research**: Multi-turn analysis (up to 5 iterations) for complex topics
- **Agent Loop**: Autonomous tool-calling (search codebase, read files, list repo) for DeepSeek and OpenAI models
- **RAG Pipeline**: FAISS retrieval + LLM generation with conversation memory
- **Wiki Cache**: Server-side caching of generated wikis with CRUD API
- **Comprehensive & Concise Modes**: Full multi-page wiki or single-page summary
- **10 Output Languages**: Documentation generation in 10 languages
- **Local Storage**: All data stored locally — repos, embeddings, and wiki cache
- **Authentication Mode**: Optional auth code to restrict frontend generation

## Quick Setup

### Step 1: Install Dependencies

```bash
# From the project root
uv sync
```

### Step 2: Configure Environment Variables

Create a `.env` file in the project root:

```
# Required
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key

# Optional — only needed for specific providers
OPENROUTER_API_KEY=your_openrouter_api_key
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
DEEPSEEK_API_KEY=your_deepseek_api_key
DASHSCOPE_API_KEY=your_dashscope_api_key

# Ollama host (default: http://localhost:11434)
OLLAMA_HOST=http://localhost:11434

# Embedding backend: openai (default), google, ollama, bedrock
DEEPWIKI_EMBEDDER_TYPE=openai

# Authentication mode (optional)
DEEPWIKI_AUTH_MODE=true
DEEPWIKI_AUTH_CODE=your_secret_code

# Server
PORT=8001
```

> **Where to get API keys:**
> - [Google AI Studio](https://makersuite.google.com/app/apikey)
> - [OpenAI Platform](https://platform.openai.com/api-keys)
> - [OpenRouter](https://openrouter.ai/keys)
> - [AWS IAM Console](https://console.aws.amazon.com/iam/)
> - [DeepSeek Platform](https://platform.deepseek.com/)
> - [Alibaba DashScope](https://dashscope.console.aliyun.com/)

If not using Ollama embeddings, you must configure an OpenAI API key for embeddings. Other API keys are only needed when using the corresponding provider.

### Step 3: Start the API Server

```bash
# From the project root
uv run python -m api.main
```

The API will be available at `http://localhost:8001`.

## How It Works

### 1. Repository Indexing

When a repository URL is provided:
- Clones the repository locally (or uses cached clone)
- Reads all files, applying inclusion/exclusion filters from `config/repo.json`
- Creates embeddings for file contents (text-embedding-3-small by default)
- Stores embeddings in a local FAISS database (`~/.adalflow/databases/`)

### 2. Smart Retrieval (RAG)

When a question is asked:
- The query is embedded and matched against the FAISS index
- Top-K relevant code snippets are retrieved (`top_k: 20`)
- Snippets are used as context for the LLM
- The LLM generates a grounded, context-aware response

### 3. Real-Time Streaming

Responses are streamed in real-time via:
- **HTTP SSE**: `POST /chat/completions/stream`
- **WebSocket**: `ws://localhost:8001/ws/chat`

The WebSocket endpoint additionally supports **Deep Research** multi-turn analysis.

### 4. Wiki Generation

The wiki generation pipeline:
1. User provides a repository URL and configuration (language, wiki type, provider, model)
2. Backend clones the repo and creates embeddings
3. Frontend requests wiki generation via the chat endpoint
4. LLM generates structured wiki pages with titles, content, file paths, and Mermaid diagrams
5. Wiki structure and pages are cached as JSON

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | API information and available endpoints |
| `GET` | `/health` | Health check |
| `POST` | `/chat/completions/stream` | HTTP SSE streaming chat |
| `WebSocket` | `/ws/chat` | WebSocket streaming chat + Deep Research |
| `GET` | `/models/config` | Available LLM providers and model configurations |
| `GET` | `/lang/config` | Supported output languages |
| `POST` | `/export/wiki` | Export wiki as JSON |
| `GET` | `/local_repo/structure` | Get local repository file structure |
| `GET` | `/api/wiki_cache` | Get cached wiki by owner/repo/language |
| `POST` | `/api/wiki_cache` | Save wiki structure and pages to cache |
| `DELETE` | `/api/wiki_cache` | Delete cached wiki |
| `GET` | `/api/processed_projects` | List all processed (cached) projects |
| `GET` | `/auth/status` | Check if authentication mode is enabled |
| `POST` | `/auth/validate` | Validate authentication code |

### HTTP SSE Chat

**`POST /chat/completions/stream`**

```json
{
  "repo_url": "https://github.com/username/repo",
  "messages": [
    {
      "role": "user",
      "content": "What does this repository do?"
    }
  ],
  "provider": "google",
  "model": "gemini-2.5-flash",
  "filePath": "optional/path/to/file.py",
  "token": "optional_access_token",
  "type": "github"
}
```

Response: Server-Sent Events stream with generated text.

### WebSocket Chat

**`ws://localhost:8001/ws/chat`**

Send a JSON message with the same structure as the HTTP endpoint. Supports Deep Research when `[DEEP RESEARCH]` is prepended to the user message.

## LLM Providers

Configuration is JSON-driven via `api/config/generator.json`:

| Provider | Default Model | Features | Key Env Var |
|---|---|---|---|
| `google` | `gemini-2.5-flash` | Fast, cost-effective | `GOOGLE_API_KEY` |
| `openai` | `gpt-5.4-mini` | Thinking/reasoning | `OPENAI_API_KEY` |
| `openrouter` | `openai/gpt-5-nano` | Multi-model proxy | `OPENROUTER_API_KEY` |
| `ollama` | `qwen3:1.7b` | Local, no API key | None |
| `bedrock` | `claude-sonnet-4-6` | Adaptive thinking | `AWS_ACCESS_KEY_ID` |
| `azure` | `gpt-4o` | Enterprise Azure | Azure credentials |
| `dashscope` | `qwen-plus` | Alibaba Qwen | `DASHSCOPE_API_KEY` |
| `deepseek` | `deepseek-v4-flash` | Thinking + tool calling | `DEEPSEEK_API_KEY` |

### Custom Model Selection

Users can select from predefined models or enter custom model IDs in the frontend. This is designed for service providers who want to offer model flexibility without code changes.

### Agent Loop (DeepSeek & OpenAI)

DeepSeek V4 and OpenAI GPT-5 models support an autonomous agent loop with three tools:

- `search_codebase` — FAISS semantic search across the repository
- `read_file` — Read a specific file from the cloned repository
- `list_repo_files` — List files matching a glob pattern

The agent loop runs up to 15 rounds, accumulating tool calls and responses.

## Embedding Backends

Controlled via `DEEPWIKI_EMBEDDER_TYPE`:

| Backend | Model | Dimensions | Batch Size |
|---|---|---|---|
| `openai` (default) | `text-embedding-3-small` | 256 | 500 |
| `google` | `gemini-embedding-001` | 768 | 100 |
| `ollama` | `nomic-embed-text` | 768 | Single-doc |
| `bedrock` | `titan-embed-text-v2` | 256 | 100 |

Configuration in `api/config/embedder.json`.

## Configuration Files

Located in `api/config/` (customizable via `DEEPWIKI_CONFIG_DIR`):

| File | Purpose |
|---|---|
| `generator.json` | LLM provider/model definitions, features, defaults |
| `embedder.json` | Embedding model, retriever (`top_k: 20`), text splitter (`chunk_size: 350`, `chunk_overlap: 100`) |
| `repo.json` | File exclusion/inclusion filters, repository size limits |
| `lang.json` | Supported output languages |

Environment variable placeholders (e.g., `${OPENAI_API_KEY}`) in config files are automatically resolved at startup.

## Storage

All data is stored locally:

| Path | Content |
|---|---|
| `~/.adalflow/repos/` | Cloned repositories |
| `~/.adalflow/databases/` | FAISS embeddings and indexes |
| `~/.adalflow/wikicache/` | Cached wiki structures and pages |

No cloud storage is used — everything runs on your machine.

## Deep Research

Deep Research is a multi-turn analysis mode. When `[DEEP RESEARCH]` is prepended to a user message, the system performs up to 5 iterations of research:

1. **Iteration 1**: Research plan — outlines approach and initial findings
2. **Iterations 2-4**: Deep dive — each round builds on previous insights
3. **Iteration 5**: Final conclusion — comprehensive answer synthesizing all findings

Each iteration uses specialized system prompts and accumulates context from previous rounds.

## Authentication Mode

Enable to require an auth code for wiki generation via the frontend:

```
DEEPWIKI_AUTH_MODE=true
DEEPWIKI_AUTH_CODE=your_secret_code
```

Note: This protects the frontend UI and cached wiki deletion, but does not prevent direct API access.

## Example Usage

```python
import requests

# HTTP SSE streaming
url = "http://localhost:8001/chat/completions/stream"
payload = {
    "repo_url": "https://github.com/AsyncFuncAI/deepwiki-open",
    "messages": [{"role": "user", "content": "Explain the architecture"}],
    "provider": "google"
}

response = requests.post(url, json=payload, stream=True)
for chunk in response.iter_content(chunk_size=None):
    if chunk:
        print(chunk.decode('utf-8'), end='', flush=True)
```
