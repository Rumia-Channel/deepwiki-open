"""Agent tools for DeepSeek V4 thinking mode with function calling.

Exposes repository analysis tools that the model can autonomously invoke:
  - search_codebase: FAISS semantic search across embedded documents
  - read_file: Read file contents from the cached repository
  - list_repo_files: List files matching a glob pattern

Each tool returns a JSON-serializable result. Output is capped at 32KB
(per Reasonix: maxToolOutputBytes = 32*1024) to prevent a single tool
call from blowing the context window.
"""

import logging
from typing import List, Dict, Any, Optional, Callable

log = logging.getLogger(__name__)

# Maximum tool output bytes before head+tail truncation.
# ~32KB ≈ 8K tokens — enough for a full file read, while preventing one
# accidental large file read from blowing the context window.
_MAX_TOOL_OUTPUT = 32 * 1024


def _truncate_output(text: str) -> str:
    """Head+tail truncate when exceeding _MAX_TOOL_OUTPUT."""
    if len(text) <= _MAX_TOOL_OUTPUT:
        return text
    keep = _MAX_TOOL_OUTPUT // 2
    head = text[:keep]
    tail = text[-keep:]
    omitted = len(text) - len(head) - len(tail)
    return head + f"\n\n…[truncated {omitted} of {len(text)} bytes]…\n\n" + tail

# Tool definitions in OpenAI/DeepSeek function-calling format
AGENT_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_codebase",
            "description": (
                "Semantically search the codebase for code chunks relevant to a query. "
                "Uses FAISS vector search over embedded document chunks. "
                "Returns the most relevant code/document chunks with file paths and content. "
                "Use this to find implementations, understand architecture, or locate specific features."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query describing what code or documentation to find."
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the full content of a specific file from the repository. "
                "Use this when you need to examine a file's complete implementation, "
                "not just the chunk fragments returned by search_codebase."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file within the repository, e.g. 'src/main.py' or 'api/config.py'."
                    }
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_repo_files",
            "description": (
                "List files in the repository matching a glob pattern. "
                "Use this to discover the project structure, find files by naming patterns, "
                "or understand the directory layout."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match files, e.g. '**/*.py' or 'src/**/*.ts'. Defaults to '**/*' to list all files."
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
]


class ToolExecutor:
    """Executes agent tool calls against the repository context."""

    def __init__(self, rag_instance=None, repo_cache_path: Optional[str] = None):
        self._rag = rag_instance
        self._repo_path = repo_cache_path

    def set_repo_path(self, path: str):
        self._repo_path = path

    def set_rag(self, rag_instance):
        self._rag = rag_instance

    def execute(self, tool_call) -> str:
        """Execute a single tool call and return the result as a string."""
        name = tool_call.function.name
        try:
            import json
            args = json.loads(tool_call.function.arguments)
        except Exception:
            args = {}

        handlers: Dict[str, Callable] = {
            "search_codebase": self._search_codebase,
            "read_file": self._read_file,
            "list_repo_files": self._list_repo_files,
        }

        handler = handlers.get(name)
        if not handler:
            return f"Unknown tool: {name}"

        try:
            result = handler(**args)
            result = _truncate_output(result)
            log.info(f"Tool {name} executed successfully ({len(result)} chars)")
            return result
        except Exception as e:
            log.error(f"Tool {name} failed: {e}")
            return f"Error executing {name}: {str(e)}"

    def _search_codebase(self, query: str) -> str:
        """FAISS semantic search."""
        if not self._rag or not self._rag.retriever:
            return "Error: retriever not initialized. Repository must be loaded first."

        results = self._rag.retriever(query)
        if not results or not results[0].doc_indices:
            return "No matching documents found."

        docs = [
            self._rag.transformed_docs[idx]
            for idx in results[0].doc_indices
        ]

        output_parts = []
        for i, doc in enumerate(docs[:10]):
            file_path = doc.meta_data.get("file_path", "unknown")
            output_parts.append(
                f"[{i}] {file_path}\n```\n{doc.text[:2000]}\n```"
            )

        return "\n\n".join(output_parts) if output_parts else "No results."

    def _read_file(self, file_path: str) -> str:
        """Read file content from the cached repository."""
        import os

        if not self._repo_path:
            return "Error: repository path not set."

        full_path = os.path.join(self._repo_path, file_path)
        if not os.path.exists(full_path):
            # Try searching for the file in the repo
            for root, _, files in os.walk(self._repo_path):
                for f in files:
                    if f == os.path.basename(file_path) or file_path in os.path.join(root, f):
                        full_path = os.path.join(root, f)
                        break
                else:
                    continue
                break
            else:
                return f"File not found: {file_path}"

        if not os.path.isfile(full_path):
            return f"Not a file: {file_path}"

        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            # Truncate if too large
            if len(content) > 16000:
                content = content[:16000] + "\n... [truncated]"
            return content
        except Exception as e:
            return f"Error reading {file_path}: {str(e)}"

    def _list_repo_files(self, pattern: str = "**/*") -> str:
        """List files matching a glob pattern."""
        import os
        import fnmatch

        if not self._repo_path:
            return "Error: repository path not set."

        matches = []
        for root, dirs, files in os.walk(self._repo_path):
            # Skip hidden dirs and common exclusions
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__", ".git", "venv", ".venv", "dist", "build")]

            rel_root = os.path.relpath(root, self._repo_path)
            if rel_root == ".":
                rel_root = ""

            for fname in files:
                rel_path = os.path.join(rel_root, fname) if rel_root else fname
                rel_path = rel_path.replace("\\", "/")
                if fnmatch.fnmatch(rel_path, pattern):
                    matches.append(rel_path)
                    if len(matches) >= 200:
                        break
            if len(matches) >= 200:
                break

        if not matches:
            return f"No files matching '{pattern}' found."

        result = "\n".join(sorted(matches)[:200])
        if len(matches) > 200:
            result += f"\n... and {len(matches) - 200} more files"
        return result
