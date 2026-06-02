"""CAG (Cache-Augmented Generation) module.

Replaces RAG/embedding with direct file context feeding.
DeepSeek's KV cache auto-caches the file prefix for subsequent requests.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from api.data_pipeline import download_repo, _extract_repo_name_from_url

logger = logging.getLogger(__name__)

# Maximum size of a single file to include in context (bytes)
MAX_FILE_SIZE_BYTES = 200 * 1024  # 200KB
# Maximum total context size to include (chars)
MAX_TOTAL_CONTEXT_CHARS = 400_000


class CAGContext:
    """Manages repo cloning and file reading for CAG-based wiki generation."""

    def __init__(self):
        self._repos: Dict[str, str] = {}  # repo_id -> local_path

    def clone_repo(
        self,
        repo_url_or_path: str,
        repo_type: str = "github",
        access_token: Optional[str] = None,
    ) -> str:
        """Clone/download a repo and return the local path."""
        from api.data_pipeline import _extract_repo_name_from_url, DatabaseManager

        db = DatabaseManager()
        repo_paths = db.prepare_repo_paths(repo_url_or_path, repo_type)
        save_repo_dir = repo_paths.get("save_repo_dir", "")

        if save_repo_dir and os.path.isdir(save_repo_dir) and os.listdir(save_repo_dir):
            logger.info(f"Repo already cloned at {save_repo_dir}")
            self._repos[repo_url_or_path] = save_repo_dir
            return save_repo_dir

        try:
            local_path = download_repo(repo_url_or_path, save_repo_dir, repo_type, access_token)
            self._repos[repo_url_or_path] = local_path
            logger.info(f"Repo cloned to {local_path}")
            return local_path
        except Exception as e:
            logger.error(f"Failed to clone repo: {e}")
            raise

    def read_files(
        self,
        repo_url_or_path: str,
        file_paths: List[str],
    ) -> Dict[str, str]:
        """Read specified files from a cloned repo. Returns {file_path: content}."""
        local_path = self._repos.get(repo_url_or_path)
        if not local_path or not os.path.isdir(local_path):
            raise ValueError(f"Repo not cloned yet: {repo_url_or_path}")

        result = {}
        total_chars = 0
        for fp in file_paths:
            if total_chars >= MAX_TOTAL_CONTEXT_CHARS:
                logger.warning(f"Reached max context size ({MAX_TOTAL_CONTEXT_CHARS} chars), stopping file read")
                break

            full_path = Path(local_path) / fp.lstrip("/")
            if not full_path.is_file():
                logger.warning(f"File not found: {full_path}")
                continue

            try:
                file_size = full_path.stat().st_size
                if file_size > MAX_FILE_SIZE_BYTES:
                    logger.warning(f"File too large ({file_size}B > {MAX_FILE_SIZE_BYTES}B): {fp}")
                    continue

                content = full_path.read_text(encoding="utf-8", errors="replace")
                result[fp] = content
                total_chars += len(content)
            except Exception as e:
                logger.warning(f"Error reading file {fp}: {e}")
                continue

        return result

    def read_all_text_files(
        self,
        repo_url_or_path: str,
        excluded_dirs: Optional[List[str]] = None,
        excluded_files: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Read all text files from a cloned repo. Returns {file_path: content}."""
        local_path = self._repos.get(repo_url_or_path)
        if not local_path or not os.path.isdir(local_path):
            raise ValueError(f"Repo not cloned yet: {repo_url_or_path}")

        excluded_dirs = excluded_dirs or []
        excluded_files = excluded_files or []

        # Common text file extensions
        text_extensions = {
            ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp", ".h", ".hpp",
            ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala", ".cs", ".vb",
            ".html", ".css", ".scss", ".less", ".xml", ".json", ".yaml", ".yml",
            ".toml", ".ini", ".cfg", ".conf", ".md", ".txt", ".rst", ".sh", ".bash",
            ".zsh", ".fish", ".ps1", ".bat", ".cmd", ".sql", ".r", ".m", ".mm",
            ".pl", ".pm", ".lua", ".dart", ".ex", ".exs", ".erl", ".hrl", ".hs",
            ".lhs", ".clj", ".cljs", ".edn", ".elm", ".vue", ".svelte", ".sol",
            ".proto", ".graphql", ".prisma", ".tf", ".hcl", ".dockerfile", ".makefile",
            ".cmake", ".gradle", ".groovy", ".jl", ".nim", ".zig", ".v", ".sv",
            ".env", ".gitignore", ".editorconfig",
        }

        result = {}
        total_chars = 0
        for root, dirs, files in os.walk(local_path):
            # Skip excluded directories
            relative_root = os.path.relpath(root, local_path)
            dirs_to_skip = []
            for d in dirs:
                rel = os.path.join(relative_root, d).replace("\\", "/")
                if any(excl.strip("/") in rel for excl in excluded_dirs):
                    dirs_to_skip.append(d)
            for d in dirs_to_skip:
                dirs.remove(d)

            for f in files:
                if total_chars >= MAX_TOTAL_CONTEXT_CHARS:
                    return result

                ext = os.path.splitext(f)[1].lower()
                if ext not in text_extensions:
                    continue

                rel_path = os.path.join(relative_root, f).replace("\\", "/")
                if any(excl in rel_path for excl in excluded_files):
                    continue

                full_path = os.path.join(root, f)
                try:
                    file_size = os.path.getsize(full_path)
                    if file_size > MAX_FILE_SIZE_BYTES:
                        continue
                    content = Path(full_path).read_text(encoding="utf-8", errors="replace")
                    result[rel_path] = content
                    total_chars += len(content)
                except Exception:
                    continue

        return result

    def build_context_block(self, file_contents: Dict[str, str]) -> str:
        """Build a CAG context block from file contents.

        This is the prefix that DeepSeek's KV cache will cache.
        All file contents are wrapped in XML for structured parsing by the LLM.
        """
        if not file_contents:
            return ""

        parts = ["<repository_context>\n"]
        for file_path, content in file_contents.items():
            parts.append(f'<file path="{file_path}">\n')
            parts.append(content)
            parts.append(f"\n</file>\n")
        parts.append("</repository_context>")
        return "\n".join(parts)


# Global CAG context instance
cag_context = CAGContext()
