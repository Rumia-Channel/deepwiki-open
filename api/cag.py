"""CAG (Cache-Augmented Generation) module.

Replaces RAG/embedding with direct file context feeding.
DeepSeek's KV cache auto-caches the file prefix for subsequent requests.
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from adalflow.utils import get_adalflow_default_root_path
from api.data_pipeline import download_repo

logger = logging.getLogger(__name__)

# Maximum size of a single file to include in context (bytes)
MAX_FILE_SIZE_BYTES = 200 * 1024  # 200KB
# Maximum total context size to include (chars)
MAX_TOTAL_CONTEXT_CHARS = 400_000


def _sanitize_path_component(name: str) -> str:
    """Strip path separators, null bytes, and traversal sequences from a path component."""
    sanitized = name.replace("\x00", "").replace("/", "_").replace("\\", "_")
    sanitized = re.sub(r"\.\.+", "_", sanitized)
    sanitized = re.sub(r"[^\w\-.]", "_", sanitized)
    sanitized = sanitized.strip("._-")
    return sanitized or "unknown"


def _extract_repo_name(repo_url_or_path: str, repo_type: str = "github") -> str:
    """Extract a unique repo name from URL or path."""
    url_parts = repo_url_or_path.rstrip("/").split("/")

    if repo_type in ["github", "gitlab", "bitbucket"] and len(url_parts) >= 5:
        owner = _sanitize_path_component(url_parts[-2])
        repo = _sanitize_path_component(url_parts[-1].replace(".git", ""))
        repo_name = f"{owner}_{repo}"
    else:
        repo_name = _sanitize_path_component(url_parts[-1].replace(".git", ""))

    return repo_name


def _get_repo_local_path(repo_url_or_path: str, repo_type: str = "github") -> str:
    """Get the local filesystem path where a repo should be cloned."""
    root_path = get_adalflow_default_root_path()
    if repo_url_or_path.startswith("https://") or repo_url_or_path.startswith("http://"):
        repo_name = _extract_repo_name(repo_url_or_path, repo_type)
        return os.path.join(root_path, "repos", repo_name)
    else:
        return repo_url_or_path


class CAGContext:
    """Manages repo cloning and file reading for CAG-based wiki generation."""

    def __init__(self):
        self._repos: Dict[str, str] = {}  # repo_url -> local_path

    def clone_repo(
        self,
        repo_url_or_path: str,
        repo_type: str = "github",
        access_token: Optional[str] = None,
    ) -> str:
        """Clone/download a repo and return the local path."""
        save_repo_dir = _get_repo_local_path(repo_url_or_path, repo_type)

        if save_repo_dir and os.path.isdir(save_repo_dir) and os.listdir(save_repo_dir):
            logger.info(f"Repo already cloned at {save_repo_dir}")
            self._repos[repo_url_or_path] = save_repo_dir
            return save_repo_dir

        try:
            os.makedirs(os.path.dirname(save_repo_dir) or save_repo_dir, exist_ok=True)
            download_repo(repo_url_or_path, save_repo_dir, repo_type, access_token)
            self._repos[repo_url_or_path] = save_repo_dir
            logger.info(f"Repo cloned to {save_repo_dir}")
            return save_repo_dir
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
