"""CAG (Cache-Augmented Generation) module.

Replaces RAG/embedding with direct file context feeding.
DeepSeek's KV cache auto-caches the shared file prefix across pages.
"""

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

try:
    import tiktoken
    _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
    def _count_tokens(text: str) -> int:
        return len(_tiktoken_enc.encode(text))
except Exception:
    def _count_tokens(text: str) -> int:
        return len(text) // 4

from adalflow.utils import get_adalflow_default_root_path
from api.data_pipeline import download_repo

logger = logging.getLogger(__name__)

# Maximum size of a single file to include in context (bytes)
MAX_FILE_SIZE_BYTES = 200 * 1024  # 200KB
# Maximum total context size to include (chars) — ~850K tokens, DeepSeek 1M window safe
MAX_TOTAL_CONTEXT_CHARS = 2_000_000
# Text file extensions to include in context
TEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp", ".h", ".hpp",
    ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala", ".cs", ".vb",
    ".html", ".css", ".scss", ".less", ".xml", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".conf", ".md", ".txt", ".rst", ".sh", ".bash",
    ".zsh", ".fish", ".ps1", ".bat", ".sql", ".r", ".m", ".mm",
    ".pl", ".pm", ".lua", ".dart", ".ex", ".exs", ".erl", ".hrl", ".hs",
    ".lhs", ".clj", ".cljs", ".edn", ".elm", ".vue", ".svelte", ".sol",
    ".proto", ".graphql", ".prisma", ".tf", ".hcl", ".dockerfile", ".makefile",
    ".cmake", ".gradle", ".groovy", ".jl", ".nim", ".zig", ".v", ".sv",
}


def _sanitize_path_component(name: str) -> str:
    sanitized = name.replace("\x00", "").replace("/", "_").replace("\\", "_")
    sanitized = re.sub(r"\.\.+", "_", sanitized)
    sanitized = re.sub(r"[^\w\-.]", "_", sanitized)
    sanitized = sanitized.strip("._-")
    return sanitized or "unknown"


def _extract_repo_name(repo_url_or_path: str, repo_type: str = "github") -> str:
    url_parts = repo_url_or_path.rstrip("/").split("/")
    if repo_type in ["github", "gitlab", "bitbucket"] and len(url_parts) >= 5:
        owner = _sanitize_path_component(url_parts[-2])
        repo = _sanitize_path_component(url_parts[-1].replace(".git", ""))
        return f"{owner}_{repo}"
    return _sanitize_path_component(url_parts[-1].replace(".git", ""))


def _get_repo_local_path(repo_url_or_path: str, repo_type: str = "github") -> str:
    root_path = get_adalflow_default_root_path()
    if repo_url_or_path.startswith("https://") or repo_url_or_path.startswith("http://"):
        repo_name = _extract_repo_name(repo_url_or_path, repo_type)
        return os.path.join(root_path, "repos", repo_name)
    return repo_url_or_path


class CAGContext:
    """Manages repo cloning and CAG context building for wiki generation.

    CAG strategy: build ONE shared context block from all source files
    (sorted by path for deterministic ordering). This block is prepended
    to every page-generation request so DeepSeek's KV cache can reuse
    the cached computation across all pages.
    """

    def __init__(self):
        self._repos: Dict[str, str] = {}         # repo_url -> local_path
        self._context_cache: Dict[str, str] = {}  # repo_url -> built context block
        self._context_file_count: Dict[str, int] = {}  # repo_url -> file count in context

    def clone_repo(
        self,
        repo_url_or_path: str,
        repo_type: str = "github",
        access_token: Optional[str] = None,
        force: bool = False,
    ) -> str:
        """Clone/download a repo, invalidating any cached context.

        Set force=True to delete existing clone and re-download.
        """
        save_repo_dir = _get_repo_local_path(repo_url_or_path, repo_type)

        # Invalidate context cache so next build_full_context reads fresh files
        self._context_cache.pop(repo_url_or_path, None)
        self._context_file_count.pop(repo_url_or_path, None)

        if force and save_repo_dir and os.path.isdir(save_repo_dir):
            logger.info(f"Force re-clone: removing {save_repo_dir}")
            shutil.rmtree(save_repo_dir, ignore_errors=True)
            self._repos.pop(repo_url_or_path, None)

        if save_repo_dir and os.path.isdir(save_repo_dir) and os.listdir(save_repo_dir):
            logger.info(f"Repo already cloned at {save_repo_dir}")
            self._repos[repo_url_or_path] = save_repo_dir
            # Init submodules if present (needed for repos with --recursive)
            try:
                subprocess.run(
                    ["git", "-C", save_repo_dir, "-c", "protocol.file.allow=never",
                     "submodule", "update", "--init", "--recursive", "--depth=1"],
                    check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
            except Exception:
                pass
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

    def get_context_block(
        self,
        repo_url_or_path: str,
        repo_type: str = "github",
        access_token: Optional[str] = None,
        force_reclone: bool = False,
    ) -> str:
        """Get the cached CAG context block, cloning the repo if needed.

        Set force_reclone=True to delete existing clone and re-download before building.
        """
        if repo_url_or_path in self._context_cache and not force_reclone:
            logger.info(
                f"CAG context cache HIT: {repo_url_or_path} "
                f"({self._context_file_count.get(repo_url_or_path, 0)} files)"
            )
            return self._context_cache[repo_url_or_path]

        # Ensure repo is cloned (force re-clone if requested)
        if force_reclone or repo_url_or_path not in self._repos or not self._repos[repo_url_or_path]:
            self.clone_repo(repo_url_or_path, repo_type, access_token, force=force_reclone)

        local_path = self._repos.get(repo_url_or_path)
        logger.info(f"CAG: building full context for {repo_url_or_path} ...")

        # Walk the repo and collect text files
        entries = []
        for root, dirs, files in os.walk(local_path):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext not in TEXT_EXTENSIONS:
                    continue
                full_path = os.path.join(root, f)
                try:
                    if os.path.getsize(full_path) > MAX_FILE_SIZE_BYTES:
                        continue
                except OSError:
                    continue
                rel_path = os.path.relpath(full_path, local_path).replace("\\", "/")
                entries.append((rel_path, full_path))

        # Sort by path for deterministic ordering (critical for KV cache reuse)
        entries.sort(key=lambda x: x[0])

        # Build context block, respecting max size
        parts = ["<repository_context>\n"]
        total_chars = 0
        file_count = 0
        for rel_path, full_path in entries:
            if total_chars >= MAX_TOTAL_CONTEXT_CHARS:
                logger.warning(
                    f"CAG: reached max context size ({MAX_TOTAL_CONTEXT_CHARS} chars), "
                    f"stopped at {file_count}/{len(entries)} files"
                )
                break
            try:
                content = Path(full_path).read_text(encoding="utf-8", errors="replace")
                parts.append(f'<file path="{rel_path}">\n')
                parts.append(content)
                parts.append("\n</file>\n")
                total_chars += len(content)
                file_count += 1
            except Exception as e:
                logger.warning(f"CAG: error reading {rel_path}: {e}")
                continue

        parts.append("</repository_context>")
        context = "\n".join(parts)

        self._context_cache[repo_url_or_path] = context
        self._context_file_count[repo_url_or_path] = file_count
        logger.info(
            f"CAG: context built — {file_count} files, "
            f"{len(context):,} chars (~{_count_tokens(context):,} tokens, estimated)"
        )
        return context

    def read_files(
        self,
        repo_url_or_path: str,
        file_paths: List[str],
    ) -> Dict[str, str]:
        """Read specific files from a cloned repo. Utility method."""
        local_path = self._repos.get(repo_url_or_path)
        if not local_path or not os.path.isdir(local_path):
            raise ValueError(f"Repo not cloned yet: {repo_url_or_path}")

        result = {}
        total_chars = 0
        for fp in file_paths:
            if total_chars >= MAX_TOTAL_CONTEXT_CHARS:
                break
            full_path = Path(local_path) / fp.lstrip("/")
            if not full_path.is_file():
                continue
            try:
                if full_path.stat().st_size > MAX_FILE_SIZE_BYTES:
                    continue
                content = full_path.read_text(encoding="utf-8", errors="replace")
                result[fp] = content
                total_chars += len(content)
            except Exception:
                continue
        return result


# Global CAG context instance
cag_context = CAGContext()
