"""Central configuration management loaded from `.env` and environment variables."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any, Literal

from dotenv import load_dotenv

DialectType = Literal["odps", "hive"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_PLACEHOLDER_API_KEYS = frozenset(
    {
        "",
        "your_api_key_here",
        "your_llm_api_key_here",
        "your_openai_api_key_here",
        "your_opencode_api_key_here",
    }
)


def _first_nonempty_env(*names: str) -> str:
    for name in names:
        val = (os.getenv(name) or "").strip()
        if val:
            return val
    return ""


@dataclass
class Settings:
    """Runtime settings for Dual-Role OpenAI-compatible LLM endpoints, concurrency, and sandbox dialect."""

    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    actor_model: str = "deepseek-v4.1-flash"
    reflector_model: str = "deepseek-v4-pro"
    max_concurrency: int = 8
    timeout_sec: float = 210.0
    dialect: DialectType = "odps"
    max_refine_turns: int = 2
    prompts_dir: Path = PROJECT_ROOT / "prompts" / "seed"
    outputs_dir: Path = PROJECT_ROOT / "outputs"

    @property
    def has_usable_api_key(self) -> bool:
        """Return True when a non-placeholder live API key is configured."""
        return bool(self.api_key and self.api_key.strip() not in _PLACEHOLDER_API_KEYS)

    @classmethod
    def from_env(
        cls,
        env_path: Path | str | None = None,
        **overrides: Any,
    ) -> "Settings":
        """Load settings from `.env` / environment variables with optional CLI overrides.

        Recognizes standard `LLM_*` and `OPENAI_*` variables as primary configuration while
        maintaining seamless backwards compatibility with `OPENCODE_*` variables.
        """
        resolved_env = Path(env_path) if env_path else (PROJECT_ROOT / ".env")
        if resolved_env.exists():
            load_dotenv(dotenv_path=resolved_env, override=False)
        else:
            load_dotenv(override=False)

        raw_dialect = (os.getenv("DEFAULT_DIALECT") or "odps").strip().lower()
        dialect: DialectType = "hive" if raw_dialect == "hive" else "odps"

        primary_key = _first_nonempty_env("LLM_API_KEY", "OPENAI_API_KEY")
        opencode_key = (os.getenv("OPENCODE_API_KEY") or "").strip()
        api_key = primary_key or opencode_key

        explicit_base_url = _first_nonempty_env("LLM_BASE_URL", "OPENAI_BASE_URL", "OPENCODE_BASE_URL")
        if explicit_base_url:
            base_url = explicit_base_url
        elif opencode_key and not primary_key:
            base_url = "https://opencode.ai/zen/go/v1"
        else:
            base_url = "https://api.openai.com/v1"

        max_conc_raw = _first_nonempty_env("LLM_MAX_CONCURRENCY", "OPENCODE_MAX_CONCURRENCY") or "8"
        timeout_raw = _first_nonempty_env("LLM_TIMEOUT_SEC", "OPENCODE_TIMEOUT_SEC") or "210"

        settings = cls(
            api_key=api_key,
            base_url=base_url,
            actor_model=(os.getenv("ACTOR_MODEL") or "deepseek-v4.1-flash").strip(),
            reflector_model=(os.getenv("REFLECTOR_MODEL") or "deepseek-v4-pro").strip(),
            max_concurrency=int(max_conc_raw),
            timeout_sec=float(timeout_raw),
            dialect=dialect,
        )

        for key, value in overrides.items():
            if value is not None and hasattr(settings, key):
                if key == "dialect":
                    value = "hive" if str(value).strip().lower() == "hive" else "odps"
                setattr(settings, key, value)

        return settings


_KNOWN_HF_LEAF_MODELS: dict[str, str] = {
    "Qwen2.5-Coder-1.5B-Instruct": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "Qwen2.5-Coder-3B-Instruct": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "Qwen2.5-Coder-7B-Instruct": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "XiYanSQL-QwenCoder-3B-2504": "XGenerationLab/XiYanSQL-QwenCoder-3B-2504",
    "XiYanSQL-QwenCoder-7B-2504": "XGenerationLab/XiYanSQL-QwenCoder-7B-2504",
}

_HOST_PATH_PREFIX_RE = re.compile(
    r"^(?:/(?:home|Users)/[^/]+/|/content/|[A-Za-z]:/(?:Users|Temp)/[^/]+/|/tmp/|/var/folders/[^/]+/)"
)


def to_portable_posix_path(path_or_str: str | Path, project_root: Path = PROJECT_ROOT) -> str:
    """Normalize any local host, Windows, or Colab filesystem path into a portable relative POSIX path."""
    if isinstance(path_or_str, Path):
        try:
            return path_or_str.resolve().relative_to(project_root.resolve()).as_posix()
        except ValueError:
            raw = str(path_or_str)
    else:
        raw = str(path_or_str)

    s = raw.strip().replace("\\", "/")
    if not s:
        return s

    root_posix = project_root.resolve().as_posix().rstrip("/")
    if root_posix and s.startswith(root_posix + "/"):
        return s[len(root_posix) + 1 :]

    if s.startswith("/content/EvoDW-RSI/"):
        return s[len("/content/EvoDW-RSI/") :]
    if s.startswith("/content/drive/MyDrive/EvoDW_RL_v2_Workspace/"):
        return "colab_workspace/" + s[len("/content/drive/MyDrive/EvoDW_RL_v2_Workspace/") :]
    if s.startswith("/content/"):
        return s[len("/content/") :]

    if s.startswith("/data/llm-models/"):
        leaf = s.rstrip("/").split("/")[-1]
        return _KNOWN_HF_LEAF_MODELS.get(leaf, f"models/{leaf}")

    repo_marker = "/EvoDW-RSI/"
    if repo_marker in s:
        return s.split(repo_marker, 1)[1]

    leaf_candidate = s.rstrip("/").split("/")[-1]
    if _HOST_PATH_PREFIX_RE.match(s) and leaf_candidate in _KNOWN_HF_LEAF_MODELS:
        return _KNOWN_HF_LEAF_MODELS[leaf_candidate]

    if _HOST_PATH_PREFIX_RE.match(s):
        cleaned = _HOST_PATH_PREFIX_RE.sub("", s)
        parts = [p for p in cleaned.split("/") if p]
        if "outputs" in parts:
            idx = parts.index("outputs")
            return "/".join(parts[idx:])
        if "colab_workspace" in parts:
            idx = parts.index("colab_workspace")
            return "/".join(parts[idx:])
        return "/".join(parts[-3:]) if len(parts) > 3 else "/".join(parts)

    return s


def sanitize_host_paths_in_obj(obj: Any, project_root: Path = PROJECT_ROOT) -> Any:
    """Recursively sanitize leaked host/Colab filesystem path strings inside JSON-serializable structures."""
    if isinstance(obj, dict):
        return {k: sanitize_host_paths_in_obj(v, project_root=project_root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_host_paths_in_obj(v, project_root=project_root) for v in obj]
    if isinstance(obj, Path):
        return to_portable_posix_path(obj, project_root=project_root)
    if isinstance(obj, str) and "\n" not in obj and len(obj) < 512:
        norm = obj.replace("\\", "/")
        if (
            norm.startswith(("/home/", "/Users/", "/content/", "/data/llm-models/"))
            or re.match(r"^[A-Za-z]:/Users/", norm)
        ):
            return to_portable_posix_path(obj, project_root=project_root)
    return obj
