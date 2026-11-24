"""Central configuration management loaded from `.env` and environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
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
