"""
Centralized configuration.

Industry pattern: never scatter os.environ["X"] calls or magic numbers
through business logic. One module owns config, validates it once at
startup, and everything else imports typed values from here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str
    claude_model: str
    chunk_size_words: int
    top_k_chunks: int


def _require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {key}. "
            f"Copy .env.example to .env and fill it in."
        )
    return value


def load_settings() -> Settings:
    return Settings(
        anthropic_api_key=_require_env("ANTHROPIC_API_KEY"),
        claude_model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5"),
        chunk_size_words=int(os.getenv("CHUNK_SIZE_WORDS", "250")),
        top_k_chunks=int(os.getenv("TOP_K_CHUNKS", "2")),
    )
