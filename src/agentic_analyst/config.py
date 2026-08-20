"""Environment loading, per-agent model selection, and optional tracing setup.

Everything that reads os.environ lives here so the rest of the package can be
tested by constructing a Settings object directly.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Repo root is three levels up from this file: src/agentic_analyst/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = PROJECT_ROOT / ".checkpoints"

DEFAULT_DATASET = DATA_DIR / "TelecomCustomerChurn.csv"

# The three structured agents need strict JSON schema support. gpt-5.4-mini is the
# current mini-class model and handles temperature=0 + strict schemas without
# spending reasoning tokens. Deliberately not a gpt-4o-family model - that line is
# on its way out through 2026.
DEFAULT_STRUCTURED_MODEL = "openai:gpt-5.4-mini"

# The narrator only ever emits prose. Groq's gpt-oss models are fast and cheap but
# tend to bleed reasoning fragments into tool-call slots, so they're kept strictly
# away from anything the graph has to parse.
DEFAULT_NARRATOR_MODEL = "groq:openai/gpt-oss-120b"


@dataclass(frozen=True)
class ModelSpec:
    """A `provider:model` pair, e.g. openai:gpt-5.4-mini."""

    provider: str
    name: str

    @classmethod
    def parse(cls, raw: str) -> "ModelSpec":
        if ":" not in raw:
            raise ValueError(
                f"Model spec {raw!r} must be 'provider:model', e.g. 'openai:gpt-5.4-mini'"
            )
        provider, _, name = raw.partition(":")
        provider = provider.strip().lower()
        name = name.strip()
        if provider not in {"openai", "groq"}:
            raise ValueError(f"Unsupported provider {provider!r} (expected openai or groq)")
        if not name:
            raise ValueError(f"Model spec {raw!r} has an empty model name")
        return cls(provider=provider, name=name)

    def __str__(self) -> str:
        return f"{self.provider}:{self.name}"


@dataclass(frozen=True)
class DriveMCPConfig:
    """Connection details for the Google Drive MCP server.

    `command` empty means no server configured, which is a supported mode - the
    pipeline falls back to writing reports locally.
    """

    command: str = ""
    args: list[str] = field(default_factory=list)
    folder_id: str = ""
    env: dict[str, str] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return bool(self.command)


@dataclass(frozen=True)
class Settings:
    query_model: ModelSpec
    analysis_model: ModelSpec
    viz_model: ModelSpec
    narrator_model: ModelSpec
    openai_api_key: str
    groq_api_key: str
    drive: DriveMCPConfig
    tracing_enabled: bool
    tracing_project: str

    def model_for(self, agent: str) -> ModelSpec:
        try:
            return getattr(self, f"{agent}_model")
        except AttributeError as exc:
            raise KeyError(f"No model configured for agent {agent!r}") from exc


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def load_settings(dotenv_path: Path | None = None) -> Settings:
    """Read .env plus the process environment into a Settings object.

    Real environment variables win over .env values, which is what you want in CI.
    """
    load_dotenv(dotenv_path or (PROJECT_ROOT / ".env"), override=False)

    openai_key = _env("OPENAI_API_KEY")
    groq_key = _env("GROQ_API_KEY")

    narrator_raw = _env("NARRATOR_MODEL", DEFAULT_NARRATOR_MODEL)
    narrator = ModelSpec.parse(narrator_raw)
    # Falling back rather than failing: a missing Groq key shouldn't stop the demo,
    # and the narrator's job doesn't actually depend on which provider serves it.
    if narrator.provider == "groq" and not groq_key:
        narrator = ModelSpec.parse(DEFAULT_STRUCTURED_MODEL)

    drive_command = _env("DRIVE_MCP_COMMAND")
    drive = DriveMCPConfig(
        command=drive_command,
        # shlex so `-y mcp-google-drive` in a single env var splits the way a shell would
        args=shlex.split(_env("DRIVE_MCP_ARGS")) if drive_command else [],
        folder_id=_env("DRIVE_MCP_FOLDER_ID"),
        env={
            k: v
            for k, v in (
                ("GOOGLE_CLIENT_ID", _env("GOOGLE_CLIENT_ID")),
                ("GOOGLE_CLIENT_SECRET", _env("GOOGLE_CLIENT_SECRET")),
            )
            if v
        },
    )

    # Accept either naming scheme; LANGCHAIN_* is what the LangSmith SDK still reads
    # from most installs, LANGSMITH_* is the newer alias.
    langsmith_key = _env("LANGCHAIN_API_KEY") or _env("LANGSMITH_API_KEY")
    tracing_flag = (_env("LANGCHAIN_TRACING_V2") or _env("LANGSMITH_TRACING") or "true").lower()
    tracing_enabled = bool(langsmith_key) and tracing_flag not in {"false", "0", "no", "off"}

    return Settings(
        query_model=ModelSpec.parse(_env("QUERY_MODEL", DEFAULT_STRUCTURED_MODEL)),
        analysis_model=ModelSpec.parse(_env("ANALYSIS_MODEL", DEFAULT_STRUCTURED_MODEL)),
        viz_model=ModelSpec.parse(_env("VIZ_MODEL", DEFAULT_STRUCTURED_MODEL)),
        narrator_model=narrator,
        openai_api_key=openai_key,
        groq_api_key=groq_key,
        drive=drive,
        tracing_enabled=tracing_enabled,
        tracing_project=_env("LANGCHAIN_PROJECT") or _env("LANGSMITH_PROJECT") or "agentic-analyst",
    )


def configure_tracing(settings: Settings) -> bool:
    """Turn LangSmith auto-tracing on, or make sure it is firmly off.

    Call this before importing anything that builds a LangChain client. The explicit
    "off" branch matters: a stale LANGCHAIN_TRACING_V2=true left in the shell with no
    API key present makes every call warn or fail, and that is a miserable first-run
    experience for someone who just cloned the repo.
    """
    if settings.tracing_enabled:
        key = _env("LANGCHAIN_API_KEY") or _env("LANGSMITH_API_KEY")
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = key
        os.environ["LANGCHAIN_PROJECT"] = settings.tracing_project
        return True

    for stale in ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING"):
        os.environ.pop(stale, None)
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    return False


def ensure_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
