"""Chat model construction.

Direct instantiation of ChatOpenAI / ChatGroq rather than init_chat_model - the
per-provider kwargs differ enough that the indirection costs more than it saves.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from .config import ModelSpec, Settings

# Structured agents run deterministic. The narrator gets a little room so the prose
# doesn't read like a template.
STRUCTURED_TEMPERATURE = 0.0
NARRATIVE_TEMPERATURE = 0.3


def build_model(spec: ModelSpec, settings: Settings, *, temperature: float) -> BaseChatModel:
    if spec.provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=spec.name,
            temperature=temperature,
            api_key=settings.openai_api_key,
            timeout=90,
            max_retries=3,
        )

    if spec.provider == "groq":
        if not settings.groq_api_key:
            raise RuntimeError(f"GROQ_API_KEY is not set but {spec} was requested.")
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=spec.name,
            temperature=temperature,
            api_key=settings.groq_api_key,
            timeout=90,
            max_retries=3,
        )

    raise ValueError(f"Unsupported provider: {spec.provider}")


# Keyed on spec+temperature rather than lru_cache over Settings: Settings holds
# list/dict fields, so it isn't hashable no matter that the dataclass is frozen.
_MODEL_CACHE: dict[tuple[str, float], BaseChatModel] = {}


def get_agent_model(agent: str, settings: Settings) -> BaseChatModel:
    """Return the model for a named agent (query / analysis / viz / narrator)."""
    spec = settings.model_for(agent)
    temperature = NARRATIVE_TEMPERATURE if agent == "narrator" else STRUCTURED_TEMPERATURE
    key = (str(spec), temperature)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = build_model(spec, settings, temperature=temperature)
    return _MODEL_CACHE[key]


def get_structured_model(agent: str, settings: Settings, schema: type):
    """Model bound to a Pydantic schema, with strict JSON-schema decoding.

    strict=True means the provider enforces the schema server-side, so a malformed
    payload can't reach the graph in the first place.
    """
    model = get_agent_model(agent, settings)
    spec = settings.model_for(agent)
    if spec.provider == "openai":
        return model.with_structured_output(schema, method="json_schema", strict=True)
    # Groq exposes json_mode rather than strict schemas. This path exists so a model
    # override doesn't hard-crash, but it isn't the recommended configuration.
    return model.with_structured_output(schema)
