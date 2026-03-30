import logging
import os

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

_llm: BaseChatModel | None = None


def get_llm() -> BaseChatModel:
    """
    Return a cached LLM instance based on environment configuration.

    Environment variables:
        LLM_PROVIDER  — openai (default) | anthropic | google
        LLM_MODEL     — model name, defaults per provider:
                          openai:    gpt-4o-mini
                          anthropic: claude-haiku-4-5-20251001
                          google:    gemini-2.0-flash

    Switching providers requires the corresponding langchain package:
        openai:    langchain-openai (already a dependency)
        anthropic: pip install langchain-anthropic
        google:    pip install langchain-google-genai
    """
    global _llm
    if _llm is not None:
        return _llm

    from dotenv import load_dotenv
    load_dotenv()

    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model = os.getenv("LLM_MODEL", "")

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        resolved_model = model or "gpt-4o-mini"
        _llm = ChatOpenAI(model=resolved_model)

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # type: ignore[import]
        resolved_model = model or "claude-haiku-4-5-20251001"
        _llm = ChatAnthropic(model=resolved_model)  # type: ignore[call-arg]

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore[import]
        resolved_model = model or "gemini-2.0-flash"
        _llm = ChatGoogleGenerativeAI(model=resolved_model)

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: {provider!r}. "
            "Valid options: openai, anthropic, google"
        )

    logger.info(f"[LLM] Using provider={provider} model={resolved_model}")
    return _llm
