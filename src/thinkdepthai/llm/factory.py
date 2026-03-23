"""LangChain model factory — creates a chat model that supports bind_tools."""

from __future__ import annotations

import os
from typing import Any

from ..config.schema import ModelConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


def create_langchain_model(config: ModelConfig) -> Any:
    """Create a LangChain chat model from config.

    Supports api_format: openai | anthropic | google.
    Also handles legacy provider:model_name prefix.
    """
    mp = config.model_provider
    api_key = mp.api_key or os.getenv("UTU_LLM_API_KEY")
    base_url = mp.base_url or os.getenv("UTU_LLM_BASE_URL")
    model_name = mp.model or os.getenv("UTU_LLM_MODEL", "gpt-4")

    if not api_key or not base_url:
        raise ValueError("UTU_LLM_API_KEY and UTU_LLM_BASE_URL must be set")

    api_format = mp.api_format or "openai"
    actual_model_name = model_name

    # Legacy provider prefix in model name (e.g. "anthropic:claude-3-5-sonnet")
    if model_name and ":" in model_name:
        prefix, actual_model_name = model_name.split(":", 1)
        prefix = prefix.lower()
        if api_format == "openai":
            api_format = prefix
            logger.info(f"Detected provider prefix: {prefix}, model: {actual_model_name}")

    logger.info(f"Using API format: {api_format}, model: {actual_model_name}")
    assert actual_model_name is not None

    if api_format == "anthropic":
        from langchain_anthropic import ChatAnthropic
        from pydantic import SecretStr

        return ChatAnthropic(
            api_key=SecretStr(api_key),
            base_url=base_url,
            model_name=actual_model_name,
            temperature=0.3,
            timeout=300,
            max_retries=0,
            stop=[],
        )
    elif api_format == "openai":
        from langchain_openai import ChatOpenAI
        from pydantic import SecretStr

        return ChatOpenAI(
            api_key=SecretStr(api_key),
            base_url=base_url,
            model=actual_model_name,
            temperature=0.3,
            timeout=300,
            max_retries=0,
            streaming=False,
        )
    elif api_format == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            api_key=api_key,
            model=actual_model_name,
            temperature=0.3,
            timeout=600,
            max_retries=0,
            base_url=base_url,
            include_thoughts=True,
        )
    else:
        raise ValueError(
            f"Unsupported api_format: {api_format}. "
            "Supported: 'openai', 'anthropic', 'google'."
        )
