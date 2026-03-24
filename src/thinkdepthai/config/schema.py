"""Minimal configuration schema for ThinkDepthAI agent."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ModelProviderConfig(BaseModel):
    """Model provider configuration."""

    type: str = "chat.completions"
    model: str | None = Field(default=None)
    base_url: str | None = Field(default=None)
    api_key: str | None = Field(default=None)
    api_format: str = "openai"  # openai | anthropic | google

    def __repr__(self) -> str:
        return (
            f"ModelProviderConfig(type={self.type!r}, model={self.model!r}, "
            f"base_url='***', api_key='***', api_format={self.api_format!r})"
        )


class RateLimitConfig(BaseModel):
    """Rate limiting and retry configuration."""

    rpm: int | None = None
    tpm: int | None = None
    max_retries: int = 5
    retry_min_wait: float = 1.0
    retry_max_wait: float = 60.0
    retry_jitter: float = 1.0


class ModelConfig(BaseModel):
    """Overall model configuration."""

    model_provider: ModelProviderConfig = Field(default_factory=ModelProviderConfig)
    model_settings: dict[str, Any] = Field(default_factory=dict)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)


class AgentProfileConfig(BaseModel):
    """Agent profile configuration."""

    name: str | None = "RCA-Agent"
    prompt_path: str | None = None
    prompt_key: str | None = None


class ToolkitConfig(BaseModel):
    """Toolkit configuration."""

    name: str | None = None
    mode: str = "builtin"
    config: dict[str, Any] = Field(default_factory=dict)
    activated_tools: list[str] | None = None


class AgentConfig(BaseModel):
    """Overall agent configuration."""

    type: str = "langgraph_rca"
    model: ModelConfig = Field(default_factory=ModelConfig)
    agent: AgentProfileConfig = Field(default_factory=AgentProfileConfig)
    toolkits: dict[str, ToolkitConfig] = Field(default_factory=dict)
    max_turns: int = 1000
