"""Prompt loading manager with caching."""

from ..utils.path import load_prompts


class PromptManager:
    """Unified prompt loading manager with caching support."""

    _cache: dict[str, dict[str, str]] = {}

    @classmethod
    def get_prompt(cls, prompt_path: str, prompt_key: str, fallback: str | None = None) -> str:
        if prompt_path not in cls._cache:
            cls._cache[prompt_path] = load_prompts(prompt_path)
        result = cls._cache[prompt_path].get(prompt_key)
        if result is None:
            if fallback is not None:
                return fallback
            raise KeyError(f"Prompt key '{prompt_key}' not found in '{prompt_path}'")
        return result

    @classmethod
    def get_prompts(cls, prompt_path: str) -> dict[str, str]:
        if prompt_path not in cls._cache:
            cls._cache[prompt_path] = load_prompts(prompt_path)
        return cls._cache[prompt_path]

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()
