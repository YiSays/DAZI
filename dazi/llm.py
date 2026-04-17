"""LLM client wrapper for Dazi.

This module owns:
- LLM client creation (create_llm) and lazy initialization (_get_llm)
- Memory/skills content injection into prompts
"""

from __future__ import annotations

import langchain_openai.chat_models.base as _lc_base
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_openai import ChatOpenAI

from dazi._singletons import (
    memory_store,
    settings_manager,
    skill_registry,
)
from dazi.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL

# ─────────────────────────────────────────────────────────
# MONKEY-PATCH: preserve reasoning_content across providers
# ─────────────────────────────────────────────────────────
# langchain_openai drops `reasoning_content` from streaming deltas and
# message serialization.  Many OpenAI-compatible providers (z.ai/GLM,
# DeepSeek, OpenRouter, etc.) use this field for extended thinking.
# These patches make ChatOpenAI handle it generically.

_orig_convert_delta = _lc_base._convert_delta_to_message_chunk
_orig_convert_msg = _lc_base._convert_message_to_dict


def _patched_convert_delta(_dict, default_class):
    """Capture reasoning_content from streaming delta into additional_kwargs."""
    msg = _orig_convert_delta(_dict, default_class)
    if isinstance(msg, AIMessageChunk):
        rc = _dict.get("reasoning_content")
        if rc:
            msg.additional_kwargs["reasoning_content"] = rc
    return msg


def _patched_convert_msg(message, api: str = "chat/completions"):
    """Inject reasoning_content from additional_kwargs into serialized assistant messages."""
    msg_dict = _orig_convert_msg(message, api=api)
    if isinstance(message, AIMessage) and msg_dict.get("role") == "assistant":
        rc = message.additional_kwargs.get("reasoning_content")
        if rc:
            msg_dict["reasoning_content"] = rc
    return msg_dict


_lc_base._convert_delta_to_message_chunk = _patched_convert_delta
_lc_base._convert_message_to_dict = _patched_convert_msg

# ─────────────────────────────────────────────────────────
# LLM CLIENT FACTORY
# ─────────────────────────────────────────────────────────


def create_llm(
    model: str | None = None,
    temperature: float = 0.0,
    streaming: bool = True,
    base_url: str | None = None,
    api_key: str | None = None,
    enable_thinking: bool = True,
) -> ChatOpenAI:
    """Create a ChatOpenAI instance with project defaults.

    Args:
        model: Override model name. Defaults to OPENAI_MODEL env var.
        temperature: Model temperature.
        streaming: Enable streaming responses.
        base_url: Override API base URL. Defaults to OPENAI_BASE_URL env var.
        api_key: Override API key. Defaults to OPENAI_API_KEY env var.
        enable_thinking: Enable extended thinking (reasoning_content) for
            providers that support it (z.ai, DeepSeek, OpenRouter, etc.).

    Returns:
        Configured ChatOpenAI instance.
    """
    kwargs: dict = {
        "model": model or OPENAI_MODEL,
        "temperature": temperature,
        "streaming": streaming,
    }
    if api_key:
        kwargs["api_key"] = api_key
    elif OPENAI_API_KEY:
        kwargs["api_key"] = OPENAI_API_KEY
    effective_base_url = base_url or OPENAI_BASE_URL
    if effective_base_url:
        kwargs["base_url"] = effective_base_url
    if enable_thinking:
        kwargs["extra_body"] = {"thinking": {"type": "enabled", "clear_thinking": False}}

    return ChatOpenAI(**kwargs)


_base_llm = None


def _get_llm():
    """Lazy LLM initialization — avoid crashing at import time if no API key."""
    global _base_llm
    if _base_llm is None:
        _base_llm = create_llm(
            model=settings_manager.get_model_name(),
            api_key=settings_manager.get_api_key(),
            base_url=settings_manager.get_api_base_url(),
            enable_thinking=settings_manager.is_thinking_enabled(),
        )
    return _base_llm


def _get_model_name() -> str:
    """Get the current model name for token counting."""
    return settings_manager.get_model_name()


# ─────────────────────────────────────────────────────────
# PROMPT CONTENT HELPERS
# ─────────────────────────────────────────────────────────


def get_memory_content(user_query: str, limit: int = 5) -> str:
    """Find relevant memories for the current query."""
    memories = memory_store.find_relevant(user_query, limit=limit)
    if not memories:
        return ""

    parts = []
    for mem in memories:
        desc = mem.description or mem.content[:100]
        parts.append(f"**[{mem.category.value}]** {desc}")

    return "\n".join(parts)


def get_skills_content() -> str:
    """Build skills listing from the registry for prompt injection."""
    skills = skill_registry.list_user_invocable()
    if not skills:
        return ""
    lines = ["Available skills (use the `skill` tool or /<name> to invoke):"]
    for s in skills:
        hint = f" {s.argument_hint}" if s.argument_hint else ""
        lines.append(f"- {s.name}{hint}: {s.description}")
    return "\n".join(lines)
