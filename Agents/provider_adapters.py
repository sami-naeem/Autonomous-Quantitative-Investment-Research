"""
Agents/provider_adapters.py — Unified adapter layer for frontier LLM providers.

Wraps OpenAI, Anthropic, and Google Gemini behind LangChain ChatModel interfaces
so the existing harness (tool binding, .invoke(), message types) works unchanged.

Configuration via environment variables:
    OPENAI_API_KEY      / OPENAI_MODEL      (default: gpt-4o)
    ANTHROPIC_API_KEY   / ANTHROPIC_MODEL    (default: claude-sonnet-4-20250514)
    GEMINI_API_KEY      / GEMINI_MODEL       (default: gemini-2.0-flash)

JSON reliability:
    - OpenAI:    uses response_format={"type": "json_object"} when json_mode=True
    - Anthropic: system prompt injection + post-hoc validation + retry
    - Gemini:    response_mime_type="application/json" when json_mode=True
"""

import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────────
_OPENAI_DEFAULT_MODEL = "gpt-4o"
_ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-20250514"
_GEMINI_DEFAULT_MODEL = "gemini-2.0-flash"

_JSON_MAX_RETRIES = 3

# ── Provider registry ──────────────────────────────────────────────────
FRONTIER_PROVIDERS = {"openai", "anthropic", "gemini"}


def is_frontier_provider(model_choice: str) -> bool:
    """Return True if model_choice is a frontier cloud provider."""
    return model_choice.lower() in FRONTIER_PROVIDERS


# =====================================================================
# OpenAI adapter
# =====================================================================
def _get_openai_llm(
    *,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    json_mode: bool = False,
):
    """Return a LangChain ChatOpenAI instance."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "langchain-openai is required for the OpenAI provider. "
            "Install: pip install langchain-openai"
        )

    api_key = os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("OPENAI_MODEL", _OPENAI_DEFAULT_MODEL)

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Set it in your .env file or export it."
        )

    kwargs: Dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

    llm = ChatOpenAI(**kwargs)

    request_id = str(uuid.uuid4())[:8]
    logger.info(
        f"[OpenAI] Initialized: model={model} temp={temperature} "
        f"max_tokens={max_tokens} json_mode={json_mode} req_id={request_id}"
    )
    return llm


# =====================================================================
# Anthropic adapter
# =====================================================================
def _get_anthropic_llm(
    *,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    json_mode: bool = False,
):
    """Return a LangChain ChatAnthropic instance."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise ImportError(
            "langchain-anthropic is required for the Anthropic provider. "
            "Install: pip install langchain-anthropic"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    model = os.environ.get("ANTHROPIC_MODEL", _ANTHROPIC_DEFAULT_MODEL)

    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Set it in your .env file or export it."
        )

    llm = ChatAnthropic(
        model=model,
        anthropic_api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    request_id = str(uuid.uuid4())[:8]
    logger.info(
        f"[Anthropic] Initialized: model={model} temp={temperature} "
        f"max_tokens={max_tokens} json_mode={json_mode} req_id={request_id}"
    )
    return llm


# =====================================================================
# Gemini adapter
# =====================================================================
def _get_gemini_llm(
    *,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    json_mode: bool = False,
):
    """Return a LangChain ChatGoogleGenerativeAI instance."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise ImportError(
            "langchain-google-genai is required for the Gemini provider. "
            "Install: pip install langchain-google-genai"
        )

    api_key = os.environ.get("GEMINI_API_KEY", "")
    model = os.environ.get("GEMINI_MODEL", _GEMINI_DEFAULT_MODEL)

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set. "
            "Set it in your .env file or export it."
        )

    kwargs: Dict[str, Any] = {
        "model": model,
        "google_api_key": api_key,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }

    if json_mode:
        kwargs["model_kwargs"] = {"response_mime_type": "application/json"}

    llm = ChatGoogleGenerativeAI(**kwargs)

    request_id = str(uuid.uuid4())[:8]
    logger.info(
        f"[Gemini] Initialized: model={model} temp={temperature} "
        f"max_tokens={max_tokens} json_mode={json_mode} req_id={request_id}"
    )
    return llm


# =====================================================================
# Unified factory
# =====================================================================
def get_frontier_llm(
    provider: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    json_mode: bool = False,
):
    """Return a LangChain ChatModel for the given frontier provider.

    Args:
        provider:    One of "openai", "anthropic", "gemini".
        temperature: Sampling temperature.
        max_tokens:  Maximum output tokens.
        json_mode:   If True, enable provider-native JSON output mode where
                     available (OpenAI json_object, Gemini response_mime_type).
                     For Anthropic, JSON compliance is handled via prompt
                     engineering + retry in the harness.

    Returns:
        A LangChain ChatModel with .invoke() / .bind_tools() support.
    """
    provider_lower = provider.lower().strip()
    common = dict(temperature=temperature, max_tokens=max_tokens, json_mode=json_mode)

    if provider_lower == "openai":
        return _get_openai_llm(**common)
    elif provider_lower == "anthropic":
        return _get_anthropic_llm(**common)
    elif provider_lower == "gemini":
        return _get_gemini_llm(**common)
    else:
        raise ValueError(
            f"Unknown frontier provider: {provider!r}. "
            f"Supported: {sorted(FRONTIER_PROVIDERS)}"
        )


# =====================================================================
# JSON validation + retry wrapper
# =====================================================================
def invoke_with_json_retry(
    llm,
    messages,
    *,
    provider: str,
    schema_hint: str = "",
    max_retries: int = _JSON_MAX_RETRIES,
) -> dict:
    """Invoke the LLM and ensure the response is valid JSON.

    Strategy per provider:
        - OpenAI (json_mode):   Usually valid on first try; retry if not.
        - Anthropic:            No native JSON mode; rely on prompt + retry.
        - Gemini (json_mode):   Usually valid; retry if not.

    Args:
        llm:          LangChain ChatModel.
        messages:     List of LangChain message objects.
        provider:     Provider name for logging.
        schema_hint:  A string describing the expected JSON schema, appended
                      to retry prompts.
        max_retries:  Number of retries on invalid JSON before raising.

    Returns:
        Parsed dict from model output.

    Raises:
        ValueError: If valid JSON cannot be obtained after retries.
    """
    from langchain_core.messages import HumanMessage

    last_raw = ""
    for attempt in range(1, max_retries + 1):
        request_id = str(uuid.uuid4())[:8]
        logger.info(
            f"[{provider}] JSON invoke attempt {attempt}/{max_retries} "
            f"(req_id={request_id})"
        )

        try:
            resp = llm.invoke(messages)
        except Exception as e:
            logger.warning(
                f"[{provider}] LLM invoke failed on attempt {attempt}: "
                f"{type(e).__name__}: {e}"
            )
            if attempt == max_retries:
                raise
            time.sleep(1)
            continue

        raw = getattr(resp, "content", "") or ""
        last_raw = raw

        # Try parsing
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON object from surrounding prose
        start = cleaned.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(cleaned)):
                if cleaned[i] == "{":
                    depth += 1
                elif cleaned[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(cleaned[start:i + 1])
                        except json.JSONDecodeError:
                            break

        if attempt < max_retries:
            schema_reminder = ""
            if schema_hint:
                schema_reminder = f"\nExpected schema:\n{schema_hint}\n"
            retry_msg = HumanMessage(
                content=(
                    "Your previous response was not valid JSON. "
                    "Return ONLY a single valid JSON object. "
                    "No prose, no markdown code fences, no trailing commas.\n"
                    f"{schema_reminder}"
                )
            )
            messages = list(messages) + [
                resp,  # include the assistant's bad response for context
                retry_msg,
            ]
            logger.warning(
                f"[{provider}] Attempt {attempt} produced invalid JSON. "
                f"Retrying with schema reminder."
            )
        else:
            raise ValueError(
                f"[{provider}] Could not obtain valid JSON after {max_retries} "
                f"attempts. Last raw output:\n{last_raw[:500]}"
            )

    # Should not reach here
    raise ValueError(f"[{provider}] JSON retry loop exited unexpectedly.")