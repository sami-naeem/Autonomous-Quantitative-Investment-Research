"""
LLM configuration and initialization module.
Creates and configures all LLM instances used in the agent pipeline.

Supports:
    - "openai"    → OpenAI via LLMConfig.MAIN_MODEL (original behavior)
    - "base"      → vLLM base Llama on Lambda endpoint
    - "finllama"  → vLLM finance-tuned Llama on Lambda endpoint
    - "custom"    → vLLM custom fine-tune on Lambda endpoint
    - "anthropic" → Anthropic Claude (new)
    - "gemini"    → Google Gemini (new)

Frontier providers (anthropic, gemini) are lazily initialized on first use
to avoid import errors when their SDKs aren't installed.

Environment variables for new providers:
    ANTHROPIC_API_KEY / ANTHROPIC_MODEL  (default: claude-sonnet-4-20250514)
    GEMINI_API_KEY    / GEMINI_MODEL     (default: gemini-2.0-flash)
"""
import logging
import httpx

from langchain_openai import ChatOpenAI
from Agents.config import LLMConfig
from Agents.utils.tools import TOOLS

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════
# Lambda / vLLM HTTP client (original)
# ═════════════════════════════════════════════════════════════════════════
def _build_lambda_http_client() -> httpx.Client:
    timeout_s = max(1.0, float(LLMConfig.LAMBDA_REQUEST_TIMEOUT))
    # Explicit transport-level timeouts protect against hangs even when model wrappers
    # fail to propagate request_timeout consistently.
    return httpx.Client(
        timeout=httpx.Timeout(timeout_s, connect=min(10.0, timeout_s)),
    )


_LAMBDA_HTTP_CLIENT = _build_lambda_http_client()


# ═════════════════════════════════════════════════════════════════════════
# Original LLM factory functions (unchanged)
# ═════════════════════════════════════════════════════════════════════════
def get_main_llm():
    """Get the main OpenAI LLM for orchestration"""
    return ChatOpenAI(
        model=LLMConfig.MAIN_MODEL,
        temperature=LLMConfig.MAIN_TEMPERATURE
    )


def get_main_llm_with_tools():
    """Get the main LLM with tools bound"""
    llm = get_main_llm()
    return llm.bind_tools(TOOLS)


def get_base_llm():
    """Get the base Llama model from vLLM endpoint"""
    return ChatOpenAI(
        base_url=LLMConfig.LAMBDA_ENDPOINT,
        api_key=LLMConfig.LAMBDA_API_KEY,
        model=LLMConfig.BASE_MODEL,
        temperature=LLMConfig.LAMBDA_TEMPERATURE,
        request_timeout=LLMConfig.LAMBDA_REQUEST_TIMEOUT,
        max_retries=LLMConfig.LAMBDA_MAX_RETRIES,
        http_client=_LAMBDA_HTTP_CLIENT,
        streaming=False,
    )


def get_finllama_llm():
    """Get the FinLLaMA model from vLLM endpoint (with LoRA adapter)"""
    return ChatOpenAI(
        base_url=LLMConfig.LAMBDA_ENDPOINT,
        api_key=LLMConfig.LAMBDA_API_KEY,
        model=LLMConfig.FINLLAMA_MODEL,
        temperature=LLMConfig.LAMBDA_TEMPERATURE,
        request_timeout=LLMConfig.LAMBDA_REQUEST_TIMEOUT,
        max_retries=LLMConfig.LAMBDA_MAX_RETRIES,
        http_client=_LAMBDA_HTTP_CLIENT,
        streaming=False,
    )


def get_custom_llm():
    """Get the custom model from vLLM endpoint (with custom LoRA adapter)"""
    return ChatOpenAI(
        base_url=LLMConfig.LAMBDA_ENDPOINT,
        api_key=LLMConfig.LAMBDA_API_KEY,
        model=LLMConfig.CUSTOM_MODEL,
        temperature=LLMConfig.LAMBDA_TEMPERATURE,
        request_timeout=LLMConfig.LAMBDA_REQUEST_TIMEOUT,
        max_retries=LLMConfig.LAMBDA_MAX_RETRIES,
        http_client=_LAMBDA_HTTP_CLIENT,
        streaming=False,
    )


# ═════════════════════════════════════════════════════════════════════════
# Frontier provider factory functions (new — lazily loaded)
# ═════════════════════════════════════════════════════════════════════════
def get_anthropic_llm():
    """Get the Anthropic Claude model.

    Requires: pip install langchain-anthropic
    Env vars: ANTHROPIC_API_KEY, ANTHROPIC_MODEL (optional)
    """
    from Agents.provider_adapters import get_frontier_llm
    return get_frontier_llm(
        "anthropic",
        temperature=getattr(LLMConfig, "MAIN_TEMPERATURE", 0.2),
        max_tokens=4096,
    )


def get_gemini_llm():
    """Get the Google Gemini model.

    Requires: pip install langchain-google-genai
    Env vars: GEMINI_API_KEY, GEMINI_MODEL (optional)
    """
    from Agents.provider_adapters import get_frontier_llm
    return get_frontier_llm(
        "gemini",
        temperature=getattr(LLMConfig, "MAIN_TEMPERATURE", 0.2),
        max_tokens=4096,
    )


# ═════════════════════════════════════════════════════════════════════════
# Module-level LLM singletons (original — eagerly initialized)
# ═════════════════════════════════════════════════════════════════════════
llm = get_main_llm()
llm_with_tools = get_main_llm_with_tools()
base_llm = get_base_llm()
finllama_llm = get_finllama_llm()
custom_llm = get_custom_llm()

# Frontier providers are NOT eagerly initialized — they are created on
# first call to get_llm_by_name("anthropic") / get_llm_by_name("gemini").
# This avoids import errors when their SDKs aren't installed and avoids
# requiring API keys that won't be used.
_frontier_cache = {}


def _get_cached_frontier(provider: str):
    """Return a cached frontier LLM, creating it on first access."""
    if provider not in _frontier_cache:
        if provider == "anthropic":
            _frontier_cache[provider] = get_anthropic_llm()
        elif provider == "gemini":
            _frontier_cache[provider] = get_gemini_llm()
        else:
            raise ValueError(f"Unknown frontier provider: {provider!r}")
    return _frontier_cache[provider]


# ═════════════════════════════════════════════════════════════════════════
# Primary dispatcher (updated)
# ═════════════════════════════════════════════════════════════════════════
# All valid model choice strings accepted by the pipeline.
ALL_MODEL_CHOICES = ["openai", "anthropic", "gemini", "base", "finllama", "custom"]

# Set of choices that are cloud frontier providers (not vLLM).
FRONTIER_PROVIDERS = {"anthropic", "gemini"}


def get_llm_by_name(model_name: str):
    """
    Get an LLM instance by its name.

    This is the primary function for getting LLMs based on user selection.

    Args:
        model_name: The model to use.
            Original:  "openai", "finllama", "base", "custom"
            New:       "anthropic", "gemini"

    Returns:
        Configured ChatModel instance (ChatOpenAI, ChatAnthropic,
        or ChatGoogleGenerativeAI depending on provider).

    Example:
        >>> signal_llm = get_llm_by_name("openai")
        >>> response = signal_llm.invoke("Extract signals from...")
        >>> claude_llm = get_llm_by_name("anthropic")
        >>> response = claude_llm.invoke("Extract signals from...")
    """
    name = (model_name or "").strip().lower()

    # ── Frontier providers (lazily initialized) ─────────────────────
    if name in FRONTIER_PROVIDERS:
        return _get_cached_frontier(name)

    # ── Original eagerly-initialized models ─────────────────────────
    model_map = {
        "finllama": finllama_llm,
        "custom": custom_llm,
        "base": base_llm,
        "openai": llm,
    }

    return model_map.get(name, llm)


def get_llm_for_task(task: str):
    """
    Get the appropriate LLM for a specific task based on configuration.

    NOTE: This function uses config-based model selection. For user-selected
    models at runtime, use get_llm_by_name() instead.

    Args:
        task: The task name. Options: "signal_extraction", "analysis", "orchestration"

    Returns:
        Configured ChatModel instance for the task
    """
    task_to_config = {
        "signal_extraction": LLMConfig.SIGNAL_EXTRACTION_MODEL,
        "analysis": LLMConfig.ANALYSIS_MODEL,
        "orchestration": LLMConfig.ORCHESTRATION_MODEL,
    }

    model_name = task_to_config.get(task, "openai")
    return get_llm_by_name(model_name)