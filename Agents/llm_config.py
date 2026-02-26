"""
LLM configuration and initialization module.
Creates and configures all LLM instances used in the agent pipeline.
"""
import httpx

from langchain_openai import ChatOpenAI
from Agents.config import LLMConfig
from Agents.utils.tools import TOOLS


def _build_lambda_http_client() -> httpx.Client:
    timeout_s = max(1.0, float(LLMConfig.LAMBDA_REQUEST_TIMEOUT))
    # Explicit transport-level timeouts protect against hangs even when model wrappers
    # fail to propagate request_timeout consistently.
    return httpx.Client(
        timeout=httpx.Timeout(timeout_s, connect=min(10.0, timeout_s)),
    )


_LAMBDA_HTTP_CLIENT = _build_lambda_http_client()


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


# Initialize LLMs at module level for reuse
llm = get_main_llm()
llm_with_tools = get_main_llm_with_tools()
base_llm = get_base_llm()
finllama_llm = get_finllama_llm()
custom_llm = get_custom_llm()


def get_llm_by_name(model_name: str):
    """
    Get an LLM instance by its name.

    This is the primary function for getting LLMs based on user selection.

    Args:
        model_name: The model to use. Options: "openai", "finllama", "base", "custom"

    Returns:
        Configured ChatOpenAI instance

    Example:
        >>> signal_llm = get_llm_by_name("openai")
        >>> response = signal_llm.invoke("Extract signals from...")
    """
    model_map = {
        "finllama": finllama_llm,
        "custom": custom_llm,
        "base": base_llm,
        "openai": llm,
    }

    return model_map.get(model_name, llm)


def get_codegen_llm():
    """
    Get the LLM for code generation (Agents 5 & 6).

    Configured independently from the main pipeline model via env vars:
      CODE_GEN_PROVIDER  — "openai" | "claude" | "custom"
      CODE_GEN_MODEL     — model name for the chosen provider
      CODE_GEN_API_KEY   — API key (falls back to provider-specific env var)
      CODE_GEN_ENDPOINT  — base URL for "custom" provider
      CODE_GEN_TEMPERATURE

    Returns:
        A LangChain chat model instance ready for code generation.
    """
    provider = LLMConfig.CODE_GEN_PROVIDER.lower()

    if provider == "claude":
        from langchain_anthropic import ChatAnthropic
        api_key = LLMConfig.CODE_GEN_API_KEY or LLMConfig.ANTHROPIC_API_KEY
        return ChatAnthropic(
            model=LLMConfig.CODE_GEN_MODEL,
            temperature=LLMConfig.CODE_GEN_TEMPERATURE,
            api_key=api_key,
        )

    if provider == "custom":
        return ChatOpenAI(
            base_url=LLMConfig.CODE_GEN_ENDPOINT,
            api_key=LLMConfig.CODE_GEN_API_KEY or "token-abc123",
            model=LLMConfig.CODE_GEN_MODEL,
            temperature=LLMConfig.CODE_GEN_TEMPERATURE,
            streaming=False,
        )

    # Default: openai
    return ChatOpenAI(
        model=LLMConfig.CODE_GEN_MODEL,
        temperature=LLMConfig.CODE_GEN_TEMPERATURE,
        api_key=LLMConfig.CODE_GEN_API_KEY or LLMConfig.OPENAI_API_KEY,
    )


def get_llm_for_task(task: str):
    """
    Get the appropriate LLM for a specific task based on configuration.

    NOTE: This function uses config-based model selection. For user-selected
    models at runtime, use get_llm_by_name() instead.

    Args:
        task: The task name. Options: "signal_extraction", "analysis", "orchestration"

    Returns:
        Configured ChatOpenAI instance for the task
    """
    task_to_config = {
        "signal_extraction": LLMConfig.SIGNAL_EXTRACTION_MODEL,
        "analysis": LLMConfig.ANALYSIS_MODEL,
        "orchestration": LLMConfig.ORCHESTRATION_MODEL,
    }

    model_name = task_to_config.get(task, "openai")
    return get_llm_by_name(model_name)
