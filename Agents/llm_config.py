"""
LLM configuration and initialization module.
Creates and configures all LLM instances used in the agent pipeline.
"""
from langchain_openai import ChatOpenAI
from Agents.config import LLMConfig
from Agents.utils.tools import TOOLS


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
        streaming=False,
    )


def get_finllama_llm():
    """Get the FinLLaMA model from vLLM endpoint (with LoRA adapter)"""
    return ChatOpenAI(
        base_url=LLMConfig.LAMBDA_ENDPOINT,
        api_key=LLMConfig.LAMBDA_API_KEY,
        model=LLMConfig.FINLLAMA_MODEL,
        temperature=LLMConfig.LAMBDA_TEMPERATURE,
        streaming=False,
    )


def get_custom_llm():
    """Get the custom model from vLLM endpoint (with custom LoRA adapter)"""
    return ChatOpenAI(
        base_url=LLMConfig.LAMBDA_ENDPOINT,
        api_key=LLMConfig.LAMBDA_API_KEY,
        model=LLMConfig.CUSTOM_MODEL,
        temperature=LLMConfig.LAMBDA_TEMPERATURE,
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
