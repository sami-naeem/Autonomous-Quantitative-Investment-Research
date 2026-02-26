"""
LangGraph workflow nodes for the agent pipeline.
Each node represents a step in the paper analysis workflow.
"""
import logging
import random
from typing import TypedDict, List, Annotated, Literal, Optional
import operator

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from langgraph.types import interrupt

from Agents.gcs_utils import get_storage_client, list_available_papers, download_paper_text
from Agents.llm_config import get_llm_by_name

logger = logging.getLogger("agent-pipeline")

# Valid model choices
MODEL_OPTIONS = ["openai", "finllama", "base", "custom"]


def _last_value(existing: str, new: str) -> str:
    """Reducer that keeps the latest non-empty value (last-write-wins)."""
    return new if new else existing


def _last_bool(existing: bool, new: bool) -> bool:
    """Reducer that always takes the latest boolean value."""
    return new


def _last_optional(existing, new):
    """Reducer for optional fields — keeps new value only when it is not None."""
    return new if new is not None else existing


class AgentState(TypedDict):
    """State schema for the agent workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    model_choice: Annotated[str, _last_value]
    paper_name: Annotated[str, _last_value]
    paper_text: Annotated[str, _last_value]
    signals: Annotated[list, operator.add]
    analysis: Annotated[str, _last_value]
    model_used: Annotated[str, _last_value]
    # New keys for simplified pipeline
    abstract: Annotated[str, _last_value]
    theme_pick: Annotated[dict, _last_value]
    factor_extract: Annotated[dict, _last_value]
    output_gcs_uri: Annotated[str, _last_value]
    # Endpoint validation result — set by validate_endpoint_node
    endpoint_valid: Annotated[bool, _last_bool]
    endpoint_error: Annotated[str, _last_value]
    # Optional seed for reproducible paper selection across coordinated runs
    random_seed: Annotated[Optional[int], _last_optional]
    # Fields used by load_paper_markdown but not previously declared in TypedDict
    paper_gcs_uri: Annotated[str, _last_value]
    paper_blob_name: Annotated[str, _last_value]
    # GCS checkpoint URIs written inline by each agent node
    triage_gcs_uri: Annotated[str, _last_value]
    factor_gcs_uri: Annotated[str, _last_value]


def select_model_node(state: AgentState) -> AgentState:
    """
    First node: prompt user to select which LLM to use for the workflow.

    Uses interrupt() to pause and wait for user input in LangGraph Studio.
    Re-interrupts when routed back from a failed endpoint validation so the
    user can pick a different model.

    Args:
        state: Current agent state

    Returns:
        Updated state with model_choice and cleared endpoint_error
    """
    model_choice = state.get("model_choice", "")
    endpoint_valid = state.get("endpoint_valid", False)
    endpoint_error = state.get("endpoint_error", "")

    # endpoint_valid defaults to False in the TypedDict before any validation runs.
    # An actual validation failure always sets endpoint_error to a non-empty string.
    # So: False + no error = not yet validated (treat as OK to bypass interrupt).
    #     False + error    = validation explicitly failed (prompt user).
    endpoint_failed = not endpoint_valid and bool(endpoint_error)

    # Skip interrupt if a valid model was pre-selected and validation hasn't failed
    if model_choice and model_choice in MODEL_OPTIONS and not endpoint_failed:
        logger.info(f"Using pre-selected model: {model_choice}")
        return {"model_choice": model_choice}

    if endpoint_failed:
        logger.warning(
            f"Endpoint validation failed for '{model_choice}': {endpoint_error}. "
            "Prompting model re-selection."
        )

    # Interrupt and wait for user input
    selection = interrupt(
        {
            "question": "Select the LLM to use for this workflow:",
            "options": MODEL_OPTIONS,
            "default": "openai",
            **({"error": endpoint_error} if endpoint_error else {}),
        }
    )

    # Validate selection
    if selection not in MODEL_OPTIONS:
        logger.warning(f"Invalid model '{selection}', defaulting to 'openai'")
        selection = "openai"

    logger.info(f"User selected model: {selection}")
    # Optimistically reset endpoint_valid so the next validation starts clean
    return {"model_choice": selection, "endpoint_valid": True}


def select_paper_node(state: AgentState) -> AgentState:
    """
    Initial node to select and download a paper from GCS bucket.

    If paper_name is provided in state, downloads that specific paper.
    Otherwise, lists available papers and auto-selects the first one.

    Args:
        state: Current agent state

    Returns:
        Updated state with paper_name and paper_text
    """
    storage_client = get_storage_client()
    if storage_client is None:
        logger.error("Failed to authenticate with GCS")
        return {
            **state,
            "paper_text": "ERROR: Failed to authenticate with GCS"
        }

    paper_name = state.get("paper_name", "")

    if not paper_name:
        # If no paper specified, list available papers
        available_papers = list_available_papers(storage_client)
        logger.info(f"Found {len(available_papers)} available papers")

        # Select from available papers (seeded if provided).
        # Integer seeds map directly to an index (seed % N) so consecutive
        # integers always yield distinct papers. random.choice() is not used
        # for integers because different seeds can collide to the same index
        # in small paper pools (e.g. seeds 3 and 13 both hit index 3 in a
        # 10-paper list).
        if available_papers:
            seed = state.get("random_seed")
            if isinstance(seed, int):
                paper_name = available_papers[seed % len(available_papers)]
            else:
                paper_name = random.Random(seed).choice(available_papers)
            logger.info(f"Selected paper (seed={seed}, pool={len(available_papers)}): {paper_name}")
        else:
            return {
                **state,
                "paper_text": "ERROR: No papers available in bucket"
            }

    # Download the paper content
    paper_text = download_paper_text(storage_client, paper_name)

    if paper_text:
        return {
            **state,
            "paper_name": paper_name,
            "paper_text": paper_text
        }
    else:
        return {
            **state,
            "paper_name": paper_name,
            "paper_text": f"ERROR: Could not download paper {paper_name}"
        }


def extract_signals(state: AgentState) -> AgentState:
    """
    Extract trading signals from paper using the user-selected LLM.

    Uses the model specified in state.model_choice to analyze the paper
    and extract quantitative trading signals that can be implemented in Python.

    Args:
        state: Current agent state with paper_text and model_choice

    Returns:
        Updated state with extracted signals and model_used
    """
    paper_text = state.get("paper_text", "")
    model_choice = state.get("model_choice", "openai")

    if paper_text.startswith("ERROR:"):
        logger.error(f"Cannot extract signals: {paper_text}")
        return {
            "signals": [f"Failed to extract signals: {paper_text}"],
            "model_used": model_choice
        }

    # Get the LLM based on user's selection
    signal_llm = get_llm_by_name(model_choice)

    logger.info(f"Extracting trading signals using {model_choice}...")
    prompt = (
        "Extract the core quantitative trading signals from this paper. "
        "You will later be using the signal to create your own working code in Python, "
        "so study the paper carefully and be specific:\n\n"
        f"{paper_text}"
    )

    try:
        response = signal_llm.invoke(prompt)
        logger.info("Successfully extracted signals")

        return {
            "signals": [response.content],
            "model_used": model_choice
        }
    except Exception as e:
        logger.error(f"Error extracting signals: {e}")

        return {
            "signals": [f"Error extracting signals: {str(e)}"],
            "model_used": model_choice
        }
