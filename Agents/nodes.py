"""
LangGraph workflow nodes for the agent pipeline.
Each node represents a step in the paper analysis workflow.
"""
import logging
from typing import TypedDict, List, Annotated, Literal
import operator

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from langgraph.types import interrupt

from Agents.gcs_utils import get_storage_client, list_available_papers, download_paper_text
from Agents.llm_config import get_llm_by_name

logger = logging.getLogger("agent-pipeline")

# Valid model choices
MODEL_OPTIONS = ["openai", "finllama", "base", "custom"]


class AgentState(TypedDict):
    """State schema for the agent workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    model_choice: str  # User-selected model: "openai", "finllama", "base", or "custom"
    paper_name: str
    paper_text: str
    signals: Annotated[list, operator.add]
    analysis: str
    model_used: str


def select_model_node(state: AgentState) -> AgentState:
    """
    First node: prompt user to select which LLM to use for the workflow.

    Uses interrupt() to pause and wait for user input in LangGraph Studio.

    Args:
        state: Current agent state

    Returns:
        Updated state with model_choice
    """
    # Check if model was already provided in initial state
    model_choice = state.get("model_choice", "")

    if model_choice and model_choice in MODEL_OPTIONS:
        logger.info(f"Using pre-selected model: {model_choice}")
        return {"model_choice": model_choice}

    # Interrupt and wait for user input
    selection = interrupt(
        {
            "question": "Select the LLM to use for this workflow:",
            "options": MODEL_OPTIONS,
            "default": "openai"
        }
    )

    # Validate selection
    if selection not in MODEL_OPTIONS:
        logger.warning(f"Invalid model '{selection}', defaulting to 'openai'")
        selection = "openai"

    logger.info(f"User selected model: {selection}")
    return {"model_choice": selection}


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

        # Select the first available paper by default
        if available_papers:
            paper_name = available_papers[0]
            logger.info(f"Auto-selected paper: {paper_name}")
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
