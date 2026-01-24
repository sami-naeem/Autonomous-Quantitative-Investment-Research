"""
Main agent pipeline for autonomous quantitative investment research.

This agent orchestrates a workflow that:
1. Prompts user to select which LLM to use
2. Selects and downloads research papers from GCS
3. Extracts quantitative trading signals using the selected LLM
4. (Future) Analyzes and implements trading strategies

Author: Your Team
Project: Autonomous Quantitative Investment Research
"""
# Load environment variables BEFORE any langchain/langgraph imports
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

import logging
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from Agents.config import LoggingConfig
from Agents.nodes import AgentState, select_model_node, select_paper_node, extract_signals
from Agents.gcs_utils import ensure_auth, list_available_papers
from Agents.utils.tools import TOOLS

# Setup logging
logging.basicConfig(
    level=LoggingConfig.LEVEL,
    format=LoggingConfig.FORMAT,
)
logger = logging.getLogger(LoggingConfig.LOGGER_NAME)


def build_graph() -> StateGraph:
    """
    Build the LangGraph workflow for the agent pipeline.

    Workflow: select_model -> select_paper -> extract_signals -> END

    Returns:
        Compiled StateGraph application
    """
    # Create graph
    graph = StateGraph(AgentState)

    # Add workflow nodes
    graph.add_node("select_model", select_model_node)
    graph.add_node("select_paper", select_paper_node)
    graph.add_node("extract_signals", extract_signals)
    graph.add_node("tools", ToolNode(TOOLS))  # For future use

    # Set entry point - model selection first
    graph.set_entry_point("select_model")

    # Define workflow edges
    graph.add_edge("select_model", "select_paper")
    graph.add_edge("select_paper", "extract_signals")
    graph.add_edge("extract_signals", END)

    # Compile (LangGraph dev provides its own checkpointer for interrupt to work)
    return graph.compile()


# Build the application
app = build_graph()


def main():
    """
    Main execution function.
    Runs the agent workflow with default settings.
    """
    logger.info("=" * 80)
    logger.info("Autonomous Quantitative Investment Research Agent")
    logger.info("=" * 80)

    # Authenticate with GCS and list available papers
    storage_client = ensure_auth()
    if storage_client:
        papers = list_available_papers(storage_client)
        logger.info(f"Found {len(papers)} available papers in GCS bucket")
        if papers:
            logger.info(f"First paper: {papers[0]}")
    else:
        logger.error("Failed to authenticate with GCS. Exiting.")
        return

    # Initialize workflow state
    initial_state = {
        "messages": [],
        "model_choice": "",  # Leave empty to prompt user, or set to "openai", "finllama", "base", "custom"
        "paper_name": "",  # Leave empty to auto-select first paper
        "paper_text": "",
        "signals": [],
        "analysis": "",
        "model_used": ""
    }

    # Run the workflow
    logger.info("Starting workflow...")
    logger.info("-" * 80)

    try:
        result = app.invoke(initial_state)

        # Display results
        logger.info("=" * 80)
        logger.info("WORKFLOW COMPLETED")
        logger.info("=" * 80)
        print(f"\nPaper: {result.get('paper_name', 'N/A')}")
        print(f"\nModel Used: {result.get('model_used', 'N/A')}")
        print(f"\nExtracted Signals:\n{'-' * 80}")
        for i, signal in enumerate(result.get('signals', []), 1):
            print(f"\n[Signal {i}]")
            print(signal)
            print("-" * 80)

    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
