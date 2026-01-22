#agent.py - main file for orchestrating the agent workflow

from __future__ import annotations
from typing import List, Annotated, TypedDict, Optional
from pathlib import Path
from dotenv import load_dotenv
import operator
import os
import logging

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

from langchain_core.messages import BaseMessage  # HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from google.cloud import storage

from Agents.utils.tools import TOOLS

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("agent-pipeline")


# ### GCS Configuration

class Config:
    PROJECT_ID = "seraphic-ripple-476917-i5"
    SOURCE_BUCKET_NAME = "raw_papers"
    SOURCE_PREFIX = "processed/"  # Where processed papers with markdown are stored
    LOCAL_TEMP_DIR = Path("./temp_papers")


def ensure_auth():
    """Ensure GCS authentication is set up correctly"""
    try:
        credentials_path = "/workspace/secrets/service-account-key.json"
        if os.path.exists(credentials_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
            logger.info("Using service account key")

        storage_client = storage.Client(project=Config.PROJECT_ID)
        list(storage_client.list_buckets(max_results=1))
        logger.info("GCS authentication successful")
        return storage_client
    except Exception as e:
        logger.error(f"GCS authentication failed: {e}")
        return None


def list_available_papers(storage_client) -> List[str]:
    """List available processed papers from GCS bucket"""
    try:
        bucket = storage_client.bucket(Config.SOURCE_BUCKET_NAME)
        blobs = list(bucket.list_blobs(prefix=Config.SOURCE_PREFIX))
        # Get unique paper folder names (processed papers have folder structure)
        paper_names = set()
        for blob in blobs:
            # Extract paper name from path like "processed/paper_name/file.md"
            parts = blob.name.replace(Config.SOURCE_PREFIX, "").split("/")
            if parts and parts[0]:
                paper_names.add(parts[0])
        return sorted(list(paper_names))
    except Exception as e:
        logger.error(f"Error listing papers: {e}")
        return []


def download_paper_text(storage_client, paper_name: str) -> Optional[str]:
    """Download and read the markdown content of a processed paper"""
    try:
        bucket = storage_client.bucket(Config.SOURCE_BUCKET_NAME)
        # Look for markdown files in the paper's folder
        prefix = f"{Config.SOURCE_PREFIX}{paper_name}/"
        blobs = list(bucket.list_blobs(prefix=prefix))

        # Find markdown file
        md_blob = None
        for blob in blobs:
            if blob.name.endswith(".md"):
                md_blob = blob
                break

        if md_blob:
            content = md_blob.download_as_text()
            logger.info(f"Downloaded paper: {paper_name}")
            return content
        else:
            logger.warning(f"No markdown file found for paper: {paper_name}")
            return None
    except Exception as e:
        logger.error(f"Error downloading paper {paper_name}: {e}")
        return None


# ### Agent State

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    paper_name: str
    paper_text: str
    signals: Annotated[list, operator.add]
    analysis: str
    model_used: str


# Initialize GCS client at module level
_storage_client = None


def get_storage_client():
    """Get or create the GCS storage client"""
    global _storage_client
    if _storage_client is None:
        _storage_client = ensure_auth()
    return _storage_client


# ### LLM Configurations

# Main agent LLM (OpenAI)
llm = ChatOpenAI(model="gpt-5.1", temperature=0.0)
llm_with_tools = llm.bind_tools(TOOLS)

# vLLM multi-LoRA endpoint LLMs
base_llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
    model="meta-llama/Llama-3.2-3B-Instruct",
    temperature=0.7,
)

finllama_llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
    model="finllama",
    temperature=0.7,
)

custom_llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
    model="custom",
    temperature=0.7,
)


# ### Workflow Nodes

def select_paper_node(state: AgentState) -> AgentState:
    """
    Initial node to select and download a paper from GCS bucket.
    The paper_name should be provided in the initial state.
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
        logger.info(f"Available papers: {available_papers}")
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
        logger.info(f"Successfully loaded paper: {paper_name} ({len(paper_text)} characters)")
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
    """Extract trading signals from paper using FinLLaMA"""
    paper_text = state.get("paper_text", "")

    if paper_text.startswith("ERROR:"):
        logger.error(f"Cannot extract signals: {paper_text}")
        return {
            **state,
            "signals": [f"Failed to extract signals: {paper_text}"],
            "model_used": "finllama"
        }

    prompt = f"Extract the core quantitative trading signals from this paper, you will later be using the signal to create your own working code in python so study the paper carefully:\n{paper_text}"
    response = finllama_llm.invoke(prompt)

    return {
        **state,
        "signals": [response.content],
        "model_used": "finllama"
    }


# OLD Judge Node
# def judge_node(state: AgentState) -> AgentState:
#     """
#     AI LLM judge node. Evaluates the agent's work and provides assessment.
#     This runs after the main agent completes its reasoning.
#     """
#     judge_llm = ChatOpenAI(model="gpt-5.1", temperature=0.0)
#
#     # Create a judge prompt to evaluate the previous agent's output
#     judge_prompt = HumanMessage(
#         content="As an AI judge, evaluate the quality and correctness of the previous analysis. "
#                 "Provide a brief assessment of the approach, methodology, and conclusions."
#     )
#
#     response = judge_llm.invoke(state["messages"] + [judge_prompt])
#     return {"messages": [response]}


# Tool Node
tool_node = ToolNode(TOOLS)

# Graph Wiring
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("select_paper", select_paper_node)
graph.add_node("extract_signals", extract_signals)
graph.add_node("tools", tool_node)
# OLD graph.add_node("judge", judge_node)

# Set entry point - workflow starts with paper selection
graph.set_entry_point("select_paper")

# Workflow: select_paper -> extract_signals -> END
graph.add_edge("select_paper", "extract_signals")
graph.add_edge("extract_signals", END)

# # OLD conditional edges for agent with tools
# graph.add_conditional_edges(
#     "agent",
#     tools_condition,
#     {
#         "tools": "tools",
#         "__end__": END,
#     },
# )

# # After tools execute, send results back into the agent for reflection / next step.
# graph.add_edge("tools", "agent")

# # OLD After judge completes, end the workflow.
# graph.add_edge("judge", END)

app = graph.compile()


# # Example usage
# if __name__ == "__main__":
#     # List available papers
#     storage_client = ensure_auth()
#     if storage_client:
#         papers = list_available_papers(storage_client)
#         print(f"Available papers: {papers}")
#
#     # Run the workflow with a specific paper
#     initial_state = {
#         "messages": [],
#         "paper_name": "",  # Leave empty to auto-select first paper, or specify a paper name
#         "paper_text": "",
#         "signals": [],
#         "analysis": "",
#         "model_used": ""
#     }
#
#     result = app.invoke(initial_state)
#     print(f"Paper: {result['paper_name']}")
#     print(f"Signals: {result['signals']}")
#     print(f"Analysis: {result['analysis']}")
