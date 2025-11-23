
from __future__ import annotations
from typing import List, Annotated, TypedDict
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from Agents.utils.tools import TOOLS


# ### Agent State

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# ### LLM Node with Tools Bound

llm = ChatOpenAI(model="gpt-5.1", temperature=0.0)
llm_with_tools = llm.bind_tools(TOOLS)


def agent_node(state: AgentState) -> AgentState:
    """
    Main reasoning node. The model decides whether to:
    - Call a quant tool (e.g. decile_sorter, fama_macbeth_ols), or
    - Respond in natural language (e.g. explain diagnostics).
    """
    response = llm_with_tools.invoke(state["messages"])
    # Because `messages` uses add_messages, returning [response]
    # appends it onto the existing conversation.
    return {"messages": [response]}


def judge_node(state: AgentState) -> AgentState:
    """
    AI LLM judge node. Evaluates the agent's work and provides assessment.
    This runs after the main agent completes its reasoning.
    """
    judge_llm = ChatOpenAI(model="gpt-5.1", temperature=0.0)

    # Create a judge prompt to evaluate the previous agent's output
    judge_prompt = HumanMessage(
        content="As an AI judge, evaluate the quality and correctness of the previous analysis. "
                "Provide a brief assessment of the approach, methodology, and conclusions."
    )

    response = judge_llm.invoke(state["messages"] + [judge_prompt])
    return {"messages": [response]}


# Tool Node
tool_node = ToolNode(TOOLS)

# Graph Wiring
graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_node("judge", judge_node)

graph.set_entry_point("agent")

# If the LLM asks for tools, go to tool node; otherwise go to judge.
graph.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        "__end__": "judge",
    },
)

# After tools execute, send results back into the agent for reflection / next step.
graph.add_edge("tools", "agent")

# After judge completes, end the workflow.
graph.add_edge("judge", END)

app = graph.compile()




