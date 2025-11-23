
from __future__ import annotations
from typing import List, Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from tools.langgraph_tools import TOOLS


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

# Tool Node
tool_node = ToolNode(TOOLS)

# Graph Wiring
graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")

# If the LLM asks for tools, go to tool node; otherwise END.
graph.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        "end": END,
    },
)

# After tools execute, send results back into the agent for reflection / next step.
graph.add_edge("tools", "agent")

app = graph.compile()




