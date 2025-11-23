"""
Test script for the agent workflow with judge node.
"""
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage
from Agents.agent import app


def test_simple_workflow():
    """Test the agent workflow with a simple query."""

    # Create initial state with a user message
    initial_state = {
        "messages": [
            HumanMessage(content="Explain what a Sharpe ratio is and why it's important in quantitative finance.")
        ]
    }

    print("Starting agent workflow...\n")
    print("=" * 80)

    # Run the workflow
    result = app.invoke(initial_state)

    # Print all messages in the conversation
    for i, message in enumerate(result["messages"]):
        print(f"\n[Message {i+1}] {message.__class__.__name__}:")
        print("-" * 80)
        print(message.content)
        print("-" * 80)

    print("\n" + "=" * 80)
    print("Workflow completed!")


def test_tool_calling_workflow():
    """Test the agent workflow with a query that might use tools."""

    initial_state = {
        "messages": [
            HumanMessage(content="Can you help me understand how to perform a decile sort analysis?")
        ]
    }

    print("Starting agent workflow with potential tool calls...\n")
    print("=" * 80)

    # Run the workflow
    result = app.invoke(initial_state)

    # Print all messages
    for i, message in enumerate(result["messages"]):
        print(f"\n[Message {i+1}] {message.__class__.__name__}:")
        print("-" * 80)
        if hasattr(message, 'tool_calls') and message.tool_calls:
            print(f"Tool Calls: {message.tool_calls}")
        print(message.content)
        print("-" * 80)

    print("\n" + "=" * 80)
    print("Workflow completed!")


if __name__ == "__main__":
    print("Test 1: Simple workflow (no tool calling)")
    print("=" * 80)
    test_simple_workflow()

    print("\n\n")

    print("Test 2: Workflow with potential tool calling")
    print("=" * 80)
    test_tool_calling_workflow()
