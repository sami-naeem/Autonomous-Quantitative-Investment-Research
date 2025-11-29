"""
Test script for the agent workflow with judge node.

Usage:
    python Agents/test_agent.py simple     # Run simple conversational test
    python Agents/test_agent.py tools      # Run sequential tests of all quant tools
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage
from Agents.agent import app


def print_result(result, test_name):
    """Helper to print workflow results."""
    print(f"\n{'='*80}")
    print(f"Test: {test_name}")
    print('='*80)

    for i, message in enumerate(result["messages"]):
        print(f"\n[Message {i+1}] {message.__class__.__name__}:")
        print("-" * 80)
        if hasattr(message, 'tool_calls') and message.tool_calls:
            print(f"Tool Calls: {message.tool_calls}\n")
        print(message.content)
        print("-" * 80)

    print("\n" + "=" * 80)
    print("Workflow completed!")
    print("=" * 80 + "\n")


def test_simple_workflow():
    """Test the agent workflow with a simple conversational query."""

    initial_state = {
        "messages": [
            HumanMessage(content="Explain what a Sharpe ratio is and why it's important in quantitative finance.")
        ]
    }

    result = app.invoke(initial_state)
    print_result(result, "Simple Workflow - Sharpe Ratio Explanation")


def create_sample_data():
    """Create sample data for testing quant tools."""
    np.random.seed(42)

    dates = pd.date_range('2020-01-01', periods=60, freq='M')
    assets = [f'STOCK_{i}' for i in range(1, 21)]

    data = []
    for date in dates:
        for asset in assets:
            data.append({
                'date': date,
                'asset': asset,
                'signal': np.random.randn(),
                'ret': np.random.randn() * 0.02,
                'industry': np.random.choice(['Tech', 'Finance', 'Healthcare', 'Energy'])
            })

    return pd.DataFrame(data)


def test_tools_sequential():
    """Test each quant tool sequentially through the agent."""

    # List of test queries for each tool/concept
    tool_tests = [
        {
            "name": "Decile Sorter",
            "query": "Explain how decile sorting works in quantitative finance and when I should use it."
        },
        {
            "name": "Z-Score Normalization",
            "query": "What is z-score normalization and why is it important for cross-sectional analysis?"
        },
        {
            "name": "Industry Neutralization",
            "query": "Explain industry neutralization in factor investing and its benefits."
        },
        {
            "name": "Fama-MacBeth Regression",
            "query": "What is Fama-MacBeth regression and when should I use it for factor analysis?"
        },
        {
            "name": "Backtesting Long-Short Portfolios",
            "query": "How do I backtest a long-short portfolio strategy? What metrics should I focus on?"
        },
        {
            "name": "Event Study Methodology",
            "query": "Explain event study methodology in finance research and its key components."
        },
        {
            "name": "Term Structure Analysis",
            "query": "What is term structure analysis for signals and how does half-life relate to signal decay?"
        },
        {
            "name": "Search Research Themes (Tool Call)",
            "query": "Search for research on momentum strategies in equity markets."
        }
    ]

    print("\n" + "="*80)
    print("SEQUENTIAL TOOL TESTING")
    print("="*80)
    print(f"Running {len(tool_tests)} tests sequentially...\n")

    for i, test in enumerate(tool_tests, 1):
        print(f"\n[{i}/{len(tool_tests)}] Testing: {test['name']}")
        print("-" * 80)

        initial_state = {
            "messages": [HumanMessage(content=test['query'])]
        }

        try:
            result = app.invoke(initial_state)
            print_result(result, f"{test['name']} - Query: {test['query'][:60]}...")
        except Exception as e:
            print(f"ERROR in {test['name']}: {str(e)}")
            print("=" * 80 + "\n")
            continue

        # Small separator between tests
        print("\n" + "~"*80 + "\n")


def main():
    """Main entry point for the test script."""

    if len(sys.argv) < 2:
        print(__doc__)
        print("\nPlease specify test mode:")
        print("  simple - Run simple conversational test")
        print("  tools  - Run sequential tests of all quant tools")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "simple":
        print("\n" + "="*80)
        print("SIMPLE WORKFLOW TEST")
        print("="*80)
        test_simple_workflow()

    elif mode == "tools":
        test_tools_sequential()

    else:
        print(f"Unknown mode: {mode}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
