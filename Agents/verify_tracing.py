"""
Verify that LangSmith tracing is working correctly.
"""
import sys
from pathlib import Path
import os

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Check if environment variables are loaded
print("Environment Variables Check:")
print("=" * 80)
print(f"LANGCHAIN_TRACING_V2: {os.getenv('LANGCHAIN_TRACING_V2')}")
print(f"LANGCHAIN_API_KEY: {'SET' if os.getenv('LANGCHAIN_API_KEY') else 'NOT SET'}")
print(f"LANGSMITH_WORKSPACE_ID: {os.getenv('LANGSMITH_WORKSPACE_ID')}")
print(f"LANGSMITH_PROJECT: {os.getenv('LANGSMITH_PROJECT')}")
print(f"OPENAI_API_KEY: {'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")
print("=" * 80)

from langchain_core.messages import HumanMessage
from Agents.agent import app

print("\nRunning a simple test with tracing enabled...")
print("=" * 80)

initial_state = {
    "messages": [
        HumanMessage(content="What is alpha in finance?")
    ]
}

result = app.invoke(initial_state)

print("\nWorkflow completed!")
print("=" * 80)
print("\nCheck your LangSmith dashboard at:")
print(f"https://smith.langchain.com/o/{os.getenv('LANGSMITH_WORKSPACE_ID')}/projects/p/{os.getenv('LANGSMITH_PROJECT')}")
print("\nYou should see a trace for this run with the agent and judge nodes.")
