"""
test_code_execution.py — Simple test: prompt finllama for code, run in sandbox.
"""
import re
from openai import OpenAI
from sandbox_executor import SandboxExecutor


def strip_fences(text):
    """Remove markdown code fences from LLM output."""
    text = text.strip()
    # Remove opening fence: ```python or ```
    text = re.sub(r'^```(?:python)?\s*\n', '', text)
    # Remove closing fence
    text = re.sub(r'\n```\s*$', '', text)
    return text.strip()


# Connect to vLLM
client = OpenAI(base_url="http://localhost:8000/v1", api_key="token-abc123")

PROMPT = (
    "Write a Python script that computes the first 20 Fibonacci numbers, "
    "stores them in a list, and prints the result as:\n"
    "METRICS: {\"fibonacci_list\": [...], \"sum\": ...}\n\n"
    "Use import json and json.dumps. "
    "Return ONLY Python code, no explanations, no markdown."
)

# Ask finllama to generate simple code
print("=== PROMPTING FINLLAMA ===")
print(PROMPT)
print()

resp = client.chat.completions.create(
    model="finllama",
    messages=[{"role": "user", "content": PROMPT}],
    temperature=0.2,
    max_tokens=512,
)

raw = resp.choices[0].message.content.strip()
code = strip_fences(raw)

print("\n=== GENERATED CODE ===")
print(code)
print("=== END CODE ===\n")

# Run in sandbox
print("=== RUNNING IN SANDBOX ===")
executor = SandboxExecutor(llm=None, max_retries=0, timeout=30)
result = executor.run(code)

print(f"\nSuccess: {result['success']}")
print(f"Metrics: {result['metrics']}")
if not result["success"]:
    print(f"Error:   {result['stderr'][:500]}")
