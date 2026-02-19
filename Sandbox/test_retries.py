"""
test_backtest_with_retries.py — Full test of the retry loop:

    1. Prompt finllama to generate a momentum backtest
    2. Run it in sandbox
    3. If it fails: diagnose error → fix code → retry (up to N times)
    4. Log everything: prompts, diagnoses, fixes, errors

Drop next to sandbox_executor.py and run:
    python3 test_backtest_with_retries.py
"""
import json
import logging
import re
import textwrap
from openai import OpenAI
from sandbox_executor import SandboxExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest_test")

DIVIDER = "=" * 70


# ─── LLM wrapper ────────────────────────────────────────────────────────
class VLLMClient:
    """
    Wraps vLLM's OpenAI-compatible endpoint.
    Exposes .invoke(messages) for SandboxExecutor compatibility.
    """
    def __init__(self, base_url="http://localhost:8000/v1", api_key="token-abc123", model="finllama"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def invoke(self, messages):
        # Convert to OpenAI format
        formatted = []
        for m in messages:
            if isinstance(m, dict):
                formatted.append(m)
            else:
                role = "user"
                cls_name = m.__class__.__name__
                if cls_name == "SystemMessage" or getattr(m, "type", "") == "system":
                    role = "system"
                elif cls_name == "HumanMessage" or getattr(m, "type", "") == "human":
                    role = "user"
                elif cls_name == "AIMessage" or getattr(m, "type", "") == "ai":
                    role = "assistant"
                formatted.append({"role": role, "content": m.content})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=formatted,
            temperature=0.2,
            max_tokens=4096,
        )

        class Resp:
            content = response.choices[0].message.content
        return Resp()


def strip_fences(text):
    """Remove markdown code fences from LLM output."""
    text = text.strip()
    text = re.sub(r'^```(?:python)?\s*\n', '', text)
    text = re.sub(r'\n```\s*$', '', text)
    return text.strip()


# ─── Main test ───────────────────────────────────────────────────────────
def main():
    MAX_RETRIES = 5

    print(f"\n{DIVIDER}")
    print(f"BACKTEST GENERATION + RETRY LOOP TEST")
    print(f"Model: finllama | Max retries: {MAX_RETRIES}")
    print(f"{DIVIDER}\n")

    # ── Connect to vLLM ─────────────────────────────────────────────
    llm = VLLMClient(model="finllama")

    print("[1] Checking vLLM connection...")
    try:
        resp = llm.invoke([{"role": "user", "content": "Say hello"}])
        print(f"    ✅ vLLM responding: {resp.content.strip()[:50]}\n")
    except Exception as e:
        print(f"    ❌ Cannot reach vLLM: {e}")
        return

    # ── Generate backtest code ──────────────────────────────────────
    PROMPT = textwrap.dedent("""\
        Write a complete, self-contained Python script that does the following:

        1. Generate synthetic daily price data:
           - 10 stocks, 3 years of business days
           - Use numpy with seed 42
           - Each stock starts at price 100
           - Daily returns are random normal (mean=0.0003, std=0.015)
           - Compute prices as: prices = 100 * np.exp(np.cumsum(returns))

        2. Simple momentum strategy:
           - Compute 60-day momentum (percent change over 60 days) for each stock
           - Each month (every 21 trading days), go long the top 3 stocks by momentum
           - Equal weight the 3 stocks, compute daily portfolio returns

        3. Compute and print metrics:
           - Sharpe ratio (annualized)
           - Total return
           - Max drawdown
           - Number of trading days
           - Print EXACTLY: print("METRICS:", json.dumps({...}))

        4. Save daily returns:
           - Save CSV with columns 'date' and 'strategy_returns' to 'returns.csv'

        RULES:
        - Use ONLY numpy, pandas, json. No other libraries.
        - All imports at the top of the file.
        - No external data files — generate everything synthetically.
        - Return ONLY Python code. No explanations. No markdown fences.
    """)

    print("[2] Prompting finllama to generate backtest code...")
    print(f"    PROMPT:\n{textwrap.indent(PROMPT, '    ')}\n")

    resp = llm.invoke([
        {"role": "system", "content": "You are a Python code generator. Return ONLY valid Python code. No markdown fences, no explanations."},
        {"role": "user", "content": PROMPT},
    ])

    code = strip_fences(resp.content.strip())
    print(f"[3] Generated {len(code)} chars of code")
    print(f"    First 5 lines:")
    for line in code.split("\n")[:5]:
        print(f"      {line}")
    print()

    # Save for inspection
    with open("generated_backtest.py", "w") as f:
        f.write(code)
    print(f"    💾 Saved to: generated_backtest.py\n")

    # ── Run in sandbox WITH retry loop ──────────────────────────────
    print(f"[4] Running in sandbox (max_retries={MAX_RETRIES}, LLM fixes errors)...\n")

    executor = SandboxExecutor(
        llm=llm,
        max_retries=MAX_RETRIES,
        timeout=120,
        extra_packages=["pandas", "numpy"],
        keep_sandbox=True,
    )

    result = executor.run(code)

    # ── Results ─────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print(f"RESULTS")
    print(f"{DIVIDER}")
    print(f"  Success:       {result['success']}")
    print(f"  Retries used:  {result['retries_used']} / {MAX_RETRIES}")

    if result["success"]:
        print(f"  Metrics:")
        if result["metrics"]:
            for k, v in result["metrics"].items():
                print(f"    {k}: {v}")
        else:
            print(f"    (no metrics parsed)")
            print(f"    stdout: {result['stdout'][:200]}")

        if result.get("returns_csv"):
            print(f"  📊 Returns CSV: {result['returns_csv']}")
        if result.get("output_files"):
            print(f"  📁 Output files: {result['output_files']}")

        # Save final code if it was modified
        if result["retries_used"] > 0:
            with open("generated_backtest_fixed.py", "w") as f:
                f.write(result["code_final"])
            print(f"  💾 Fixed code saved to: generated_backtest_fixed.py")

    else:
        print(f"\n  ❌ FAILED after {result['retries_used']} retries")
        print(f"  Last error:\n{textwrap.indent(result['stderr'][:500], '    ')}")

    # Show error history
    if result["error_history"]:
        print(f"\n  Error history ({len(result['error_history'])} attempts):")
        for i, err in enumerate(result["error_history"], 1):
            # Show just the last line of traceback (the actual error)
            err_lines = err.strip().split("\n")
            last_line = err_lines[-1] if err_lines else err[:100]
            print(f"    Attempt {i}: {last_line[:120]}")

    if result.get("sandbox_path"):
        print(f"\n  Sandbox: {result['sandbox_path']}")
        print(f"  Clean up: rm -rf {result['sandbox_path']}")

    print(f"\n{DIVIDER}")
    if result["success"]:
        print("🎉 SUCCESS")
    else:
        print("💥 FAILED — check generated_backtest.py and error history above")
    print(DIVIDER)

    return result


if __name__ == "__main__":
    main()
