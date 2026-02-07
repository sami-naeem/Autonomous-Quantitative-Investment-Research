"""
Agent 7: Sandbox Code Executor for the Paper Replication Pipeline.

This module provides:
  - SandboxExecutor: a class that creates an isolated Python venv,
    installs dependencies, runs generated backtest code, and iterates
    on errors using an LLM — all using free, subprocess-based tooling
    suitable for Lambda AI instances.
  - agent7_sandbox_execute_node: a LangGraph node that plugs into
    the existing pipeline after `finalize_replication_code_node`.

Usage (standalone):
    from sandbox_executor import SandboxExecutor
    executor = SandboxExecutor(llm=my_llm, max_retries=3)
    result = executor.run(code_string)
    # result = {"success": bool, "metrics": {...}, "stdout": "...", ...}

Usage (LangGraph):
    graph.add_node("agent7_sandbox", agent7_sandbox_execute_node)
    graph.add_edge("finalize", "agent7_sandbox")
    graph.add_edge("agent7_sandbox", END)
"""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("sandbox_executor")


# ── Standard-library modules that should never be pip-installed ──────────
STDLIB_MODULES: Set[str] = {
    "abc", "argparse", "ast", "asyncio", "base64", "bisect", "calendar",
    "cmath", "collections", "colorsys", "concurrent", "configparser",
    "contextlib", "copy", "csv", "ctypes", "dataclasses", "datetime",
    "decimal", "difflib", "dis", "email", "enum", "errno", "faulthandler",
    "fileinput", "fnmatch", "fractions", "ftplib", "functools", "gc",
    "getpass", "gettext", "glob", "gzip", "hashlib", "heapq", "hmac",
    "html", "http", "imaplib", "importlib", "inspect", "io", "ipaddress",
    "itertools", "json", "keyword", "linecache", "locale", "logging",
    "lzma", "mailbox", "math", "mimetypes", "mmap", "multiprocessing",
    "numbers", "operator", "os", "pathlib", "pdb", "pickle", "pkgutil",
    "platform", "plistlib", "pprint", "profile", "pstats", "queue",
    "random", "re", "readline", "reprlib", "sched", "secrets", "select",
    "shelve", "shlex", "shutil", "signal", "site", "smtplib", "socket",
    "socketserver", "sqlite3", "ssl", "stat", "statistics", "string",
    "struct", "subprocess", "sys", "sysconfig", "syslog", "tempfile",
    "textwrap", "threading", "time", "timeit", "token", "tokenize",
    "traceback", "tracemalloc", "turtle", "types", "typing", "unicodedata",
    "unittest", "urllib", "uuid", "venv", "warnings", "wave", "weakref",
    "webbrowser", "xml", "xmlrpc", "zipfile", "zipimport", "zlib",
    # common sub-packages people import
    "os.path", "collections.abc", "concurrent.futures", "email.mime",
    "http.client", "urllib.parse", "urllib.request", "xml.etree",
}

# ── Known import-name → PyPI-name mismatches ────────────────────────────
IMPORT_TO_PYPI: Dict[str, str] = {
    "sklearn":       "scikit-learn",
    "cv2":           "opencv-python",
    "PIL":           "Pillow",
    "yaml":          "pyyaml",
    "bs4":           "beautifulsoup4",
    "dateutil":      "python-dateutil",
    "attr":          "attrs",
    "dotenv":        "python-dotenv",
    "google.cloud":  "google-cloud-storage",
}


# ═════════════════════════════════════════════════════════════════════════
# SandboxExecutor
# ═════════════════════════════════════════════════════════════════════════
class SandboxExecutor:
    """
    Creates a temporary Python venv, installs dependencies, executes
    generated code, and optionally asks an LLM to fix errors.

    Parameters
    ----------
    llm : optional
        A LangChain-compatible LLM instance (must have `.invoke()`).
        If None, the executor will not attempt to auto-fix errors.
    max_retries : int
        Number of LLM-assisted fix attempts on failure (default 3).
    timeout : int
        Maximum wall-clock seconds for each code execution (default 300).
    extra_packages : list[str]
        Additional pip packages to always install (e.g. ["pandas", "numpy"]).
    keep_sandbox : bool
        If True, don't delete the temp directory after execution (debugging).
    """

    def __init__(
        self,
        llm=None,
        max_retries: int = 3,
        timeout: int = 300,
        extra_packages: Optional[List[str]] = None,
        keep_sandbox: bool = False,
    ):
        self.llm = llm
        self.max_retries = max_retries
        self.timeout = timeout
        self.extra_packages = extra_packages or []
        self.keep_sandbox = keep_sandbox

        # Populated during .run()
        self._sandbox_dir: Optional[Path] = None
        self._venv_dir: Optional[Path] = None
        self._python: Optional[str] = None
        self._pip: Optional[str] = None

    # ── public API ───────────────────────────────────────────────────────
    def run(self, code: str) -> Dict[str, Any]:
        """
        Execute *code* in an isolated venv. Returns a result dict:
            {
                "success": bool,
                "metrics": dict | None,       # parsed JSON metrics on success
                "stdout": str,                 # captured stdout
                "stderr": str,                 # captured stderr
                "code_final": str,             # the (possibly LLM-fixed) code
                "retries_used": int,
                "sandbox_path": str,           # path to temp dir (if keep_sandbox)
                "error_history": [str, ...],   # stderr from each failed attempt
            }
        """
        result = {
            "success": False,
            "metrics": None,
            "stdout": "",
            "stderr": "",
            "code_final": code,
            "retries_used": 0,
            "sandbox_path": "",
            "error_history": [],
        }

        try:
            self._create_sandbox()
            self._create_venv()
            self._install_dependencies(code)

            current_code = code

            for attempt in range(1 + self.max_retries):
                logger.info(f"[Sandbox] Execution attempt {attempt + 1}/{1 + self.max_retries}")
                stdout, stderr, returncode = self._execute(current_code)

                if returncode == 0:
                    # ── Success ──────────────────────────────────────
                    metrics = self._extract_metrics(stdout)
                    result.update(
                        success=True,
                        metrics=metrics,
                        stdout=stdout,
                        stderr=stderr,
                        code_final=current_code,
                        retries_used=attempt,
                    )
                    logger.info(f"[Sandbox] Code executed successfully on attempt {attempt + 1}")
                    return result

                # ── Failure ──────────────────────────────────────────
                logger.warning(f"[Sandbox] Attempt {attempt + 1} failed:\n{stderr[:500]}")
                result["error_history"].append(stderr)

                if attempt < self.max_retries and self.llm is not None:
                    logger.info("[Sandbox] Asking LLM to fix the code...")
                    fixed = self._ask_llm_to_fix(current_code, stderr, attempt + 1)
                    if fixed and fixed.strip() != current_code.strip():
                        current_code = fixed
                        # Re-scan for any new imports
                        self._install_dependencies(current_code)
                    else:
                        logger.warning("[Sandbox] LLM returned unchanged or empty code.")
                else:
                    logger.error("[Sandbox] Max retries reached or no LLM available.")

            # All attempts exhausted
            result.update(
                stdout=stdout,
                stderr=stderr,
                code_final=current_code,
                retries_used=self.max_retries,
            )
            return result

        except Exception as exc:
            logger.exception("[Sandbox] Unexpected error in executor")
            result["stderr"] = str(exc)
            return result

        finally:
            if self._sandbox_dir and not self.keep_sandbox:
                self._cleanup()
            elif self._sandbox_dir:
                result["sandbox_path"] = str(self._sandbox_dir)

    # ── sandbox lifecycle ────────────────────────────────────────────────
    def _create_sandbox(self):
        self._sandbox_dir = Path(tempfile.mkdtemp(prefix="agent7_sandbox_"))
        logger.info(f"[Sandbox] Created temp dir: {self._sandbox_dir}")

    def _create_venv(self):
        self._venv_dir = self._sandbox_dir / "venv"
        logger.info("[Sandbox] Creating virtual environment...")
        subprocess.run(
            [sys.executable, "-m", "venv", str(self._venv_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        self._python = str(self._venv_dir / "bin" / "python")
        self._pip = str(self._venv_dir / "bin" / "pip")

        # Upgrade pip silently
        subprocess.run(
            [self._pip, "install", "--upgrade", "pip"],
            capture_output=True,
            text=True,
        )
        logger.info("[Sandbox] Virtual environment ready.")

    def _cleanup(self):
        if self._sandbox_dir and self._sandbox_dir.exists():
            shutil.rmtree(self._sandbox_dir, ignore_errors=True)
            logger.info(f"[Sandbox] Cleaned up {self._sandbox_dir}")

    # ── dependency management ────────────────────────────────────────────
    @staticmethod
    def detect_imports(code: str) -> Set[str]:
        """
        Parse top-level import names from Python source.
        Returns the set of root package names (e.g. 'pandas', 'sklearn').
        """
        packages: Set[str] = set()
        # Matches: import X, from X import Y, from X.sub import Y
        import_re = re.compile(
            r"^\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE
        )
        for m in import_re.finditer(code):
            packages.add(m.group(1))
        return packages

    def _resolve_pip_names(self, import_names: Set[str]) -> List[str]:
        """Convert import names to pip-installable package names, filtering stdlib."""
        pip_names: List[str] = []
        for name in sorted(import_names):
            if name in STDLIB_MODULES:
                continue
            pip_name = IMPORT_TO_PYPI.get(name, name)
            pip_names.append(pip_name)
        return pip_names

    def _install_dependencies(self, code: str):
        imports = self.detect_imports(code)
        pip_names = self._resolve_pip_names(imports)

        # Merge any extra packages the caller always wants
        all_packages = list(set(pip_names + self.extra_packages))
        if not all_packages:
            return

        logger.info(f"[Sandbox] Installing: {all_packages}")
        proc = subprocess.run(
            [self._pip, "install", "--quiet"] + all_packages,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if proc.returncode != 0:
            logger.warning(f"[Sandbox] pip install warnings:\n{proc.stderr[:500]}")
        else:
            logger.info("[Sandbox] Dependencies installed.")

    # ── code execution ───────────────────────────────────────────────────
    def _execute(self, code: str) -> tuple:
        """
        Write code to a file inside the sandbox and run it.
        Returns (stdout, stderr, returncode).
        """
        script_path = self._sandbox_dir / "backtest.py"
        script_path.write_text(code, encoding="utf-8")

        try:
            proc = subprocess.run(
                [self._python, str(script_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(self._sandbox_dir),
                env={
                    **os.environ,
                    "PYTHONPATH": str(self._sandbox_dir),
                },
            )
            return proc.stdout, proc.stderr, proc.returncode

        except subprocess.TimeoutExpired:
            return "", f"TIMEOUT: Script exceeded {self.timeout}s limit.", 1

    # ── metrics extraction ───────────────────────────────────────────────
    @staticmethod
    def _extract_metrics(stdout: str) -> Optional[Dict[str, Any]]:
        """
        Try to find and parse a JSON dict from stdout.

        The generated code is expected to print a JSON object as its final
        output.  We search for the *last* JSON-like block in stdout.
        """
        if not stdout.strip():
            return None

        # Strategy 1: look for an explicit METRICS marker
        marker_re = re.compile(r"METRICS:\s*(\{.*\})", re.S)
        m = marker_re.search(stdout)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        # Strategy 2: find the last {...} block in stdout
        brace_blocks = re.findall(r"\{[^{}]*\}", stdout)
        for block in reversed(brace_blocks):
            try:
                parsed = json.loads(block)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

        # Strategy 3: try the entire stdout as JSON
        try:
            parsed = json.loads(stdout.strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        return None

    # ── LLM error-fixing loop ────────────────────────────────────────────
    def _ask_llm_to_fix(self, code: str, error: str, attempt: int) -> Optional[str]:
        """Send the broken code + traceback to the LLM and return fixed code."""
        if self.llm is None:
            return None

        system = textwrap.dedent("""\
            You are a Python debugging assistant for quantitative finance backtests.

            You will receive:
            1. Python source code that failed to execute
            2. The error traceback

            Your job:
            - Diagnose the root cause of the error
            - Return ONLY the complete, corrected Python source code
            - Do NOT include markdown fences, explanations, or anything other
              than valid Python
            - Preserve the original logic and intent of the code
            - If data files are missing, create realistic synthetic/mock data
              so the backtest can run end-to-end
            - The script MUST print a JSON dict of metrics as its final output,
              prefixed with "METRICS:" — for example:
              print("METRICS:", json.dumps({"sharpe": 1.23, "total_return": 0.45}))
            - Make sure all imports are at the top of the file

            IMPORTANT: Return ONLY valid Python code. No markdown. No explanations.
        """)

        human = (
            f"## Attempt {attempt} — the following code failed.\n\n"
            f"### CODE:\n```python\n{code}\n```\n\n"
            f"### ERROR:\n```\n{error[-3000:]}\n```\n\n"
            f"Return the corrected Python code."
        )

        # Build messages — use langchain message classes if available,
        # otherwise fall back to plain dicts (works with mock LLMs and
        # many LangChain-compatible providers).
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content=system),
                HumanMessage(content=human),
            ]
        except ImportError:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": human},
            ]

        try:
            resp = self.llm.invoke(messages)
            raw = getattr(resp, "content", "") or ""
            # If resp is a plain string (some providers), use it directly
            if not raw and isinstance(resp, str):
                raw = resp
            return self._strip_markdown_fences(raw)
        except Exception as exc:
            logger.error(f"[Sandbox] LLM fix request failed: {exc}")
            return None

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Remove ```python ... ``` wrappers if present."""
        text = text.strip()
        if text.startswith("```"):
            # Remove opening fence
            first_nl = text.index("\n") if "\n" in text else 3
            text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()


# ═════════════════════════════════════════════════════════════════════════
# LangGraph Node
# ═════════════════════════════════════════════════════════════════════════
def agent7_sandbox_execute_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: execute the replication_code in a sandbox.

    Reads from state:
        - replication_code (str): the Python module from finalize node
        - model_choice (str): which LLM to use for error fixing

    Writes to state:
        - sandbox_result (dict): full execution result
        - backtest_metrics (dict | None): parsed metrics on success
    """
    from Agents.llm_config import get_llm_by_name

    code = state.get("replication_code", "")
    model_choice = state.get("model_choice", "openai")

    if not code.strip():
        logger.error("[Agent7] No replication_code found in state.")
        return {
            **state,
            "sandbox_result": {"success": False, "stderr": "No code to execute."},
            "backtest_metrics": None,
        }

    # Get the same LLM the rest of the pipeline is using
    llm = get_llm_by_name(model_choice)

    executor = SandboxExecutor(
        llm=llm,
        max_retries=3,
        timeout=300,
        extra_packages=["pandas", "numpy"],  # always available
    )

    result = executor.run(code)

    logger.info(f"[Agent7] Sandbox finished — success={result['success']}, "
                f"retries_used={result['retries_used']}")

    if result["success"]:
        logger.info(f"[Agent7] Metrics: {result['metrics']}")
    else:
        logger.error(f"[Agent7] Execution failed after all retries.")
        logger.error(f"[Agent7] Last error:\n{result['stderr'][:500]}")

    return {
        **state,
        "sandbox_result": result,
        "backtest_metrics": result.get("metrics"),
    }


# ═════════════════════════════════════════════════════════════════════════
# Graph Integration Helper
# ═════════════════════════════════════════════════════════════════════════
def patch_graph_with_sandbox(graph_builder):
    """
    Convenience function: adds agent7 to an existing StateGraph builder
    BEFORE it is compiled.

    Usage:
        graph = StateGraph(AgentState)
        # ... add all existing nodes/edges ...
        # REMOVE the old:  graph.add_edge("finalize", END)
        patch_graph_with_sandbox(graph)
        app = graph.compile()
    """
    from langgraph.graph import END

    graph_builder.add_node("agent7_sandbox", agent7_sandbox_execute_node)
    graph_builder.add_edge("finalize", "agent7_sandbox")
    graph_builder.add_edge("agent7_sandbox", END)


# ═════════════════════════════════════════════════════════════════════════
# CLI / Quick Test
# ═════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    """Quick smoke test — runs a trivial script in the sandbox."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    test_code = textwrap.dedent("""\
        import json
        import numpy as np
        import pandas as pd

        # Synthetic backtest
        np.random.seed(42)
        dates = pd.date_range("2015-01-01", periods=252 * 5, freq="B")
        returns = np.random.normal(0.0004, 0.012, len(dates))
        cumulative = (1 + pd.Series(returns, index=dates)).cumprod()

        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        total_return = float(cumulative.iloc[-1] - 1)
        max_dd = float((cumulative / cumulative.cummax() - 1).min())

        metrics = {
            "sharpe_ratio": round(sharpe, 4),
            "total_return": round(total_return, 4),
            "max_drawdown": round(max_dd, 4),
            "n_days": len(dates),
        }
        print("METRICS:", json.dumps(metrics))
    """)

    executor = SandboxExecutor(llm=None, max_retries=0, timeout=60)
    result = executor.run(test_code)

    print("\n" + "=" * 60)
    print(f"Success: {result['success']}")
    print(f"Metrics: {json.dumps(result['metrics'], indent=2)}")
    print(f"Retries: {result['retries_used']}")
    if result["stderr"]:
        print(f"Stderr:  {result['stderr'][:300]}")
    print("=" * 60)
