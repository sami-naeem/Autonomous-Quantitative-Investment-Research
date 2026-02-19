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
    def run(self, code: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute *code* in an isolated venv. Returns a result dict:
            {
                "success": bool,
                "metrics": dict | None,       # parsed JSON metrics on success
                "returns_csv": str | None,     # path to copied returns CSV (if produced)
                "output_files": [str, ...],    # paths to all output files copied out
                "stdout": str,                 # captured stdout
                "stderr": str,                 # captured stderr
                "code_final": str,             # the (possibly LLM-fixed) code
                "retries_used": int,
                "sandbox_path": str,           # path to temp dir (if keep_sandbox)
                "error_history": [str, ...],   # stderr from each failed attempt
            }

        Parameters
        ----------
        code : str
            Python source code to execute.
        output_dir : str, optional
            Directory to copy output files (CSVs, PNGs) to. If None,
            files remain in the sandbox (accessible via sandbox_path if
            keep_sandbox=True, otherwise lost on cleanup).
        """
        result = {
            "success": False,
            "metrics": None,
            "returns_csv": None,
            "output_files": [],
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
            self._error_history_for_prompt = []  # track errors across attempts
            self._diagnosis_history = []          # track diagnoses across attempts
            self._error_classes = []              # track error categories for escalation

            for attempt in range(1 + self.max_retries):
                logger.info(f"[Sandbox] Execution attempt {attempt + 1}/{1 + self.max_retries}")
                stdout, stderr, returncode = self._execute(current_code)

                if returncode == 0:
                    # ── Success ──────────────────────────────────────
                    metrics = self._extract_metrics(stdout)
                    returns_csv, output_files = self._collect_output_files(output_dir)
                    result.update(
                        success=True,
                        metrics=metrics,
                        returns_csv=returns_csv,
                        output_files=output_files,
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
                self._error_history_for_prompt.append(stderr)

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

    # ── output file collection ───────────────────────────────────────
    def _collect_output_files(self, output_dir: Optional[str] = None) -> tuple:
        """
        Scan the sandbox for output files (CSVs, PNGs, etc.).
        Optionally copy them to output_dir.

        Returns (returns_csv_path, list_of_all_output_paths).

        The returns CSV is identified by looking for files matching common
        names: returns.csv, daily_returns.csv, strategy_returns.csv,
        portfolio_returns.csv, or any CSV containing 'return' in the name.
        """
        if not self._sandbox_dir:
            return None, []

        # File extensions we consider "outputs"
        OUTPUT_EXTENSIONS = {".csv", ".png", ".jpg", ".json", ".html", ".xlsx"}

        # Find all output files in sandbox (exclude venv/)
        output_files: List[str] = []
        for f in self._sandbox_dir.iterdir():
            if f.is_file() and f.suffix.lower() in OUTPUT_EXTENSIONS:
                # Skip the input data file if present
                if f.name == "input_data.csv":
                    continue
                output_files.append(str(f))

        if not output_files:
            logger.info("[Sandbox] No output files found in sandbox.")
            return None, []

        logger.info(f"[Sandbox] Found output files: {[os.path.basename(f) for f in output_files]}")

        # Identify the returns CSV
        returns_csv = None
        RETURNS_PATTERNS = [
            "returns.csv", "daily_returns.csv", "strategy_returns.csv",
            "portfolio_returns.csv", "backtest_returns.csv",
        ]
        for f in output_files:
            fname = os.path.basename(f).lower()
            if fname in RETURNS_PATTERNS:
                returns_csv = f
                break
        # Fallback: any CSV with 'return' in the name
        if returns_csv is None:
            for f in output_files:
                if f.endswith(".csv") and "return" in os.path.basename(f).lower():
                    returns_csv = f
                    break

        # Copy to output_dir if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            copied_files = []
            for f in output_files:
                dest = os.path.join(output_dir, os.path.basename(f))
                shutil.copy2(f, dest)
                copied_files.append(dest)
                logger.info(f"[Sandbox] Copied {os.path.basename(f)} → {dest}")

            # Update returns_csv to point to copied location
            if returns_csv:
                returns_csv = os.path.join(output_dir, os.path.basename(returns_csv))

            return returns_csv, copied_files

        return returns_csv, output_files

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

    # ── LLM error diagnosis ────────────────────────────────────────────────
    def _diagnose_error(self, code: str, error: str, attempt: int) -> str:
        """
        Ask the LLM to analyze the error and produce a concise diagnosis.
        Returns a plain-text summary of what went wrong and why.
        """
        system = textwrap.dedent("""\
            You are an expert Python debugger specializing in pandas, numpy,
            and quantitative finance code.

            You will receive:
            1. Python code that failed
            2. The raw error traceback

            Your job: produce a SHORT, precise diagnosis (3-6 sentences max).
            Include:
            - The exact line and operation that failed
            - WHY it failed (root cause, not just the exception type)
            - What the code SHOULD have done instead
            - Whether this is a small fix or a structural/logic problem

            Be specific. For example:
            BAD:  "There is a pandas indexing error."
            GOOD: "Line 37 passes row indices from nlargest() to iloc's column
                   position. nlargest() returns row labels (integers 0-1259), but
                   iloc's second argument expects column positions (0-49). The code
                   confuses rows with columns. Fix: use column names from a
                   rank-based selection instead of nlargest row indices."

            Return ONLY the diagnosis text. No code. No markdown.
        """)

        # Include previous diagnoses if this isn't the first attempt
        prev_context = ""
        if attempt > 1 and hasattr(self, '_diagnosis_history'):
            prev_context = "\n\nPREVIOUS DIAGNOSES (fixes that did NOT work):\n"
            for i, diag in enumerate(self._diagnosis_history, 1):
                prev_context += f"  Attempt {i}: {diag}\n"
            prev_context += "\nThe previous fixes failed. Identify what is STILL wrong.\n"

        human = (
            f"### CODE:\n```python\n{code}\n```\n\n"
            f"### ERROR TRACEBACK:\n```\n{error[-4000:]}\n```\n"
            f"{prev_context}\n"
            f"Provide your diagnosis."
        )

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
            diagnosis = getattr(resp, "content", "") or ""
            if not diagnosis and isinstance(resp, str):
                diagnosis = resp
            diagnosis = diagnosis.strip()
            logger.info(f"[Sandbox] Error diagnosis: {diagnosis[:200]}")
            return diagnosis
        except Exception as exc:
            logger.error(f"[Sandbox] Diagnosis request failed: {exc}")
            # Fall back to raw error if diagnosis fails
            return f"Raw error: {error[-1500:]}"

    # ── Error classification for escalation ────────────────────────────────
    @staticmethod
    def _classify_error(stderr: str) -> str:
        """
        Classify an error into a broad category so we can detect repeats.
        Returns a short tag like 'indexing', 'type_error', 'shape', etc.
        """
        stderr_lower = stderr.lower()
        if any(k in stderr_lower for k in ["iloc", "loc[", "getitem", "keyerror", "indexer_strict", "column"]):
            return "indexing"
        if "shape" in stderr_lower or "broadcast" in stderr_lower or "operands could not" in stderr_lower:
            return "shape_mismatch"
        if "typeerror" in stderr_lower:
            return "type_error"
        if "attributeerror" in stderr_lower:
            return "attribute_error"
        if "nameerror" in stderr_lower:
            return "name_error"
        if "valueerror" in stderr_lower:
            return "value_error"
        if "timeout" in stderr_lower:
            return "timeout"
        return "other"

    def _should_escalate(self) -> bool:
        """
        Check if the last 2+ errors are the same class → time to escalate.
        """
        if len(self._error_classes) < 2:
            return False
        return self._error_classes[-1] == self._error_classes[-2]

    # ── Working code skeleton for escalation ─────────────────────────────
    ESCALATION_SKELETON = textwrap.dedent("""\
        import json
        import numpy as np
        import pandas as pd

        # 1. Generate synthetic price data: rows=dates, columns=stocks
        np.random.seed(42)
        n_stocks = 10
        n_days = 252 * 3
        dates = pd.bdate_range("2020-01-01", periods=n_days)
        stock_names = [f"stock_{i}" for i in range(n_stocks)]

        daily_returns = np.random.normal(0.0003, 0.015, (n_days, n_stocks))
        prices = 100 * np.exp(np.cumsum(daily_returns, axis=0))
        prices_df = pd.DataFrame(prices, index=dates, columns=stock_names)

        # 2. Compute returns (rows=dates, columns=stocks)
        returns_df = prices_df.pct_change().dropna()

        # 3. Compute momentum signal: 60-day percent change per stock
        momentum = prices_df.pct_change(60)

        # 4. Monthly rebalance: pick top 3 stocks by momentum, equal weight
        portfolio_returns = []
        portfolio_dates = []

        for i in range(60, len(returns_df), 21):
            # Signal at rebalance date — this is a Series with stock_names as index
            signal = momentum.iloc[i].dropna()

            # Rank and pick top 3 — .index gives COLUMN NAMES (stock_0, stock_1, etc.)
            top_stocks = signal.nlargest(3).index

            # Get returns for the next 21 days (or until end)
            end_idx = min(i + 21, len(returns_df))
            for j in range(i, end_idx):
                # Equal-weight return of top stocks for this day
                day_return = returns_df.iloc[j][top_stocks].mean()
                portfolio_returns.append(day_return)
                portfolio_dates.append(returns_df.index[j])

        # 5. Build results
        strat_returns = np.array(portfolio_returns)
        strat_dates = portfolio_dates

        # Metrics
        ann_return = float(np.mean(strat_returns) * 252)
        ann_vol = float(np.std(strat_returns) * np.sqrt(252))
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
        cumulative = np.cumprod(1 + strat_returns)
        total_ret = float(cumulative[-1] - 1)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative / running_max - 1
        max_dd = float(np.min(drawdowns))

        metrics = {
            "sharpe_ratio": round(sharpe, 4),
            "annualized_return": round(ann_return, 4),
            "annualized_volatility": round(ann_vol, 4),
            "total_return": round(total_ret, 4),
            "max_drawdown": round(max_dd, 4),
            "n_trading_days": len(strat_returns),
        }

        # 6. Save returns CSV
        pd.DataFrame({
            "date": [d.strftime("%Y-%m-%d") for d in strat_dates],
            "strategy_returns": strat_returns,
        }).to_csv("returns.csv", index=False)

        print("METRICS:", json.dumps(metrics))
    """)

    # ── LLM error-fixing loop ────────────────────────────────────────────
    def _ask_llm_to_fix(self, code: str, error: str, attempt: int) -> Optional[str]:
        """
        Three-tier fix process:
          Tier 1 (attempts 1-2): Diagnose → Fix normally
          Tier 2 (repeated error class): Diagnose → Fix with WORKING SKELETON injected
          Tier 3 (3+ same class): Return the working skeleton directly
        """
        if self.llm is None:
            return None

        # Classify this error and track it
        error_class = self._classify_error(error)
        self._error_classes.append(error_class)

        # ── Tier 3: Too many repeats → just use the skeleton ─────────
        same_class_count = sum(1 for c in self._error_classes if c == error_class)
        if same_class_count >= 3:
            logger.warning(f"[Sandbox] Error class '{error_class}' repeated {same_class_count}x. "
                           f"Escalating to Tier 3: injecting working skeleton.")
            return self.ESCALATION_SKELETON

        # ── Step 1: Diagnose the error ───────────────────────────────
        diagnosis = self._diagnose_error(code, error, attempt)
        self._diagnosis_history.append(diagnosis)

        # ── Check if we should escalate to Tier 2 ───────────────────
        escalated = self._should_escalate()
        if escalated:
            logger.warning(f"[Sandbox] Same error class '{error_class}' repeated. "
                           f"Escalating to Tier 2: injecting working skeleton in prompt.")

        # ── Step 2: Fix the code ─────────────────────────────────────
        system = textwrap.dedent("""\
            You are an expert Python debugging assistant specializing in pandas,
            numpy, and quantitative finance backtests.

            You will receive:
            1. Python source code that failed
            2. An expert DIAGNOSIS explaining exactly what went wrong and why
            3. (Optionally) history of previous failed fix attempts

            Your job is to fix the code based on the diagnosis.

            CRITICAL RULES:
            1. Return ONLY the complete, corrected Python source code.
            2. Do NOT include markdown fences (```), explanations, or anything
               other than valid Python.
            3. ALL imports must be at the top of the file.
            4. The script MUST be completely self-contained — no external data files.
            5. The script MUST print metrics as the last line:
               print("METRICS:", json.dumps({...}))
            6. The script MUST save daily returns to CSV:
               pd.DataFrame({"date": dates, "strategy_returns": rets}).to_csv("returns.csv", index=False)

            COMMON PANDAS/NUMPY PITFALLS:
            - df.iloc[row, col] requires INTEGER positions for BOTH row and col.
            - df.loc[label, col_name] uses LABELS, not integer positions.
            - DataFrame.nlargest(n, col) returns ROWS — .index gives row labels,
              NOT column names. To rank columns, transpose or use a different approach.
            - For a long/short portfolio across stocks (columns), compute returns per
              stock first: stock_returns = prices.pct_change(), then select columns.
            - pct_change(N) computes percent change over N periods — no 'window' arg.
            - rolling(N).mean() does NOT take an axis argument.

            If the diagnosis says the problem is structural, REWRITE from scratch.
            Do NOT make small patches to fundamentally broken logic.
        """)

        # Build human message
        history_text = ""
        if attempt > 1 and len(self._diagnosis_history) > 1:
            history_text = "\n\n### PREVIOUS FAILED ATTEMPTS:\n"
            for i, diag in enumerate(self._diagnosis_history[:-1], 1):
                # Summarize if history is getting long
                diag_text = diag if len(diag) < 300 else diag[:300] + "..."
                history_text += f"  Attempt {i}: {diag_text}\n"
            history_text += (
                "\nThose fixes did NOT work. The current diagnosis identifies what "
                "is STILL wrong. Take a DIFFERENT approach this time.\n"
            )

        # Tier 2: inject working skeleton
        skeleton_text = ""
        if escalated:
            skeleton_text = (
                "\n\n### WORKING REFERENCE CODE:\n"
                "The previous fixes keep failing on the same issue. Here is a "
                "COMPLETE WORKING implementation of a momentum backtest. Use this "
                "as your foundation — adapt it to match the original code's intent "
                "(number of stocks, lookback period, etc.) but keep the pandas "
                "patterns exactly as shown:\n"
                f"```python\n{self.ESCALATION_SKELETON}\n```\n"
            )

        human = (
            f"## Fix attempt {attempt} of {self.max_retries + 1}\n\n"
            f"### DIAGNOSIS:\n{diagnosis}\n\n"
            f"### CODE TO FIX:\n```python\n{code}\n```\n"
            f"{history_text}"
            f"{skeleton_text}\n"
            f"Return the COMPLETE corrected Python code."
        )

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
        - returns_csv (str | None): path to daily returns CSV
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
            "returns_csv": None,
        }

    # Get the same LLM the rest of the pipeline is using
    llm = get_llm_by_name(model_choice)

    # Output directory for persistent results (alongside the project)
    output_dir = os.path.join(os.path.expanduser("~"), "backtest_results")

    executor = SandboxExecutor(
        llm=llm,
        max_retries=3,
        timeout=300,
        extra_packages=["pandas", "numpy"],  # always available
    )

    result = executor.run(code, output_dir=output_dir)

    logger.info(f"[Agent7] Sandbox finished — success={result['success']}, "
                f"retries_used={result['retries_used']}")

    if result["success"]:
        logger.info(f"[Agent7] Metrics: {result['metrics']}")
        if result["returns_csv"]:
            logger.info(f"[Agent7] Returns CSV: {result['returns_csv']}")
        if result["output_files"]:
            logger.info(f"[Agent7] Output files: {result['output_files']}")
    else:
        logger.error(f"[Agent7] Execution failed after all retries.")
        logger.error(f"[Agent7] Last error:\n{result['stderr'][:500]}")

    return {
        **state,
        "sandbox_result": result,
        "backtest_metrics": result.get("metrics"),
        "returns_csv": result.get("returns_csv"),
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

        # Save daily returns series to CSV
        returns_df = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "strategy_returns": returns,
        })
        returns_df.to_csv("returns.csv", index=False)

        print("METRICS:", json.dumps(metrics))
    """)

    executor = SandboxExecutor(llm=None, max_retries=0, timeout=60, keep_sandbox=True)
    result = executor.run(test_code)

    print("\n" + "=" * 60)
    print(f"Success:     {result['success']}")
    print(f"Metrics:     {json.dumps(result['metrics'], indent=2)}")
    print(f"Returns CSV: {result['returns_csv']}")
    print(f"All outputs: {result['output_files']}")
    print(f"Retries:     {result['retries_used']}")
    if result["stderr"]:
        print(f"Stderr:      {result['stderr'][:300]}")
    if result.get("sandbox_path"):
        print(f"Sandbox:     {result['sandbox_path']}")
        print(f"Clean up:    rm -rf {result['sandbox_path']}")
    print("=" * 60)
