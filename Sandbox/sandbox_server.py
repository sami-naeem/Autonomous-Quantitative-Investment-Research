"""
FastAPI Sandbox Server for Remote Code Execution.

This module provides a FastAPI endpoint that receives Python code from the
LangGraph pipeline, executes it in an isolated sandbox, and writes artifacts
to GCS.

IMPORTANT: This is a PURE execution environment. No LLM-based auto-fixing.
Retry decisions are made by the LangGraph agent, not the sandbox.

Usage:
    uvicorn Sandbox.sandbox_server:app --host 0.0.0.0 --port 9000

Environment:
    SHARADAR_DATA_DIR: Path to local Sharadar data (required for backtests)
    GCS_OUTPUT_BUCKET_NAME: GCS bucket for output artifacts
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from Agents.config import GCSConfig
from Agents.gcs_utils import get_storage_client, upload_agent_output

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("sandbox_server")

# Initialize FastAPI app
app = FastAPI(
    title="QuantAgent Sandbox Server",
    description="Pure execution endpoint for quantitative backtesting (no LLM auto-fix)",
    version="1.1.0",
)

# Global storage client (initialized on first use)
_storage_client = None

# Standard-library modules that should never be pip-installed
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
    "os.path", "collections.abc", "concurrent.futures", "email.mime",
    "http.client", "urllib.parse", "urllib.request", "xml.etree",
}

# Known import-name to PyPI-name mismatches
IMPORT_TO_PYPI: Dict[str, str] = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "attr": "attrs",
    "dotenv": "python-dotenv",
    "google.cloud": "google-cloud-storage",
}


def _get_storage_client():
    """Get or initialize the GCS storage client."""
    global _storage_client
    if _storage_client is None:
        _storage_client = get_storage_client()
    return _storage_client


def detect_imports(code: str) -> Set[str]:
    """Parse top-level import names from Python source."""
    import re
    packages: Set[str] = set()
    import_re = re.compile(
        r"^\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE
    )
    for m in import_re.finditer(code):
        packages.add(m.group(1))
    return packages


def resolve_pip_names(import_names: Set[str]) -> List[str]:
    """Convert import names to pip-installable package names."""
    pip_names: List[str] = []
    for name in sorted(import_names):
        if name in STDLIB_MODULES:
            continue
        pip_name = IMPORT_TO_PYPI.get(name, name)
        pip_names.append(pip_name)
    return pip_names


class CodeExecutor:
    """
    Simple code executor that runs Python in an isolated venv.
    No LLM-based fixing - pure execution only.
    """
    
    def __init__(self, timeout: int = 300):
        self.timeout = timeout
        self._sandbox_dir: Optional[Path] = None
        self._venv_dir: Optional[Path] = None
        
    def run(self, code: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute code in isolated venv.
        Returns dict with success, metrics, stdout, stderr, etc.
        """
        import shutil
        result = {
            "success": False,
            "metrics": None,
            "returns_csv": None,
            "stdout": "",
            "stderr": "",
        }
        
        try:
            # Create sandbox
            self._sandbox_dir = Path(tempfile.mkdtemp(prefix="sandbox_"))
            self._venv_dir = self._sandbox_dir / "venv"
            
            # Create venv
            logger.info("[Executor] Creating virtual environment...")
            subprocess.run(
                [sys.executable, "-m", "venv", str(self._venv_dir)],
                check=True, capture_output=True, text=True
            )
            
            # Get python/pip paths (cross-platform)
            if sys.platform == "win32":
                python = str(self._venv_dir / "Scripts" / "python.exe")
                pip = str(self._venv_dir / "Scripts" / "pip.exe")
            else:
                python = str(self._venv_dir / "bin" / "python")
                pip = str(self._venv_dir / "bin" / "pip")
            
            # Upgrade pip
            subprocess.run([pip, "install", "--upgrade", "pip"], 
                         capture_output=True, text=True)
            
            # Install dependencies
            imports = detect_imports(code)
            pip_packages = resolve_pip_names(imports)
            if pip_packages:
                logger.info(f"[Executor] Installing: {pip_packages}")
                proc = subprocess.run(
                    [pip, "install", "--quiet"] + pip_packages,
                    capture_output=True, text=True, timeout=180
                )
                if proc.returncode != 0:
                    logger.warning(f"[Executor] pip warnings: {proc.stderr[:500]}")
            
            # Write code to file
            script_path = self._sandbox_dir / "backtest.py"
            script_path.write_text(code, encoding="utf-8")
            
            # Execute code
            logger.info("[Executor] Running backtest code...")
            proc = subprocess.run(
                [python, str(script_path)],
                capture_output=True, text=True, timeout=self.timeout,
                cwd=str(self._sandbox_dir),
                env={**os.environ, "PYTHONPATH": str(self._sandbox_dir)},
            )
            
            result["stdout"] = proc.stdout
            result["stderr"] = proc.stderr
            
            if proc.returncode == 0:
                result["success"] = True
                result["metrics"] = self._extract_metrics(proc.stdout)
                result["returns_csv"] = self._collect_returns_csv(output_dir)
                logger.info("[Executor] Execution successful")
            else:
                result["success"] = False
                logger.warning(f"[Executor] Execution failed: {proc.stderr[:500]}")
                
        except subprocess.TimeoutExpired:
            result["stderr"] = f"TIMEOUT: Script exceeded {self.timeout}s limit"
            logger.error("[Executor] Execution timed out")
        except Exception as e:
            result["stderr"] = str(e)
            logger.exception(f"[Executor] Execution error: {e}")
        finally:
            # Cleanup
            if self._sandbox_dir and self._sandbox_dir.exists():
                shutil.rmtree(self._sandbox_dir, ignore_errors=True)
        
        return result
    
    def _extract_metrics(self, stdout: str) -> Optional[Dict[str, Any]]:
        """Extract JSON metrics from stdout."""
        import re
        if not stdout.strip():
            return None
        
        # Look for METRICS: marker
        marker_re = re.compile(r"METRICS:\s*(\{.*\})", re.S)
        m = marker_re.search(stdout)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        
        # Find last {...} block
        brace_blocks = re.findall(r"\{[^{}]*\}", stdout)
        for block in reversed(brace_blocks):
            try:
                parsed = json.loads(block)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
        
        return None
    
    def _collect_returns_csv(self, output_dir: Optional[str]) -> Optional[str]:
        """Copy returns.csv to output_dir if it exists."""
        import shutil
        if not self._sandbox_dir:
            return None
        
        returns_path = self._sandbox_dir / "returns.csv"
        if not returns_path.exists():
            return None
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            dest = os.path.join(output_dir, "returns.csv")
            shutil.copy2(returns_path, dest)
            return dest
        return str(returns_path)


# =============================================================================
# Pydantic Models
# =============================================================================

class ExecuteRequest(BaseModel):
    """Request model for the /execute endpoint."""
    code: str = Field(..., description="Python source code to execute")
    paper_name: str = Field(..., description="Identifier for the paper being replicated")


class ExecuteResponse(BaseModel):
    """Response model for the /execute endpoint."""
    status: str = Field(..., description="Execution status: 'success' or 'failure'")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Parsed backtest metrics")
    stdout: str = Field(default="", description="Captured stdout from execution")
    stderr: str = Field(default="", description="Captured stderr from execution")
    artifact_json_gcs_uri: Optional[str] = Field(None, description="GCS URI of metrics JSON artifact")
    artifact_code_gcs_uri: Optional[str] = Field(None, description="GCS URI of code snapshot")
    returns_csv_gcs_uri: Optional[str] = Field(None, description="GCS URI of returns CSV")
    execution_time_ms: Optional[int] = Field(None, description="Total execution time in milliseconds")


# =============================================================================
# Artifact Upload Helpers
# =============================================================================

def _write_artifacts_to_gcs(
    paper_name: str,
    code: str,
    result: Dict[str, Any],
) -> Dict[str, str]:
    """Write execution artifacts to GCS."""
    storage_client = _get_storage_client()
    if storage_client is None:
        logger.error("Cannot write artifacts: GCS storage client not available")
        return {}
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_path = f"replication_outputs/{paper_name}/{timestamp}"
    
    uris = {}
    
    # 1. Upload metrics JSON
    try:
        metrics_payload = {
            "paper_name": paper_name,
            "timestamp": timestamp,
            "status": "success" if result.get("success") else "failure",
            "metrics": result.get("metrics"),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "code": code,
        }
        json_blob_name = f"{base_path}_metrics.json"
        uris["artifact_json_gcs_uri"] = upload_agent_output(
            storage_client, json_blob_name, metrics_payload
        )
        logger.info(f"Uploaded metrics JSON to {uris['artifact_json_gcs_uri']}")
    except Exception as e:
        logger.error(f"Failed to upload metrics JSON: {e}")
    
    # 2. Upload code snapshot
    try:
        code_blob_name = f"{base_path}_code.py"
        bucket = storage_client.bucket(GCSConfig.OUTPUT_BUCKET_NAME)
        blob = bucket.blob(code_blob_name)
        blob.upload_from_string(code, content_type="text/plain")
        uris["artifact_code_gcs_uri"] = f"gs://{GCSConfig.OUTPUT_BUCKET_NAME}/{code_blob_name}"
        logger.info(f"Uploaded code snapshot to {uris['artifact_code_gcs_uri']}")
    except Exception as e:
        logger.error(f"Failed to upload code snapshot: {e}")
    
    # 3. Upload returns CSV if produced
    returns_csv_path = result.get("returns_csv")
    if returns_csv_path and os.path.exists(returns_csv_path):
        try:
            csv_blob_name = f"{base_path}_returns.csv"
            bucket = storage_client.bucket(GCSConfig.OUTPUT_BUCKET_NAME)
            blob = bucket.blob(csv_blob_name)
            blob.upload_from_filename(returns_csv_path)
            uris["returns_csv_gcs_uri"] = f"gs://{GCSConfig.OUTPUT_BUCKET_NAME}/{csv_blob_name}"
            logger.info(f"Uploaded returns CSV to {uris['returns_csv_gcs_uri']}")
        except Exception as e:
            logger.error(f"Failed to upload returns CSV: {e}")
    
    return uris


# =============================================================================
# API Endpoints
# =============================================================================

@app.post("/execute", response_model=ExecuteResponse)
async def execute(req: ExecuteRequest) -> ExecuteResponse:
    """
    Execute Python backtest code in an isolated sandbox.
    
    This is a PURE execution endpoint - no LLM-based auto-fixing.
    The LangGraph agent decides whether to retry based on the response.
    
    The code is expected to:
    - Read Sharadar data from local CSV files (path in SHARADAR_DATA_DIR env var)
    - Save returns to "returns.csv" in the working directory
    - Print metrics as JSON: print("METRICS:", json.dumps({...}))
    """
    import time
    start_time = time.time()
    
    logger.info(f"[Execute] Received request for paper={req.paper_name}")
    
    # Validate Sharadar data directory
    sharadar_data_dir = os.getenv("SHARADAR_DATA_DIR", "/home/ubuntu/data/sharadar")
    if not os.path.exists(sharadar_data_dir):
        logger.error(f"SHARADAR_DATA_DIR does not exist: {sharadar_data_dir}")
        raise HTTPException(
            status_code=500,
            detail=f"Sharadar data directory not found: {sharadar_data_dir}"
        )
    
    # Create output directory
    output_dir = os.path.join(os.path.expanduser("~"), "backtest_results", req.paper_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Execute code (pure execution, no LLM)
    executor = CodeExecutor(timeout=300)
    
    try:
        logger.info(f"[Execute] Starting execution for {req.paper_name}")
        result = executor.run(req.code, output_dir=output_dir)
        execution_time_ms = int((time.time() - start_time) * 1000)
        
        logger.info(
            f"[Execute] Completed: success={result['success']}, "
            f"time={execution_time_ms}ms"
        )
        
    except Exception as e:
        execution_time_ms = int((time.time() - start_time) * 1000)
        logger.exception(f"[Execute] Execution failed: {e}")
        result = {
            "success": False,
            "metrics": None,
            "returns_csv": None,
            "stdout": "",
            "stderr": str(e),
        }
    
    # Write artifacts to GCS
    logger.info(f"[Execute] Writing artifacts to GCS")
    artifact_uris = _write_artifacts_to_gcs(
        paper_name=req.paper_name,
        code=req.code,
        result=result,
    )
    
    # Build response
    response = ExecuteResponse(
        status="success" if result["success"] else "failure",
        metrics=result.get("metrics"),
        stdout=result.get("stdout", ""),
        stderr=result.get("stderr", ""),
        artifact_json_gcs_uri=artifact_uris.get("artifact_json_gcs_uri"),
        artifact_code_gcs_uri=artifact_uris.get("artifact_code_gcs_uri"),
        returns_csv_gcs_uri=artifact_uris.get("returns_csv_gcs_uri"),
        execution_time_ms=execution_time_ms,
    )
    
    logger.info(f"[Execute] Request completed for {req.paper_name}")
    return response


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "sandbox-server"}


@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint with API info."""
    return {
        "service": "QuantAgent Sandbox Server",
        "version": "1.1.0",
        "description": "Pure execution endpoint - no LLM auto-fix",
        "endpoints": {
            "execute": "POST /execute - Execute Python backtest code",
            "health": "GET /health - Health check",
        },
        "config": {
            "sharadar_data_dir": os.getenv("SHARADAR_DATA_DIR", "/home/ubuntu/data/sharadar"),
            "gcs_output_bucket": GCSConfig.OUTPUT_BUCKET_NAME,
        },
    }


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("=" * 60)
    logger.info("QuantAgent Sandbox Server Starting (Pure Execution Mode)")
    logger.info("=" * 60)
    
    # Check environment
    sharadar_dir = os.getenv("SHARADAR_DATA_DIR", "/home/ubuntu/data/sharadar")
    if os.path.exists(sharadar_dir):
        logger.info(f"✓ Sharadar data directory: {sharadar_dir}")
    else:
        logger.warning(f"✗ Sharadar data directory not found: {sharadar_dir}")
    
    # Initialize GCS client
    client = _get_storage_client()
    if client:
        logger.info(f"✓ GCS client initialized")
        logger.info(f"  Output bucket: {GCSConfig.OUTPUT_BUCKET_NAME}")
    else:
        logger.warning(f"✗ GCS client not available")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("SANDBOX_SERVER_PORT", "9000"))
    host = os.getenv("SANDBOX_SERVER_HOST", "0.0.0.0")
    
    logger.info(f"Starting sandbox server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
