"""
Configuration module for the agent pipeline.
Loads environment variables and defines project settings.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
#env_path = Path(__file__).parent.parent / ".env"
#load_dotenv(dotenv_path=env_path)

env_path = Path("Agents/agent.py").resolve().parent.parent / ".env"
load_dotenv(env_path, override=True)


class GCSConfig:
    """Google Cloud Storage configuration"""
    PROJECT_ID = "agentic-investment-research"
    SOURCE_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "investment-research").strip()
    SOURCE_PREFIX = "all_jkp_papers_final_processed/"  # Where processed papers with markdown are stored
    OUTPUT_BUCKET_NAME = os.getenv("GCS_OUTPUT_BUCKET_NAME", "investment-research-output").strip()
    LOCAL_TEMP_DIR = Path("./temp_papers")
    CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    # Fallback for Lambda/server environment
    LAMBDA_CREDENTIALS_PATH = "/workspace/secrets/service-account-key.json"


class LLMConfig:
    """LLM endpoint and model configuration"""
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MAIN_MODEL = "gpt-4o-mini"
    MAIN_TEMPERATURE = 0.0

    # vLLM Lambda endpoint
    LAMBDA_ENDPOINT = os.getenv("LAMBDA_ENDPOINT", "http://132.145.196.74:8000/v1")
    LAMBDA_API_KEY = os.getenv("LAMBDA_API_KEY", "token-abc123")

    # Model base path on Lambda server
    MODEL_BASE_PATH = "/models"

    BASE_MODEL = f"{MODEL_BASE_PATH}/{os.getenv('BASE_MODEL_NAME', 'llama-3.1-8b')}"
    # LoRA adapters are registered by name only (no path prefix) in vLLM
    FINLLAMA_MODEL = f"{MODEL_BASE_PATH}/{os.getenv('FINLLAMA_MODEL_NAME', 'finllama')}"
    CUSTOM_MODEL = os.getenv('CUSTOM_MODEL_NAME', 'custom')

    # Model temperature
    LAMBDA_TEMPERATURE = 0.7
    LAMBDA_REQUEST_TIMEOUT = float(os.getenv("LAMBDA_REQUEST_TIMEOUT", "120"))
    LAMBDA_MAX_RETRIES = int(os.getenv("LAMBDA_MAX_RETRIES", "1"))

    # Workflow-specific model assignments
    # Change this single setting to switch ALL tasks to a different model
    # Options: "finllama", "custom", "base", "openai"
    ACTIVE_MODEL = "openai"

    # All tasks use the same model for consistent testing/comparison
    SIGNAL_EXTRACTION_MODEL = ACTIVE_MODEL
    ANALYSIS_MODEL = ACTIVE_MODEL
    ORCHESTRATION_MODEL = ACTIVE_MODEL

    # Anthropic (Claude) API key — used when CODE_GEN_PROVIDER="claude"
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

    # Code generation LLM (Agents 5 & 6) — independent of the main pipeline model
    # Set CODE_GEN_PROVIDER to switch between providers without touching other agents.
    # Options: "openai" | "claude" | "custom"
    CODE_GEN_PROVIDER = os.getenv("CODE_GEN_PROVIDER", "openai")
    # Model name for the chosen provider (e.g. "gpt-4o", "claude-opus-4-6", "llama-3.1-8b")
    CODE_GEN_MODEL = os.getenv("CODE_GEN_MODEL", "gpt-4o-mini")
    # API key for the code-gen provider.
    #   openai  → falls back to OPENAI_API_KEY if not set
    #   claude  → falls back to ANTHROPIC_API_KEY if not set
    #   custom  → falls back to "token-abc123" (vLLM dummy key) if not set
    CODE_GEN_API_KEY = os.getenv("CODE_GEN_API_KEY", "")
    # Base URL for "custom" provider (e.g. a second vLLM endpoint, Ollama, etc.)
    CODE_GEN_ENDPOINT = os.getenv("CODE_GEN_ENDPOINT", "")
    CODE_GEN_TEMPERATURE = float(os.getenv("CODE_GEN_TEMPERATURE", "0.1"))
    # How many times to re-send code to the sandbox after a runtime error
    SANDBOX_MAX_RETRIES = int(os.getenv("SANDBOX_MAX_RETRIES", "2"))

    # Sandbox endpoint for code execution (Phase 2/3)
    # This is the HTTP endpoint where generated code is sent for execution
    # The sandbox machine must have Sharadar data on local disk
    SANDBOX_ENDPOINT_URL = os.getenv("SANDBOX_ENDPOINT_URL", "http://localhost:9000")


class LangSmithConfig:
    """LangSmith tracing configuration"""
    TRACING_ENABLED = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    API_KEY = os.getenv("LANGCHAIN_API_KEY")
    PROJECT_NAME = os.getenv("LANGCHAIN_PROJECT", "autonomous-quant-research")


class BM25Config:
    """BM25 retrieval configuration for get_matching_content tool.

    Tune these via environment variables without code changes:
      BM25_SCORE_THRESHOLD  — minimum score to consider a chunk relevant.
                              Lower (e.g. 0.3) = more results, higher false-positive risk.
                              Higher (e.g. 1.0) = fewer but more precise results.
      BM25_TOP_K            — number of top-scoring chunks to return per query.
    """
    SCORE_THRESHOLD: float = float(os.getenv("BM25_SCORE_THRESHOLD") or "0.5")
    TOP_K: int = int(os.getenv("BM25_TOP_K") or "2")


class LoggingConfig:
    """Logging configuration"""
    LEVEL = "INFO"
    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOGGER_NAME = "agent-pipeline"
