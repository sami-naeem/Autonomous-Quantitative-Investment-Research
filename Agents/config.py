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
load_dotenv(env_path)


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
    LAMBDA_ENDPOINT = os.getenv("LAMBDA_ENDPOINT", "http://209.20.159.74:8000/v1")
    LAMBDA_API_KEY = os.getenv("LAMBDA_API_KEY", "token-abc123")

    # Model base path on Lambda server
    MODEL_BASE_PATH = "/models"

    BASE_MODEL = f"{MODEL_BASE_PATH}/{os.getenv('BASE_MODEL_NAME', 'llama-3.1-8b')}"
    # LoRA adapters are registered by name only (no path prefix) in vLLM
    FINLLAMA_MODEL = os.getenv('FINLLAMA_MODEL_NAME', 'finllama')
    CUSTOM_MODEL = os.getenv('CUSTOM_MODEL_NAME', 'custom')

    # Model temperature
    LAMBDA_TEMPERATURE = 0.7

    # Workflow-specific model assignments
    # Change this single setting to switch ALL tasks to a different model
    # Options: "finllama", "custom", "base", "openai"
    ACTIVE_MODEL = "openai"

    # All tasks use the same model for consistent testing/comparison
    SIGNAL_EXTRACTION_MODEL = ACTIVE_MODEL
    ANALYSIS_MODEL = ACTIVE_MODEL
    ORCHESTRATION_MODEL = ACTIVE_MODEL


class LangSmithConfig:
    """LangSmith tracing configuration"""
    TRACING_ENABLED = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    API_KEY = os.getenv("LANGCHAIN_API_KEY")
    PROJECT_NAME = os.getenv("LANGCHAIN_PROJECT", "autonomous-quant-research")


class LoggingConfig:
    """Logging configuration"""
    LEVEL = "INFO"
    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOGGER_NAME = "agent-pipeline"
