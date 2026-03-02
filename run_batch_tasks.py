"""
Batch runner: iterate over all papers in GCS and run the agent pipeline.
Run from project root: python run_batch_tasks.py
"""

import argparse
import json
import logging
import re
from pathlib import Path
from google.cloud import storage
from Agents.agent import app
from Agents.config import GCSConfig

logger = logging.getLogger(__name__)

LOCAL_RESULTS = Path("results/")
LOCAL_RESULTS.mkdir(exist_ok=True)


class AgentOutputColorFormatter(logging.Formatter):
    RESET = "\x1b[0m"
    AGENT1 = "\x1b[96m"
    AGENT2 = "\x1b[93m"
    _AGENT1_PATTERN = re.compile(r"\bAgent\s*1\b", re.IGNORECASE)
    _AGENT2_PATTERN = re.compile(r"\bAgent\s*2\b", re.IGNORECASE)

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if self._AGENT1_PATTERN.search(message):
            return f"{self.AGENT1}{message}{self.RESET}"
        if self._AGENT2_PATTERN.search(message):
            return f"{self.AGENT2}{message}{self.RESET}"
        return message


def configure_colored_logging() -> None:
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=logging.INFO)

    for handler in root_logger.handlers:
        existing_formatter = handler.formatter
        fmt = (
            existing_formatter._fmt
            if existing_formatter and getattr(existing_formatter, "_fmt", None)
            else "%(levelname)s:%(name)s:%(message)s"
        )
        datefmt = existing_formatter.datefmt if existing_formatter else None
        handler.setFormatter(AgentOutputColorFormatter(fmt=fmt, datefmt=datefmt))


def get_gcs_client() -> storage.Client:
    creds_path = GCSConfig.CREDENTIALS_PATH or GCSConfig.LAMBDA_CREDENTIALS_PATH
    if creds_path and Path(creds_path).exists():
        return storage.Client.from_service_account_json(creds_path)
    return storage.Client(project=GCSConfig.PROJECT_ID)


def list_papers() -> list[str]:
    client = get_gcs_client()
    bucket = client.bucket(GCSConfig.SOURCE_BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=GCSConfig.SOURCE_PREFIX)

    paper_names = set()
    for blob in blobs:
        if not blob.name.endswith(".md"):
            continue
        parts = blob.name.replace(GCSConfig.SOURCE_PREFIX, "").split("/")
        if parts and parts[0]:
            paper_names.add(parts[0])

    return sorted(list(paper_names))


def download_paper_text(paper_name: str) -> str:
    """Download the .md file(s) for a given paper folder."""
    client = get_gcs_client()
    bucket = client.bucket(GCSConfig.SOURCE_BUCKET_NAME)
    prefix = f"{GCSConfig.SOURCE_PREFIX}{paper_name}/"
    blobs = list(bucket.list_blobs(prefix=prefix))

    md_blobs = [b for b in blobs if b.name.endswith(".md")]
    if not md_blobs:
        raise FileNotFoundError(f"No .md files found for {paper_name}")

    texts = [b.download_as_text() for b in md_blobs]
    return "\n\n".join(texts)


def run_pipeline(
    paper_name: str,
    paper_text: str,
    model_choice: str = "base",
    use_ground_truth_theme: bool = False,
    eval_policy_path: str = "",
) -> dict:
    initial_state = {
        "messages": [],
        "model_choice": model_choice,
        "paper_name": paper_name,
        "paper_text": paper_text,
        "paper_gcs_uri": "",
        "paper_blob_name": "",
        "abstract": "",
        "theme_pick": {},
        "factor_extract": {},
        "use_ground_truth_theme": use_ground_truth_theme,
        "eval_policy_path": eval_policy_path,
    }
    final_state = app.invoke(initial_state)

    logger.info(f"Final state keys: {list(final_state.keys())}")
    logger.info(f"theme_pick type: {type(final_state.get('theme_pick'))}, value: {final_state.get('theme_pick')}")
    logger.info(f"factor_extract type: {type(final_state.get('factor_extract'))}, value: {final_state.get('factor_extract')}")

    return {
        "paper_id": paper_name,
        "model_choice": model_choice,
        "theme_pick": json.dumps(final_state.get("theme_pick", {}), indent=2),
        "factor_extract": json.dumps(final_state.get("factor_extract", {}), indent=2),
        "output_gcs_uri": final_state.get("output_gcs_uri", ""),
    }


def run_all(
    model_choice: str = "base",
    start_index: int = 0,
    use_ground_truth_theme: bool = False,
    eval_policy_path: str = "",
):
    papers = list_papers()
    logger.info(f"Found {len(papers)} papers")

    if start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if start_index > len(papers):
        raise ValueError(f"--start-index ({start_index}) is greater than number of papers ({len(papers)})")
    if start_index > 0:
        logger.info(f"Skipping first {start_index} papers; starting from index {start_index}")

    all_results = []

    for i, paper_name in enumerate(papers[start_index:], start=start_index):
        logger.info(f"[{i+1}/{len(papers)}] Processing: {paper_name}")
        try:
            text = download_paper_text(paper_name)
            result = run_pipeline(
                paper_name, text, model_choice,
                use_ground_truth_theme=use_ground_truth_theme,
                eval_policy_path=eval_policy_path,
            )
            all_results.append(result)

            (LOCAL_RESULTS / f"{paper_name}.json").write_text(json.dumps(result, indent=2))

        except Exception as e:
            logger.error(f"Failed on {paper_name}: {e}")
            all_results.append({"paper_id": paper_name, "model_choice": model_choice, "error": str(e)})

    summary_path = LOCAL_RESULTS / "all_results.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    logger.info(f"Done. {len(all_results)} results written.")
    return all_results


if __name__ == "__main__":
    configure_colored_logging()

    parser = argparse.ArgumentParser(description="Batch paper pipeline runner")
    parser.add_argument(
        "--model",
        choices=["openai", "anthropic", "gemini", "base", "finllama", "custom"],
        default="base",
        help="LLM to use for the pipeline (default: base)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="0-based paper index to start from (e.g., 18 skips the first 18 papers)",
    )
    parser.add_argument(
        "--ground-truth-theme",
        action="store_true",
        default=False,
        help=(
            "Agent 1 runs normally, but Agent 2 receives the ground-truth "
            "jkp_theme from the eval policy instead of Agent 1's output."
        ),
    )
    parser.add_argument(
        "--eval-policy-path",
        type=str,
        default=None,
        help="Path to eval policy JSON (default: eval/osp_eval_policy_v003.json)",
    )
    args = parser.parse_args()

    results = run_all(
        model_choice=args.model,
        start_index=args.start_index,
        use_ground_truth_theme=args.ground_truth_theme,
        eval_policy_path=args.eval_policy_path or "",
    )


# Example usage:
#   python run_batch_tasks.py
#   python run_batch_tasks.py --model base
#   python run_batch_tasks.py --model openai
#   python run_batch_tasks.py --model finllama  # type: ignore
#   python run_batch_tasks.py --model custom    # pyright: ignore[reportUndefinedVariable]
#   python run_batch_tasks.py --start-index 18
#   python run_batch_tasks.py --model base --start-index 18  # skip first 18 papers
#   python run_batch_tasks.py --model finllama --start-index 18  # skip first 18 papers

#   python run_batch_tasks.py --model openai --start-index 18
#   python run_batch_tasks.py --help