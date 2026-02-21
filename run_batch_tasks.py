"""
Batch runner: iterate over all papers in GCS and run the agent pipeline.
Run from project root: python run_batch_tasks.py
"""

import argparse
import json
import logging
from pathlib import Path
from google.cloud import storage
from sympy import python

from Agents.agent import app
from Agents.config import GCSConfig

logger = logging.getLogger(__name__)

LOCAL_RESULTS = Path("results/")
LOCAL_RESULTS.mkdir(exist_ok=True)


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


def run_pipeline(paper_name: str, paper_text: str, model_choice: str = "base") -> dict:
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


def run_all(model_choice: str = "base"):
    papers = list_papers()
    logger.info(f"Found {len(papers)} papers")

    all_results = []

    for i, paper_name in enumerate(papers):
        logger.info(f"[{i+1}/{len(papers)}] Processing: {paper_name}")
        try:
            text = download_paper_text(paper_name)
            result = run_pipeline(paper_name, text, model_choice)
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
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Batch paper pipeline runner")
    parser.add_argument(
        "--model",
        choices=["openai", "base", "finllama", "custom"],
        default="base",
        help="LLM to use for the pipeline (default: base)",
    )
    args = parser.parse_args()

    results = run_all(model_choice=args.model)


# Example usage:
#    python run_batch_tasks.py --model custom # pyright: ignore[reportUndefinedVariable]
#    python run_batch_tasks.py --model openai
#   python run_batch_tasks.py --model finllama # type: ignore
#   python run_batch_tasks.py --model base   # default, same as plain: python run_batch_tasks.py
#   python run_batch_tasks.py --help         # shows all options