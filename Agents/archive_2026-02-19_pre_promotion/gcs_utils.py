"""
Google Cloud Storage utilities for paper management.
Handles authentication, listing, and downloading papers from GCS.
"""
import os
import logging
from typing import List, Optional
from google.cloud import storage

from Agents.config import GCSConfig

logger = logging.getLogger("agent-pipeline")

# Module-level storage client (singleton pattern)
_storage_client = None


def ensure_auth() -> Optional[storage.Client]:
    """
    Ensure GCS authentication is set up correctly.

    Returns:
        storage.Client if authentication succeeds, None otherwise
    """
    try:
        # Check for Lambda/server environment credentials first
        if os.path.exists(GCSConfig.LAMBDA_CREDENTIALS_PATH):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCSConfig.LAMBDA_CREDENTIALS_PATH
            logger.info("Using Lambda service account key")

        # Create storage client
        storage_client = storage.Client(project=GCSConfig.PROJECT_ID)

        # Test authentication by listing buckets
        list(storage_client.list_buckets(max_results=1))
        logger.info("GCS authentication successful")

        return storage_client
    except Exception as e:
        logger.error(f"GCS authentication failed: {e}")
        return None


def get_storage_client() -> Optional[storage.Client]:
    """
    Get or create the GCS storage client (singleton pattern).

    Returns:
        storage.Client instance or None if authentication fails
    """
    global _storage_client
    if _storage_client is None:
        _storage_client = ensure_auth()
    return _storage_client


def list_available_papers(storage_client: storage.Client) -> List[str]:
    """
    List available processed papers from GCS bucket.

    Args:
        storage_client: Authenticated GCS storage client

    Returns:
        Sorted list of paper names (folder names in the processed/ directory)
    """
    try:
        bucket = storage_client.bucket(GCSConfig.SOURCE_BUCKET_NAME)
        blobs = list(bucket.list_blobs(prefix=GCSConfig.SOURCE_PREFIX))

        # Get unique paper folder names (processed papers have folder structure)
        paper_names = set()
        for blob in blobs:
            # Extract paper name from path like "processed/paper_name/file.md"
            parts = blob.name.replace(GCSConfig.SOURCE_PREFIX, "").split("/")
            if parts and parts[0]:
                paper_names.add(parts[0])

        return sorted(list(paper_names))
    except Exception as e:
        logger.error(f"Error listing papers: {e}")
        return []


def download_paper_text(storage_client: storage.Client, paper_name: str) -> Optional[str]:
    """
    Download and read the markdown content of a processed paper.

    Args:
        storage_client: Authenticated GCS storage client
        paper_name: Name of the paper folder to download

    Returns:
        Paper content as markdown string, or None if not found
    """
    try:
        bucket = storage_client.bucket(GCSConfig.SOURCE_BUCKET_NAME)

        # Look for markdown files in the paper's folder
        prefix = f"{GCSConfig.SOURCE_PREFIX}{paper_name}/"
        blobs = list(bucket.list_blobs(prefix=prefix))

        # Find markdown file
        md_blob = None
        for blob in blobs:
            if blob.name.endswith(".md"):
                md_blob = blob
                break

        if md_blob:
            content = md_blob.download_as_text()
            logger.info(f"Downloaded paper: {paper_name} ({len(content)} characters)")
            return content
        else:
            logger.warning(f"No markdown file found for paper: {paper_name}")
            return None
    except Exception as e:
        logger.error(f"Error downloading paper {paper_name}: {e}")
        return None
