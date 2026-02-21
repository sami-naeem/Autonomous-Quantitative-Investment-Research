"""
Google Cloud Storage utilities for paper management.
Handles authentication, listing, and downloading papers from GCS.

This version is SAFE for service accounts without
`storage.buckets.list` permission.
"""

import json
import os
from typing import Any, Dict, List, Optional
from google.cloud import storage

from Agents.config import GCSConfig

# Module-level storage client (singleton)
_storage_client: Optional[storage.Client] = None


def ensure_auth() -> Optional[storage.Client]:
    """
    Ensure GCS authentication is set up correctly.

    - NEVER calls list_buckets()
    - Verifies access using bucket.exists() and list_blobs()
    """
    try:
        # Prefer explicit credentials path if it exists (Lambda-style)
        if os.path.exists(GCSConfig.LAMBDA_CREDENTIALS_PATH):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
                GCSConfig.LAMBDA_CREDENTIALS_PATH
            )

        client = storage.Client(project=GCSConfig.PROJECT_ID)

        # ✅ Bucket-scoped auth check (NO project-wide listing)
        bucket = client.bucket(GCSConfig.SOURCE_BUCKET_NAME)

        if not bucket.exists():
            raise PermissionError(
                f"Cannot access bucket '{GCSConfig.SOURCE_BUCKET_NAME}' "
                "(missing storage.buckets.get or bucket does not exist)."
            )

        # ✅ Object-level permission check
        next(
            bucket.list_blobs(
                prefix=GCSConfig.SOURCE_PREFIX,
                max_results=1,
            ),
            None,
        )

        return client

    except Exception:
        return None

def get_storage_client() -> Optional[storage.Client]:
    """
    Get or create the GCS storage client (singleton pattern).
    """
    global _storage_client
    if _storage_client is None:
        _storage_client = ensure_auth()
    return _storage_client


def list_available_papers(storage_client: storage.Client) -> List[str]:
    """
    List available processed papers from the configured bucket/prefix.
    """
    try:
        bucket = storage_client.bucket(GCSConfig.SOURCE_BUCKET_NAME)

        blobs = bucket.list_blobs(prefix=GCSConfig.SOURCE_PREFIX)

        paper_names = set()
        for blob in blobs:
            # expected path: <prefix>/<paper_name>/<file>.md
            remainder = blob.name.replace(GCSConfig.SOURCE_PREFIX, "")
            parts = remainder.split("/")
            if parts and parts[0]:
                paper_names.add(parts[0])

        return sorted(paper_names)

    except Exception:
        return []


def upload_agent_output(
    storage_client: storage.Client,
    blob_name: str,
    data: Dict[str, Any],
    bucket_name: Optional[str] = None,
) -> str:
    """
    Serialize `data` as JSON and upload it to the output GCS bucket.

    Args:
        storage_client: Authenticated GCS client.
        blob_name: Destination path inside the bucket, e.g. ``<paper_name>/<timestamp>.json``.
        data: JSON-serializable dict to upload.
        bucket_name: Override bucket; defaults to ``GCSConfig.OUTPUT_BUCKET_NAME``.

    Returns:
        The full GCS URI of the uploaded object (``gs://bucket/blob``).

    Raises:
        Exception: Re-raised as-is so callers can decide whether to fail hard or warn.
    """
    target_bucket = bucket_name or GCSConfig.OUTPUT_BUCKET_NAME
    bucket = storage_client.bucket(target_bucket)
    blob = bucket.blob(blob_name)
    payload = json.dumps(data, indent=2, default=str, ensure_ascii=False)
    blob.upload_from_string(payload, content_type="application/json")
    return f"gs://{target_bucket}/{blob_name}"


def download_paper_text(
    storage_client: storage.Client,
    paper_name: str,
) -> Optional[str]:
    """
    Download markdown content for a given paper folder.
    """
    try:
        bucket = storage_client.bucket(GCSConfig.SOURCE_BUCKET_NAME)

        prefix = f"{GCSConfig.SOURCE_PREFIX}{paper_name}/"
        blobs = bucket.list_blobs(prefix=prefix)

        for blob in blobs:
            if blob.name.endswith(".md"):
                return blob.download_as_text()

        return None

    except Exception:
        return None