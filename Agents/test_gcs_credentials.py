"""
Simple GCS credentials test script.
Run this to verify your credentials have access to the bucket.
"""
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.cloud import storage
from Agents.config import GCSConfig


def test_credentials():
    """Test GCS credentials and bucket access."""
    print("=" * 50)
    print("GCS Credentials Test")
    print("=" * 50)

    # Show configuration
    print(f"\nConfiguration:")
    print(f"  Project ID: {GCSConfig.PROJECT_ID}")
    print(f"  Bucket: {GCSConfig.SOURCE_BUCKET_NAME}")
    print(f"  Prefix: {GCSConfig.SOURCE_PREFIX}")
    print(f"  Credentials path: {GCSConfig.CREDENTIALS_PATH}")
    print(f"  Lambda credentials path: {GCSConfig.LAMBDA_CREDENTIALS_PATH}")

    # Check which credentials file exists
    creds_path = None
    if GCSConfig.CREDENTIALS_PATH and os.path.exists(GCSConfig.CREDENTIALS_PATH):
        creds_path = GCSConfig.CREDENTIALS_PATH
        print(f"\n✓ Found credentials at: {creds_path}")
    elif os.path.exists(GCSConfig.LAMBDA_CREDENTIALS_PATH):
        creds_path = GCSConfig.LAMBDA_CREDENTIALS_PATH
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
        print(f"\n✓ Found Lambda credentials at: {creds_path}")
    else:
        print(f"\n✗ No credentials file found!")
        print(f"  Checked: {GCSConfig.CREDENTIALS_PATH}")
        print(f"  Checked: {GCSConfig.LAMBDA_CREDENTIALS_PATH}")
        return False

    # Test 1: Create client
    print("\n--- Test 1: Create Storage Client ---")
    try:
        client = storage.Client(project=GCSConfig.PROJECT_ID)
        print("✓ Storage client created successfully")
    except Exception as e:
        print(f"✗ Failed to create client: {e}")
        return False

    # Test 2: List buckets (tests basic auth)
    print("\n--- Test 2: List Buckets ---")
    try:
        buckets = list(client.list_buckets(max_results=5))
        print(f"✓ Can list buckets. Found {len(buckets)} bucket(s):")
        for b in buckets:
            print(f"    - {b.name}")
    except Exception as e:
        print(f"✗ Failed to list buckets: {e}")
        return False

    # Test 3: Access target bucket
    print(f"\n--- Test 3: Access Bucket '{GCSConfig.SOURCE_BUCKET_NAME}' ---")
    try:
        bucket = client.bucket(GCSConfig.SOURCE_BUCKET_NAME)
        if bucket.exists():
            print(f"✓ Bucket '{GCSConfig.SOURCE_BUCKET_NAME}' exists and is accessible")
        else:
            print(f"✗ Bucket '{GCSConfig.SOURCE_BUCKET_NAME}' does not exist")
            return False
    except Exception as e:
        print(f"✗ Failed to access bucket: {e}")
        return False

    # Test 4: List files in processed/ folder
    print(f"\n--- Test 4: List Files in '{GCSConfig.SOURCE_PREFIX}' ---")
    try:
        blobs = list(bucket.list_blobs(prefix=GCSConfig.SOURCE_PREFIX, max_results=10))
        if blobs:
            print(f"✓ Found {len(blobs)} file(s) in '{GCSConfig.SOURCE_PREFIX}':")
            for blob in blobs[:10]:
                print(f"    - {blob.name}")
            if len(blobs) == 10:
                print("    ... (showing first 10 only)")
        else:
            print(f"⚠ No files found in '{GCSConfig.SOURCE_PREFIX}' (folder may be empty)")
    except Exception as e:
        print(f"✗ Failed to list files: {e}")
        return False

    print("\n" + "=" * 50)
    print("✓ All tests passed! Credentials are working.")
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = test_credentials()
    sys.exit(0 if success else 1)
