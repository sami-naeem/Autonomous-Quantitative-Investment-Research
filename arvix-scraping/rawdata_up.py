import arxiv
import os
from pathlib import Path
from google.cloud import storage
from dotenv import load_dotenv
import tempfile
import time
import argparse
from typing import Optional

# Load environment variables from .env file
load_dotenv()

class ArxivGCSDownloader:
    """Download arXiv papers and upload them to Google Cloud Storage"""
    
    def __init__(self, bucket_name: Optional[str] = None, gcs_folder: str = "papers"):
        """
        Initialize the downloader with GCS credentials

        Args:
            bucket_name: GCS bucket name (defaults to env variable GCS_BUCKET_NAME)
            gcs_folder: Folder path within the bucket (default: "papers")
        """
        # Load GCS configuration from environment
        self.bucket_name = bucket_name or os.getenv("GCS_BUCKET_NAME")
        self.gcs_folder = gcs_folder

        # Initialize GCS client
        # GOOGLE_APPLICATION_CREDENTIALS env var should point to service account JSON
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if credentials_path and not os.path.exists(credentials_path):
            raise FileNotFoundError(f"Credentials file not found: {credentials_path}")

        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.bucket_name)

        # Rate limiting configuration (per arXiv guidelines)
        self.requests_per_burst = 4
        self.burst_sleep = 1  # seconds

        print(f"Initialized GCS client for bucket: {self.bucket_name}")
        print(f"Papers will be uploaded to: gs://{self.bucket_name}/{self.gcs_folder}/")
    
    def download_and_upload_papers(
        self,
        category: str = "q-fin",
        max_papers: int = 100,
        start_index: int = 0,
        local_temp_storage: bool = False,
        temp_dir: str = "./papers_temp"
    ):
        """
        Download papers from arXiv and upload to GCS

        Args:
            category: arXiv category (e.g., "q-fin.MF")
            max_papers: Maximum number of papers to download
            start_index: Index to start downloading from (useful for resuming)
            local_temp_storage: If True, keep local copies in temp_dir
            temp_dir: Directory for temporary local storage
        """
        # Create temp directory if keeping local copies
        if local_temp_storage:
            Path(temp_dir).mkdir(exist_ok=True)

        # Initialize arXiv client with rate limiting and better retry handling
        # Use larger page_size and more retries per arxiv.py best practices
        client = arxiv.Client(
            page_size=50,  # Larger page size reduces total requests
            delay_seconds=0,  # We implement our own rate limiting
            num_retries=3  # Increased retries for flaky API
        )
        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=max_papers + start_index,  # Request total papers including skipped ones
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        download_count = 0
        upload_count = 0
        error_count = 0
        skipped_count = 0
        current_index = 0
        requests_in_burst = 0

        print(f"\nStarting download for category: {category}")
        print(f"Starting from index: {start_index}")
        print(f"Maximum papers to process: {max_papers}")
        print(f"Rate limiting: {self.requests_per_burst} requests/burst, {self.burst_sleep}s sleep/burst")
        print(f"Using arXiv's recommended export endpoint\n")

        try:
            for result in client.results(search):
                # Skip papers before start_index
                if current_index < start_index:
                    current_index += 1
                    skipped_count += 1
                    continue

                # Stop if we've processed max_papers
                if download_count >= max_papers:
                    break

                current_index += 1

                try:
                    # Rate limiting: sleep after every burst
                    if requests_in_burst >= self.requests_per_burst:
                        time.sleep(self.burst_sleep)
                        requests_in_burst = 0

                    # Generate filename from paper ID
                    filename = f"{result.entry_id.split('/abs/')[-1].replace('/', '_')}.pdf"

                    # Use temporary directory for download
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        # Download paper to temp location
                        temp_path = os.path.join(tmpdirname, filename)
                        result.download_pdf(dirpath=tmpdirname, filename=filename)
                        download_count += 1
                        requests_in_burst += 1

                        # Upload to GCS
                        gcs_path = f"{self.gcs_folder}/{filename}"
                        blob = self.bucket.blob(gcs_path)

                        # Check if file already exists
                        if blob.exists():
                            print(f"⊙ [{current_index}] Already exists: {result.title[:50]}... (skipping)")
                            continue

                        # Upload with metadata
                        blob.metadata = {
                            'title': result.title,
                            'authors': ', '.join([author.name for author in result.authors]),
                            'published': str(result.published),
                            'arxiv_id': result.entry_id,
                            'categories': ', '.join(result.categories)
                        }

                        blob.upload_from_filename(temp_path)
                        upload_count += 1

                        print(f"✓ [{current_index}] Uploaded: {result.title[:50]}...")
                        print(f"  → gs://{self.bucket_name}/{gcs_path}")

                        # Optionally keep local copy
                        if local_temp_storage:
                            local_path = os.path.join(temp_dir, filename)
                            with open(temp_path, 'rb') as src, open(local_path, 'wb') as dst:
                                dst.write(src.read())

                except Exception as e:
                    error_count += 1
                    print(f"✗ [{current_index}] Failed: {result.title[:50] if hasattr(result, 'title') else 'Unknown'}")
                    print(f"  Error: {e}")

        except arxiv.UnexpectedEmptyPageError as e:
            # This likely means we've reached the end of available papers
            print(f"\n⚠ Reached end of available papers at index {current_index}")
            print(f"  This category may have ~{current_index} total papers")
            print(f"  ArXiv API details: {e}")

        # Print summary
        print(f"\n{'='*60}")
        print(f"Download Summary:")
        print(f"  Skipped (start_index): {skipped_count}")
        print(f"  Downloaded: {download_count}")
        print(f"  Uploaded to GCS: {upload_count}")
        print(f"  Errors: {error_count}")
        print(f"  Last processed index: {current_index}")
        print(f"  GCS Location: gs://{self.bucket_name}/{self.gcs_folder}/")
        print(f"{'='*60}")

        return {
            'skipped': skipped_count,
            'downloaded': download_count,
            'uploaded': upload_count,
            'errors': error_count,
            'last_index': current_index
        }
    
    def list_uploaded_papers(self, max_results: int = 10):
        """List papers in the GCS bucket"""
        print(f"\nListing papers in gs://{self.bucket_name}/{self.gcs_folder}/")
        blobs = self.storage_client.list_blobs(
            self.bucket_name, 
            prefix=self.gcs_folder,
            max_results=max_results
        )
        
        for blob in blobs:
            if blob.name.endswith('.pdf'):
                print(f"  • {blob.name}")
                if blob.metadata:
                    print(f"    Title: {blob.metadata.get('title', 'N/A')[:60]}")
        
# Example usage
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Download arXiv papers and upload to Google Cloud Storage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download first 100 papers from q-fin.MF category
  python rawdata_up.py --category q-fin.MF --max-papers 100

  # Resume from index 500 (skip first 500 papers)
  python rawdata_up.py --category q-fin.MF --max-papers 1000 --start-index 500

Rate Limiting:
  This script follows arXiv's usage guidelines:
  - Uses export.arxiv.org (recommended for programmatic access)
  - Rate limit: 4 requests per second with 1 second sleep per burst
        """
    )

    parser.add_argument(
        '--category',
        type=str,
        default='q-fin.MF',
        help='arXiv category to download (e.g., q-fin.MF, cs.AI, etc.)'
    )

    parser.add_argument(
        '--max-papers',
        type=int,
        default=5000,
        help='Maximum number of papers to download (default: 5000)'
    )

    parser.add_argument(
        '--start-index',
        type=int,
        default=0,
        help='Index to start downloading from, useful for resuming (default: 0)'
    )

    args = parser.parse_args()

    # Initialize downloader (uses environment variables)
    downloader = ArxivGCSDownloader()

    # Download and upload papers
    stats = downloader.download_and_upload_papers(
        category=args.category,
        max_papers=args.max_papers,
        start_index=args.start_index,
        local_temp_storage=False
    )

    # List some uploaded papers
    print("\n")
    downloader.list_uploaded_papers(max_results=10)

    # Print final stats for easy reference
    print(f"\nTo resume from where you left off, use:")
    print(f"  python rawdata_up.py --category {args.category} --start-index {stats['last_index']}")
