#local mineru pipeline script for processing papers with http-client backend and vLLM server, with background uploads and logging
#process_test.py
import os
import shutil
import subprocess
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from tqdm import tqdm
from google.cloud import storage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mineru_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mineru-pipeline")

class Config:
    PROJECT_ID = "seraphic-ripple-476917-i5"
    SOURCE_BUCKET_NAME = "raw_papers"
    DESTINATION_BUCKET_NAME = "raw_papers"
    SOURCE_PREFIX = "papers/"
    DEST_PREFIX = "processed/"

    LOCAL_TEMP_DIR = Path("./temp_papers")
    LOCAL_OUTPUT_DIR = Path("./processed_papers")

    MAX_WORKERS = 4
    DOWNLOAD_BATCH_SIZE = 20  # Download N papers at a time
    UPLOAD_BATCH_SIZE = 4  # Upload after every N processed papers

    PROCESSED_LOG = Path("processed_files.txt")
    DO_CLEANUP = True

    VLLM_SERVER_URL = "http://localhost:30000"

def ensure_auth():
    """Ensure GCS authentication is set up correctly"""
    try:
        credentials_path = "/workspace/secrets/service-account-key.json"
        if os.path.exists(credentials_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
            logger.info(f"Using service account key")
        
        storage_client = storage.Client(project=Config.PROJECT_ID)
        list(storage_client.list_buckets(max_results=1))
        logger.info("GCS authentication successful")
        return storage_client
    except Exception as e:
        logger.error(f"GCS authentication failed: {e}")
        return None

def get_already_processed():
    """Get set of already processed files from log"""
    already_processed = set()
    if Config.PROCESSED_LOG.exists():
        with open(Config.PROCESSED_LOG, "r") as f:
            already_processed = set(line.strip() for line in f)
        logger.info(f"Found {len(already_processed)} already processed papers")
    return already_processed

def record_processed(paper_names):
    """Record processed paper names to log file"""
    with open(Config.PROCESSED_LOG, "a") as f:
        for name in paper_names:
            f.write(f"{name}\n")

def download_blob(blob_info):
    """Download a single blob from GCS"""
    blob_name, client, bucket = blob_info
    try:
        blob = bucket.blob(blob_name)
        local_path = Config.LOCAL_TEMP_DIR / Path(blob_name).name
        blob.download_to_filename(local_path)
        return local_path
    except Exception as e:
        logger.error(f"Error downloading {blob_name}: {e}")
        return None

def upload_folder(upload_info):
    """Upload all files in a folder to GCS"""
    local_folder_path, paper_name, client, bucket = upload_info
    local_folder = Path(local_folder_path)
    
    if not local_folder.exists():
        return 0
    
    if not any(local_folder.rglob("*")):
        return 0
    
    uploaded_count = 0
    for file_path in local_folder.rglob("*"):
        if file_path.is_file():
            try:
                rel_path = file_path.relative_to(Config.LOCAL_OUTPUT_DIR)
                dest_blob_name = f"{Config.DEST_PREFIX}{rel_path}"
                blob = bucket.blob(dest_blob_name)
                blob.upload_from_filename(file_path)
                uploaded_count += 1
            except Exception as e:
                logger.error(f"Error uploading {file_path}: {e}")
    
    return uploaded_count

def check_mineru_installation():
    """Check if MinerU is properly installed"""
    try:
        result = subprocess.run("mineru --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"MinerU version: {result.stdout.strip()}")
            return True
    except:
        pass
    return False

def check_vllm_server():
    """Check if vLLM server is running"""
    try:
        result = subprocess.run(
            f"curl -s {Config.VLLM_SERVER_URL}/v1/models",
            shell=True,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            logger.info(f"vLLM server is running at {Config.VLLM_SERVER_URL}")
            return True
        else:
            logger.error(f"vLLM server not responding at {Config.VLLM_SERVER_URL}")
            return False
    except Exception as e:
        logger.error(f"vLLM server check failed: {e}")
        return False

def process_single_paper(pdf_path):
    """Process a single paper with MinerU using http-client backend"""
    if pdf_path is None:
        return None
    
    paper_name = pdf_path.stem
    cmd = (f"mineru -p \"{pdf_path}\" -o {Config.LOCAL_OUTPUT_DIR} "
           f"-b vlm-http-client -u {Config.VLLM_SERVER_URL}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            logger.warning(f"{paper_name}: processing failed ({elapsed:.2f}s) - {result.stderr[:100]}")
            return None
        
        output_dir = Config.LOCAL_OUTPUT_DIR / paper_name
        
        if output_dir.exists():
            all_files = list(output_dir.rglob("*"))
            actual_files = [f for f in all_files if f.is_file()]
            
            if actual_files:
                logger.info(f"{paper_name}: ✓ {len(actual_files)} files ({elapsed:.2f}s)")
                return paper_name
            else:
                logger.warning(f"{paper_name}: no files generated ({elapsed:.2f}s)")
                return None
        else:
            logger.warning(f"{paper_name}: output directory not created ({elapsed:.2f}s)")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error(f"{paper_name}: processing timed out (>300s)")
        return None
    except Exception as e:
        logger.error(f"{paper_name}: exception - {e}")
        return None

def upload_batch_background(batch, storage_client, dest_bucket):
    """Upload batch in background thread"""
    # Record as processed first (in case upload fails)
    record_processed(batch)

    # Upload the batch
    upload_infos = []
    for paper_name in batch:
        paper_dir = Config.LOCAL_OUTPUT_DIR / paper_name
        if paper_dir.exists() and any(paper_dir.rglob("*")):
            upload_infos.append((paper_dir, paper_name, storage_client, dest_bucket))

    if upload_infos:
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            results = list(executor.map(upload_folder, upload_infos))
            batch_uploaded = sum(results)

        logger.info(f"Background upload complete: {batch_uploaded} files ({len(batch)} papers)")

        # Clean up uploaded papers if cleanup is enabled
        if Config.DO_CLEANUP:
            for paper_name in batch:
                paper_dir = Config.LOCAL_OUTPUT_DIR / paper_name
                if paper_dir.exists():
                    shutil.rmtree(paper_dir, ignore_errors=True)

        return batch_uploaded
    return 0

def process_papers_list(pdf_paths, storage_client, dest_bucket, batch_size=4):
    """Process a list of papers individually via http-client with batched uploads"""
    if not pdf_paths:
        return []

    valid_paths = [p for p in pdf_paths if p is not None]
    if not valid_paths:
        return []

    logger.info(f"Processing {len(valid_paths)} papers individually with background uploads every {batch_size} papers...")

    processed = []
    batch_buffer = []
    total_uploaded = 0
    upload_thread = None  # Track background upload thread

    for idx, path in enumerate(valid_paths, 1):
        result = process_single_paper(path)
        if result:
            processed.append(result)
            batch_buffer.append(result)

            # Upload when batch is full or at the end
            if len(batch_buffer) >= batch_size or idx == len(valid_paths):
                # Wait for previous upload to complete before starting new one
                if upload_thread:
                    logger.info("Waiting for previous upload to complete...")
                    upload_thread.join()
                    logger.info("Previous upload finished")

                logger.info(f"Starting background upload of {len(batch_buffer)} papers...")

                # Start upload in background thread
                batch_copy = batch_buffer.copy()  # Copy to avoid race conditions
                upload_thread = Thread(
                    target=upload_batch_background,
                    args=(batch_copy, storage_client, dest_bucket)
                )
                upload_thread.start()

                batch_buffer = []

    # Wait for final upload to complete
    if upload_thread:
        logger.info("Waiting for final upload to complete...")
        upload_thread.join()
        logger.info("Final upload complete")

    logger.info(f"All processing and uploads complete")
    return processed

def run_pipeline():
    """Main pipeline function"""
    logger.info("Starting MinerU GCS Pipeline (individual http-client processing)")
    
    if not check_mineru_installation():
        logger.error("MinerU not installed")
        return
    
    if not check_vllm_server():
        logger.error("vLLM server not running. Start with: mineru-vllm-server --host 0.0.0.0 --port 30000")
        return
    
    os.makedirs(Config.LOCAL_TEMP_DIR, exist_ok=True)
    os.makedirs(Config.LOCAL_OUTPUT_DIR, exist_ok=True)
    
    storage_client = ensure_auth()
    if not storage_client:
        return
    
    already_processed = get_already_processed()
    
    source_bucket = storage_client.bucket(Config.SOURCE_BUCKET_NAME)
    dest_bucket = storage_client.bucket(Config.DESTINATION_BUCKET_NAME)
    
    logger.info(f"Listing PDFs in {Config.SOURCE_BUCKET_NAME}/{Config.SOURCE_PREFIX}")
    blobs = list(source_bucket.list_blobs(prefix=Config.SOURCE_PREFIX))
    pdf_blobs = [blob for blob in blobs if blob.name.lower().endswith('.pdf')]
    
    logger.info(f"Found {len(pdf_blobs)} total PDFs, {len(already_processed)} already processed")
    
    new_pdf_blobs = [blob for blob in pdf_blobs if Path(blob.name).stem not in already_processed]
    logger.info(f"Found {len(new_pdf_blobs)} new papers to process")
    
    if not new_pdf_blobs:
        logger.info("No new papers to process")
        return

    # Process in download batches to avoid filling disk
    total_papers = len(new_pdf_blobs)
    total_processed = 0

    for batch_start in range(0, total_papers, Config.DOWNLOAD_BATCH_SIZE):
        batch_end = min(batch_start + Config.DOWNLOAD_BATCH_SIZE, total_papers)
        batch_blobs = new_pdf_blobs[batch_start:batch_end]

        logger.info(f"\n{'='*60}")
        logger.info(f"DOWNLOAD BATCH {batch_start//Config.DOWNLOAD_BATCH_SIZE + 1}: Papers {batch_start+1}-{batch_end} of {total_papers}")
        logger.info(f"{'='*60}")

        # Download this batch
        logger.info(f"Downloading {len(batch_blobs)} PDF files...")
        blob_infos = [(blob.name, storage_client, source_bucket) for blob in batch_blobs]

        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            local_paths = list(tqdm(
                executor.map(download_blob, blob_infos),
                total=len(batch_blobs),
                desc="Downloading"
            ))

        # Process this batch
        logger.info(f"Processing {len(local_paths)} papers with vLLM server...")
        batch_processed = process_papers_list(local_paths, storage_client, dest_bucket, batch_size=Config.UPLOAD_BATCH_SIZE)
        total_processed += len(batch_processed)
        logger.info(f"Batch complete: {len(batch_processed)}/{len(local_paths)} papers processed")

        # Cleanup downloaded PDFs from this batch
        if Config.DO_CLEANUP:
            logger.info("Cleaning up downloaded PDFs from this batch...")
            shutil.rmtree(Config.LOCAL_TEMP_DIR, ignore_errors=True)
            os.makedirs(Config.LOCAL_TEMP_DIR, exist_ok=True)
            logger.info("Batch cleanup complete")

    logger.info(f"\n{'='*60}")
    logger.info(f"PIPELINE COMPLETE: {total_processed}/{total_papers} papers successfully processed")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    run_pipeline()