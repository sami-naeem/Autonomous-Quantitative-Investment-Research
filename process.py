import os
import tempfile
import shutil
import subprocess
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
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
    
    BATCH_SIZE = 10
    MAX_WORKERS = 4
    MINERU_WORKERS = 4
    
    PROCESSED_LOG = Path("processed_files.txt")
    DO_CLEANUP = True
    MINERU_BACKEND = "vlm-vllm-engine"

def ensure_auth():
    """Ensure GCS authentication is set up correctly"""
    try:
        credentials_path = "/workspace/secrets/service-account-key.json"
        if os.path.exists(credentials_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
            logger.info(f"Using service account key: {credentials_path}")
        
        storage_client = storage.Client(project=Config.PROJECT_ID)
        list(storage_client.list_buckets(max_results=1))
        logger.info("GCS authentication successful")
        return storage_client
    except Exception as e:
        logger.error(f"GCS authentication failed: {e}")
        logger.error("Ensure service account has 'Storage Admin' role")
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
    logger.debug(f"Recorded {len(paper_names)} papers to processed log")

def download_blob(blob_info):
    """Download a single blob from GCS"""
    blob_name, client, bucket = blob_info
    try:
        blob = bucket.blob(blob_name)
        local_path = Config.LOCAL_TEMP_DIR / Path(blob_name).name
        blob.download_to_filename(local_path)
        logger.debug(f"Downloaded {blob_name} to {local_path}")
        return local_path
    except Exception as e:
        logger.error(f"Error downloading {blob_name}: {e}")
        return None

def upload_folder(upload_info):
    """Upload all files in a folder to GCS"""
    local_folder_path, paper_name, client, bucket = upload_info
    local_folder = Path(local_folder_path)
    
    if not local_folder.exists():
        logger.warning(f"Output directory not found for {paper_name}")
        return 0
        
    if not any(local_folder.rglob("*")):
        logger.warning(f"No processed files found for {paper_name}")
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
                logger.debug(f"Uploaded {file_path} to {dest_blob_name}")
            except Exception as e:
                logger.error(f"Error uploading {file_path}: {e}")
    
    return uploaded_count

def check_mineru_installation():
    """Check if MinerU is properly installed and available"""
    try:
        result = subprocess.run("mineru --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"MinerU version: {result.stdout.strip()}")
            return True
        else:
            logger.error(f"MinerU check failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error checking MinerU: {e}")
        return False

def check_vllm_server():
    """Check if vLLM server is running (only for vlm-vllm-engine backend)"""
    try:
        result = subprocess.run(
            "curl -s http://localhost:30000/v1/models",
            shell=True,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            logger.info("vLLM server is running")
            return True
        else:
            logger.warning("vLLM server not responding")
            return False
    except Exception as e:
        logger.warning(f"vLLM server check failed: {e}")
        return False

def process_papers_batch(pdf_paths):
    """Process a batch of papers with MinerU"""
    if not pdf_paths:
        return []
    
    valid_paths = [p for p in pdf_paths if p is not None]
    if not valid_paths:
        logger.warning("No valid PDF paths to process in batch")
        return []
    
    input_paths_str = " ".join([f'"{path}"' for path in valid_paths])
    
    # For vlm-vllm-engine backend, don't use --workers flag as it's not supported
    if Config.MINERU_BACKEND == "vlm-vllm-engine":
        cmd = f"mineru -p {input_paths_str} -o {Config.LOCAL_OUTPUT_DIR} -b {Config.MINERU_BACKEND}"
    else:
        cmd = (f"mineru -p {input_paths_str} -o {Config.LOCAL_OUTPUT_DIR} -b {Config.MINERU_BACKEND} "
               f"--workers {Config.MINERU_WORKERS}")
    
    logger.info(f"Running MinerU batch: {len(valid_paths)} papers using '{Config.MINERU_BACKEND}' backend")
    logger.debug(f"MinerU command: {cmd}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
        elapsed = time.time() - start_time
        
        logger.info(f"Batch processed in {elapsed:.2f} seconds")
        
        if result.returncode != 0:
            logger.error(f"MinerU error: {result.stderr}")
            logger.debug(f"Stdout: {result.stdout}")
            return []
        
        processed = []
        for path in valid_paths:
            paper_name = path.stem
            
            # For vlm-vllm-engine backend, output is in paper_name/vlm/
            # Look for any files in that directory structure
            if Config.MINERU_BACKEND == "vlm-vllm-engine":
                output_dir = Config.LOCAL_OUTPUT_DIR / paper_name / "vlm"
            else:
                output_dir = Config.LOCAL_OUTPUT_DIR / paper_name
            
            logger.debug(f"Checking output for {paper_name} at {output_dir}")
            
            if output_dir.exists():
                # Recursively find all files in the vlm directory (includes images/, md files, etc)
                all_files = list(output_dir.rglob("*"))
                actual_files = [f for f in all_files if f.is_file()]
                
                if actual_files:
                    logger.info(f"Output for {paper_name}: {len(actual_files)} files generated")
                    processed.append(paper_name)
                else:
                    logger.warning(f"No output files generated for {paper_name}")
            else:
                logger.warning(f"Output directory not created for {paper_name}")
        
        if not processed:
            logger.error("No papers successfully processed in this batch")
            failure_log = Config.LOCAL_OUTPUT_DIR / "mineru_failure.log"
            with open(failure_log, "w") as f:
                f.write(f"Command: {cmd}\n")
                f.write(f"Return code: {result.returncode}\n")
                f.write(f"Stdout:\n{result.stdout}\n")
                f.write(f"Stderr:\n{result.stderr}\n")
            logger.info(f"Wrote failure log to {failure_log}")
        
        return processed
    except subprocess.TimeoutExpired:
        logger.error("MinerU processing timed out after 600 seconds")
        return []
    except Exception as e:
        logger.error(f"Exception during MinerU processing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def run_pipeline():
    """Main pipeline function that can be called directly from a notebook"""
    logger.info("Starting MinerU GCS Pipeline")
    
    if not check_mineru_installation():
        logger.error("MinerU is not properly installed. Exiting.")
        return
    
    if Config.MINERU_BACKEND == "vlm-vllm-engine":
        if not check_vllm_server():
            logger.error("vLLM server not running. Start it with: mineru-vllm-server --host 0.0.0.0 --port 30000")
            return
    
    os.makedirs(Config.LOCAL_TEMP_DIR, exist_ok=True)
    os.makedirs(Config.LOCAL_OUTPUT_DIR, exist_ok=True)
    
    storage_client = ensure_auth()
    if storage_client is None:
        logger.error("Failed to authenticate with GCS. Exiting.")
        return
    
    already_processed = get_already_processed()
    
    try:
        source_bucket = storage_client.bucket(Config.SOURCE_BUCKET_NAME)
        dest_bucket = storage_client.bucket(Config.DESTINATION_BUCKET_NAME)
        logger.info(f"Source bucket: {Config.SOURCE_BUCKET_NAME}")
        logger.info(f"Destination bucket: {Config.DESTINATION_BUCKET_NAME}")
    except Exception as e:
        logger.error(f"Failed to initialize GCS buckets: {e}")
        return
    
    try:
        logger.info(f"Listing PDFs in {Config.SOURCE_BUCKET_NAME}/{Config.SOURCE_PREFIX}")
        blobs = list(source_bucket.list_blobs(prefix=Config.SOURCE_PREFIX))
        pdf_blobs = [blob for blob in blobs if blob.name.lower().endswith('.pdf')]
        
        if not pdf_blobs:
            logger.warning(f"No PDF files found in {Config.SOURCE_BUCKET_NAME}/{Config.SOURCE_PREFIX}")
            return
        
        logger.info(f"Found {len(pdf_blobs)} PDF files")
        
        new_pdf_blobs = []
        for blob in pdf_blobs:
            paper_name = Path(blob.name).stem
            if paper_name not in already_processed:
                new_pdf_blobs.append(blob)
        
        logger.info(f"Found {len(new_pdf_blobs)} new papers to process")
        
        if not new_pdf_blobs:
            logger.info("No new papers to process. Exiting.")
            return
    except Exception as e:
        logger.error(f"Error listing blobs: {e}")
        return
    
    try:
        logger.info("Downloading PDF files...")
        blob_infos = [(blob.name, storage_client, source_bucket) for blob in new_pdf_blobs]
        
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            local_paths = list(tqdm(
                executor.map(download_blob, blob_infos),
                total=len(new_pdf_blobs),
                desc="Downloading"
            ))
        
        logger.info(f"Processing PDF files with MinerU using {Config.MINERU_BACKEND} backend...")
        all_processed = []
        for i in range(0, len(local_paths), Config.BATCH_SIZE):
            batch = local_paths[i:i+Config.BATCH_SIZE]
            batch_num = (i // Config.BATCH_SIZE) + 1
            total_batches = (len(local_paths) + Config.BATCH_SIZE - 1) // Config.BATCH_SIZE
            logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            processed_names = process_papers_batch(batch)
            all_processed.extend(processed_names)
            
            # Only record papers that were actually successfully processed
            if processed_names:
                record_processed(processed_names)
        
        logger.info("Uploading processed files...")
        
        if not all_processed:
            logger.error("No papers were successfully processed. Skipping upload.")
            Config.DO_CLEANUP = False
            return
        
        upload_infos = []
        for paper_name in all_processed:
            paper_output_dir = Config.LOCAL_OUTPUT_DIR / paper_name
            if paper_output_dir.exists() and any(paper_output_dir.rglob("*")):
                upload_infos.append((paper_output_dir, paper_name, storage_client, dest_bucket))
                file_count = sum(1 for _ in paper_output_dir.rglob('*') if _.is_file())
                logger.info(f"Queuing upload for {paper_name} ({file_count} files)")
            else:
                logger.warning(f"No output directory or empty directory for {paper_name}")
        
        if not upload_infos:
            logger.error("No output directories found for upload. Skipping upload.")
            Config.DO_CLEANUP = False
            return
        
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            results = list(tqdm(
                executor.map(upload_folder, upload_infos),
                total=len(upload_infos),
                desc="Uploading"
            ))
            total_uploaded = sum(results)
        
        logger.info(f"Pipeline complete: Processed {len(all_processed)} papers, uploaded {total_uploaded} files")
        
    except Exception as e:
        logger.error(f"Error in pipeline: {e}")
        Config.DO_CLEANUP = False
    finally:
        if Config.DO_CLEANUP:
            logger.info("Cleaning up temporary files...")
            try:
                shutil.rmtree(Config.LOCAL_TEMP_DIR)
                shutil.rmtree(Config.LOCAL_OUTPUT_DIR)
                logger.info("Cleanup complete")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
        else:
            logger.info("Skipping cleanup as requested or due to errors")

if __name__ == "__main__":
    try:
        import argparse
        parser = argparse.ArgumentParser(description='Process papers from GCS with MinerU')
        parser.add_argument('--batch-size', type=int, help='Number of papers to process in each batch')
        parser.add_argument('--workers', type=int, help='Number of MinerU worker threads (ignored for vlm-vllm-engine)')
        parser.add_argument('--no-cleanup', action='store_true', help='Skip cleanup of temporary files')
        parser.add_argument('--backend', choices=['pipeline', 'vlm-transformers', 'vlm-vllm-engine'], 
                          help='MinerU backend to use')
        
        args, unknown = parser.parse_known_args()
        
        if args.batch_size:
            Config.BATCH_SIZE = args.batch_size
        if args.workers:
            Config.MINERU_WORKERS = args.workers
        if args.no_cleanup:
            Config.DO_CLEANUP = False
        if args.backend:
            Config.MINERU_BACKEND = args.backend
    except:
        pass
    
    run_pipeline()