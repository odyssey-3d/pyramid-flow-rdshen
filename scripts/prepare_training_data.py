import pandas as pd
import jsonlines
import requests
from pathlib import Path
import concurrent.futures
from tqdm import tqdm
import os
import hashlib
from PIL import Image
import io
import shutil
import psutil
import cv2
import tempfile
import csv
from datetime import datetime
import isodate
import logging
import argparse

# Constants for storage management
SSD_PATH = Path("/mnt/ssd/datasets")
LAION_PATH = SSD_PATH / "relaion2B-en-research-safe"
WEBVID_PATH = SSD_PATH / "webvid-10M/data/train/partitions"
PROCESSED_PATH = SSD_PATH / "processed_data"
TARGET_FREE_SPACE = 500 * 1024 * 1024 * 1024  # Keep 500GB free as buffer

def get_available_space(path):
    """Get available space in bytes at given path"""
    stats = shutil.disk_usage(path)
    return stats.free

def ensure_free_space(path, required_space, is_video=False):
    """Ensure there's enough free space by removing oldest files if necessary"""
    if get_available_space(path) < required_space:
        # If we need space for video, first try removing images since videos are priority
        if is_video:
            images = sorted(
                (PROCESSED_PATH / "images").glob('**/*.jpg'),
                key=lambda x: x.stat().st_mtime
            )
            for img in images:
                img.unlink()
                if get_available_space(path) >= required_space:
                    return
        
        # If still need space, remove oldest files of the respective type
        files = sorted(
            (PROCESSED_PATH / ("videos" if is_video else "images")).glob('**/*.*'),
            key=lambda x: x.stat().st_mtime
        )
        for file in files:
            file.unlink()
            if get_available_space(path) >= required_space:
                break

def validate_video(video_path):
    """Validate video file and check its properties"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logging.warning(f"Failed to open video: {video_path}")
            return False
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Adjusted validation criteria for WebVid dataset
        valid = (
            width >= 336 and  # Most videos are 596x336
            height >= 316 and # Some are 600x316
            frame_count >= 16 and 
            fps >= 10
        )
        
        if not valid:
            logging.warning(f"Video failed validation: {video_path}")
            logging.warning(f"Specs: {width}x{height} @ {fps}fps, {frame_count} frames")
        
        cap.release()
        return valid
        
    except Exception as e:
        logging.error(f"Error validating video {video_path}: {e}")
        return False

def download_and_save_video(url, save_dir, video_id=None):
    """Download video and save to disk, returning path if successful"""
    try:
        ensure_free_space(save_dir, 100 * 1024 * 1024, is_video=True)
        
        filename = f"{video_id}.mp4" if video_id else f"{hashlib.md5(url.encode()).hexdigest()}.mp4"
        save_path = save_dir / filename
        
        if save_path.exists() and validate_video(save_path):
            return save_path

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            response = requests.get(url, stream=True, timeout=30)
            if response.status_code != 200:
                return None
                
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
            
            temp_file.flush()
            
            if validate_video(temp_file.name):
                shutil.move(temp_file.name, save_path)
                return save_path
            else:
                os.unlink(temp_file.name)
                return None
            
    except Exception as e:
        if 'temp_file' in locals():
            try:
                os.unlink(temp_file.name)
            except:
                pass
        return None

def download_and_save_image(url, save_dir):
    """Download image and save to disk, returning path if successful"""
    try:
        ensure_free_space(save_dir, 1024 * 1024, is_video=False)
        
        response = requests.get(url, timeout=10, stream=True)
        if response.status_code != 200:
            return None
        
        filename = hashlib.md5(url.encode()).hexdigest() + '.jpg'
        save_path = save_dir / filename
        
        if save_path.exists():
            return save_path

        content = io.BytesIO()
        content_length = 0
        for chunk in response.iter_content(chunk_size=8192):
            content_length += len(chunk)
            if content_length > 10 * 1024 * 1024:  # Skip images > 10MB
                return None
            content.write(chunk)
        
        content.seek(0)
        
        img = Image.open(content)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        if img.width < 384 or img.height < 384:
            return None
            
        img.save(save_path, 'JPEG', quality=90)
        return save_path
        
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

def setup_logging(processed_path):
    """Setup logging to file"""
    log_file = processed_path / f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        filename=str(log_file),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def process_webvid_csv(csv_path, output_dir, video_dir, processed_files_log):
    """Process a single WebVid CSV file"""
    processed_files = set()
    if Path(processed_files_log).exists():
        with open(processed_files_log) as f:
            processed_files = set(line.strip() for line in f)
    
    output_jsonl = output_dir / f"{csv_path.stem}_processed.jsonl"
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                futures = {
                    executor.submit(
                        download_and_save_video,
                        row['contentUrl'],
                        video_dir,
                        row['videoid']
                    ): (row['videoid'], row['contentUrl'], row['name'])
                    for row in rows
                }
                
                try:
                    for future in tqdm(concurrent.futures.as_completed(futures),
                                     total=len(futures),
                                     desc=f"Processing {csv_path.stem}"):
                        video_id, url, caption = futures[future]
                        try:
                            video_path = future.result()
                            if video_path:
                                entry = {
                                    "video": str(video_path.absolute()),
                                    "text": caption
                                }
                                with jsonlines.open(output_jsonl, mode='a') as writer:
                                    writer.write(entry)
                                logging.info(f"Successfully processed video {video_id}: {url} -> {video_path.absolute()}")
                            else:
                                logging.warning(f"Failed to process video {video_id}: {url} - File validation failed")
                        except Exception as e:
                            logging.error(f"Error processing video {video_id} from {url}: {str(e)}")
                            
                except KeyboardInterrupt:
                    raise
        
        main_jsonl = PROCESSED_PATH / "video_text.jsonl"
        if output_jsonl.exists():
            with jsonlines.open(output_jsonl) as reader:
                with jsonlines.open(main_jsonl, mode='a') as writer:
                    writer.write_all(reader)
        
        with open(processed_files_log, 'a') as f:
            f.write(f"{csv_path.stem}\n")
            
    except Exception as e:
        logging.error(f"Error processing CSV file {csv_path}: {str(e)}")

def process_laion_parquet(parquet_path, output_dir, image_dir, processed_files_log):
    """Process a single LAION parquet file"""
    processed_files = set()
    if Path(processed_files_log).exists():
        with open(processed_files_log) as f:
            processed_files = set(line.strip() for line in f)
    
    output_jsonl = output_dir / f"{parquet_path.stem}_processed.jsonl"
    
    try:
        table = pd.read_parquet(parquet_path)
        chunk_size = 5000
        for i in range(0, len(table), chunk_size):
            chunk = table.iloc[i:i + chunk_size]
            
            chunk = chunk[
                (chunk['width'] >= 384) & 
                (chunk['height'] >= 384) &
                (chunk['punsafe'] < 0.1) &
                (chunk['pwatermark'] < 0.5)
            ]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                futures = {
                    executor.submit(
                        download_and_save_image,
                        row['url'],
                        image_dir
                    ): (row['url'], row['caption'])
                    for _, row in chunk.iterrows()
                }
                
                for future in concurrent.futures.as_completed(futures):
                    url, caption = futures[future]
                    try:
                        image_path = future.result()
                        if image_path:
                            entry = {
                                "image": str(image_path.absolute()),
                                "text": caption
                            }
                            with jsonlines.open(output_jsonl, mode='a') as writer:
                                writer.write(entry)
                            logging.info(f"Successfully processed image: {url} -> {image_path.absolute()}")
                        else:
                            logging.warning(f"Failed to process image: {url} - File validation failed")
                    except Exception as e:
                        logging.error(f"Error processing image {url}: {str(e)}")
        
        main_jsonl = PROCESSED_PATH / "image_text.jsonl"
        if output_jsonl.exists():
            with jsonlines.open(output_jsonl) as reader:
                with jsonlines.open(main_jsonl, mode='a') as writer:
                    writer.write_all(reader)
        
        with open(processed_files_log, 'a') as f:
            f.write(f"{parquet_path.stem}\n")
            
    except Exception as e:
        logging.error(f"Error processing parquet file {parquet_path}: {str(e)}")

def main():
    # Setup command line arguments
    parser = argparse.ArgumentParser(description='Prepare training data from WebVid and LAION datasets')
    parser.add_argument('--num_video_files', type=int, default=50,
                      help='Number of WebVid CSV files to process (default: 50)')
    parser.add_argument('--num_image_files', type=int, default=50,
                      help='Number of LAION parquet files to process (default: 50)')
    args = parser.parse_args()
    
    PROCESSED_PATH.mkdir(exist_ok=True)
    setup_logging(PROCESSED_PATH)
    
    video_dir = PROCESSED_PATH / "videos"
    image_dir = PROCESSED_PATH / "images"
    video_jsonl_dir = PROCESSED_PATH / "video_jsonl"
    image_jsonl_dir = PROCESSED_PATH / "image_jsonl"
    
    for dir in [video_dir, image_dir, video_jsonl_dir, image_jsonl_dir]:
        dir.mkdir(exist_ok=True, parents=True)
    
    # video_log = PROCESSED_PATH / "processed_video_files.txt"
    image_log = PROCESSED_PATH / "processed_image_files.txt"
    
    # Clear the processed files logs to reprocess all files
    # if video_log.exists():
    #     video_log.unlink()
    if image_log.exists():
        image_log.unlink()
    
    # Create or clear the main JSONL files
    # main_video_jsonl = PROCESSED_PATH / "video_text.jsonl"
    main_image_jsonl = PROCESSED_PATH / "image_text.jsonl"
    # if main_video_jsonl.exists():
    #     main_video_jsonl.unlink()
    if main_image_jsonl.exists():
        main_image_jsonl.unlink()
    
    # Process limited number of WebVid CSV files
    # csv_files = sorted(WEBVID_PATH.glob("*.csv"))[:args.num_video_files]
    # logging.info(f"Processing {len(csv_files)} WebVid files")
    
    # for csv_file in tqdm(csv_files, desc="Processing WebVid files"):
    #     if get_available_space(SSD_PATH) < TARGET_FREE_SPACE:
    #         logging.warning("Storage space low, stopping video processing")
    #         break
    #     process_webvid_csv(csv_file, video_jsonl_dir, video_dir, video_log)
    
    # Process limited number of LAION parquet files
    parquet_files = sorted(LAION_PATH.glob("*.snappy.parquet"))[:args.num_image_files]
    logging.info(f"Processing {len(parquet_files)} LAION files")
    
    for parquet_file in tqdm(parquet_files, desc="Processing LAION files"):
        if get_available_space(SSD_PATH) < TARGET_FREE_SPACE:
            logging.warning("Storage space low, stopping image processing")
            break
        process_laion_parquet(parquet_file, image_jsonl_dir, image_dir, image_log)

if __name__ == "__main__":
    main() 