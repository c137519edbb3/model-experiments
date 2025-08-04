import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.parse import parse_video_list
from utils.dataset import download_ucf_dataset
from utils.logger import setup_logger

logger = setup_logger(__name__, log_level="INFO", log_to_file=True, log_to_console=True)

import argparse

def main():
    parser = argparse.ArgumentParser(description="Download UCF dataset videos.")
    parser.add_argument(
        "--download-path",
        type=str,
        default="downloads",
    )
    args = parser.parse_args()

    anomaly_test_file = "data/Anomaly_Test.txt"
    
    logger.info(f"Parsing video list from {anomaly_test_file}")
    video_names = parse_video_list(anomaly_test_file)
    
    logger.info(f"Found {len(video_names)} videos in Anomaly_Test.txt")
    logger.info(video_names)
    
    download_dir = args.download_dir
    
    logger.info(f"Starting download to {download_dir}")
    try:
        malform = download_ucf_dataset(download_dir, video_names)
        while malform:
            print(f"\ngot {len(malform)} files, reiterating...")
            malform = download_ucf_dataset(malform)

        logger.info("Download completed successfully!")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise

if __name__ == "__main__":
    main()