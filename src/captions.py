import sys
import os
import json

from utils.parse import *
from utils.logger import setup_logger

logger = setup_logger(__name__, log_level="INFO", log_to_file=True, log_to_console=True)

def main():
    file_paths = ["data/mask_UCFCrime_Val.txt", "data/mask_UCFCrime_Train.txt", "data/mask_UCFCrime_Test.txt"]

    processed = load_json_data(PROCESSED_PATH, set())
    memory = load_json_data(MEMORY_PATH, {})
    
    logger.info(f"Loaded {len(processed)} processed videos and {len(memory)} contexts")
    
    try:
        for file_path in file_paths:
            metadatas = parse_meta_data(file_path)        
            logger.info(f"{file_path} Total videos found: {len(metadatas)}")

            for video_name, metadata in metadatas.items():
                if video_name in processed:
                    continue
                
                logger.info(f"Processing video {video_name}:")
                response = llm_output("\n".join(metadata))
                logger.info(f"{response}\n-------------")

                context, normal, anomaly = extract_tags(response)
                if not context:
                    raise ValueError("Context could not be extracted from LLM response.")
                else:
                    memory[context] = [normal, anomaly]

                    processed.add(video_name)
                
                save_json_data(list(processed), PROCESSED_PATH)
                save_json_data(memory, MEMORY_PATH)
                
                logger.info(f"Saved progress after processing {video_name}")

    except Exception as e:
        logger.warning(f"Pipeline error: {e}")
    finally:
        logger.info(f"Pipeline completed: {len(processed)} processed videos, {len(memory)} contexts")
        

if __name__ == "__main__":
    main() 