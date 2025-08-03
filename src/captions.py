import sys
import os
import json

from utils.parse import *
from utils.logger import setup_logger

logger = setup_logger(__name__, log_level="INFO", log_to_file=True, log_to_console=True)
PROCESSED_PATH = "processed.json"
MEMORY_PATH = "memory.json"

def load_json_data(filename: str, default_value):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"Loaded {filename} with {len(data) if isinstance(data, (dict, list, set)) else 'data'}")
            
            if isinstance(default_value, set) and isinstance(data, list):
                return set(data)
            
            return data
    else:
        logger.info(f"File {filename} not found, using default value")
        return default_value

def save_json_data(data, filename: str):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved data to {filename}")

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