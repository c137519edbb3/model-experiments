from groq import Groq
from .prompts import SYSTEM_PROMPT
from dotenv import load_dotenv
from .logger import setup_logger
import os

load_dotenv()
model = os.getenv("GROQ_MODEL")
api_key = os.getenv("GROQ_API_KEY")

logger = setup_logger(__name__, log_level="INFO", log_to_file=True, log_to_console=False)

def llm_output(metadata: str):
    logger.info("Starting LLM processing")
    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": metadata
                }
            ],
            temperature=0,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
            seed=4594,
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in LLM processing: {e}")
        raise


def parse_meta_data(file_path: str) -> dict:
    logger.info(f"Starting to parse metadata from file: {file_path}")
    video_segments = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split(' ')
                if len(parts) < 4:
                    logger.warning(f"Line {line_num}: Invalid format, skipping")
                    continue
                
                video_name = parts[0]
                start_time = parts[1]
                end_time = parts[2]
                description = ' '.join(parts[3:])
                if description.startswith('##'):
                    description = description[2:]
                
                if video_name not in video_segments:
                    video_segments[video_name] = []
                
                formatted_segment = f"{start_time} - {end_time} :- {description}"
                video_segments[video_name].append(formatted_segment)

        logger.info(f"Successfully parsed {len(video_segments)} videos from {file_path}")
        sorted(video_segments)
        return video_segments

    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return {}


def parse_video_list(file_path: str) -> list:
    video_names = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue

                video_name = os.path.basename(line)
                video_name = os.path.splitext(video_name)[0]
                
                video_names.append(video_name)
        return video_names

    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return []
    
def extract_tags(response: str) -> tuple[str, list, list]:    
    try:
        context = ""
        if "<CONTEXT>" in response and "</CONTEXT>" in response:
            context_start = response.find("<CONTEXT>") + len("<CONTEXT>")
            context_end = response.find("</CONTEXT>")
            context = response[context_start:context_end].strip()
            logger.debug(f"Extracted context: {context}")
        
        normal_events = []
        if "<NORMAL>" in response and "</NORMAL>" in response:
            normal_start = response.find("<NORMAL>") + len("<NORMAL>")
            normal_end = response.find("</NORMAL>")
            normal_content = response[normal_start:normal_end].strip()
            
            try:
                normal_content = normal_content.strip("[]")
                normal_events = [event.strip().strip('"').strip("'") for event in normal_content.split(",")]
                normal_events = [event for event in normal_events if event]
                logger.debug(f"Extracted {len(normal_events)} normal events")
            except Exception as e:
                logger.warning(f"Error parsing normal events: {e}")
                normal_events = []
        
        anomaly_events = []
        if "<ANOMALY>" in response and "</ANOMALY>" in response:
            anomaly_start = response.find("<ANOMALY>") + len("<ANOMALY>")
            anomaly_end = response.find("</ANOMALY>")
            anomaly_content = response[anomaly_start:anomaly_end].strip()
            
            try:
                anomaly_content = anomaly_content.strip("[]")
                anomaly_events = [event.strip().strip('"').strip("'") for event in anomaly_content.split(",")]
                anomaly_events = [event for event in anomaly_events if event]
                logger.debug(f"Extracted {len(anomaly_events)} anomaly events")
            except Exception as e:
                logger.warning(f"Error parsing anomaly events: {e}")
                anomaly_events = []
                
        return context, normal_events, anomaly_events
        
    except Exception as e:
        logger.error(f"Error extracting tags from response: {e}")
        return "", [], []

