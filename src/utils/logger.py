import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
import sys

def setup_logger(name: str = None, log_level: str = "INFO", log_to_file: bool = True, log_to_console: bool = True):
    if name is None:
        name = __name__
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{name.replace('.', '_')}_{timestamp}.log"
        log_filepath = os.path.join(logs_dir, log_filename)    
        file_handler = RotatingFileHandler(
            log_filepath, 
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        general_log_filepath = os.path.join(logs_dir, 'app.log')
        general_file_handler = RotatingFileHandler(
            general_log_filepath,
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        general_file_handler.setLevel(getattr(logging, log_level.upper()))
        general_file_handler.setFormatter(formatter)
        logger.addHandler(general_file_handler)
    
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def get_logger(name: str = None):    
    if name is None:
        name = __name__
    
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger = setup_logger(name)
    
    return logger 