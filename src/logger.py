import logging
import os
from datetime import datetime

PATH_DIR = 'logs'    # Create a Directory
os.makedirs(PATH_DIR,exist_ok=True)

# Create a log file
LOG_FILE = os.path.join(PATH_DIR,f"log{datetime.now().strftime('%Y-%m-%d')}.log")

#Configure the logger
logging.basicConfig(
    filename=LOG_FILE,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    level= logging.INFO # only Logs with level INFO or higher will be recorded
)

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger

