
from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessor
from src.model_training import ModelTraining
from utils.common_functions import read_yaml,load_data
from config.paths_config import *


if __name__ == "__main__":
    ######## DATA INGESTION #############
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()

    ######## DATA PREPROCESSING #########
    processor = DataProcessor(TRAIN_FILE_PATH,TEST_FILE_PATH,CONFIG_PATH,PROCESSED_DIR)
    processor.process()

    ######## MODEL TRAINING #############
    training = ModelTraining(PROCESSED_TRAIN_DATA_PATH,PROCESSED_TEST_DATA_PATH,MODEL_OUTPUT_PATH)
    training.run_model()
