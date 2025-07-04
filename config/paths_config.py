import os


################################ DATA INGESTION ####################################

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

RAW_DIR = os.path.join(BASE_DIR, "artifact", "raw")

RAW_FILE_PATH = os.path.join(RAW_DIR,'raw.csv')
TRAIN_FILE_PATH = os.path.join(RAW_DIR,'train.csv')
TEST_FILE_PATH = os.path.join(RAW_DIR,'test.csv')


CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

############################# DATA PROCESSING ######################################


PROCESSED_DIR = os.path.join(BASE_DIR, "artifact", "processed")
PROCESSED_TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR,"processed_train.csv")
PROCESSED_TEST_DATA_PATH = os.path.join(PROCESSED_DIR,"processed_test.csv")


############################ MODEL PATH #############################################

MODEL_OUTPUT_DIR = os.path.join(BASE_DIR,"artifact","models")
MODEL_OUTPUT_PATH = os.path.join(MODEL_OUTPUT_DIR,"lgbm_model.pkl")

