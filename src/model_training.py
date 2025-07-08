#IMPORTING LIBRARIES
import os
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from config.model_params import *
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
import joblib
from utils.common_functions import read_yaml,load_data
from scipy.stats import randint
import mlflow
import mlflow.sklearn


logger = get_logger(__name__)


class ModelTraining:
    def __init__(self,train_path,test_path,model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        self.params = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

    def load_and_split_data(self):
        try:
            logger.info(f"Loading the  train data from {self.train_path} and performing splits")
            train = load_data(self.train_path)

            logger.info(f"Loading the  train data from {self.test_path} and performing splits")
            test = load_data(self.test_path)

            x_train = train.drop(columns=['booking_status'],axis = 1)
            y_train = train['booking_status']
            x_test = test.drop(columns=['booking_status'],axis = 1)
            y_test = test['booking_status']
            logger.info("Successfully loaded the data and performed splitting")
            return x_train,y_train,x_test,y_test
        
        except Exception as e:
            logger.error(f"couldn't load the data amd perform splitting {e}")
            raise CustomException("Failed to load and split data",e)
        
    def model_training(self,x_train,y_train):
        try:
            logger.info("Initializing the model")
            lg = LGBMClassifier(random_state=self.random_search_params['random_state'])

            logger.info("Starting model training")

            random_search = RandomizedSearchCV(estimator=lg,
                                               param_distributions=self.params,
                                               n_iter=self.random_search_params['n_iter'],
                                               cv=self.random_search_params['cv'],
                                               verbose=self.random_search_params['verbose'],
                                               n_jobs=self.random_search_params['n_jobs'],
                                               random_state=self.random_search_params['random_state'],
                                               scoring=self.random_search_params['scoring']
                                               )
            
            logger.info("Starting Random search")
            random_search.fit(x_train,y_train)
            best_params = random_search.best_params_

            logger.info(f"The best parameters are {best_params}")

            best_model = random_search.best_estimator_

            return best_model
        
        except Exception as e:
            logger.error(f"Couldn't perform model training {e}")
            raise CustomException("Failed to perform model_training",e)
        

    def model_testing(self,model,x_test,y_test):
        try:
            logger.info("Starting model testing")
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test,y_pred)
            precision = precision_score(y_test,y_pred)
            recall = recall_score(y_test,y_pred)
            f1 = f1_score(y_test,y_pred)
            roc = roc_auc_score(y_test,y_pred)

            logger.info(f"Accuracy for LGBM model is {accuracy}")
            logger.info(f"Precision for LGBM model is {precision}")
            logger.info(f"Recall for LGBM model is {recall}")
            logger.info(f"F1 score for LGBM model is {f1}")
            logger.info(f"ROC AUC score for LGBM model is {roc}")


            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc
                }

        except Exception as e:
            logger.error(f"Couldn't perform model testing {e}")
            raise CustomException("Failed to perform model testing",e)
        

    def save_model(self,model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path),exist_ok=True)
            logger.info("saving the model")

            joblib.dump(model,self.model_output_path)
            logger.info(f"The model saved to {self.model_output_path}")

        except Exception as e:
            logger.error(f"Couldn't save the model {e}")
            raise CustomException("Failed to save the model",e)
        

    
    def run_model(self):
        try:
            with mlflow.start_run():
                logger.info("Starting model training pipeline")

                logger.info("Starting mlflow experimentation")

                logger.info("Logging the training and testing dataset to MLFLOW")
                mlflow.log_artifact(self.train_path,artifact_path="datasets")
                mlflow.log_artifact(self.test_path,artifact_path="datasets")

                x_train,y_train,x_test,y_test = self.load_and_split_data()
                best_model = self.model_training(x_train,y_train)
                metrics = self.model_testing(best_model,x_test,y_test)
                self.save_model(best_model)

                logger.info("Logging the model into MLFLOW")    
                mlflow.log_artifact(self.model_output_path)

                logger.info("Logging the parameters into MLFLOW")    
                mlflow.log_params(best_model.get_params())

                logger.info("Logging metrics into MLFLOW")
                mlflow.log_metrics(metrics)



                logger.info("Successfully performed model training and testing")

        except Exception as e:
            logger.error(f"Couldn't run the model training pipeline {e}")
            raise CustomException("Failed to run the model training pipeline",e)


if __name__ == "__main__":
    training = ModelTraining(PROCESSED_TRAIN_DATA_PATH,PROCESSED_TEST_DATA_PATH,MODEL_OUTPUT_PATH)
    training.run_model()

    


