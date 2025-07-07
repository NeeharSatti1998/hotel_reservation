import pandas as pd
import os
from src.logger import get_logger
import numpy as np
from config.paths_config import *
from src.custom_exception import CustomException
from utils.common_functions import read_yaml,load_data
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier


logger = get_logger(__name__)


class DataProcessor:
    def __init__(self,train_path,test_path,config_path,processed_dir):

        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir

        self.config = read_yaml(config_path)

        if not os.path.exists(os.path.join(self.processed_dir)):
            os.makedirs(self.processed_dir)

    
    def processed_data(self,df):
        try:
            logger.info("Starting Data Processing")

            logger.info("Dropping unnecessary columns")
            df.drop(columns=['Booking_ID','Unnamed: 0'],axis=1,inplace=True)

            logger.info("Converting the numeric into categorical columns where deemed necessarey")
            df['required_car_parking_space'] = df['required_car_parking_space'].astype('object')
            df['repeated_guest'] = df['repeated_guest'].astype('object')
            df['arrival_year'] = df['arrival_year'].astype('object')
            df['arrival_month'] = df['arrival_month'].astype('object')
            
            arrivate_date_cat = []
            for i in df['arrival_date']:
                if i <=12:
                    arrivate_date_cat.append('start_month')
                elif i>12 and i<21:
                    arrivate_date_cat.append('mid_month')
                else:
                    arrivate_date_cat.append('end_month')

            df['arrival_date'] = arrivate_date_cat

            cat_cols = self.config["data_processing"] ["categorical_columns"]
            num_cols = self.config["data_processing"] ["numerical_columns"]

            logger.info("Handling skew")
            skewness = df[num_cols].skew()
            for i in num_cols:
                if skewness[i] > 1 or skewness[i] <-1:
                    df[i] = np.log1p(df[i])

            logger.info("Encoding Categorical values")
            room_map = {'Room_Type 1':1,'Room_Type 2':2,'Room_Type 3':3,'Room_Type 4':4,'Room_Type 5':5,'Room_Type 6':6,'Room_Type 7':7}
            df['room_type_reserved'] = df['room_type_reserved'].map(room_map)
            
            logger.info("group rare categories into others")
            df['market_segment_type'] = df['market_segment_type'].replace(['Corporate','Aviation','Complimentary'],'other')

            df=pd.get_dummies(df,columns=['type_of_meal_plan','required_car_parking_space','market_segment_type','repeated_guest','arrival_year','arrival_month','arrival_date'],drop_first=True)
            df['booking_status'] = df['booking_status'].map({'Not_Canceled':0,'Canceled' : 1})

            logger.info("Converting boolean to integer")

            for col in df.columns:
                if df[col].dtypes == 'bool':
                    df[col]=df[col].astype('int')

            logger.info("Converted all categorical columns into numerical")
            logger.info(f"The dataframe :{print(df.head())}")

            return df
        except Exception as e:
            logger.error(f"Coudn't Preprocess the data {e}")
            raise CustomException("Failed to Preprocess data",e)
        
    
    def balanced_data(self,df):

        try:
            logger.info("Starting data balancing process")
            logger.info(f"Before data balancing: {print(df['booking_status'].value_counts())}")
            x = df.drop(columns=['booking_status'], axis=1)
            y = df['booking_status']

            smote = SMOTE(random_state=42)
            x_res, y_res = smote.fit_resample(x,y)

            df_balanced = pd.DataFrame(x_res,columns=x_res.columns)
            df_balanced['booking_status'] = y_res
            df = df_balanced.copy()
            logger.info("Succesfully completed data balancing")
            logger.info(f"After data balancing: {print(df['booking_status'].value_counts())}")

            return df
        except Exception as e:
            logger.error(f"Couldn't balance the data {e}")
            raise CustomException("Failed to balance data",e)
        

    def feauture_selection(self,df):

        try:
            logger.info("Starting feature selection")

            rf = RandomForestClassifier()
            x = df.drop(columns=['booking_status'], axis=1)
            y = df['booking_status']

            rf.fit(x,y)
            feature_importance = rf.feature_importances_
            f_df = pd.DataFrame({'features' : x.columns,'importance' : feature_importance})
            logger.info(f"The feauture importance is as follows {print(f_df.sort_values(by='importance',ascending=False))}")
            importance = self.config["data_processing"]["importance"]
            f_df = f_df[f_df['importance']>importance]
            imp_features = f_df['features'].to_list()
            df = df[imp_features + ['booking_status']]

            return df
        
        except Exception as e:
            logger.error(f"Couldn't select the important features {e}")
            raise CustomException("Failed to select important features",e)
        

    
    def save_data(self,df,path):

        try:
            logger.info("Saving the data in processed folder")
            df.to_csv(path,index = False)

            logger.info(f"Saved successfullt to {path}")

        except Exception as e:
            logger.error(f"Could not save the data {e} ")
            raise CustomException ("Failed to save data",e)
        

    def process(self):

        try:
            logger.info("loading Raw csv file")

            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            train_df = self.processed_data(train_df)
            test_df = self.processed_data(test_df)

            train_df = self.balanced_data(train_df)

            train_df = self.feauture_selection(train_df)

            selected_features = [col for col in train_df.columns if col != 'booking_status']
            test_df = test_df[selected_features+['booking_status']]


            self.save_data(train_df,PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df,PROCESSED_TEST_DATA_PATH)

            logger.info("Data pre-processing completed successfully")

        except Exception as e:
            logger.error(f"Error during preprocessing pipeline {e}")
            raise CustomException(f"Error while data pre-processing pipeline{e}")
        


if __name__ == '__main__':
    processor = DataProcessor(TRAIN_FILE_PATH,TEST_FILE_PATH,CONFIG_PATH,PROCESSED_DIR)
    processor.process()
