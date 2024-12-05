from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import pandas as pd
import numpy as np

import os
import sys

from utils import save_object

# Add the directory to sys.path
sys.path.append(r"S:\ML_Projects\DaimondPricePrediction\src")

from exception import CustomException
from logger import logging

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformed_object(self):
        try:
            logging.info("Data Transformation started")

            # categorical columns and numerical columns 
            categorical_col = ["cut",'color','clarity']
            numerical_Col = ['carat','depth','table','x','y','z']

            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info("Data Transformation Pipeline Initated")

            num_pipeline = Pipeline(
                steps=(
                    ("imputer",SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                )
            )

            cat_pipeline = Pipeline(
                steps=(
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("ordinalencoder",OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ("scaler",StandardScaler())
                )
            )

            preprocessor = ColumnTransformer([
                ("num_pipeline",num_pipeline,numerical_Col),
                ("cat_pipeline",cat_pipeline,categorical_col)
            ])

            logging.info("Pipeline completed")

            return preprocessor


        except Exception as e:
            logging.info("Error Occured while Creating preprocessor pipeline")
            raise CustomException(e,sys)


    def initate_data_transformation(self,train_data_path,test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Reading Train and test data completed")
            logging.info(f"Train DataFrame Head : \n{train_df.head().to_string()}")
            logging.info(f"Test DataFrame Head : \n{test_df.head().to_string()}")

            logging.info("Obtaining Preprocessor Object")

            preprocessor_obj = self.get_data_transformed_object()

            target_column = "price"
            drop_column = [target_column,"id"]

            #seprating Dependent Feature and Independent Features
            input_feature_train_df = train_df.drop(drop_column,axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(drop_column,axis=1)
            target_feature_test_df = test_df[target_column]

            # Apply Transformation
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            # concat the dependent and Independent feature in one arr
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info('Processsor pickle in created and saved')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Error Ocuured While Initating Data Transformation")
            raise CustomException(e,sys)