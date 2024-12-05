import os
import sys 
import sys
import os

# Add the directory to sys.path
sys.path.append(r"S:\ML_Projects\DaimondPricePrediction\src")

# Now import the logger module
from logger import logging
from exception import CustomException
import pandas as pd

from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation
from components.model_trainer import ModelTrainer


if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initate_data_ingestion()
    print(train_data_path)
    print(test_data_path) # src\piprlines\training_pipeline.py
    data_transformation = DataTransformation()
    train_arr,test_arr,preprocessor_obj = data_transformation.initate_data_transformation(train_data_path,test_data_path)

    model_trainer = ModelTrainer()
    model_trainer.initate_model_training(train_arr,test_arr)