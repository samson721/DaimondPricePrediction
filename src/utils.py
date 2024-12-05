import pandas as pd
import numpy as np
import pickle
import os
import sys

# Add the directory to sys.path
sys.path.append(r"S:\ML_Projects\DaimondPricePrediction\src")

from logger import logging
from exception import CustomException

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)

        pass
    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        logging.info("Model Evaluation Initaited")
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)

            

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report
    
    except Exception as e:
        logging.info("Error Occured While Evaluating Model")
        raise CustomException(e,sys)
    


def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)