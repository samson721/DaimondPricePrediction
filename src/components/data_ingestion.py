import os 
import sys

# Add the directory to sys.path
sys.path.append(r"S:\ML_Projects\DaimondPricePrediction\src")
from logger import logging
from exception import CustomException

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts","train.csv")
    test_data_path:str = os.path.join("artifacts","test.csv")
    raw_data_path:str = os.path.join("artifacts","raw.csv")


# Create the Data Ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initate_data_ingestion(self):
        logging.info("Data Ingestion starts")

        try:
            df = pd.read_csv(os.path.join("Notebooks/data","gemstone.csv"))
            logging.info("Data read as pandas Dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,  index=False)

            logging.info("Train test split")
            train_set,test_set = train_test_split(df,test_size=0.30,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of Data Completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        
        except Exception as e:
            logging.info("error occured while data ingestion")