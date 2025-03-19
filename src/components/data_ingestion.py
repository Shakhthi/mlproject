import os
import sys

from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts", "train.csv")
    test_data_path:str = os.path.join("artifacts", "test.csv")
    raw_data_path:str = os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
    
    def intiate_data_ingestion(self):
        logging.info("Entered into data ingestion component.")

        try:
            data = pd.read_csv("notebook\data\stud.csv")
            logging.info("Read the data as Dataframe")

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)

            data.to_csv(self.data_ingestion_config.train_data_path, header=True, index=False)

            logging.info("train_test_split separation initiated.")
            train_data, test_data = train_test_split(data, train_size=0.8, random_state=42)

            train_data.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
            test_data.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)
            logging.info("Data ingestion is Completed.")

            return (self.data_ingestion_config.train_data_path,
                    self.data_ingestion_config.test_data_path)
        
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.intiate_data_ingestion()