import os
import sys
import pdb
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'rawData.csv')

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion component')
        try:
            data = pd.read_csv('notebook\data\student-por.csv')
            logging.info('Read the dataset as dataframe')
            #pdb.set_trace()

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info('Train test initiated')
            train_set, test_set = train_test_split(data, test_size=0.3, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path)
            test_set.to_csv(self.ingestion_config.test_data_path)
            logging.info('Data Ingestion Completed')


            return(
                self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)
        

