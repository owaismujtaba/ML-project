import sys
import os
from dataclasses import dataclass
import pdb
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, format_labels

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformer_object(self):
        '''
        Transformation of data
        '''
        try:
            numerical_columns = ['age', 
                                 'Medu', 
                                 'Fedu',
                                 'traveltime', 
                                 'studytime', 
                                 'failures',
                                 'famrel',
                                 'freetime',
                                 'goout',
                                 'Dalc',
                                 'Walc',
                                 'health',
                                 'absences'
            ]

            categorical_columns = [
                'sex',
                'address',
                'famsize',
                'Pstatus',
                'Mjob',
                'Fjob',
                'reason',
                'guardian',
                'schoolsup',
                'famsup',
                'paid',
                'activities',
                'nursery',
                'higher',
                'internet',
                'romantic',
            ]   

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            logging.info('Numercial columns scaling completed')
            categorical_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info('Categorical columns encoding completed')
            #pdb.set_trace()
            preprocessor = ColumnTransformer(
                [
                    ('numerical_pipeline', numerical_pipeline, numerical_columns),
                    ('categorical_pipeline', categorical_pipeline, categorical_columns)
                ]
            )
            logging.info('get data transformation completed')
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path, sep=';')
            test_data = pd.read_csv(test_path, sep=';')
            logging.info('Read train and test completed')

            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = 'G3'
            
            input_feature_train_data = train_data.drop(columns=[target_column_name], axis=1)
            target_feature_train_data = train_data[target_column_name]
            target_feature_train_data = format_labels(target_feature_train_data)
            input_feature_test_data = test_data.drop(columns=[target_column_name], axis=1)
            target_feature_test_data = test_data[target_column_name]
            target_feature_test_data = format_labels(target_feature_test_data)


            logging.info('Appling preprocessing steps to train and test data')
            #pdb.set_trace()
            input_feature_train_data = preprocessing_obj.fit_transform(input_feature_train_data)
            input_feature_test_data = preprocessing_obj.transform(input_feature_test_data)

            train_arr = np.c_[input_feature_train_data, np.array(target_feature_train_data)]
            test_arr = np.c_[input_feature_test_data, np.array(target_feature_test_data)]

            logging.info('Saving transformation object')
            save_object(
                file_path = self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)