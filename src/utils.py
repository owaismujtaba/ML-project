import os
import sys
import dill
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(x_train, y_train, x_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            y_train_predictions = model.predict(x_train)
            y_test_predictions = model.predict(x_test)

            train_model_score = r2_score(y_train, y_train_predictions)
            test_model_score = r2_score(y_test, y_test_predictions)

            report[list(models.keys())[i]] = test_model_score
        print(test_model_score)
        return report
    except Exception as e:
        raise CustomException(e, sys)
    

def format_labels(labels):
    new_labels = []
    for label in labels:
        new_labels.append(float(label[0]))

    return new_labels