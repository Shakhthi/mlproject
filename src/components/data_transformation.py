import os
import sys
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import split_features, save_object

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation():
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_transformer_object(self):
        try:
            num_features, cat_features = split_features()
            num_features.remove("math_score")

            logging.info(f"num_features:{num_features}")
            logging.info(f"categorical features:{cat_features}")
            logging.info("numerical & categorical features splitted.")

            num_pipeline = Pipeline(
                                    steps =[("imputer", SimpleImputer(strategy="median")),
                                            ("scaler", StandardScaler())
                                            ]
            )
            logging.info("numerical feature pipeline created")

            cat_pipeline = Pipeline(
                                    steps = [
                                        ("imputer", SimpleImputer(strategy="most_frequent")),
                                        ("OHencoder", OneHotEncoder()),
                                        ("scaler", StandardScaler(with_mean=False))
                                    ]
            )
            logging.info("categorical feature pipeline created.")

            preprocessor = ColumnTransformer(
                                            [
                                                ("num_pipe", num_pipeline, num_features),
                                                ("cat_pipe", cat_pipeline, cat_features)
                                             ]
            )
            logging.info("preprocessor object created.")

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformer(self, train_path, test_path):
        try:
            logging.info("Data transformer initiated.")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train & test data Completed.")

            preprocesor_obj = self.get_transformer_object()
            logging.info("preprocessor object obtained.")

            target_col = "math_score"
            input_train_df = train_df.drop(columns = [target_col], axis=1)
            target_train_df = train_df[target_col]

            input_test_df = test_df.drop(columns = [target_col], axis=1)
            target_test_df = test_df[target_col]

            logging.info("Applying preprocessor object to the training and testing data.")
            train_df_arr = preprocesor_obj.fit_transform(input_train_df)

            test_df_arr = preprocesor_obj.transform(input_test_df)

            train_arr = np.c_[train_df_arr, np.array(target_train_df)]
            test_arr = np.c_[test_df_arr, np.array(target_test_df)]

            logging.info("Saved preprocessor Object.")

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocesor_obj)

            logging.info("Data Transformation Completed successfully.")

            return (
                    train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e, sys)