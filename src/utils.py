import os
import sys

import dill
import pandas as pd

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(name=dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)
    

def split_features():
    df = pd.read_csv("notebook/data/stud.csv")

    num_cols, cat_cols = [], []
    for col in df.columns:
        if df[col].dtype != "O":
            num_cols.append(col)
        else:
            cat_cols.append(col)
    return (num_cols, cat_cols)

if __name__ == "__main__":
    nums, cats = split_features()
    print(f"num_features:{nums}")
    print(f"cat_features:{cats}")