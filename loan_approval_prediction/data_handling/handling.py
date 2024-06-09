import pandas as pd
import joblib
from loan_approval_prediction.config import config
import pathlib


def load_dataset(file_name):
    filepath = pathlib.Path.joinpath(config.DATA_PATH,file_name)
    _data = pd.read_csv(filepath)
    return _data


# Serialization
def save_pipeline(pipeline_to_save):
    save_path = pathlib.Path.joinpath(config.SAVE_MODEL_PATH,config.MODEL_NAME)
    joblib.dump(pipeline_to_save, save_path)
    print("Model saved successfully")


# Deserialization
def load_pipeline(pipeline_to_load):
    save_path = pathlib.Path.joinpath(config.SAVE_MODEL_PATH,config.MODEL_NAME)
    model_loaded = joblib.load(save_path)
    print("model loaded successfully")
    return model_loaded
