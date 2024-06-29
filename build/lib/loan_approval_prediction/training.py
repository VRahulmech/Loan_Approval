'''
from data_handling import handling
from config import config
import pipeline as pipe
'''
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parent.parent
sys.path.append(str(PACKAGE_ROOT))


from loan_approval_prediction.data_handling import handling
from loan_approval_prediction.config import config
import loan_approval_prediction.pipeline as pipe


def train_model():
    train_data = handling.load_dataset(config.TRAIN_DF)
    train_y = train_data[config.TARGET].map({'N': 0, 'Y': 1})
    pipe.classification_pipeline.fit(train_data[config.FEATURES], train_y)
    handling.save_pipeline(pipe.classification_pipeline)


if __name__ == "__main__":
    train_model()
