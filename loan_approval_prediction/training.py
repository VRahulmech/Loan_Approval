from loan_approval_prediction.data_handling import handling
from loan_approval_prediction.config import config
import loan_approval_prediction.processing.preprocessing as pp
import loan_approval_prediction.pipeline as pipe


def train_model():
    train_data = handling.load_dataset(config.TRAIN_DF)
    train_y = train_data[config.TARGET].map({'N': 0, 'Y': 1})
    pipe.classification_pipeline.fit(train_data[config.FEATURES], train_y)
    handling.save_pipeline(pipe.classification_pipeline)


if __name__ == "__main__":
    train_model()
