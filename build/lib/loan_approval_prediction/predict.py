from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PACKAGE_ROOT))

from loan_approval_prediction.config import config
from loan_approval_prediction.data_handling import handling
import numpy as np


def predict(dataset=None):
    if dataset == None:
        test_data = handling.load_dataset(config.TEST_DF)
    else:
        test_data = dataset
    loaded_model = handling.load_pipeline(config.SAVED_MODELS_PATH)
    pred = loaded_model.predict(test_data[config.FEATURES])
    output = np.where(pred == 1, 'Y', 'N')
    return output


if __name__ == '__main__':
    print(predict())