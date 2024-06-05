import pathlib
import loan_approval_prediction


PACKAGE_ROOT = pathlib.Path(loan_approval_prediction.__file__).resolve().parent

DATA_PATH = pathlib.Path.joinpath(PACKAGE_ROOT, "datasets")

SAVED_MODELS_PATH = pathlib.Path.joinpath(PACKAGE_ROOT, "trained_models")

TRAIN_DF = "train.csv"
TEST_DF = "test.csv"

TARGET = ""