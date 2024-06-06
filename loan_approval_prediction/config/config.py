import pathlib
import loan_approval_prediction


PACKAGE_ROOT = pathlib.Path(loan_approval_prediction.__file__).resolve().parent

DATA_PATH = pathlib.Path.joinpath(PACKAGE_ROOT, "datasets")

SAVED_MODEL_NAME = "loan_classification.pkl"
SAVED_MODELS_PATH = pathlib.Path.joinpath(PACKAGE_ROOT, "trained_models")

TRAIN_DF = "train.csv"
TEST_DF = "test.csv"

TARGET = "Loan_Status"

FEATURES = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
            'Credit_History', 'Property_Area']

NUM_FEATURES = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

CAT_FEATURES = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']


FEATURES_TO_ENCODE = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']

FEATURE_TO_MODIFY = 'ApplicantIncome'
FEATURE_TO_ADD = 'CoapplicantIncome'

DROP_FEATURES = ['CoapplicantIncome']

LOG_FEATURES = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']