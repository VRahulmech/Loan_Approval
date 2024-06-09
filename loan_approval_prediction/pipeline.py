import loan_approval_prediction.processing.preprocessing as pp
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from loan_approval_prediction.config import config
from sklearn.linear_model import LogisticRegression


classification_pipeline = Pipeline(
    [
        ("MeanImputation", pp.MeanImputer(config.NUM_FEATURES)),
        ("ModeImputation", pp.ModeImputer(config.CAT_FEATURES)),
        ("AddingVariables", pp.AddingVariables(config.FEATURE_TO_MODIFY, config.FEATURE_TO_ADD)),
        ("DropColumns", pp.DropColumns(config.DROP_FEATURES)),
        ("LabelEncoder", LabelEncoder()),
        ("LogTransform", pp.LogTransformation()),
        ("MinMaxScaler", MinMaxScaler()),
        ("LogisticRegression", LogisticRegression(random_state=42))
    ]
)