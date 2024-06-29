import pytest


import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from loan_approval_prediction import predict

@pytest.fixture
def single_pred():
    pred = predict.predict()
    return pred[0]


def test_pred_none(single_pred):
    assert single_pred is not None


def test_pred_str(single_pred):
    assert isinstance(single_pred, str)


