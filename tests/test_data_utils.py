import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_utils import load_data


import pytest
import os
from data_utils import load_data

def test_load_data():
    # Create a path to a small sample dataset or mock dataset
    sample_data_path = "tests/sample_data.csv"  # Make sure this file exists in the tests directory
    
    if not os.path.exists(sample_data_path):
        pytest.fail("Sample data file does not exist")

    X_train, X_test, y_train, y_test = load_data(sample_data_path)
    assert X_train.shape[0] > 0, "Training set is empty"
    assert X_test.shape[0] > 0, "Test set is empty"

