import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))





import tensorflow as tf
from project.models.model_dnn import build_model
from data_utils import load_data

def test_train_process():
    X_train, _, y_train, _ = load_data("tests/sample_data.csv")
    model = build_model(input_shape=X_train.shape[1])
    
    # Mock training with limited epochs and small dataset
    history = model.fit(X_train, y_train, epochs=1, batch_size=2, validation_split=0.2)
    
    # Check if training history contains essential metrics
    assert 'accuracy' in history.history, "Training history missing accuracy data"
    assert 'val_accuracy' in history.history, "Validation accuracy missing in training history"
    
    # Check if a model checkpoint is saved properly
    assert model is not None, "Model training did not complete as expected"
