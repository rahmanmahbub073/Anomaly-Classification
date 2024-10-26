import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



from project.models.model_dnn import build_model
import tensorflow as tf

def test_build_model():
    model = build_model(input_shape=15)  # Use a dummy input shape matching the dataset
    
    # Check model compilation
    assert model.compiled_loss is not None, "Model did not compile correctly"
    
    # Check if the number of layers is as expected
    assert len(model.layers) >= 3, "Model architecture seems incomplete"
    
    # Check output layer
    assert model.layers[-1].units == 1, "Output layer should have 1 unit for binary classification"
    assert model.layers[-1].activation.__name__ == 'sigmoid', "Output layer should use sigmoid activation"
