import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# tests/test_logging_utils.py

import pytest
from utils.logging_utils import save_plots
import os


import matplotlib.pyplot as plt
import os

def save_plots(history):
    # Ensure the plots directory exists
    if not os.path.exists("logs/plots"):
        os.makedirs("logs/plots")
    
    # Plot accuracy
    plt.figure()
    plt.plot(history['accuracy'], label='train accuracy')      # Access dictionary directly
    plt.plot(history['val_accuracy'], label='validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('logs/plots/accuracy.png')
    plt.close()  # Close the figure to free memory

    # Plot loss
    plt.figure()
    plt.plot(history['loss'], label='train loss')
    plt.plot(history['val_loss'], label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('logs/plots/loss.png')
    plt.close()  # Close the figure to free memory




def test_save_plots():
    # Mock data to simulate training history
    mock_history = {
        'accuracy': [0.5, 0.6, 0.7],
        'val_accuracy': [0.4, 0.5, 0.6],
        'loss': [0.8, 0.7, 0.6],
        'val_loss': [0.9, 0.8, 0.7]
    }
    
    # Ensure plot directory exists
    if not os.path.exists("logs/plots"):
        os.makedirs("logs/plots")
    
    # Call the function to save plots
    save_plots(mock_history)
    
    # Verify the plots were saved
    assert os.path.exists("logs/plots/accuracy.png"), "Accuracy plot not saved"
    assert os.path.exists("logs/plots/loss.png"), "Loss plot not saved"
