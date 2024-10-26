
# evaluate.py
import tensorflow as tf
from data_utils import load_data
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

def save_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.savefig("logs/confusion_matrix.png")
    plt.close()

def save_classification_report(report):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.axis('tight')
    table_data = []
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            table_data.append([label, metrics['precision'], metrics['recall'], metrics['f1-score'], metrics['support']])
    table = ax.table(cellText=table_data, colLabels=["Class", "Precision", "Recall", "F1-Score", "Support"], loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)
    plt.title("Classification Report")
    plt.savefig("logs/classification_report.png")
    plt.close()

def evaluate_model():
    model = tf.keras.models.load_model("logs/saved_model.keras")
    _, X_test, _, y_test = load_data("data/DupliRowsReduced.csv")
    X_test = np.array(X_test).reshape(-1, X_test.shape[1], 1)

    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    results = {"test_accuracy": test_accuracy, "test_loss": test_loss}
    with open("logs/test_results.json", "w") as f:
        json.dump(results, f)

    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    save_confusion_matrix(cm, class_names=["Class 0", "Class 1"])

    report = classification_report(y_test, y_pred, output_dict=True)
    save_classification_report(report)

if __name__ == "__main__":
    evaluate_model()


















# # evaluate.py
# import tensorflow as tf
# from data_utils import load_data
# from sklearn.metrics import confusion_matrix, classification_report
# import numpy as np
# import pandas as pd
# import json
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# def save_confusion_matrix(cm, labels, filename):
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title('Confusion Matrix')
#     plt.savefig(filename)
#     plt.close()

# def save_classification_report(report, filename):
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap='coolwarm')
#     plt.title("Classification Report")
#     plt.savefig(filename)
#     plt.close()

# def save_training_history(history, filename_prefix="logs/history"):
#     # Plot accuracy
#     plt.figure()
#     plt.plot(history['accuracy'], label='Train Accuracy')
#     plt.plot(history['val_accuracy'], label='Validation Accuracy')
#     plt.title('Model Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.savefig(f"{filename_prefix}_accuracy.png")
#     plt.close()

#     # Plot loss
#     plt.figure()
#     plt.plot(history['loss'], label='Train Loss')
#     plt.plot(history['val_loss'], label='Validation Loss')
#     plt.title('Model Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.savefig(f"{filename_prefix}_loss.png")
#     plt.close()

# def evaluate_model():
#     # Load the trained model
#     model = tf.keras.models.load_model("logs/saved_model.keras")

#     # Load the dataset
#     _, X_test, _, y_test = load_data("data/DupliRowsReduced.csv")

#     # Reshape test data if necessary
#     X_test = np.array(X_test).reshape(-1, X_test.shape[1], 1)

#     # Evaluate the model on the test set
#     test_loss, test_accuracy = model.evaluate(X_test, y_test)
#     print(f"Test Accuracy: {test_accuracy:.4f}")
#     print(f"Test Loss: {test_loss:.4f}")

#     # Save evaluation results
#     results = {
#         "test_accuracy": test_accuracy,
#         "test_loss": test_loss
#     }

#     with open("logs/test_results.json", "w") as f:
#         json.dump(results, f)

#     # Get model predictions
#     y_pred = model.predict(X_test)
#     y_pred = (y_pred > 0.5).astype(int)  # Threshold for binary classification

#     # Confusion Matrix and Classification Report
#     cm = confusion_matrix(y_test, y_pred)
#     report = classification_report(y_test, y_pred, output_dict=True)
    
#     # Print Confusion Matrix and Classification Report
#     print("Confusion Matrix:")
#     print(cm)
#     print("\nClassification Report:")
#     print(classification_report(y_test, y_pred))

#     # Save Confusion Matrix and Classification Report as images
#     save_confusion_matrix(cm, labels=[0, 1], filename="logs/confusion_matrix.png")
#     save_classification_report(report, filename="logs/classification_report.png")

#     # Save training history (if applicable)
#     if 'history' in locals():
#         save_training_history(history.history)

# if __name__ == "__main__":
#     evaluate_model()







# import tensorflow as tf
# from data_utils import load_data
# from sklearn.metrics import confusion_matrix, classification_report
# import numpy as np
# import json
# import matplotlib.pyplot as plt
# import seaborn as sns

# def plot_confusion_matrix_and_save(y_true, y_pred, filename):
#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(10, 7))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
#     plt.title("Confusion Matrix")
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.savefig(filename)
#     plt.close()

# def plot_classification_report_and_save(report, filename):
#     plt.figure(figsize=(8, 6))
#     plt.text(0.01, 0.05, str(report), {'fontsize': 12}, fontproperties="monospace")
#     plt.title("Classification Report")
#     plt.axis("off")
#     plt.savefig(filename)
#     plt.close()

# def evaluate_model():
#     # Load the trained model
#     model = tf.keras.models.load_model("logs/saved_model.keras")

#     # Load the dataset
#     _, X_test, _, y_test = load_data("data/DupliRowsReduced.csv")

#     # Evaluate the model on the test set
#     test_loss, test_accuracy = model.evaluate(X_test, y_test)
#     print(f"Test Accuracy: {test_accuracy:.4f}")
#     print(f"Test Loss: {test_loss:.4f}")

#     # Save evaluation results
#     results = {
#         "test_accuracy": test_accuracy,
#         "test_loss": test_loss
#     }

#     with open("logs/test_results.json", "w") as f:
#         json.dump(results, f)

#     # Get model predictions
#     y_pred = model.predict(X_test)
#     y_pred = (y_pred > 0.5).astype(int)  # Adjust threshold for binary classification

#     # Save confusion matrix as PNG
#     plot_confusion_matrix_and_save(y_test, y_pred, "logs/confusion_matrix.png")

#     # Generate classification report
#     report = classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"], digits=4)
#     print("\nClassification Report:")
#     print(report)

#     # Save classification report as PNG
#     plot_classification_report_and_save(report, "logs/classification_report.png")

# if __name__ == "__main__":
#     evaluate_model()
