# train.py

import os
import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np
import json

from data_utils import load_data
from utils.logging_utils import save_plots

# Dynamic model loader
def load_model(model_name, input_shape):
    try:
        model_module = __import__(f"models.{model_name}", fromlist=["build_model"])
        model = model_module.build_model(input_shape=input_shape)
        return model
    except ModuleNotFoundError as e:
        print(f"Error: Could not find the model '{model_name}' in 'models/' directory.")
        raise e

def train(model_name="model"):
    # Setup for multi-GPU support
    strategy = tf.distribute.MirroredStrategy()  # Using all available GPUs
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    # Load the dataset
    X_train, X_test, y_train, y_test = load_data("data/DupliRowsReduced.csv")

    # Reshape data for Conv1D (samples, features, 1)
    X_train = np.array(X_train).reshape(-1, X_train.shape[1], 1)
    X_test = np.array(X_test).reshape(-1, X_test.shape[1], 1)
    y_train, y_test = np.array(y_train), np.array(y_test)

    # K-Fold Cross-Validation setup
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_no = 1
    all_scores = []

    # Train using multi-GPU strategy
    with strategy.scope():
        for train_idx, val_idx in kfold.split(X_train):
            print(f"\nTraining Fold {fold_no}")

            # Split the data
            X_train_fold, y_train_fold = X_train[train_idx], y_train[train_idx]
            X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]

            # Load and compile model dynamically
            model = load_model(model_name, input_shape=(X_train.shape[1], 1))
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                          loss='binary_crossentropy', metrics=['accuracy'])

            # Define callbacks
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

            # Train the model
            history = model.fit(X_train_fold, y_train_fold, 
                                epochs=20, 
                                batch_size=16, 
                                validation_data=(X_val_fold, y_val_fold),
                                callbacks=[early_stopping, lr_scheduler])

            # Evaluate the model
            val_loss, val_accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)
            print(f"Fold {fold_no} - Validation Accuracy: {val_accuracy:.4f} - Validation Loss: {val_loss:.4f}")
            all_scores.append(val_accuracy)

            # Save plots for each fold
            save_plots(history, fold=fold_no)
            fold_no += 1

    # Report cross-validation results
    print(f"\nCross-Validation Average Accuracy: {np.mean(all_scores):.4f}")

    # Final model training on full dataset
    with strategy.scope():
        print("\nTraining final model on full dataset...")
        model_final = load_model(model_name, input_shape=(X_train.shape[1], 1))
        model_final.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                            loss='binary_crossentropy', metrics=['accuracy'])
        model_final.fit(X_train, y_train, epochs=20, batch_size=16, 
                        validation_data=(X_test, y_test), 
                        callbacks=[early_stopping, lr_scheduler])

        # Save the trained model
        model_final.save("logs/saved_model.keras")

        # Evaluate on test set
        test_loss, test_accuracy = model_final.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")

        # Save test results
        results = {"test_accuracy": test_accuracy, "test_loss": test_loss}
        with open("logs/test_results.json", "w") as f:
            json.dump(results, f)

if __name__ == "__main__":
    # Pass model name dynamically when running the script
    train(model_name="model_dnn")  # You can change the model name as needed










# # train.py
# import tensorflow as tf
# from data_utils import load_data
# from sklearn.model_selection import KFold
# import numpy as np
# import json
# from utils.logging_utils import save_plots
# import importlib

# def load_model(model_name, input_shape):
#     """Dynamically loads the model from models/ directory based on the model name."""
#     try:
#         # Import model module dynamically
#         model_module = importlib.import_module(f"models.{model_name}")
#         model = model_module.build_model(input_shape=input_shape)
#         return model
#     except ModuleNotFoundError:
#         raise ValueError(f"Model '{model_name}' not found in models directory.")
#     except AttributeError:
#         raise ValueError(f"Model '{model_name}' does not have a 'build_model' function.")

# def train(model_name="model_resnet"):
#     # Load dataset
#     X_train, X_test, y_train, y_test = load_data("data/DupliRowsReduced.csv")
#     X_train = np.array(X_train).reshape(-1, X_train.shape[1], 1)
#     X_test = np.array(X_test).reshape(-1, X_test.shape[1], 1)
#     y_train, y_test = np.array(y_train), np.array(y_test)

#     # K-Fold cross-validation setup
#     kfold = KFold(n_splits=5, shuffle=True, random_state=42)
#     fold_no = 1
#     all_scores = []

#     for train_idx, val_idx in kfold.split(X_train):
#         print(f"\nTraining Fold {fold_no}")

#         # Prepare train/val splits
#         X_train_fold, y_train_fold = X_train[train_idx], y_train[train_idx]
#         X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]

#         # Load model dynamically
#         model = load_model(model_name, input_shape=(X_train.shape[1], 1))
#         model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
#                       loss='binary_crossentropy', metrics=['accuracy'])

#         # Define callbacks
#         early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
#         lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

#         # Train the model
#         history = model.fit(X_train_fold, y_train_fold, 
#                             epochs=20, 
#                             batch_size=16, 
#                             validation_data=(X_val_fold, y_val_fold),
#                             callbacks=[early_stopping, lr_scheduler])

#         # Evaluate model on validation fold
#         val_loss, val_accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)
#         print(f"Fold {fold_no} - Validation Accuracy: {val_accuracy:.4f} - Validation Loss: {val_loss:.4f}")
#         all_scores.append(val_accuracy)
        
#         # Save plots for each fold
#         save_plots(history, fold=fold_no)
#         fold_no += 1

#     print(f"\nCross-Validation Average Accuracy: {np.mean(all_scores):.4f}")

#     # Train final model on the full dataset after cross-validation
#     model_final = load_model(model_name, input_shape=(X_train.shape[1], 1))
#     model_final.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
#                         loss='binary_crossentropy', metrics=['accuracy'])
#     model_final.fit(X_train, y_train, epochs=20, batch_size=16, 
#                     validation_data=(X_test, y_test), 
#                     callbacks=[early_stopping, lr_scheduler])

#     # Save the trained model
#     model_final.save("logs/saved_model.keras")

#     # Evaluate on test set
#     test_loss, test_accuracy = model_final.evaluate(X_test, y_test, verbose=0)
#     print(f"Test Accuracy: {test_accuracy:.4f}")
#     print(f"Test Loss: {test_loss:.4f}")

#     # Save test results
#     results = {"test_accuracy": test_accuracy, "test_loss": test_loss}
#     with open("logs/test_results.json", "w") as f:
#         json.dump(results, f)

# if __name__ == "__main__":
#     # Pass in model name as argument
#     train(model_name="model_unet")  # Example: "model_resnet", "model_inception", "model_tcn"



# import tensorflow as tf
# from models.model import build_model
# from data_utils import load_data
# from sklearn.model_selection import KFold
# import numpy as np
# import json
# from utils.logging_utils import save_plots

# def train():
#     # Load the dataset
#     X_train, X_test, y_train, y_test = load_data("data/DupliRowsReduced.csv")
    
#     # Ensure the data is in NumPy format (in case it's a DataFrame)
#     X_train = np.array(X_train)
#     y_train = np.array(y_train)
#     X_test = np.array(X_test)
#     y_test = np.array(y_test)

#     # Define number of folds for cross-validation
#     kfold = KFold(n_splits=5, shuffle=True, random_state=42)

#     fold_no = 1
#     all_scores = []
#     history_all_folds = []

#     for train_idx, val_idx in kfold.split(X_train):
#         print(f"\nTraining Fold {fold_no}")

#         # Use NumPy array slicing based on train_idx and val_idx
#         X_train_fold = X_train[train_idx]
#         y_train_fold = y_train[train_idx]
#         X_val_fold = X_train[val_idx]
#         y_val_fold = y_train[val_idx]

#         # Create model
#         model = build_model(input_shape=X_train.shape[1])

#         # Define callbacks
#         early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
#         lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

#         # Train the model
#         history = model.fit(X_train_fold, y_train_fold, 
#                             epochs=10, 
#                             batch_size=32, 
#                             validation_data=(X_val_fold, y_val_fold),
#                             callbacks=[early_stopping, lr_scheduler])

#         # Evaluate the model on the validation fold
#         val_loss, val_accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)
#         print(f"Fold {fold_no} - Validation Accuracy: {val_accuracy:.4f} - Validation Loss: {val_loss:.4f}")
#         all_scores.append(val_accuracy)
#         history_all_folds.append(history.history)
        
#         # Save plots for each fold
#         save_plots(history, fold=fold_no)

#         fold_no += 1

#     # Cross-validation results
#     print(f"\nCross-Validation Average Accuracy: {np.mean(all_scores):.4f}")

#     # Train the model on the full dataset after cross-validation (optional)
#     print("\nTraining final model on full dataset...")
#     model_final = build_model(input_shape=X_train.shape[1])
#     model_final.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), 
#                     callbacks=[early_stopping, lr_scheduler])

#     # Save the trained model
#     model_final.save("logs/saved_model.keras")

#     # Evaluate on test set
#     test_loss, test_accuracy = model_final.evaluate(X_test, y_test, verbose=0)
#     print(f"Test Accuracy: {test_accuracy:.4f}")
#     print(f"Test Loss: {test_loss:.4f}")

#     # Save test results
#     results = {
#         "test_accuracy": test_accuracy,
#         "test_loss": test_loss
#     }

#     with open("logs/test_results.json", "w") as f:
#         json.dump(results, f)

#     # Save all history (optional for further analysis)
#     with open("logs/history_all_folds.json", "w") as f:
#         json.dump(history_all_folds, f)

# if __name__ == "__main__":
#     train()



















# import tensorflow as tf
# from models.model import build_model
# from data_utils import load_data
# from sklearn.model_selection import KFold
# import numpy as np
# import json
# from utils.logging_utils import save_plots

# def train():
#     # Load the dataset
#     X_train, X_test, y_train, y_test = load_data("data/DupliRowsReduced.csv")
    
#     # Ensure the data is in NumPy format (in case it's a DataFrame)
#     X_train = np.array(X_train)
#     y_train = np.array(y_train)
#     X_test = np.array(X_test)
#     y_test = np.array(y_test)

#     # Define number of folds for cross-validation
#     kfold = KFold(n_splits=5, shuffle=True, random_state=42)

#     fold_no = 1
#     all_scores = []

#     for train_idx, val_idx in kfold.split(X_train):
#         print(f"\nTraining Fold {fold_no}")

#         # Use NumPy array slicing based on train_idx and val_idx
#         X_train_fold = X_train[train_idx]
#         y_train_fold = y_train[train_idx]
#         X_val_fold = X_train[val_idx]
#         y_val_fold = y_train[val_idx]

#         # Create model
#         model = build_model(input_shape=X_train.shape[1])

#         # Define callbacks
#         early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
#         lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

#         # Train the model
#         history = model.fit(X_train_fold, y_train_fold, 
#                             epochs=50, 
#                             batch_size=32, 
#                             validation_data=(X_val_fold, y_val_fold),
#                             callbacks=[early_stopping, lr_scheduler])

#         # Evaluate the model on the validation fold
#         val_loss, val_accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)
#         print(f"Fold {fold_no} - Validation Accuracy: {val_accuracy:.4f} - Validation Loss: {val_loss:.4f}")
#         all_scores.append(val_accuracy)
        
#         # Save plots for each fold
#         save_plots(history, fold=fold_no)

#         fold_no += 1

#     print(f"\nCross-Validation Average Accuracy: {np.mean(all_scores):.4f}")

#     # Train the model on the full dataset after cross-validation (optional)
#     print("\nTraining final model on full dataset...")
#     model_final = build_model(input_shape=X_train.shape[1])
#     model_final.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), 
#                     callbacks=[early_stopping, lr_scheduler])

#     # Save the trained model
#     model_final.save("logs/saved_model.keras")

#     # Evaluate on test set
#     test_loss, test_accuracy = model_final.evaluate(X_test, y_test, verbose=0)
#     print(f"Test Accuracy: {test_accuracy:.4f}")
#     print(f"Test Loss: {test_loss:.4f}")

#     # Save test results
#     results = {
#         "test_accuracy": test_accuracy,
#         "test_loss": test_loss
#     }

#     with open("logs/test_results.json", "w") as f:
#         json.dump(results, f)

# if __name__ == "__main__":
#     train()
