
# data_utils.py for resnet
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path, test_size=0.2, random_state=42):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Split features and labels
    X = df.drop(columns=['Label'])
    y = df['Label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test





# # data_utils.py for default

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# def load_data(filepath):
#     data = pd.read_csv(filepath)
#     X = data.drop(columns=["Label"])
#     y = data["Label"]
    
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
    
#     return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
