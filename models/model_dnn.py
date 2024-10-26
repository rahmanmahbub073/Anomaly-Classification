from tensorflow.keras import layers, models, regularizers

def build_model(input_shape):
    model = models.Sequential()
    
    # Example model architecture with dropout and L2 regularization
    model.add(layers.Dense(128, activation='relu', input_shape=(input_shape,),
                           kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.5))  # Add dropout layer to prevent overfitting
    
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.5))  # Dropout again
    
    model.add(layers.Dense(1, activation='sigmoid'))  # For binary classification
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
















# import tensorflow as tf
# from tensorflow.keras import layers, models, regularizers

# def build_model(input_shape):
#     model = models.Sequential([
#         layers.Input(shape=(input_shape,)),
#         # Adding noise to the input layer
#         layers.GaussianNoise(0.1),  # Adding noise to the inputs
        
#         # L2 regularization and increased dropout
#         layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
#         layers.Dropout(0.4),  # Increased dropout rate to 40%

#         layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
#         layers.Dropout(0.4),  # Increased dropout rate to 40%

#         layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
#     ])

#     # Compile the model with an appropriate optimizer
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
    
#     return model



