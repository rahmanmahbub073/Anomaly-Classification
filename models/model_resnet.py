# models/model.py
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, Add, GlobalAveragePooling1D, Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def residual_block(x, filters, kernel_size=5, stride=1, dropout_rate=0.7, l2_reg=1e-4):
    shortcut = x
    x = Conv1D(filters, kernel_size=kernel_size, strides=stride, padding="same", kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(filters, kernel_size=kernel_size, strides=stride, padding="same", kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Dropout(dropout_rate)(x)
    return ReLU()(x)

def build_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # Initial Convolutional Block with lower filter size
    x = Conv1D(32, kernel_size=5, padding="same", kernel_regularizer=l2(1e-4), activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Residual Blocks with Increased Dropout
    x = residual_block(x, 32, kernel_size=5, dropout_rate=0.7)
    x = residual_block(x, 32, kernel_size=5, dropout_rate=0.7)
    
    # Global Pooling and Output Layer
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    return model

