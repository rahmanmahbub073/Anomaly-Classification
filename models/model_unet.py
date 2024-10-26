# models/model_unet.py

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, UpSampling1D, Concatenate, Input, BatchNormalization, Activation, Cropping1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model

def build_model(input_shape):
    inputs = Input(shape=input_shape)

    # Encoding path
    c1 = Conv1D(64, 3, padding="same")(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Activation("relu")(c1)
    p1 = MaxPooling1D(pool_size=2, padding="same")(c1)

    c2 = Conv1D(128, 3, padding="same")(p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation("relu")(c2)
    p2 = MaxPooling1D(pool_size=2, padding="same")(c2)

    c3 = Conv1D(256, 3, padding="same")(p2)
    c3 = BatchNormalization()(c3)
    c3 = Activation("relu")(c3)
    p3 = MaxPooling1D(pool_size=2, padding="same")(c3)

    c4 = Conv1D(512, 3, padding="same")(p3)
    c4 = BatchNormalization()(c4)
    c4 = Activation("relu")(c4)
    p4 = MaxPooling1D(pool_size=2, padding="same")(c4)

    # Bottleneck
    bn = Conv1D(1024, 3, padding="same")(p4)
    bn = BatchNormalization()(bn)
    bn = Activation("relu")(bn)

    # Decoding path with UpSampling and Cropping1D if necessary
    u6 = UpSampling1D(size=2)(bn)
    if u6.shape[1] > c4.shape[1]:
        u6 = Cropping1D(cropping=(0, u6.shape[1] - c4.shape[1]))(u6)
    elif u6.shape[1] < c4.shape[1]:
        c4 = Cropping1D(cropping=(0, c4.shape[1] - u6.shape[1]))(c4)
    u6 = Concatenate()([u6, c4])

    c6 = Conv1D(512, 3, padding="same")(u6)
    c6 = BatchNormalization()(c6)
    c6 = Activation("relu")(c6)

    u7 = UpSampling1D(size=2)(c6)
    if u7.shape[1] > c3.shape[1]:
        u7 = Cropping1D(cropping=(0, u7.shape[1] - c3.shape[1]))(u7)
    elif u7.shape[1] < c3.shape[1]:
        c3 = Cropping1D(cropping=(0, c3.shape[1] - u7.shape[1]))(c3)
    u7 = Concatenate()([u7, c3])

    c7 = Conv1D(256, 3, padding="same")(u7)
    c7 = BatchNormalization()(c7)
    c7 = Activation("relu")(c7)

    u8 = UpSampling1D(size=2)(c7)
    if u8.shape[1] > c2.shape[1]:
        u8 = Cropping1D(cropping=(0, u8.shape[1] - c2.shape[1]))(u8)
    elif u8.shape[1] < c2.shape[1]:
        c2 = Cropping1D(cropping=(0, c2.shape[1] - u8.shape[1]))(c2)
    u8 = Concatenate()([u8, c2])

    c8 = Conv1D(128, 3, padding="same")(u8)
    c8 = BatchNormalization()(c8)
    c8 = Activation("relu")(c8)

    u9 = UpSampling1D(size=2)(c8)
    if u9.shape[1] > c1.shape[1]:
        u9 = Cropping1D(cropping=(0, u9.shape[1] - c1.shape[1]))(u9)
    elif u9.shape[1] < c1.shape[1]:
        c1 = Cropping1D(cropping=(0, c1.shape[1] - u9.shape[1]))(c1)
    u9 = Concatenate()([u9, c1])

    c9 = Conv1D(64, 3, padding="same")(u9)
    c9 = BatchNormalization()(c9)
    c9 = Activation("relu")(c9)

    # Output layer with Global Average Pooling to match the target shape
    x = GlobalAveragePooling1D()(c9)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    return model
