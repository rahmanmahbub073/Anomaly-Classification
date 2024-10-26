import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Model



def inception_module(input_tensor, filters):
    conv_3 = Conv1D(filters, kernel_size=3, padding='same', activation='relu')(input_tensor)
    conv_5 = Conv1D(filters, kernel_size=5, padding='same', activation='relu')(input_tensor)
    conv_7 = Conv1D(filters, kernel_size=7, padding='same', activation='relu')(input_tensor)
    output = tf.keras.layers.concatenate([conv_3, conv_5, conv_7])
    return output

def build_model(input_shape):
    inputs = Input(shape=input_shape)
    x = inception_module(inputs, filters=32)
    x = BatchNormalization()(x)
    x = inception_module(x, filters=32)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    return model
