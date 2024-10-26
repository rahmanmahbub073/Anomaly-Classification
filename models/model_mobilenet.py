from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Model

from tensorflow.keras.layers import DepthwiseConv2D, Reshape

def build_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Reshape((input_shape[0], 1, 1))(inputs)
    x = DepthwiseConv2D(kernel_size=(3, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(64, kernel_size=1)(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs, outputs)
    return model
