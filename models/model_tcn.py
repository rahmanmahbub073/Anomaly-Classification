from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, GlobalAveragePooling1D, Dense, SpatialDropout1D, Add
from tensorflow.keras.models import Model

def residual_block_tcn(x, filters, dilation_rate):
    shortcut = x
    x = Conv1D(filters, kernel_size=3, padding='causal', dilation_rate=dilation_rate, activation='relu')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout1D(0.3)(x)
    x = Conv1D(filters, kernel_size=3, padding='causal', dilation_rate=dilation_rate, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    return x

def build_model(input_shape):
    inputs = Input(shape=input_shape)
    x = residual_block_tcn(inputs, filters=32, dilation_rate=1)
    x = residual_block_tcn(x, filters=32, dilation_rate=2)
    x = residual_block_tcn(x, filters=32, dilation_rate=4)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    return model
