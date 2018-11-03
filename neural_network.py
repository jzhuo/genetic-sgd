"""
This is a *potential* template for the neural network. 

There are pros and cons to wrapping sklearn and keras objects.
"""

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.initializers import constant
from keras import optimizers


def build_NN(input_size, hidden_layer_size, output_size, learning_rate, weights=None):
    """Build single hidden layer network in Keras and return it."""
    
    model = Sequential()
    model.add(
        Dense(
            input_size,
            Activation=None,
            bias_initializer=constant(1),
        )
    )
    model.add(
        Dense(
            hidden_layer_size,
            input_dim=input_size,
            Activation='tanh',
            bias_initializer=constant(1),
        )
    )
    model.add(
        Dense(
            output_size, 
            input_dim=hidden_layer_size, 
            Activation=None
        )
    )
    optimizer = optimizers.SGD(lr=learning_rate)

    if weights is not None:
        model.set_weights(weights)

    model.compile(optimizer=optimizer, loss="mse")

    return model