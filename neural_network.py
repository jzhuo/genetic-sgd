"""
This is a *potential* template for the neural network. 

There are pros and cons to wrapping sklearn and keras objects.
"""

from keras.models import Sequential
from keras.layers import Dense, Activation, Input
from keras.initializers import constant
from keras import initializers
from keras import optimizers
import main


def build_nn(
    input_size, hidden_layer_size, output_size, learning_rate, weights=None
):
    """Build single hidden layer network in Keras and return it."""

    model = Sequential()
    # input layer
    # model.add(Dense(input_size, activation=None, bias_initializer=constant(1)))
    # inputs = Input(shape=(input_size,))
    # model.add(inputs)
    # fan-in initialization
    minval = -0.5 / input_size
    maxval = 0.5 / input_size
    fan_in_init = initializers.RandomUniform(
        minval=minval, maxval=maxval, seed=main.SEED
    )
    model.add(
        Dense(
            hidden_layer_size,
            input_dim=input_size,
            activation="tanh",
            bias_initializer=constant(1),
            kernel_initializer=fan_in_init,
        )
    )
    # output layer
    minval = -0.5 / hidden_layer_size
    maxval = 0.5 / hidden_layer_size
    fan_in_init = initializers.RandomUniform(
        minval=minval, maxval=maxval, seed=main.SEED
    )
    model.add(
        Dense(1, bias_initializer=constant(1), kernel_initializer=fan_in_init)
    )

    optimizer = optimizers.SGD(lr=learning_rate)

    if weights is not None:
        model.set_weights(weights)

    model.compile(optimizer=optimizer, loss="mse")

    from keras.utils import plot_model

    plot_model(
        model, to_file="model.png", show_layer_names=True, show_shapes=True
    )
    return model


def build_sklearn_nn(
    input_size, hidden_layer_size, output_size, learning_rate, weights=None
):
    from sklearn.neural_network import MLPRegressor

    model = MLPRegressor(
        hidden_layer_sizes=(hidden_layer_size,),
        activation="tanh",
        solver="sgd",
        alpha=0.0,
        batch_size=32,
        learning_rate="constant",
        learning_rate_init=learning_rate,
    )

    return model
