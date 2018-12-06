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


class NeuralNetwork:
    
    def __init__(
        self,
        input_size=3,
        hidden_layer_size=10,
        output_size=1, 
        learning_rate=1e-3, 
        epochs=10,
        verbose=0,
        weights=None
    ):
        self.input_size=input_size
        self.hidden_layer_size=hidden_layer_size
        self.output_size=output_size
        self.learning_rate=learning_rate
        self.weights=weights
        self.model=None
        self.build_neural_network()
        self.epochs=epochs
        self.verbose=verbose
   
    def build_neural_network(self):
        """Build single hidden layer network in Keras and return it."""

        model = Sequential()
        # input layer
        # model.add(Dense(input_size, activation=None, bias_initializer=constant(1)))
        # inputs = Input(shape=(input_size,))
        # model.add(inputs)
        # fan-in initialization
        minval = -0.5 / self.input_size
        maxval = 0.5 / self.input_size
        fan_in_init = initializers.RandomUniform(
            minval=minval, maxval=maxval, seed=main.SEED
        )
        model.add(
            Dense(
                self.hidden_layer_size,
                input_dim=self.input_size,
                activation="tanh",
                bias_initializer=constant(1),
                kernel_initializer=fan_in_init,
            )
        )
        # output layer
        minval = -0.5 / self.hidden_layer_size
        maxval = 0.5 / self.hidden_layer_size
        fan_in_init = initializers.RandomUniform(
            minval=minval, maxval=maxval, seed=main.SEED
        )
        model.add(
            Dense(1, bias_initializer=constant(1), kernel_initializer=fan_in_init)
        )

        optimizer = optimizers.SGD(lr=self.learning_rate)

        if self.weights is not None:
            model.set_weights(self.weights)

        model.compile(optimizer=optimizer, loss="mse")

        # from keras.utils import plot_model

        # plot_model(
        #     model, to_file="model.png", show_layer_names=True, show_shapes=True
        # )
        self.model = model

    def fit(self, train_x, train_y):
        self.model.fit(train_x, train_y, epochs=self.epochs, verbose=self.verbose)

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=False):
        """Return the params dictionary."""
        params = {
            "input_size": self.input_size,
            "hidden_layer_size": self.hidden_layer_size,
            "output_size": self.output_size,
            # "weights": self.weights,
            "learning_rate": self.learning_rate
        }
        return params

    def set_params(self, **params):
        self.input_size=params["input_size"]
        self.hidden_layer_size=params["hidden_layer_size"]
        self.output_size=params["output_size"]
        self.learning_rate=params["learning_rate"]
        # self.weights=params["weights"]
        self.model=None
        self.build_neural_network()
