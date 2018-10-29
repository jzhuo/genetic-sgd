"""
This is a *potential* template for the neural network. 

There are pros and cons to wrapping sklearn and keras objects.
"""

from sklearn.neural_network import MLPRegressor
from keras.models import Sequential
from keras.layers import Dense, Activation


class NeuralNetwork:
    def __init__(self, hidden_layer_sizes=[]):
        pass
