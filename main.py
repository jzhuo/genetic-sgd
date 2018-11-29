"""
This is the main functtion for the files.

It should contain functions that
* Load the dataset or datasets to be used
* Initialize the genetic algorithm and its population
* Train on the dataset
"""

import pandas as pd

SEED = 42


def load_dataset(name):
    frame = pd.read_csv(name, index_col=0)
    return frame


def plot_comparison():
    import matplotlib.pyplot as plt

    pass


def initialize_ga(hybrid, input_size, hidden_layer_size, output_size):
    from genetic_algorithm import GeneticAlgorithm

    GeneticAlgorithm(hybrid, input_size)
    pass


def initiallize_one_nn(input_size):
    pass


def cross_validate_on_dataset(estimator, data):
    from sklearn.model_selection import cross_val_score, GridSearchCV

    pass


if __name__ == "__main__":
    # load data
    data = load_dataset("data/ripple_0.0_50_200")
    # init ga
    input_size = data.shape[1] - 1
    output_size = 1
    # pass to cross validate
    pass
