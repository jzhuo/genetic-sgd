"""
This is the main functtion for the files.

It should contain functions that
* Load the dataset or datasets to be used
* Initialize the genetic algorithm and its population
* Train on the dataset
"""

import pandas as pd


def load_dataset(name):
    frame = pd.read_csv(name, index_col=0)
    return frame


def plot_comparison():
    import matplotlib.pyplot as plt

    pass


def initiallize_ga(args):
    pass


def initiallize_one_nn(args):
    pass


def cross_validate_on_dataset(estimator):
    pass


if __name__ == "__main__":
    pass
