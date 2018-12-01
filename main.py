"""
This is the main functtion for the files.

It should contain functions that
* Load the dataset or datasets to be used
* Initialize the genetic algorithm and its population
* Train on the dataset
"""

import pandas as pd
import genetic_algorithm

SEED = 42


def load_dataset(name):
    frame = pd.read_csv(name, index_col=0)
    return frame


def plot_comparison():
    import matplotlib.pyplot as plt

    pass


def initiallize_one_nn(input_size):
    pass


def cross_validate_on_dataset(estimator, data):
    from sklearn.model_selection import cross_val_score, GridSearchCV

    input_size = data.shape[1] - 1
    param_grid = {
        "hybrid": [True],
        "input_size": [input_size],
        "hidden_layer_size": [input_size * index for index in range(1, 5)],
        "output_size": [1],
        "population_size": [5, 10],
        "selection_size": [2, 4],
        "learning_rate": [1e-3],
        "epochs": [10, 20],
        "generations": [25],
        "cases": [["mse", "l2", "l1", "time"]],
        "verbose": [1],
    }
    grid = GridSearchCV(
        estimator, param_grid, scoring="neg_mean_squared_error", cv=5, n_jobs=1
    )
    input_cols = list(data.columns)
    input_cols.remove("y")
    y = data["y"].values
    X = data[input_cols].values
    grid.fit(X, y)
    pass


if __name__ == "__main__":
    # load data
    data = load_dataset("data/ripple_0.0_50_200")
    # init ga
    input_size = data.shape[1] - 1
    hidden_layer_size = 5
    output_size = 1
    population_size = 6
    selection_size = 2
    learning_rate = 1e-3
    epochs = 10
    generations = 5
    estimator = genetic_algorithm.GeneticAlgorithm(
        True,
        input_size,
        hidden_layer_size,
        output_size,
        population_size,
        selection_size,
        learning_rate,
        epochs,
        generations,
    )
    cross_validate_on_dataset(estimator, data)

    # pass to cross validate
    # pass
