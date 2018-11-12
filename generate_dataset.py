"""
Library to generate the datasets for training and testing on.

Contains functions that:
* generate N input points X in D space
* calculate N output points Y of function F
* function F can be chosen from global list of functions
"""
import numpy as np


def ripple(X):
    return np.sin(10 * (X[0] ** 2 + X[1] ** 2)) / 10


def uball(X):
    return 10 / (5 + np.sum(np.square(np.subtract(X, 3))))


functions = {"ripple": ripple, "uball": uball}


def generate_inputs(loc, scale, shape):
    """
    Shape is (N,D).
    
    Use np.random.normal()
    Returns np.array() with .shape = (N,D).
    """
    return np.random.normal(loc, scale, shape)


def calculate_outputs(inputs, func_name):
    """Given (N,D) input, create (N,) outputs)."""
    outputs = []
    for X in inputs:
        y = functions[func_name](X)
        outputs.append(y)
    outputs = np.array(outputs)
    assert outputs.shape[0] == inputs.shape[0]
    return outputs


def create_and_save_data(loc, scale, shape, func_name):
    import pandas as pd

    inputs = generate_inputs(loc, scale, shape)
    outputs = calculate_outputs(inputs, func_name)
    inputs = pd.DataFrame(inputs)
    outputs = pd.Series(outputs)
    inputs["y"] = outputs

    inputs.to_csv(
        "data/"
        + str(func_name)
        + "_"
        + str(loc)
        + "_"
        + str(scale)
        + "_"
        + str(shape[0])
    )


if __name__ == "__main__":
    # create_and_save_data(0.0, 50, (200, 5), "uball")
    import sys

    # python generate_dataset.py 0.0 50 200 2 ripple
    #                            loc scale samples dim func
    create_and_save_data(
        float(sys.argv[1]),
        int(sys.argv[2]),
        (int(sys.argv[3]), int(sys.argv[4])),
        str(sys.argv[5]),
    )

    # merge inputs and outputs into a single (N,D+1) array
    # save to file with name e.g. 'data/ripple_200.ext'
