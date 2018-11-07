"""
Library to generate the datasets for training and testing on.

Contains functions that:
* generate N input points X in D space
* calculate N output points Y of function F
* function F can be chosen from global list of functions
"""
import numpy as np


def ripple(inputs):
    x, y = inputs
    return np.sin(10 * (x ** 2 + y ** 2)) / 10

def uball(inputs):
    X, y = inputs
    return 10 / (5 + np.sum(np.square(np.subtract(X, 3))))


functions = {"ripple": ripple, "uball": uball}


def generate_inputs(shape):
    """
    Shape is (N,D).
    
    Use np.random.normal()
    Returns np.array() with .shape = (N,D).
    """
    return NotImplemented


def calculate_outputs(inputs, func_name):
    """Given (N,D) input, create (N,) outputs)."""
    outputs = np.array()
    for inp in inputs:
        y = functions[func_name](inputs)
        outputs.append(y)
    assert outputs.shape[0] == inputs.shape[0]
    return outputs


if __name__ == "__main__":
    inputs = generate_inputs((200, 2))
    outputs = calculate_outputs(inputs, "ripple")
    # merge inputs and outputs into a single (N,D+1) array
    # save to file with name e.g. 'data/ripple_200.ext'
    pass
