"""
This file contains the class that provides the genetic algorithm search. 
"""
import neural_network
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

W1_INDEX = 0
W2_INDEX = 2
ALPHABET = [
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
]
ALPHABET += [a.upper() for a in ALPHABET]


class GeneticAlgorithm:
    def __init__(
        self,
        hybrid=True,
        input_size=2,
        hidden_layer_size=5,
        output_size=1,
        population_size=6,
        selection_size=2,
        learning_rate=1e-3,
        epochs=10,
        generations=10,
        cases=["mse", "l2", "l1", "time"],
        # ****** make sure to toggle verbosity during training!! *****
        verbose=0,
    ):
        """
        If learning rate is 0, the algorithm is just regular mutation.

        Cases can include:
        *  mse
        *  run time
        *  l2 norm of weights
        *  l1 norm of weights
        """
        self.hybrid = hybrid
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.population_size = population_size
        self.selection_size = selection_size
        self.learning_rate = learning_rate
        self.cases = cases
        self.epochs = epochs
        self.generations = generations
        self.scale_of_mutation = 1.0  # future work
        self.verbose = verbose
        self.best_model = None
        self.best_mse = float("inf")
        self.population = []
        self.graph = nx.Graph()
        self.generation = 0
        self.init_population()
        pass

    def print_graph(self, labels=False):
        def hierarchy():
            """
            Reference: 
            
            https://stackoverflow.com/questions/29586520/can-one-get-hierarchical-graphs-from-networkx-with-python-3/29597209
            https://networkx.github.io/documentation/latest/reference/generated/networkx.drawing.nx_pylab.draw.html#networkx.drawing.nx_pylab.draw
            """
            pos = {}
            row, col, last_len = 0, 0, 1
            max_col, max_row = 1, 1
            for name in self.graph.nodes():
                if len(name) > last_len:
                    last_len = len(name)
                    row += 1
                    col = 0
                x = row
                y = col
                pos[name] = (x, y)
                col += 1
                max_col = max(max_col, col)
                max_row = max(max_row, row)
            return pos, max_row, max_col

        pos, max_row, max_col = hierarchy()
        plt.figure(figsize=(max_row, max_col / 2))
        nx.draw(self.graph, pos=pos, node_color="paleturquoise", edge_color="gray")
        if labels:
            text = nx.draw_networkx_labels(self.graph, pos)
            for _, t in text.items():
                t.set_rotation(25)
        plt.savefig("family_tree.png")

    def init_population(self):
        """Init population of NNs according to hyper parameters."""
        if self.population_size > 52:
            raise Warning(
                "Visualization for population sizes greater than 52 is currently unsupported."
            )
        self.population = []
        for index in range(self.population_size):
            name = ALPHABET[index]
            self.population.append(
                neural_network.NeuralNetwork(
                    self.input_size,
                    self.hidden_layer_size,
                    self.output_size,
                    self.learning_rate,
                    epochs=self.epochs,
                    name=name,
                    verbose=self.verbose,
                )
            )
            self.graph.add_node(name)
        # self.print_graph()

    def select(self, test_x, test_y):
        """Using cases, apply lexicase selection to population."""

        def check_case(case, estimator):
            """Return score for estimator on case."""
            if case == "mse":
                # predict y_hat using test_x
                y_hat = estimator.predict(test_x)
                # compute a diff with test_y and y_hat
                mse = np.mean((test_y - y_hat) ** 2)
                # updating best model
                if mse < self.best_mse:
                    self.best_model = estimator
                    self.best_mse = mse
                return mse
            elif case == "l1":
                weights = estimator.get_weights()
                w1 = weights[W1_INDEX]
                w2 = weights[W2_INDEX]
                # compute l2 norm with the weight matricies
                l2 = np.linalg.norm(w1, ord=1) + np.linalg.norm(w2, ord=1)
                return l2
            elif case == "l2":
                weights = estimator.get_weights()
                w1 = weights[W1_INDEX]
                w2 = weights[W2_INDEX]
                # compute l2 norm with the weight matricies
                l2 = np.linalg.norm(w1, ord=2) + np.linalg.norm(w2, ord=2)
                return l2
            elif case == "time":
                return float("inf")

        selected = []
        while len(selected) < self.selection_size:
            pool = set(self.population) - set(selected)
            case_bests = []  # to randomly pick from later
            for case in self.cases:
                best = {}
                for estimator in pool:
                    score = check_case(case, estimator)
                    best[str(score)] = estimator
                key = sorted(best.keys())[0]
                case_bests.append(best[key])
            random_pick = np.random.randint(0, len(case_bests))
            selected.append(case_bests[random_pick])
        self.population = selected
        # now recombine

    def mutate(self, train_x, train_y):
        """Apply mutation to population, or subset passed."""
        for estimator in self.population:  # for each selected estimator
            if self.hybrid:  # hybrid GA
                estimator.fit(train_x, train_y)  # SGD
            else:  # normal GA
                weights = estimator.get_weights()
                # BUG: assuming mutable
                for matrix in weights:
                    noise = np.random.normal(
                        loc=0.0, scale=self.scale_of_mutation, size=matrix.shape
                    )
                    matrix += noise
                estimator.set_weights(weights)

    def recombine(self):
        """Recombine the passed subset of the population."""
        children = []
        child_number = 0
        num_parents = len(self.population)
        for _ in range(self.population_size):
            first, second = np.random.choice(num_parents, 2, replace=False)
            left_parent = self.population[first]
            right_parent = self.population[second]
            l_name = left_parent.name
            r_name = right_parent.name
            left = left_parent.get_weights()
            right = right_parent.get_weights()
            child = []
            for matrix in range(4):  # hardcoding 1 hidden layer
                # dealing with weight matrix
                w_l = left[matrix].T
                w_r = right[matrix].T
                height = w_l.shape[0]
                assert w_l.shape == w_r.shape
                # how many rows come from left
                split = int(np.random.uniform(0, height))
                # randomly select rows
                indices = np.random.choice(height, split, replace=False)
                child_matrix = []
                for row in range(height):
                    if row in indices:
                        child_matrix.append(w_l[row])
                    else:
                        child_matrix.append(w_r[row])
                child_matrix = np.array(child_matrix).T
                child.append(child_matrix)
            name = l_name + r_name + str(child_number)
            children.append((child, name, l_name, r_name))
            child_number += 1

        new_population = []
        for weights, name, l_name, r_name in children:
            # if l_name == r_name:
            #     raise ValueError("No Cloning!")
            new_population.append(
                neural_network.NeuralNetwork(
                    self.input_size,
                    self.hidden_layer_size,
                    self.output_size,
                    self.learning_rate,
                    weights=weights,
                    epochs=self.epochs,
                    name=name,
                    verbose=self.verbose,
                )
            )
            self.graph.add_node(name)
            self.graph.add_edges_from([(l_name, name), (r_name, name)])
        self.population = new_population
        self.generation += 1
        self.print_graph(False)

    def fit(self, train_x, train_y):
        """Run the algorithm."""
        # init population of N networks, run = 0
        #     do
        #     evaluate all N networks
        #     select the K best parents using Lexicase
        #     run SGD as mutation on selected parents
        #     save parent with lowest validation error
        #     recombine to produce N new network
        #         while run < generation limit
        #     return best saved
        height = train_x.shape[0]
        print(height)
        split = int(np.ceil(height / 5))
        indices = np.random.choice(height, split, replace=False)
        test_x, test_y, X, y = [], [], [], []
        for row in range(height):
            if row in indices:  # in test set
                test_x.append(train_x[row])
                test_y.append(train_y[row])
            else:  # in train set
                X.append(train_x[row])
                y.append(train_y[row])
        test_x, test_y, X, y = (
            np.array(test_x),
            np.array(test_y),
            np.array(X),
            np.array(y),
        )
        print(test_x.shape, test_y.shape)
        print(X.shape, y.shape)
        self.init_population()
        for gen_index in range(self.generations):
            print("Generation:", gen_index)
            print("Best Model MSE:", self.best_mse)
            self.select(test_x, test_y)
            self.mutate(X, y)
            self.recombine()

    def predict(self, X):
        return self.best_model.predict(X)

    def get_params(self, deep=False):
        """Return the params dictionary."""
        params = {
            "hybrid": self.hybrid,
            "input_size": self.input_size,
            "hidden_layer_size": self.hidden_layer_size,
            "output_size": self.output_size,
            "population_size": self.population_size,
            "selection_size": self.selection_size,
            "learning_rate": self.learning_rate,
            "cases": self.cases,
            "epochs": self.epochs,
            "generations": self.generations,
            "verbose": self.verbose,
            # "best_model": self.best_model,
            # "population": self.population,
        }
        return params

    def set_params(self, **params):
        """Set params dictionary."""
        self.hybrid = params["hybrid"]
        self.input_size = params["input_size"]
        self.hidden_layer_size = params["hidden_layer_size"]
        self.output_size = params["output_size"]
        self.population_size = params["population_size"]
        self.selection_size = params["selection_size"]
        self.learning_rate = params["learning_rate"]
        self.cases = params["cases"]
        self.epochs = params["epochs"]
        self.generations = params["generations"]
        self.verbose = params["verbose"]
        # self.best_model = params["best_model"]
        # self.population = params["population"]
        self.best_model = None
        self.best_mse = float("inf")
        self.population = []
        self.init_population()

    def write(self):
        pass

    def visualize(self):
        pass


if __name__ == "__main__":
    import main

    data = main.load_dataset("data/ripple_0.0_50_200")
    # init ga
    input_size = data.shape[1] - 1
    hidden_layer_size = 5
    output_size = 1
    population_size = 10
    selection_size = 4
    learning_rate = 1e-3
    epochs = 10
    generations = 10
    estimator = GeneticAlgorithm(
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
    X, y = main.split_data(data)
    estimator.fit(X, y)
    print(estimator)
    import pickle

    with open("test", "wb") as f:
        pickle.dump(estimator, f)

