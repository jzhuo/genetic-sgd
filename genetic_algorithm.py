"""
This file contains the class that provides the genetic algorithm search. 
"""
import neural_network
import numpy as np


class GeneticAlgorithm:
    def __init__(
        self,
        hybrid,
        input_size,
        hidden_layer_size,
        output_size,
        population_size,
        selection_size,
        learning_rate,
        epochs,
        generations,
        cases=["mse", "l2", "l1", "time"],
        verbose=1,
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
        self.verbose = verbose
        self.best_model = None
        self.population = []
        pass

    def init_population(self):
        """Init population of NNs according to hyper parameters."""

        self.population = []

        for _ in self.population_size:
            self.population.append(
                neural_network.build_nn(
                    self.input_size,
                    self.hidden_layer_size,
                    self.output_size,
                    self.learning_rate,
                )
            )

    def select(self, test_x, test_y):
        """Using cases, apply lexicase selection to population."""

        def check_case(case, estimator):
            """Return score for estimator on case."""
            if case == "mse":
                # predict y_hat using test_x
                y_hat = estimator.predict(test_x)
                # compute a diff with test_y and y_hat
                mse = (test_y - y_hat) ** 2
                return mse
            elif case == "l2":
                w1, w2 = estimator.get_weights()
                # compute l2 norm with the weight matricies
                l2 = np.linalg.norm(w1, ord=2) + np.linalg.norm(w2, ord=2)
                return l2
            elif case == "l1":
                w1, w2 = estimator.get_weights()
                # compute l2 norm with the weight matricies
                l2 = np.linalg.norm(w1, ord=1) + np.linalg.norm(w2, ord=1)
                return l2

        selected = []
        num_cases = len(self.cases)
        num_to_select_per_case = (
            self.selection_size - self.selection_size % num_cases
        ) / num_cases

        for case in self.cases:
            best = []
            for estimator in self.population:
                score = check_case(case, estimator)
                best.append((score, estimator))
            best.sort(key=lambda x: x[0])  # robust
            selected += best[:num_to_select_per_case]

        models_to_keep = [tup[1] for tup in selected]
        self.population = models_to_keep

    def mutate(self, train_x, train_y):
        """Apply mutation to population, or subset passed."""
        for estimator in self.population:
            if self.hybrid:
                estimator.fit(train_x, train_y, epochs=self.epochs)
            else:
                weights = estimator.get_weights()
                # BUG: assuming mutable
                for matrix in weights:
                    noise = np.random.normal(
                        loc=0.0, scale=1.0, size=matrix.shape
                    )
                    matrix += noise
                estimator.set_weights(weights)

    def recombine(self):
        """Recombine the passed subset of the population."""
        children = []
        num_parents = len(self.population)
        for _ in range(self.population_size):
            first, second = np.random.randint(0, num_parents, 2)
            left = self.population[first]
            right = self.population[second]
            left = left.get_weights()
            right = right.get_weights()
            child = []
            for matrix in [0, 1]:  # hardcoding 1 hidden layer
                height = left[matrix].shape[0]
                # how many rows come from left
                split = np.random.uniform(0, height)
                # randomly select rows
                indices = np.random.choice(height, split, replace=False)
                child_matrix = []
                for row in range(height):
                    if row in indices:
                        child_matrix.append(left[row])
                    else:
                        child_matrix.append(right[row])
                child.append(child_matrix)
            children.append(child)

        new_population = []
        for weights in children:
            new_population.append(
                neural_network.build_nn(
                    self.input_size,
                    self.hidden_layer_size,
                    self.output_size,
                    self.learning_rate,
                    weights=weights,
                )
            )
        self.population = new_population

    def fit(self, train_x, train_y, test_x=None, test_y=None):
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
        self.init_population()
        for _ in range(self.generations):
            self.select(test_x, test_y)
            self.mutate(train_x, train_y)
            self.recombine()

    def get_params(self):
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
            "best_model": self.best_model,
            "population": self.population,
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
        self.best_model = params["best_model"]
        self.population = params["population"]
