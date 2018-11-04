"""
This file contains the class that provides the genetic algorithm search. 
"""
import neural_network


class GeneticAlgorithm:
    def __init__(
        self,
        population_size, 
        selection_size,
        learning_rate,
        epochs,
        generations,
        cases=["mse", "l2"],
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
        self.population_size = population_size
        self.selection_size = selection_size # confused about the difference between this and pop size
        self.learning_rate = learning_rate
        self.cases = cases
        self.epochs = epochs
        self.generations = generations
        self.verbose = verbose
        self.best_model = None
        self.population = self.init_population()
        pass

    def init_population(self):
        """Init population of NNs according to hyper parameters."""

        # TODO: retrieve net size from data file
        sample_size = 7 # x values from data set
        hidden_size = 7 # assuming we will keep this the same size as input layer
        output_size = 1 # y value from data set?

        population = []

        for i in self.population_size:
            population.append(neural_network.build_NN(sample_size, hidden_size, output_size, self.learning_rate))

        return population

    def select(self):
        """Using cases, apply lexicase selection to population."""

        # TODO: Modify self.population according to lexicase args

        raise NotImplementedError

    def mutate(self):
        """Apply mutation to population, or subset passed."""
        # TODO: mutate the population

        # do SGD or a standard mutation of random add/subs from weights

        # NOTE: model.get_weights() gives list of matrices
        # NOTE: model.set_weights(weights) assigns the passed weights

        raise NotImplementedError

    def recombine(self):
        """Recombine the passed subset of the population."""

        # TODO: recombine parents to produce new population of population_size
        # NOTE: model.get_weights() gives list of matrices
        # NOTE: model.set_weights(weights) assigns the passed weights
        # reassign to self.population

        raise NotImplementedError

    def fit(self, train_x, train_y, test_x=None, test_y=None):
        """Run the algorithm."""

        # TODO: implement the full algorithm here

        raise NotImplementedError
