"""
This file contains the class that provides the genetic algorithm search. 
"""


class GeneticAlgorithm:
    def __init__(
        self, population_size, learning_rate, cases=["mse", "l2"], verbose=1
    ):
        """
        If learning rate is 0, the algorithm is just regular mutation.
        """
        self.population_size = population_size
        self.population = self.init_population()
        self.learning_rate = learning_rate
        self.cases = cases
        self.verbose = verbose
        pass

    def init_population(self):
        """Init population of NNs according to hyper parameters."""

        # TODO: Init population of NNs to self.population

        return [NotImplemented]

    def select(self):
        """Using cases, apply lexicase selection to population."""

        # TODO: Modify self.population according to lexicase args

        raise NotImplementedError

    def mutate(self, subset_indices=None):
        """Apply mutation to population, or subset passed."""
        indices = (
            subset_indices
            if subset_indices is not None
            else range(0, len(self.population))
        )
        # TODO: mutate the population (or passed, selected subset)
        # reassign pop to self.population

        # NOTE: model.get_weights() gives list of matrices
        # NOTE: model.set_weights(weights) assigns the passed weights

        raise NotImplementedError

    def recombine(self, subset_indices=None):
        """Recombine the passed subset of the population."""
        parents = (
            self.population[subset_indices]
            if subset_indices is not None
            else self.population
        )

        # TODO: recombine parents to produce new population of population_size
        # NOTE: model.get_weights() gives list of matrices
        # NOTE: model.set_weights(weights) assigns the passed weights
        # reassign to self.population

        raise NotImplementedError

    def fit(self, train_x, train_y):
        """Run the algorithm."""

        # TODO: implement the full algorithm here dude

        raise NotImplementedError
