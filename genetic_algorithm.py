"""
This file contains the class that provides the genetic algorithm search. 
"""


class GeneticAlgorithm:
    def __init__(self, population_size, sgd_mutation, cases=["mse", "l2"]):
        self.population_size = population_size
        self.sgd_mutation = sgd_mutation
        self.population = self.init_population()
        self.cases = cases
        pass

    def init_population(self):
        """Init population of NNs according to hyper parameters."""

        # TODO: Init population of NNs

        return [NotImplemented]

    def select(self):
        """Using cases, apply lexicase selection to population."""

        # TODO: Modify self.population according to lexicase args

        raise NotImplementedError

    def mutate(self, subset_indices=None):
        """Apply mutation to population, or subset passed."""
        pop = (
            self.population[subset_indices]
            if subset_indices is not None
            else self.population
        )

        # TODO: mutate the population (or passed, selected subset)

        raise NotImplementedError

    def recombine(self, subset_indices=None):
        """Recombine the passed subset of the population."""
        parents = (
            self.population[subset_indices]
            if subset_indices is not None
            else self.population
        )

        # TODO: recombine parents to produce new population of population_size

        raise NotImplementedError

    def run_algorithm(self):
        """Run the algorithm."""
        raise NotImplementedError
