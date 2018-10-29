import unittest
from genetic_algorithm import GeneticAlgorithm


class TestGeneticAlgorithm(unittest.TestCase):
    def __init__(self):
        self.ga = GeneticAlgorithm(hybrid=False)
        self.hga = GeneticAlgorithm(hybrid=True)
        pass

    def population_init(self):
        """Verify ga.population is initialized properly."""
        pass

    def single_case_selection(self):
        """Verify lexicase selection on cases occurs properly."""
        pass

    def multiple_case_selection(self):
        """Verify lexicase selection on cases occurs properly."""
        pass

    def mutation(self):
        """Verify method for mutation successfully mutates."""
        pass

    def hybrid_mutation(self):
        """Verify hybrid mutation is SGD."""
        pass

    def recombination(self):
        """Verify recombination occurs at proper split points."""
        pass
