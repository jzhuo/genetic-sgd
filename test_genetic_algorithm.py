import unittest
import numpy as np
from genetic_algorithm import GeneticAlgorithm


class TestGeneticAlgorithm(unittest.TestCase):
    
    def __init__(self):
        # initializing variables for self.ga
        self.expected_input_size = 2
        self.expected_hidden_size = 5
        self.expected_output_size = 1
        self.population_size = 10
        self.selection_size = 2
        self.learning_rate = 1e-3
        self.epochs = 5
        self.generations = 2
        # self.cases = ["mse", "l2"]
        # self.verbose = 1 

        self.ga = GeneticAlgorithm(
            False,
            self.expected_input_size,
            self.expected_hidden_size,
            self.expected_output_size,
            self.population_size, 
            self.selection_size,
            self.learning_rate,
            self.epochs,
            self.generations,
        )
        self.hga = GeneticAlgorithm(
            True,
            self.expected_input_size,
            self.expected_hidden_size,
            self.expected_output_size,
            self.population_size, 
            self.selection_size,
            self.learning_rate,
            self.epochs,
            self.generations,
        )

    def population_init(self):
        """Verify ga.population is initialized properly."""
        self.assertEqual(self.population_size, len(self.ga.population))
        pass

    def lexicase_selection(self):
        """Verify lexicase selection on cases occurs properly."""
        self.ga.select()
        self.assertEqual(self.selection_size, len(self.ga.population))
        pass

    def mutation(self):
        """Verify that mutation changes weights."""
        for index in range(len(self.ga.population)):
            prev_weights = self.ga.population[index].get_weights()
            self.ga.mutate()
            self.assertNotEqual(prev_weights, self.ga.population[index].get_weights)
        pass

    def recombination(self):
        """Verify recombination occurs at proper split points."""
        self.assertEqual(self.selection_size, len(self.ga.population))
        # sum of weights calculated from first 
        weight_sum = 0
        for pop in range(len(self.ga.population)):
            weight_sum += np.sum(self.ga.population[pop].get_weights())
        self.ga.recombine()
        # tests for population size
        self.assertEqual(self.population_size, len(self.ga.population))
        # tests for sum of weights post recombination
        for index in range(len(self.ga.population)/2):
            new_weight_sum = (np.sum(self.ga.population[index].get_weights())
                                + np.sum(self.ga.population[index+1].get_weights()))
            self.assertEqual(weight_sum, new_weight_sum)
        pass


if __name__ == "__main__":
    unittest.main()
