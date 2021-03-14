import random

import six
import sys

sys.modules['sklearn.externals.six'] = six
import mlrose_hiive
from optimization_problem import base


class TSP(base.BaseOptimizationProblem):
    def __init__(self, output_directory):
        super().__init__(output_directory)
        self.output_directory = output_directory
        self.optimization_problem = type(self).__name__

    def get_problem(self, size):
        coords = [[0 for x in range(2)] for x in range(size)]
        for i in range(0, len(coords)):
            coords[i][0] = random.randint(0, size)
            coords[i][1] = random.randint(0, size)

        fitness = mlrose_hiive.TravellingSales(coords=coords)

        # state = list(range(size))
        #
        # random.shuffle(state)
        #
        # fitness.evaluate(state)
        problem = mlrose_hiive.TSPOpt(length=size, fitness_fn=fitness, maximize=True)
        return problem



