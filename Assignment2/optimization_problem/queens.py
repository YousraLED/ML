import random

import six
import sys

sys.modules['sklearn.externals.six'] = six
import mlrose_hiive
from optimization_problem import base
import numpy as np


class Queens(base.BaseOptimizationProblem):
    def __init__(self, output_directory):
        super().__init__(output_directory)
        self.output_directory = output_directory
        self.optimization_problem = type(self).__name__

    def get_problem(self, size):
        fitness = mlrose_hiive.Queens()
        # state = np.array(list(range(size)))
        # random.shuffle(state)
        problem = mlrose_hiive.DiscreteOpt(length=size, fitness_fn=fitness, maximize=True)
        return problem



