import six
import sys

sys.modules['sklearn.externals.six'] = six
import mlrose_hiive
from optimization_problem import base
import numpy as np


class FourPeaks(base.BaseOptimizationProblem):
    def __init__(self, output_directory):
        super().__init__(output_directory)
        self.output_directory = output_directory
        self.optimization_problem = type(self).__name__

    def get_problem(self, size):
        fitness = mlrose_hiive.FourPeaks(t_pct=0.25)
        # #state = [[0 for x in range(2)] for x in range(size)]
        # state = np.array([1, 0])
        # state = np.tile(state, int(size/2))
        #
        # # state = [1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0]
        #
        # print('length:',  len(state))
        #
        # fitness.evaluate(state)
        problem = mlrose_hiive.DiscreteOpt(length=size, fitness_fn=fitness, maximize=True)
        return problem



