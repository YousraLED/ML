from time import time

# import six
# import sys
#
# sys.modules['sklearn.externals.six'] = six

import mlrose
import numpy as np
from matplotlib import pyplot as plt
from utils import timing

np.random.seed(0)
random_value = 1


@timing
def random_hill_climb(problem):
    # Solve problem using simulated annealing
    time_start = time()
    best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(problem,
                                                                       max_attempts=100,
                                                                       max_iters=5000,
                                                                       curve=True,
                                                                       random_state=random_value)
    time_end = time()
    duration = round(time_end - time_start, 5)
    return best_state, best_fitness, fitness_curve, duration


@timing
def simulated_annealing(problem):
    # Define decay schedule
    schedule = mlrose.ExpDecay()
    # Solve problem using simulated annealing
    time_start = time()
    best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problem,
                                                                         schedule=schedule,
                                                                         max_attempts=100,
                                                                         max_iters=5000,
                                                                         curve=True,
                                                                         random_state=random_value)
    time_end = time()
    duration = round(time_end - time_start, 5)
    return best_state, best_fitness, fitness_curve, duration


@timing
def genetic_alg(problem):
    # Solve problem using genetic algorithm
    time_start = time()
    best_state, best_fitness, fitness_curve = mlrose.genetic_alg(problem,
                                                                 pop_size=200,
                                                                 mutation_prob=0.1,
                                                                 max_attempts=100,
                                                                 max_iters=5000,
                                                                 curve=True,
                                                                 random_state=random_value)
    time_end = time()
    duration = round(time_end - time_start, 5)
    return best_state, best_fitness, fitness_curve, duration


@timing
def mimic(problem):
    # Solve problem using genetic algorithm
    time_start = time()
    best_state, best_fitness, fitness_curve = mlrose.mimic(problem,
                                                           pop_size=200,
                                                           keep_pct=0.2,
                                                           max_attempts=100,
                                                           max_iters=5000,
                                                           curve=True,
                                                           random_state=random_value)
    time_end = time()
    duration = round(time_end - time_start, 5)
    return best_state, best_fitness, fitness_curve, duration


# # fitness = mlrose.OneMax()
# # fitness = mlrose.FlipFlop()
fitness = mlrose.FourPeaks(t_pct=0.15)

state = np.array([0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1,
                  0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1,
                  0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1,
                  0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1,
                  0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1,
                  0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1])
fitness_eval = fitness.evaluate(state)
print(fitness_eval)

# edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
# fitness = mlrose.MaxKColor(edges)
# state = np.array([0, 1, 0, 1, 1])
# fitness_eval = fitness.evaluate(state)

# fitness = mlrose.Queens()
# state = np.array([1, 4, 1, 3, 5, 5, 2, 7])

problem = mlrose.DiscreteOpt(length=len(state), fitness_fn=fitness, maximize=True)

best_state, best_fitness, fitness_curve = random_hill_climb(problem)
# print(best_state)
print(best_fitness)
plt.plot(fitness_curve)
plt.show()

best_state, best_fitness, fitness_curve = simulated_annealing(problem)
# print(best_state)
print(best_fitness)
plt.plot(fitness_curve)
plt.show()

best_state, best_fitness, fitness_curve = genetic_alg(problem)
# print(best_state)
print(best_fitness)
plt.plot(fitness_curve)
plt.show()

best_state, best_fitness, fitness_curve = mimic(problem)
# print(best_state)
print(best_fitness)
plt.plot(fitness_curve)
plt.show()