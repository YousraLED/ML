import six
import sys

sys.modules['sklearn.externals.six'] = six
import mlrose_hiive
from utils import timing
from time import time



@timing
def random_hill_climb(problem, max_attempts, max_iters, random_value):
    time_start = time()
    best_state, best_fitness, fitness_curve = mlrose_hiive.random_hill_climb(problem,
                                                                             max_attempts=max_attempts,
                                                                             max_iters=max_iters,
                                                                             curve=True,
                                                                             random_state=random_value)
    time_end = time()
    duration = round(time_end - time_start, 5)
    return best_state, best_fitness, fitness_curve, duration


@timing
def simulated_annealing(problem, max_attempts, max_iters, schedule, random_value):
    time_start = time()
    best_state, best_fitness, fitness_curve = mlrose_hiive.simulated_annealing(problem,
                                                                               schedule=schedule,
                                                                               max_attempts=max_attempts,
                                                                               max_iters=max_iters,
                                                                               curve=True,
                                                                               random_state=random_value)
    time_end = time()
    duration = round(time_end - time_start, 5)
    return best_state, best_fitness, fitness_curve, duration


@timing
def genetic_alg(problem, max_attempts, max_iters, pop_size, pop_breed_percent, mutation_prob, random_value):
    time_start = time()
    best_state, best_fitness, fitness_curve = mlrose_hiive.genetic_alg(problem,
                                                                       pop_size=pop_size,
                                                                       pop_breed_percent=pop_breed_percent,
                                                                       mutation_prob=mutation_prob,
                                                                       max_attempts=max_attempts,
                                                                       max_iters=max_iters,
                                                                       curve=True,
                                                                       random_state=random_value)
    time_end = time()
    duration = round(time_end - time_start, 5)
    return best_state, best_fitness, fitness_curve, duration


@timing
def mimic(problem, max_attempts, max_iters, pop_size, keep_pct, random_value):
    time_start = time()
    best_state, best_fitness, fitness_curve = mlrose_hiive.mimic(problem,
                                                                 pop_size=pop_size,
                                                                 keep_pct=keep_pct,
                                                                 max_attempts=max_attempts,
                                                                 max_iters=max_iters,
                                                                 curve=True,
                                                                 random_state=random_value)

    time_end = time()
    duration = round(time_end - time_start, 5)
    return best_state, best_fitness, fitness_curve, duration
