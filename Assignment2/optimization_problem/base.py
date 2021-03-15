import os

import six
import sys

sys.modules['sklearn.externals.six'] = six
import mlrose_hiive
import numpy as np
from matplotlib import pyplot as plt
from functools import wraps

import algorithm.optimization_algorithms as oa
from algorithm import parameters_rhc
from algorithm import parameters_sa
from algorithm import parameters_ga
from algorithm import parameters_mimic
import pandas as pd

from pathlib import Path

"""
todo 
    - Store fitness curve for each algorithm and setting into file  
    - Store duration for each algorithm and size in file 
    - Store best fitness value for each algorithm and size into file  

    output/algorithm/run_x/fitness_curve.csv (settings in each column) x - iteration, y - fitness curve 
    output/algorithm/run_x/best_fitness.csv (settings in each column)  x - size, y - best fitness per setting   
    output/algorithm/run_x/duration.csv x - size, y - duration per setting  

    algorithm, run, size, settings, duration, best_fitness

    algorithm, run, size(=100), iteration, fitness curve (setting)      
"""


class BaseOptimizationProblem:
    def __init__(self, output_directory):
        self.output_directory = output_directory
        self.optimization_problem = type(self).__name__

    def set_random_seed_for_run(self, run):
        np.random.seed(run)

    def get_problem(self, size):
        pass

    def setup_random_seed(self, run):
        random_value = run
        self.set_random_seed_for_run(random_value)
        print(f"run {run}")
        return random_value

    def get_algorithm_results(self, algorithm_name, size, max_attempt, max_iterations, output_directory, runs):
        best_fitnesss, fitness_curves, durations = np.array([]), np.array([]), np.array([])
        run_fitness_curve_full_length = pd.DataFrame(index=list(range(1, max_iterations + 1)))

        best_fitness_columns = ['algorithm_name', 'size', 'settings', 'best_fitness', 'duration']
        fitness_columns_curve = ['algorithm_name', 'size']
        df_best_fitness_data = pd.DataFrame(columns=best_fitness_columns)

        problem = self.get_problem(size)
        if algorithm_name == 'random_hill_climb':
            setting = 'default'
            fitness_columns_curve.append(setting)
            df_fitness_curve = pd.DataFrame(columns=fitness_columns_curve)
            for run in range(runs):
                random_value = self.setup_random_seed(run)
                run_best_state, run_best_fitness, run_fitness_curve, run_duration = oa.random_hill_climb(problem,
                                                                                                         max_attempt,
                                                                                                         max_iterations,
                                                                                                         random_value)

                best_fitnesss = np.append(best_fitnesss, run_best_fitness)
                durations = np.append(durations, run_duration)

                if run_fitness_curve_full_length.empty:
                    run_fitness_curve_full_length.loc[0:len(run_fitness_curve), 0] = run_fitness_curve[:, 0]
                    run_fitness_curve_full_length.ffill(inplace=True)
                else:
                    run_fitness_curve_full_lengthx = pd.DataFrame(index=list(range(1, max_iterations + 1)))
                    run_fitness_curve_full_lengthx.loc[0:len(run_fitness_curve), 0] = run_fitness_curve[:, 0]
                    run_fitness_curve_full_lengthx.ffill(inplace=True)
                    run_fitness_curve_full_length += run_fitness_curve_full_lengthx

            best_fitness, duration = round(np.average(best_fitnesss), 1), round(np.average(durations), 7)
            run_best_fitness_info = pd.DataFrame([(algorithm_name, size, setting, best_fitness, duration)], columns=best_fitness_columns)

            df_best_fitness_data = df_best_fitness_data.append(run_best_fitness_info, ignore_index=True)

            run_fitness_curve_full_length = round((run_fitness_curve_full_length / runs), 0)

            df_fitness_curve[setting] = run_fitness_curve_full_length.values.flatten()
            df_fitness_curve['algorithm_name'] = algorithm_name
            df_fitness_curve['size'] = size

        elif algorithm_name == 'simulated_annealing':
            ces = parameters_sa.ce
            fitness_columns_curve.extend([f"ce={x}" for x in ces])
            df_fitness_curve = pd.DataFrame(columns=fitness_columns_curve)
            for ce in ces:
                setting = f"ce={ce}"
                for run in range(runs):
                    random_value = self.setup_random_seed(run)
                    schedule = mlrose_hiive.GeomDecay(decay=ce)
                    run_best_state, run_best_fitness, run_fitness_curve, run_duration = oa.simulated_annealing(problem,
                                                                                                               max_attempt,
                                                                                                               max_iterations,
                                                                                                               schedule,
                                                                                                               random_value)
                    best_fitnesss = np.append(best_fitnesss, run_best_fitness)
                    durations = np.append(durations, run_duration)

                    if run_fitness_curve_full_length.empty:
                        run_fitness_curve_full_length.loc[0:len(run_fitness_curve), 0] = run_fitness_curve[:, 0]
                        run_fitness_curve_full_length.ffill(inplace=True)
                    else:
                        run_fitness_curve_full_lengthx = pd.DataFrame(index=list(range(1, max_iterations + 1)))
                        run_fitness_curve_full_lengthx.loc[0:len(run_fitness_curve), 0] = run_fitness_curve[:, 0]
                        run_fitness_curve_full_lengthx.ffill(inplace=True)
                        run_fitness_curve_full_length += run_fitness_curve_full_lengthx

                best_fitness, duration = round(np.average(best_fitnesss), 1), round(np.average(durations), 7)
                run_best_fitness_info = pd.DataFrame([(algorithm_name, size, setting, best_fitness, duration)], columns=best_fitness_columns)
                df_best_fitness_data = df_best_fitness_data.append(run_best_fitness_info, ignore_index=True)
                run_fitness_curve_full_length = round((run_fitness_curve_full_length / runs), 0)

                df_fitness_curve[setting] = run_fitness_curve_full_length.values.flatten()
                df_fitness_curve['algorithm_name'] = algorithm_name
                df_fitness_curve['size'] = size

        elif algorithm_name == 'genetic_algorithm':
            pop_sizes = parameters_ga.pop_sizes
            pop_breed_percent = parameters_ga.pop_breed_percents
            mutation_probs = parameters_ga.mutation_probs
            params = np.array(np.meshgrid(pop_sizes, pop_breed_percent, mutation_probs)).T.reshape(-1, 3)

            fitness_columns_curve.extend([f"mutation_probs={x}" for x in mutation_probs])
            df_fitness_curve = pd.DataFrame(columns=fitness_columns_curve)

            for pop_size, pop_breed_percent, mutation_prob in params:
                setting = f"mutation_probs={mutation_prob}"
                for run in range(runs):
                    random_value = self.setup_random_seed(run)
                    print(pop_size, pop_breed_percent, mutation_prob)
                    run_best_state, run_best_fitness, run_fitness_curve, run_duration = oa.genetic_alg(problem,
                                                                                                       max_attempt,
                                                                                                       max_iterations,
                                                                                                       pop_size,
                                                                                                       pop_breed_percent,
                                                                                                       mutation_prob,
                                                                                                       random_value)
                    best_fitnesss = np.append(best_fitnesss, run_best_fitness)
                    durations = np.append(durations, run_duration)

                    if run_fitness_curve_full_length.empty:
                        run_fitness_curve_full_length.loc[0:len(run_fitness_curve), 0] = run_fitness_curve[:, 0]
                        run_fitness_curve_full_length.ffill(inplace=True)
                    else:
                        run_fitness_curve_full_lengthx = pd.DataFrame(index=list(range(1, max_iterations + 1)))
                        run_fitness_curve_full_lengthx.loc[0:len(run_fitness_curve), 0] = run_fitness_curve[:, 0]
                        run_fitness_curve_full_lengthx.ffill(inplace=True)
                        run_fitness_curve_full_length += run_fitness_curve_full_lengthx

                best_fitness, duration = round(np.average(best_fitnesss), 1), round(np.average(durations), 7)
                run_best_fitness_info = pd.DataFrame([(algorithm_name, size, setting, best_fitness, duration)],
                                                     columns=best_fitness_columns)
                df_best_fitness_data = df_best_fitness_data.append(run_best_fitness_info, ignore_index=True)
                run_fitness_curve_full_length = round((run_fitness_curve_full_length / runs), 0)

                df_fitness_curve[setting] = run_fitness_curve_full_length.values.flatten()
                df_fitness_curve['algorithm_name'] = algorithm_name
                df_fitness_curve['size'] = size
        elif algorithm_name == 'mimic':
            pop_sizes = parameters_mimic.pop_sizes
            keep_pcts = parameters_mimic.keep_pcts

            params = np.array(np.meshgrid(pop_sizes, keep_pcts)).T.reshape(-1, 2)

            fitness_columns_curve.extend([f"keep_pct={x}" for x in keep_pcts])
            df_fitness_curve = pd.DataFrame(columns=fitness_columns_curve)
            for pop_size, keep_pct in params:
                setting = f"keep_pct={keep_pct}"
                for run in range(runs):
                    random_value = self.setup_random_seed(run)
                    print(pop_size, keep_pct)
                    run_best_state, run_best_fitness, run_fitness_curve, run_duration = oa.mimic(problem,
                                                                                                 max_attempt,
                                                                                                 max_iterations,
                                                                                                 pop_size,
                                                                                                 keep_pct,
                                                                                                 random_value)
                    best_fitnesss = np.append(best_fitnesss, run_best_fitness)
                    durations = np.append(durations, run_duration)

                    if run_fitness_curve_full_length.empty:
                        run_fitness_curve_full_length.loc[0:len(run_fitness_curve), 0] = run_fitness_curve[:, 0]
                        run_fitness_curve_full_length.ffill(inplace=True)
                    else:
                        run_fitness_curve_full_lengthx = pd.DataFrame(index=list(range(1, max_iterations + 1)))
                        run_fitness_curve_full_lengthx.loc[0:len(run_fitness_curve), 0] = run_fitness_curve[:, 0]
                        run_fitness_curve_full_lengthx.ffill(inplace=True)
                        run_fitness_curve_full_length += run_fitness_curve_full_lengthx

                best_fitness, duration = round(np.average(best_fitnesss), 1), round(np.average(durations), 7)
                run_best_fitness_info = pd.DataFrame([(algorithm_name, size, setting, best_fitness, duration)],
                                                     columns=best_fitness_columns)
                df_best_fitness_data = df_best_fitness_data.append(run_best_fitness_info, ignore_index=True)
                run_fitness_curve_full_length = round((run_fitness_curve_full_length / runs), 0)

                df_fitness_curve[setting] = run_fitness_curve_full_length.values.flatten()
                df_fitness_curve['algorithm_name'] = algorithm_name
                df_fitness_curve['size'] = size
        return df_best_fitness_data, df_fitness_curve

    def init_results(self, algorithms):
        results = {}
        for algorithm in algorithms:
            results[algorithm] = {'fitness_curve': [], 'best_fitness': [], 'duration': []}
        return results

    def run_experiments(self, algorithms, output_directory, runs=3):
        self.output_directory = output_directory

        max_iterations = 2000
        max_size = 100
        max_attempt = 100
        sizes = range(10, 110, 10)

        for algorithm in algorithms:
            run_best_fitness_columns = ['algorithm_name', 'size', 'settings', 'best_fitness', 'duration']
            algorithm_best_fitness_data = pd.DataFrame(columns=run_best_fitness_columns)

            algorithm_fitness_curve_data = pd.DataFrame()
            for size in sizes:
                print(f"size: {size}, max_attempts: {max_attempt}")
                df_best_fitness_data, df_fitness_curve = self.get_algorithm_results(algorithm,
                                                                                    size,
                                                                                    max_attempt,
                                                                                    max_iterations,
                                                                                    output_directory,
                                                                                    runs)

                algorithm_best_fitness_data = algorithm_best_fitness_data.append(df_best_fitness_data,
                                                                                 ignore_index=True)

                if size == max_size:
                    if not algorithm_fitness_curve_data.empty:
                        algorithm_fitness_curve_data = df_fitness_curve.append(df_fitness_curve, ignore_index=True)
                    else:
                        algorithm_fitness_curve_data = df_fitness_curve

            self.algorithm_directory = rf"{output_directory}\{self.optimization_problem}\{algorithm}"
            Path( self.algorithm_directory).mkdir(parents=True, exist_ok=True)
            filepath_best_fitness_data = rf"{ self.algorithm_directory}\best_fitness_data.csv"
            filepath_fitness_curve = rf"{ self.algorithm_directory}\fitness_curve_data.csv"
            algorithm_best_fitness_data.to_csv(path_or_buf=filepath_best_fitness_data, index=False)
            algorithm_fitness_curve_data.to_csv(path_or_buf=filepath_fitness_curve, index=False)

            print(algorithm_best_fitness_data)
            print(algorithm_fitness_curve_data)

    def plot_algorithm_data(self):
        # todo plot fitness vs iteration for each hyperparam in the same plot
        # todo plot time vs size for each hyperparam in the same plot
        # todo plot fitness params for each hyperparam in the same plot
        optimization_problem_directories = [f.path for f in os.scandir(self.output_directory) if f.is_dir()] #[x[0] for x in os.walk(self.algorithm_directory)]
        print(optimization_problem_directories)

        for optimization_problem_directory in optimization_problem_directories:
            optimization_problem = optimization_problem_directory.split("\\")[1]
            print(optimization_problem)

            algorithm_directories = [f.path for f in os.scandir(self.output_directory + "\\" + optimization_problem) if f.is_dir()]

            print(algorithm_directories)

            for algorithm_directory in algorithm_directories:
                algorithm = algorithm_directory.split("\\")[2]

                directory = rf"{self.output_directory}\{optimization_problem}\{algorithm}"
                filepath_best_fitness_data = rf"{directory}\best_fitness_data.csv"
                filepath_fitness_curve = rf"{directory}\fitness_curve_data.csv"

                algorithm_best_fitness_data = pd.read_csv(filepath_best_fitness_data, sep=',')
                columns_to_exclude = ['algorithm_name']
                columns = algorithm_best_fitness_data.columns
                columns = [column for column in columns if column not in columns_to_exclude]
                algorithm_best_fitness_data = algorithm_best_fitness_data[columns]

                best_fitnesses = algorithm_best_fitness_data.pivot_table('best_fitness', ['size'], 'settings')
                print(best_fitnesses)
                ax = best_fitnesses.plot(title=f'{optimization_problem} - {algorithm}')
                ax.set_xlabel("Size")
                ax.set_ylabel("Fitness")
                fig = ax.get_figure()
                fig.savefig(rf"{directory}\fitness_vs_size_curve.png")

                durations = algorithm_best_fitness_data.pivot_table('duration', ['size'], 'settings')
                print(durations)
                ax = durations.plot(title=f'{optimization_problem} - {algorithm}')
                ax.set_xlabel("Size")
                ax.set_ylabel("Duration (sec)")
                fig = ax.get_figure()
                fig.savefig(rf"{directory}\duration_vs_size_curve.png")

                algorithm_fitness_curve_data = pd.read_csv(filepath_fitness_curve, sep=',')
                columns_to_exclude = ['algorithm_name', 'size']
                columns = algorithm_fitness_curve_data.columns
                columns = [column for column in columns if column not in columns_to_exclude]
                algorithm_fitness_curve_data = algorithm_fitness_curve_data[columns]
                ax = algorithm_fitness_curve_data.plot(title=f'{optimization_problem} - {algorithm}')
                ax.set_xlabel("Iterations")
                ax.set_ylabel("Fitness")

                fig = ax.get_figure()
                fig.savefig(rf"{directory}\fitness_curve.png")

    def plot_all_algorithm_data_for_comparison(self):
        # todo plot fitness vs iteration for each algorithm optimal param in the same plot
        # todo plot time vs size for each algorithm optimal param in the same plot
        # todo plot fitness params for each algorithm optimal param in the same plot
        optimization_problem_directories = [f.path for f in os.scandir(self.output_directory) if
                                            f.is_dir()]
        print(optimization_problem_directories)

        algorithm_optimals = {
            'random_hill_climb': 'default',
            'simulated_annealing': 'ce=0.9',
            'genetic_algorithm': 'mutation_probs=0.9',
            'mimic': 'keep_pct=0.9'
        }

        algorithms = list(algorithm_optimals.keys())
        all_algorithms_best_fitness = pd.DataFrame(columns=algorithms)
        all_algorithms_durations = pd.DataFrame(columns=algorithms)
        all_algorithms_fitness_curves = pd.DataFrame(columns=algorithms)

        for optimization_problem_directory in optimization_problem_directories:
            optimization_problem = optimization_problem_directory.split("\\")[1]
            print(optimization_problem)

            algorithm_directories = [f.path for f in os.scandir(self.output_directory + "\\" + optimization_problem) if
                                     f.is_dir()]

            print(algorithm_directories)

            for algorithm_directory in algorithm_directories:
                algorithm = algorithm_directory.split("\\")[2]

                directory = rf"{self.output_directory}\{optimization_problem}\{algorithm}"
                filepath_best_fitness_data = rf"{directory}\best_fitness_data.csv"
                filepath_fitness_curve = rf"{directory}\fitness_curve_data.csv"

                algorithm_best_fitness_data = pd.read_csv(filepath_best_fitness_data, sep=',')
                columns_to_exclude = ['algorithm_name']
                columns = algorithm_best_fitness_data.columns
                columns = [column for column in columns if column not in columns_to_exclude]
                algorithm_best_fitness_data = algorithm_best_fitness_data[columns]

                best_fitnesses = algorithm_best_fitness_data.pivot_table('best_fitness', ['size'], 'settings')
                all_algorithms_best_fitness[algorithm] = best_fitnesses[algorithm_optimals[algorithm]]


                durations = algorithm_best_fitness_data.pivot_table('duration', ['size'], 'settings')
                all_algorithms_durations[algorithm] = durations[algorithm_optimals[algorithm]]

                algorithm_fitness_curve_data = pd.read_csv(filepath_fitness_curve, sep=',')
                columns_to_exclude = ['algorithm_name', 'size']
                columns = algorithm_fitness_curve_data.columns
                columns = [column for column in columns if column not in columns_to_exclude]
                algorithm_fitness_curve_data = algorithm_fitness_curve_data[columns]
                all_algorithms_fitness_curves[algorithm] = algorithm_fitness_curve_data[algorithm_optimals[algorithm]]

            print('all_algorithms_best_fitness')
            print(all_algorithms_best_fitness)
            print('all_algorithms_durations')
            print(all_algorithms_durations)
            print('all_algorithms_fitness_curves')
            print(all_algorithms_fitness_curves)

            directory = rf"{self.output_directory}\{optimization_problem}"
            ax = all_algorithms_best_fitness.plot(title=f'Fitness vs Size')
            ax.set_xlabel("Size")
            ax.set_ylabel("Fitness")
            fig = ax.get_figure()
            fig.savefig(rf"{directory}\0-all_fitness_vs_size_curve.png")

            ax = all_algorithms_durations.plot(title=f'Duration vs Size')
            ax.set_xlabel("Size")
            ax.set_ylabel("Duration (sec)")
            fig = ax.get_figure()
            fig.savefig(rf"{directory}\1-all_duration_vs_size_curve.png")

            ax = all_algorithms_fitness_curves.plot(title=f'Fitness vs Iterations')
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Fitness")
            fig = ax.get_figure()
            fig.savefig(rf"{directory}\2-all_fitness_curve.png")
