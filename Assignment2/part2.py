from optimization_problem.flipflop import FlipFlop
from optimization_problem.tsp import TSP
from optimization_problem.fourpeaks import FourPeaks

output_directory = "output"
algorithms = ['random_hill_climb', 'simulated_annealing', 'genetic_algorithm', 'mimic']


op = FlipFlop(output_directory)
op.run_experiments(algorithms, output_directory)
op.plot_algorithm_data()
op.plot_all_algorithm_data_for_comparison()


op = TSP(output_directory)
op.run_experiments(algorithms, output_directory)
op.plot_algorithm_data()
op.plot_all_algorithm_data_for_comparison()


op = FourPeaks(output_directory)
op.run_experiments(algorithms, output_directory)
op.plot_algorithm_data()
op.plot_all_algorithm_data_for_comparison()

