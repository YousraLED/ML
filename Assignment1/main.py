"""
    todo for each algorithm, run baseline experiment (default params for initial comparison)
        Each algorithm has the following:
            - Pre-processing
                - Data type conversion
                - Scaling
                - Feature selection
            -  Hyper-parameters
                - Default configuration
                - Tuned configuration
            - Evaluation
                - Training time
                - Prediction time
                - Training memory
                - Testing memory
                - Iterations till convergence
                - Prediction accuracy
"""
import json
import os
from pathlib import Path
from src.eda import EDA
from src.algorithms.boosting import AdaBoost
from src.algorithms.nerual_network import NeuralNetwork
from src.algorithms.decision_tree import DecisionTree
from src.algorithms.support_vector_machines import SupportVectorMachines
from src.algorithms.k_nearest_neighbor import KNearestNeighbor
import numpy as np

root_directory = os.path.dirname(__file__)
config_file_path = fr"{root_directory}\datasets\config_datasets.ini"  # "config_datasets.ini"  #
eda = EDA(config_file_path, root_directory)
eda.load_data_using_configuration()
datasets = eda.get_data()

test_size = 0.3

classifiers = [{'name': 'DecisionTree',
                'include': True,
                'classifier': DecisionTree,
                'apply_data_scaling': True,
                'apply_dimensionality_reduction': False},
               {'name': 'SupportVectorMachines',
                'include': True,
                'classifier': SupportVectorMachines,
                'apply_data_scaling': True,
                'apply_dimensionality_reduction': False},
               {'name': 'KNearestNeighbor',
                'include': True,
                'classifier': KNearestNeighbor,
                'apply_data_scaling': True,
                'apply_dimensionality_reduction': False},
               {'name': 'AdaBoost',
                'include': True,
                'classifier': AdaBoost,
                'apply_data_scaling': True,
                'apply_dimensionality_reduction': False},
               {'name': 'NeuralNetwork',
                'include': True,
                'classifier': NeuralNetwork,
                'apply_data_scaling': True,
                'apply_dimensionality_reduction': False}]


classifier_experiment_configurations = {
    "DecisionTree": {
        "plot_option": None,
        "params": {"max_depth": range(1, 30)}},
    "AdaBoost": {
        "plot_option": None,
        "params": {"n_estimators": np.linspace(100, 1000, num=10, endpoint=True).astype(np.int64)}},
    "SupportVectorMachines": {
        "plot_option": None,
        "params": {"C": 10.0 ** -np.arange(-3, 3),
                   "kernel": ["rbf", 'linear']}},
    "KNearestNeighbor": {
        "plot_option": None,
        "params": {"n_neighbors": range(1, 30)}},
    "NeuralNetwork": dict(plot_option="UseLogAxis",
                          params=dict(activation=["logistic", "relu"], alpha=[1e-4, 1e-2, 1e-1, 1e-0],
                                      learning_rate_init=[5e-2, 1e-1, 1]))
}


def evaluate_classifiers():
    print("Evaluating baseline classifiers performance")
    for classifier_info in classifiers:
        include_classifier = classifier_info['include']

        if not include_classifier:
            continue

        classifier_name = classifier_info['name']
        classifier = classifier_info['classifier']
        apply_data_scaling = classifier_info['apply_data_scaling']
        apply_dimensionality_reduction = classifier_info['apply_dimensionality_reduction']

        print('******************************************************\n')
        print(classifier_name)

        for dataset_name, dataset in datasets.items():
            print()
            print(dataset_name)
            data_x, data_y = dataset['X_DATA'], dataset['Y_DATA']
            apply_data_balancing = eda.dataset_configurations[dataset_name]['REQUIRES_BALANCING']

            clf = classifier(data_x=data_x,
                             data_y=data_y,
                             test_size=test_size,
                             apply_data_scaling=apply_data_scaling,
                             apply_data_balancing=apply_data_balancing,
                             apply_dimensionality_reduction=apply_dimensionality_reduction)

            classifier_parameters = None
            clf.initialize_processing(classifier_parameters=classifier_parameters)
            clf.train()
            clf.predict()
            clf.evaluate()


def get_optimal_classifier_hyper_parameters():
    optimal_classifier_parameters = dict()

    print("Extract best hyper parameters for each classifier and dataset")
    for classifier_info in classifiers:
        include_classifier = classifier_info['include']

        if not include_classifier:
            continue

        classifier_name = classifier_info['name']
        classifier = classifier_info['classifier']
        apply_data_scaling = classifier_info['apply_data_scaling']
        apply_dimensionality_reduction = classifier_info['apply_dimensionality_reduction']

        print('******************************************************\n')
        print(classifier_name)

        for dataset_name, dataset in datasets.items():
            print()
            print(dataset_name)
            data_x, data_y = dataset['X_DATA'], dataset['Y_DATA']
            apply_data_balancing = eda.dataset_configurations[dataset_name]['REQUIRES_BALANCING']

            clf = classifier(data_x=data_x,
                             data_y=data_y,
                             test_size=test_size,
                             apply_data_scaling=apply_data_scaling,
                             apply_data_balancing=apply_data_balancing,
                             apply_dimensionality_reduction=apply_dimensionality_reduction)

            classifier_parameters = None
            clf.initialize_processing(classifier_parameters=classifier_parameters)
            optimal_parameters = clf.get_optimal_grid_search_hyper_parameters()
            optimal_parameters_name = f"{classifier_name}_{dataset_name}"

            clean_optimal_parameters = {key.replace("classifier__", ""): item for key, item in
                                        optimal_parameters.items()}
            optimal_classifier_parameters[optimal_parameters_name] = clean_optimal_parameters
            print(optimal_classifier_parameters[optimal_parameters_name])

    print(optimal_classifier_parameters)

    return optimal_classifier_parameters


def evaluate_optimal_classifiers(classifier_dataset_optimal_hyper_params_filepath, output_directory):
    with open(classifier_dataset_optimal_hyper_params_filepath) as f:
        classifier_dataset_optimal_hyper_params = json.load(f)

    for classifier_info in classifiers:
        include_classifier = classifier_info['include']

        if not include_classifier:
            continue

        classifier_name = classifier_info['name']
        classifier = classifier_info['classifier']
        apply_data_scaling = classifier_info['apply_data_scaling']
        apply_dimensionality_reduction = classifier_info['apply_dimensionality_reduction']

        print('******************************************************\n')
        print(classifier_name)

        for dataset_name, dataset in datasets.items():
            print()
            print(dataset_name)
            data_x, data_y = dataset['X_DATA'], dataset['Y_DATA']
            apply_data_balancing = eda.dataset_configurations[dataset_name]['REQUIRES_BALANCING']

            classifier_dataset_output_directory = rf"{output_directory}\{classifier_name}\{dataset_name}"
            Path(classifier_dataset_output_directory).mkdir(parents=True, exist_ok=True)

            clf = classifier(data_x=data_x,
                             data_y=data_y,
                             test_size=test_size,
                             apply_data_scaling=apply_data_scaling,
                             apply_data_balancing=apply_data_balancing,
                             apply_dimensionality_reduction=apply_dimensionality_reduction,
                             output_directory=classifier_dataset_output_directory)

            classifier_parameters = classifier_dataset_optimal_hyper_params[f"{classifier_name}_{dataset_name}"]
            clf.initialize_processing(classifier_parameters=classifier_parameters)
            clf.train()
            clf.predict()
            clf.evaluate()
            clf.plot_learning_curves()


def run_experiments_classifiers(classifier_dataset_optimal_hyper_params_filepath, output_directory):
    with open(classifier_dataset_optimal_hyper_params_filepath) as f:
        classifier_dataset_optimal_hyper_params = json.load(f)

    for classifier_info in classifiers:
        include_classifier = classifier_info['include']

        if not include_classifier:
            continue

        classifier_name = classifier_info['name']
        classifier = classifier_info['classifier']
        apply_data_scaling = classifier_info['apply_data_scaling']
        apply_dimensionality_reduction = classifier_info['apply_dimensionality_reduction']

        print('******************************************************\n')
        print(classifier_name)

        for dataset_name, dataset in datasets.items():
            plot_option = classifier_experiment_configurations[classifier_name]["plot_option"]
            classifier_experiment_configuration = classifier_experiment_configurations[classifier_name]["params"]

            print()
            print(dataset_name)
            data_x, data_y = dataset['X_DATA'], dataset['Y_DATA']
            apply_data_balancing = eda.dataset_configurations[dataset_name]['REQUIRES_BALANCING']

            classifier_dataset_output_directory = rf"{output_directory}\{classifier_name}\{dataset_name}"
            Path(classifier_dataset_output_directory).mkdir(parents=True, exist_ok=True)

            clf = classifier(data_x=data_x,
                             data_y=data_y,
                             test_size=test_size,
                             apply_data_scaling=apply_data_scaling,
                             apply_data_balancing=apply_data_balancing,
                             apply_dimensionality_reduction=apply_dimensionality_reduction,
                             output_directory=classifier_dataset_output_directory)

            classifier_parameters = classifier_dataset_optimal_hyper_params[f"{classifier_name}_{dataset_name}"]
            clf.initialize_processing(classifier_parameters=classifier_parameters)
            clf.run_experiments(classifier_experiment_configuration, plot_option)


if __name__ == "__main__":
    #evaluate_classifiers()

    ####################################################################################################
    results_directory = fr"{root_directory}\results"
    results_hyperparameters_directory = fr"{results_directory}\hyperparameters"

    results_hyperparameters_file_path = fr"{results_hyperparameters_directory}\optimal_classifier_parameters_new.json" # todo change
    classifier_results_directory = fr"{root_directory}\results\algorithms"

    evaluate_optimal_classifiers(results_hyperparameters_file_path, classifier_results_directory)

    run_experiments_classifiers(results_hyperparameters_file_path, classifier_results_directory)

    # #####################################################################################################
    optimal_classifier_parameters = get_optimal_classifier_hyper_parameters()
    print(optimal_classifier_parameters)

    with open(results_hyperparameters_file_path, 'w') as fp:
        print(optimal_classifier_parameters, file=fp)
        #json.dump(optimal_classifier_parameters, fp, sort_keys=True, indent=4)

    # ####################################################################################################

