import json
import os
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier
from sklearn.mixture import GaussianMixture
from sklearn.random_projection import GaussianRandomProjection

from src.algorithms.linear_discriminant_analysis import LDA
from src.algorithms.random_projection import RandomProjection
from src.dataset import Dataset
from src.algorithms.kmeans_clustering import KmeansClustering
from src.algorithms.expectation_maximization import ExpectationMaximization
from src.algorithms.principla_component_analysis import PrincipalComponentAnalysis
from src.algorithms.indpendent_component_analysis import IndependentComponentAnalysis

from src.algorithms.neural_network import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from scipy import stats
import seaborn as sns

root_directory = os.path.dirname(__file__)
config_file_path = fr"{root_directory}\datasets\config_datasets.ini"
dt = Dataset(config_file_path, root_directory)
dt.load_data_using_configuration()

test_size = 0.3
seed = 100

classifiers = [{'name': 'KmeanClustering',
                'include': False,
                'classifier': KmeansClustering,
                'apply_data_scaling': False,
                'dimensionality_reducer': None},
               {'name': 'ExpectationMaximization',
                'include': False,
                'classifier': ExpectationMaximization,
                'apply_data_scaling': False,
                'dimensionality_reducer': None},
               {'name': 'PCA',
                'include': False,
                'classifier': PrincipalComponentAnalysis,
                'apply_data_scaling': True,
                'dimensionality_reducer': None},
               {'name': 'ICA',
                'include': False,
                'classifier': IndependentComponentAnalysis,
                'apply_data_scaling': True,
                'dimensionality_reducer': None},
               {'name': 'RP',
                'include': False,
                'classifier': RandomProjection,
                'apply_data_scaling': True,
                'dimensionality_reducer': None},
               {'name': 'LDA',
                'include': True,
                'classifier': LDA,
                'apply_data_scaling': True,
                'dimensionality_reducer': None},
               {'name': 'NeuralNetwork',
                'include': False,
                'classifier': NeuralNetwork,
                'apply_data_scaling': True,
                'dimensionality_reducer': None}
               ]

classifier_experiment_configurations = {
    "KmeanClustering": {
        "plot_option": None,
        "params": {"n_clusters": range(1, 11)}},
    "ExpectationMaximization": {
        "plot_option": None,
        "params": {"n_components": range(1, 5),
                   "cv_types": ['spherical', 'tied', 'diag', 'full'],
                   "calculate_bic": True}},
    "PCA": {
        "plot_option": None,
        "params": {"n_components": range(1, 5)}},
    "ICA": {
        "plot_option": None,
        "params": {"n_components": range(1, 5)}},
    "RP": {
        "plot_option": None,
        "params": None},
    "LDA": {
        "plot_option": None,
        "params": None},
    "NeuralNetwork": {
        "plot_option": None,
        "params": None}
}

dataset_algorithm_settings = {
    "DIABETES": {"ExpectationMaximization": {"test_size": 0.2, "reg_covar": 1e-6, "n_components": range(1, 5),
                                             "dimensionality_reducer": None},
                 "KmeanClustering": {"test_size": 0.2,
                                     "dimensionality_reducer": None},
                 "PCA": {"test_size": 0.2, "n_components": range(1, 16), "best_component_num": 10,
                         "dimensionality_reducer": None},
                 "ICA": {"test_size": 0.2, "n_components": range(1, 16), "best_pcs": [10, 11],
                         "dimensionality_reducer": None},
                 "RP": {"test_size": 0.2, "n_components": range(1, 16), "dimensionality_reducer": None},
                 "LDA": {"test_size": 0.2, "n_components": range(1, 10), "dimensionality_reducer": None},
                 "NeuralNetwork": {"test_size": 0.3,
                                   "activation": "tanh",
                                   "alpha": 0.0001,
                                   "hidden_layer_sizes": [8, 8],
                                   "learning_rate": 0.05,
                                   "max_iter": 4000,
                                   "solver": "adam",
                                   "dimensionality_reducer": FastICA(n_components=12, max_iter=4000, whiten=True,
                                                                     random_state=100)}},
    "PD_SPEECH_FEATURES": {"ExpectationMaximization": {"test_size": 0.2, "reg_covar": 0.9, "n_components": range(1, 4),
                                                       "dimensionality_reducer": None},
                           "KmeanClustering": {"test_size": 0.2, "dimensionality_reducer": None},
                           "PCA": {"test_size": 0.2, "n_components": range(1, 100), "best_component_num": 80,
                                   "dimensionality_reducer": None},
                           "ICA": {"test_size": 0.2, "n_components": range(1, 100), "best_pcs": [50, 51],
                                   "dimensionality_reducer": None},
                           "RP": {"test_size": 0.2, "n_components": range(1, 100), "dimensionality_reducer": None},
                           "LDA": {"test_size": 0.2, "n_components": range(1, 100), "dimensionality_reducer": None},
                           "NeuralNetwork": {"test_size": 0.3,
                                             "activation": "relu",
                                             "alpha": 0.01,
                                             "hidden_layer_sizes": [100],
                                             "learning_rate": 1.0,
                                             "max_iter": 4000,
                                             "solver": "sgd",
                                             "dimensionality_reducer": FastICA(n_components=50, max_iter=1000,
                                                                               whiten=True,
                                                                               random_state=100)}}
}

##############################################################
classifiers_clustering_algorithm_and_reduction_settings = [{'name': 'KmeanClustering',
                                                            'include': False,
                                                            'classifier': KmeansClustering,
                                                            'apply_data_scaling': True,
                                                            'dimensionality_reducer': None},
                                                           {'name': 'ExpectationMaximization',
                                                            'include': False,
                                                            'classifier': ExpectationMaximization,
                                                            'apply_data_scaling': True,
                                                            'dimensionality_reducer': None},
                                                           {'name': 'NeuralNetwork',
                                                            'include': True,
                                                            'classifier': NeuralNetwork,
                                                            'apply_data_scaling': True,
                                                            'dimensionality_reducer': None}
                                                           ]

classifier_clustering_algorithm_and_reduction_settings = {
    "KmeanClustering": {
        "plot_option": None,
        "params": {"n_clusters": range(1, 11)}},
    "ExpectationMaximization": {
        "plot_option": None,
        "params": {"n_components": range(1, 5),
                   "cv_types": ['diag'],
                   "calculate_bic": False}},
    "PCA": {
        "plot_option": None,
        "params": {"n_components": range(1, 5)}},
    "ICA": {
        "plot_option": None,
        "params": {"n_components": range(1, 5)}},
    "RP": {
        "plot_option": None,
        "params": None},
    "LDA": {
        "plot_option": None,
        "params": None},
    "NeuralNetwork": {
        "plot_option": None,
        "params": None}
}

dataset_clustering_algorithm_and_reduction_settings = {
    "DIABETES": {"ExpectationMaximization": {"test_size": 0.2,
                                             "reg_covar": 1e-6,
                                             "n_components": [2],
                                             "dimensionality_reducers": {
                                                 "PCA": PCA(n_components=10, random_state=seed),
                                                 "ICA": FastICA(n_components=10, random_state=seed, whiten=True,
                                                                max_iter=2000,
                                                                tol=1e-2),
                                                 "RP": GaussianRandomProjection(n_components=8, random_state=25),
                                                 "LDA": FastICA(tol=1e-4)
                                             }},
                 "KmeanClustering": {"test_size": 0.2,
                                     "dimensionality_reducers": {
                                         "PCA": PCA(n_components=10, random_state=seed),
                                         "ICA": FastICA(n_components=10, random_state=seed, whiten=True, max_iter=2000,
                                                        tol=1e-2),
                                         "RP": GaussianRandomProjection(n_components=8, random_state=25),
                                         "LDA": FastICA(tol=1e-4)
                                     }},
                 "NeuralNetwork": {"test_size": 0.2,
                                   "activation": "tanh",
                                   "alpha": 0.0001,
                                   "hidden_layer_sizes": [8, 8],
                                   "learning_rate": 0.05,
                                   "max_iter": 4000,
                                   "solver": "adam",
                                   "dimensionality_reducers": {
                                       "PCA": PCA(n_components=10, random_state=seed),
                                       "ICA": FastICA(n_components=10, random_state=seed, whiten=True, max_iter=2000,
                                                      tol=1e-2),
                                       "RP": GaussianRandomProjection(n_components=8, random_state=25),
                                       "LDA": FastICA(tol=1e-4),
                                       "KmeanClustering": KMeans(random_state=seed, init='k-means++', n_init=10)
                                       #  "ExpectationMaximization": VotingClassifier([('gmm',GaussianMixture(random_state=seed, n_init=20, init_params='random'))])
                                   }}
                 },

    "PD_SPEECH_FEATURES": {"ExpectationMaximization": {"test_size": 0.2,
                                                       "reg_covar": 0.9,
                                                       "n_components": [2],
                                                       "dimensionality_reducers": {"PCA": PCA(n_components=10),
                                                                                   "ICA": FastICA(
                                                                                       n_components=65,
                                                                                       random_state=seed,
                                                                                       whiten=True,
                                                                                       max_iter=1000, tol=1e-2),
                                                                                   "RP": GaussianRandomProjection(
                                                                                       n_components=50,
                                                                                       random_state=50),
                                                                                   "LDA": LinearDiscriminantAnalysis(
                                                                                       tol=1e-4)
                                                                                   }},
                           "KmeanClustering": {"test_size": 0.2,
                                               "dimensionality_reducers": {"PCA": PCA(n_components=70),
                                                                           "ICA": FastICA(
                                                                               n_components=65,
                                                                               random_state=seed,
                                                                               whiten=True,
                                                                               max_iter=1000, tol=1e-2),
                                                                           "RP": GaussianRandomProjection(
                                                                               n_components=50, random_state=50),
                                                                           "LDA": LinearDiscriminantAnalysis(tol=1e-4),
                                                                           }},
                           "NeuralNetwork": {"test_size": 0.2,
                                             "activation": "relu",
                                             "alpha": 0.01,
                                             "hidden_layer_sizes": [100],
                                             "learning_rate": 1.0,
                                             "max_iter": 4000,
                                             "solver": "sgd",
                                             "dimensionality_reducers": {
                                                 "PCA": PCA(n_components=40),
                                                 "ICA": FastICA(
                                                     n_components=65,
                                                     random_state=seed,
                                                     whiten=True,
                                                     max_iter=1000, tol=1e-2),
                                                 "RP": GaussianRandomProjection(
                                                     n_components=50, random_state=50),
                                                 "LDA": LinearDiscriminantAnalysis(tol=1e-4),
                                                 "KmeanClustering": KMeans(random_state=seed, init='k-means++',
                                                                           n_init=10)
                                                 # "ExpectationMaximization": VotingClassifier([('gmm',GaussianMixture(random_state=seed, n_init=20, init_params='random'))])
                                             }},
                           }
}


def evaluate_algorithms(classifiers, classifier_experiment_configurations, dataset_algorithm_settings, dt,
                        output_directory):
    print("Evaluating clustering algorithms")
    datasets = dt.get_data()
    for classifier_info in classifiers:
        include_classifier = classifier_info['include']

        if not include_classifier:
            continue

        classifier_name = classifier_info['name']
        classifier = classifier_info['classifier']
        apply_data_scaling = classifier_info['apply_data_scaling']
        dimensionality_reducer = classifier_info['dimensionality_reducer']

        print('******************************************************\n')
        print(classifier_name)

        for dataset_name, dataset in datasets.items():
            print()
            print(dataset_name)
            data_x, data_y = dataset['X_DATA'], dataset['Y_DATA']
            apply_data_balancing = dt.dataset_configurations[dataset_name]['REQUIRES_BALANCING']
            classifier_experiment_configuration = classifier_experiment_configurations[classifier_name]["params"]

            classifier_dataset_output_directory = rf"{output_directory}\{classifier_name}\{dataset_name}"
            Path(classifier_dataset_output_directory).mkdir(parents=True, exist_ok=True)

            dimensionality_reducer = dataset_algorithm_settings[dataset_name][classifier_name]["dimensionality_reducer"]

            clf = classifier(data_x=data_x,
                             data_y=data_y,
                             test_size=dataset_algorithm_settings[dataset_name][classifier_name]['test_size'],
                             apply_data_scaling=apply_data_scaling,
                             apply_data_balancing=apply_data_balancing,
                             dimensionality_reducer=dimensionality_reducer,
                             output_directory=classifier_dataset_output_directory)

            classifier_parameters = None
            clf.initialize_processing(classifier_parameters=classifier_parameters)
            clf.run_experiments(classifier_experiment_configuration,
                                dataset_name,
                                dataset_algorithm_settings[dataset_name][classifier_name])


def evaluate_clustering_algorithms_with_reduction(classifiers,
                                                  classifier_experiment_configurations,
                                                  dataset_clustering_algorithm_and_reduction_settings, dt,
                                                  output_directory):
    print("Evaluating clustering algorithms with reduction")
    datasets = dt.get_data()
    for classifier_info in classifiers:
        include_classifier = classifier_info['include']

        if not include_classifier:
            continue

        classifier_name = classifier_info['name']
        classifier = classifier_info['classifier']
        apply_data_scaling = classifier_info['apply_data_scaling']

        print('******************************************************\n')
        print(classifier_name)

        for dataset_name, dataset in datasets.items():
            dimensionality_reducers = \
                dataset_clustering_algorithm_and_reduction_settings[dataset_name][classifier_name][
                    "dimensionality_reducers"]
            for dimensionality_reducer_name, dimensionality_reducer in dimensionality_reducers.items():
                print()
                print(dataset_name, dimensionality_reducer_name)
                data_x, data_y = dataset['X_DATA'], dataset['Y_DATA']
                apply_data_balancing = dt.dataset_configurations[dataset_name]['REQUIRES_BALANCING']
                classifier_experiment_configuration = classifier_experiment_configurations[classifier_name]["params"]

                classifier_dataset_output_directory = rf"{output_directory}\{classifier_name}\{dataset_name}"
                Path(classifier_dataset_output_directory).mkdir(parents=True, exist_ok=True)

                clf = classifier(data_x=data_x,
                                 data_y=data_y,
                                 test_size=
                                 dataset_clustering_algorithm_and_reduction_settings[dataset_name][classifier_name][
                                     'test_size'],
                                 apply_data_scaling=apply_data_scaling,
                                 apply_data_balancing=apply_data_balancing,
                                 dimensionality_reducer=dimensionality_reducer,
                                 output_directory=classifier_dataset_output_directory)

                classifier_parameters = None
                clf.initialize_processing(classifier_parameters=classifier_parameters)
                clf.run_experiments(classifier_experiment_configuration,
                                    f"{dataset_name} - {dimensionality_reducer_name}",
                                    dataset_clustering_algorithm_and_reduction_settings[dataset_name][classifier_name])


output_directory = fr"{root_directory}\results\algorithms"
evaluate_algorithms(classifiers, classifier_experiment_configurations, dataset_algorithm_settings, dt, output_directory)

output_directory = fr"{root_directory}\results\algorithms_with_reduction"
evaluate_clustering_algorithms_with_reduction(classifiers_clustering_algorithm_and_reduction_settings,
                                              classifier_clustering_algorithm_and_reduction_settings,
                                              dataset_clustering_algorithm_and_reduction_settings,
                                              dt,
                                              output_directory)

output_directory = fr"{root_directory}\results\algorithms_with_clustering"
evaluate_clustering_algorithms_with_reduction(classifiers_clustering_algorithm_and_reduction_settings,
                                              classifier_clustering_algorithm_and_reduction_settings,
                                              dataset_clustering_algorithm_and_reduction_settings,
                                              dt,
                                              output_directory)
