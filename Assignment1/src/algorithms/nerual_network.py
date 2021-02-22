from sklearn.neural_network import MLPClassifier

from src.algorithms.base import BaseClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd


class NeuralNetwork(BaseClassifier):
    def __init__(self,
                 data_x: pd.DataFrame,
                 data_y: pd.DataFrame,
                 test_size: float = 0.3,
                 apply_data_scaling: bool = True,
                 apply_data_balancing: bool = True,
                 apply_dimensionality_reduction: bool = True,
                 output_directory: str = ""):

        super().__init__(data_x=data_x,
                         data_y=data_y,
                         test_size=test_size,
                         apply_data_scaling=apply_data_scaling,
                         apply_data_balancing=apply_data_balancing,
                         apply_dimensionality_reduction=apply_dimensionality_reduction,
                         output_directory=output_directory)

        self.classifier_name = type(self).__name__

    def _set_classifier(self, parameters):
        hidden_layers_count = int(len(self.data_x.columns.values)*(1/2))

        # print(hidden_layers_count)
        #hidden_layer_sizes = (hidden_layers_count, hidden_layers_count)
        hidden_layer_sizes = hidden_layers_count

        self.classifier = MLPClassifier(max_iter=2000,
                                        # hidden_layer_sizes=hidden_layer_sizes,
                                        random_state=self.seed)

        if parameters:
            self.classifier.set_params(**parameters)

    def _set_hyper_parameters_ranges(self):
        activation = ['logistic', 'tanh', 'relu']
        alpha = [1e-4, 1e-2, 1e-1, 1e-0]

        hidden_layers1 = (100)
        hidden_layers2 = int(len(self.data_x.columns.values) * (2 / 3))
        hidden_layers3 = int(len(self.data_x.columns.values) * (1 / 2))
        hidden_layer_sizes = [hidden_layers1, (hidden_layers3, hidden_layers3)] #, (hidden_layers3, hidden_layers3, hidden_layers3)]

        solver = ['lbfgs', 'sgd', 'adam']
        learning_rate_init = [5e-2, 1e-1, 1]
        max_iter = [4000]  # , 3000, 6000]

        self.classifier_parameters_to_tune = {
            'classifier__activation': activation,
            'classifier__alpha': alpha,
            'classifier__hidden_layer_sizes': hidden_layer_sizes,
            'classifier__learning_rate_init': learning_rate_init,
            'classifier__solver': solver,
            'classifier__max_iter': max_iter
        }
