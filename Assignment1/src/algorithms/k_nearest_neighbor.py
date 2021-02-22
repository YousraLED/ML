from sklearn.neighbors import KNeighborsClassifier

from src.algorithms.base import BaseClassifier
import numpy as np
import pandas as pd


class KNearestNeighbor(BaseClassifier):
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
        self.classifier = KNeighborsClassifier()

        if parameters:
            self.classifier.set_params(**parameters)

    def _set_hyper_parameters_ranges(self):
        n_neighbors = list(range(3, 21))
        distances = [1, 2, 3, 4, 5]

        self.classifier_parameters_to_tune = {
            'classifier__n_neighbors': n_neighbors,
            'classifier__p': distances
        }
