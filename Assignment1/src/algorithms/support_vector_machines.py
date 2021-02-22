from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

from src.algorithms.base import BaseClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd


class SupportVectorMachines(BaseClassifier):
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
        # ok to be more aggressive here
        self.classifier = SVC(random_state=self.seed)

        if parameters:
            self.classifier.set_params(**parameters)

    def _set_hyper_parameters_ranges(self):
        C = 10.0 ** -np.arange(-5, 5)
        kernel = ['linear', 'poly', 'rbf', 'sigmoid']
        degree = [1, 2, 3, 4, 5]
        gamma = ['auto', 'scale']

        self.classifier_parameters_to_tune = {
            'classifier__C': C,
            'classifier__kernel': kernel,
            'classifier__degree': degree,
            'classifier__gamma': gamma}
