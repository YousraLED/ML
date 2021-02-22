from sklearn.model_selection import cross_val_score

from src.algorithms.base import BaseClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import TREE_LEAF
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DecisionTree(BaseClassifier):
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
        self.classifier = DecisionTreeClassifier(random_state=self.seed)

        if parameters:
            self.classifier.set_params(**parameters)

    def _set_hyper_parameters_ranges(self):
        max_depth_start, max_depth_end = 1, 100
        max_depths = list(range(max_depth_start, max_depth_end))
        criteria = ['entropy', 'gini']

        self.classifier_parameters_to_tune = {
            'classifier__max_depth': max_depths,
            'classifier__criterion': criteria}

