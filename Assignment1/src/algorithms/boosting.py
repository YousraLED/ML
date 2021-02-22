from sklearn.ensemble import AdaBoostClassifier

from src.algorithms.base import BaseClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd


class AdaBoost(BaseClassifier):
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
        self.classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3,
                                                                    random_state=self.seed), random_state=self.seed)

        if parameters:
            self.classifier.set_params(**parameters)

    def _set_hyper_parameters_ranges(self):
        n_estimators = np.linspace(100, 1000, 10, endpoint=True).astype(np.int64)
        learning_rate = np.arange(0.5, 2.0, 0.25)
        algorithm = ['SAMME', 'SAMME.R']

        self.classifier_parameters_to_tune = {
            'classifier__n_estimators': n_estimators,
            'classifier__learning_rate': learning_rate,
            'classifier__algorithm': algorithm}
