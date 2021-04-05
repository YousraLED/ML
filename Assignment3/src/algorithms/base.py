import pandas as pd
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve, ShuffleSplit, \
    StratifiedKFold
# from sklearn.pipeline import Pipeline
from imblearn.pipeline import make_pipeline, Pipeline
from imblearn.over_sampling import ADASYN, SMOTE
from src.utils import timing
import numpy as np
import matplotlib.pyplot as plt

"""
References: 
https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html 
"""


class BaseClassifier:
    def __init__(self,
                 data_x: pd.DataFrame,
                 data_y: pd.DataFrame,
                 test_size: float = 0.8,
                 apply_data_scaling: bool = True,
                 apply_data_balancing: bool = True,
                 dimensionality_reducer=None,
                 output_directory: str = ""):

        self.seed = 100
        self.data_x = data_x
        self.data_y = data_y
        self.test_size = test_size

        self.apply_data_scaling = apply_data_scaling
        self.apply_data_balancing = apply_data_balancing
        self.dimensionality_reducer = dimensionality_reducer

        self.encoder = None
        self.scaler = None

        self.data_balancer = None
        self.classifier = None
        self.classifier_parameters_to_tune = None
        self.classifier_parameters_baseline = None
        self.classifier_parameters_optimal = None

        self.categorical_features = []
        self.categorical_transformer_pipeline = None

        self.numeric_features = []
        self.numerical_transformer_pipeline = None

        self.data_preprocessor_pipeline = None
        self.complete_processing_pipeline = None

        self.train_x, self.train_y = None, None
        self.test_x, self.test_y = None, None
        self.validation_x, self.validation_y = None, None
        self.predicted_y = None

        self.statistics = None

        self.classifier_name = ''
        self.output_file_path = output_directory

    def initialize_processing(self, classifier_parameters):
        self._create_processing_pipeline(classifier_parameters)
        self._prepare_data_for_testing_and_training()

    def _prepare_data_for_testing_and_training(self):
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.data_x,
                                                                                self.data_y,
                                                                                test_size=self.test_size,
                                                                                random_state=self.seed,
                                                                                shuffle=True)

    def _create_processing_pipeline(self, classifier_parameters):
        self._set_classifier(classifier_parameters)
        # self._set_data_dimensionality_reducer()
        self._set_numeric_data_scaler()
        self._set_encoder()
        self._set_data_balancer()

        self._set_categorical_features()
        self._set_numeric_features()

        self._create_categorical_transformer_pipeline()
        self._create_numerical_transformer_pipeline()
        self._create_complete_data_processing_pipeline()

    def _set_classifier(self, parameters):
        pass

    def _set_data_balancer(self):
        self.data_balancer = ADASYN()

    def _set_encoder(self):
        self.encoder = preprocessing.OrdinalEncoder()

    def _set_numeric_data_scaler(self):
        self.scaler = preprocessing.StandardScaler()

    def _set_numeric_features(self):
        numerical_types = ['float64', 'float32', 'int32', 'int64']
        data_types = dict(self.data_x.dtypes)
        self.numeric_features = [key for key in data_types if data_types[key] in numerical_types]

    def _set_categorical_features(self):
        categorical_types = ['object']
        data_types = dict(self.data_x.dtypes)
        self.categorical_features = [key for key in data_types if data_types[key] in categorical_types]

    def _create_categorical_transformer_pipeline(self):
        transformer_steps = []

        # Handle missing data
        imputer = ('imputer', SimpleImputer(strategy='constant', fill_value='missing'))
        transformer_steps.append(imputer)

        if self.encoder:
            encoder = ('encoder', self.encoder)
            transformer_steps.append(encoder)

        self.categorical_transformer_pipeline = Pipeline(steps=transformer_steps)

    def _create_numerical_transformer_pipeline(self):
        transformer_steps = []

        # Handle missing data
        imputer = ('imputer', SimpleImputer(strategy='median'))
        transformer_steps.append(imputer)

        if self.apply_data_scaling:
            scaler = ('scaler', self.scaler)
            transformer_steps.append(scaler)

        self.numerical_transformer_pipeline = Pipeline(steps=transformer_steps)

    def _create_complete_data_processing_pipeline(self):
        data_processing_steps = []

        data_type_transformer_step = self._get_data_type_transformer_step()
        data_processing_steps.append(data_type_transformer_step)

        if self.dimensionality_reducer:
            dimensionality_reduction_step = ('dimred', self.dimensionality_reducer)
            data_processing_steps.append(dimensionality_reduction_step)

        if self.apply_data_balancing:
            training_data_balancing_step = ('balance', self.data_balancer)
            data_processing_steps.append(training_data_balancing_step)

        classifier_step = ('classifier', self.classifier)
        data_processing_steps.append(classifier_step)

        self.complete_processing_pipeline = Pipeline(steps=data_processing_steps)

    def _get_preprocessor_pipeline(self):
        data_processing_steps = []

        data_type_transformer_step = self._get_data_type_transformer_step()
        data_processing_steps.append(data_type_transformer_step)

        if self.dimensionality_reducer:
            dimensionality_reduction_step = ('dimred', self.dimensionality_reducer)
            data_processing_steps.append(dimensionality_reduction_step)

        if self.apply_data_balancing:
            training_data_balancing_step = ('balance', self.data_balancer)
            data_processing_steps.append(training_data_balancing_step)

        preprocessor_pipeline = Pipeline(steps=data_processing_steps)
        return preprocessor_pipeline


    def _get_data_type_transformer_step(self):
        numerical_columns_transformer = ('numerical',
                                         self.numerical_transformer_pipeline,
                                         self.numeric_features)

        categorical_columns_transformer = ('categorical',
                                           self.categorical_transformer_pipeline,
                                           self.categorical_features)

        data_type_transformers = ColumnTransformer(transformers=[numerical_columns_transformer,
                                                                 categorical_columns_transformer])

        data_type_transformer_step = ('preprocessor', data_type_transformers)
        return data_type_transformer_step

    def _set_hyper_parameters_ranges(self):
        pass

    @timing
    def train(self):
        self.complete_processing_pipeline.fit(self.train_x, self.train_y)

    @timing
    def predict(self):
        self.predicted_y = self.complete_processing_pipeline.predict(self.test_x)

    def run_experiments(self, parameter_groups, dataset_name):
        pass

