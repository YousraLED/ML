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
                 apply_dimensionality_reduction: bool = True,
                 output_directory: str = ""):

        self.seed = 100
        self.data_x = data_x
        self.data_y = data_y
        self.test_size = test_size

        self.apply_data_scaling = apply_data_scaling
        self.apply_data_balancing = apply_data_balancing
        self.apply_dimensionality_reduction = apply_dimensionality_reduction

        self.dimensionality_reducer = None

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
        self._set_data_dimensionality_reducer()
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

    def _set_data_dimensionality_reducer(self):
        if self.apply_dimensionality_reduction:
            self.dimensionality_reducer = PCA()

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

    def evaluate(self, split=5, cv=5):
        # # https://scikit-learn.org/stable/modules/cross_validation.html
        # accuracy = round(accuracy_score(self.test_y, self.predicted_y), 5)
        # print(f"Accuracy: {accuracy}")

        # score = round(roc_auc_score(self.test_y, self.predicted_y), 5)
        # print(f"ROC Score: {score}")

        # https://scikit-learn.org/stable/modules/cross_validation.html
        scores = cross_val_score(self.complete_processing_pipeline,
                                 self.train_x,
                                 self.train_y,
                                 cv=cv,
                                 scoring='f1_macro')
        # print(f"scores: {scores}")
        print("%0.2f f1_macro score with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    @timing
    def get_optimal_grid_search_hyper_parameters(self, jobs=3):
        self._set_hyper_parameters_ranges()

        classifier = GridSearchCV(estimator=self.complete_processing_pipeline,
                                  param_grid=self.classifier_parameters_to_tune,
                                  n_jobs=jobs,
                                  scoring='f1_macro')

        classifier.fit(self.train_x, self.train_y)
        print("Best parameters on training set\n")
        print(classifier.best_params_)

        print()
        print("Classification Report")
        predict_y = classifier.predict(self.test_x)
        print(classification_report(self.test_y, predict_y))
        print()
        return classifier.best_params_

    def plot_learning_curves(self, jobs=3, train_sizes=np.linspace(.1, 1.0, 5), cv=5):
        # Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
        # cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=self.seed)

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(self.complete_processing_pipeline,
                                                                              self.train_x,
                                                                              self.train_y,
                                                                              cv=cv,
                                                                              n_jobs=jobs,
                                                                              train_sizes=train_sizes,
                                                                              scoring='f1_macro',
                                                                              shuffle=True,
                                                                              return_times=True)
        mean_train_scores = np.mean(train_scores, axis=1)
        std_train_scores = np.std(train_scores, axis=1)
        mean_test_scores = np.mean(test_scores, axis=1)
        std_test_scores = np.std(test_scores, axis=1)
        mean_train_times = np.mean(fit_times, axis=1)
        std_train_times = np.std(fit_times, axis=1)

        self._plot_training_learning_curve(mean_test_scores,
                                           mean_train_scores,
                                           std_test_scores,
                                           std_train_scores,
                                           train_sizes)

        self._plot_training_time_versus_samples(mean_train_times, std_train_times, train_sizes)
        self._plot_training_time_versus_score( mean_train_times,mean_test_scores, std_test_scores)

    def _plot_training_learning_curve(self, mean_test_scores,
                                      mean_train_scores,
                                      std_test_scores,
                                      std_train_scores,
                                      train_sizes):
        title = f"Learning Curve - {self.classifier_name}"
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.grid()
        ax.fill_between(train_sizes, mean_train_scores - std_train_scores,
                        mean_train_scores + std_train_scores, alpha=0.1,
                        color="r")
        ax.fill_between(train_sizes, mean_test_scores - std_test_scores,
                        mean_test_scores + std_test_scores, alpha=0.1,
                        color="g")
        ax.plot(train_sizes, mean_train_scores, 'o-', color="r", label="Training score")
        ax.plot(train_sizes, mean_test_scores, 'o-', color="g", label="Cross-validation score")
        ax.set_xlabel("Training samples")
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.legend(loc="best")
        fig.savefig(rf'{self.output_file_path}/learning_curve_{self.classifier_name}.png')  # save the figure to file
        plt.close(fig)

    def _plot_training_time_versus_samples(self, mean_train_times,
                                           std_train_times,
                                           train_sizes):

        title = f"Training Time - {self.classifier_name}"
        fig, ax = plt.subplots(nrows=1, ncols=1)

        ax.grid()
        ax.plot(train_sizes, mean_train_times, 'o-')
        ax.fill_between(train_sizes, mean_train_times - std_train_times, mean_train_times + std_train_times, alpha=0.1)
        ax.set_xlabel("Training samples")
        ax.set_ylabel("Training Time")
        ax.set_title(title)
        ax.legend(loc="best")
        fig.savefig(rf'{self.output_file_path}/training_time_{self.classifier_name}.png')  # save the figure to file
        plt.close(fig)

    def _plot_training_time_versus_score(self, mean_train_times,
                                         mean_test_scores,
                                         std_test_scores):

        title = f"Model Performance - {self.classifier_name}"
        fig, ax = plt.subplots(nrows=1, ncols=1)

        ax.grid()
        ax.plot(mean_train_times, mean_test_scores, 'o-')
        ax.fill_between(mean_train_times, mean_test_scores - std_test_scores, mean_test_scores + std_test_scores,
                        alpha=0.1)
        ax.set_xlabel("Training Time")
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.legend(loc="best")
        fig.savefig(
            rf'{self.output_file_path}/score_vs_training_time_{self.classifier_name}.png')  # save the figure to file
        plt.close(fig)

    def run_experiments(self, parameter_groups, plot_option, cv=5):
        for parameter_name, parameter_group in parameter_groups.items():
            cv_scores_list = []
            cv_scores_std = []
            cv_scores_mean = []
            accuracy_scores = []

            title = f'{self.classifier_name} - Accuracy vs {parameter_name}'

            for parameter in parameter_group:
                formatted_params = {f'classifier__{parameter_name}': parameter}
                self.complete_processing_pipeline.set_params(**formatted_params)
                cv_scores = cross_val_score(self.complete_processing_pipeline, self.train_x, self.train_y, cv=cv)
                cv_scores_list.append(cv_scores)
                cv_scores_mean.append(cv_scores.mean())
                cv_scores_std.append(cv_scores.std())

                accuracy_scores.append(
                    self.complete_processing_pipeline.fit(self.test_x, self.test_y).score(self.test_x, self.test_y))

            cv_scores_mean = np.array(cv_scores_mean)
            cv_scores_std = np.array(cv_scores_std)
            accuracy_scores = np.array(accuracy_scores)
            print(accuracy_scores)
            print(cv_scores_mean)
            self.plot_cross_validation_on_parameters(parameter_name, parameter_group, cv_scores_mean, cv_scores_std,
                                                     accuracy_scores, title, plot_option)

    # function for plotting cross-validation results
    def plot_cross_validation_on_parameters(self, parameter_name,
                                            parameters,
                                            cv_scores_mean,
                                            cv_scores_std,
                                            accuracy_scores,
                                            title,
                                            plot_option):
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        if plot_option:
            plt.xscale('log')
        ax.plot(parameters, cv_scores_mean, '-o', label='mean cross-validation accuracy', alpha=0.9)
        ax.fill_between(parameters, cv_scores_mean - 2 * cv_scores_std, cv_scores_mean + 2 * cv_scores_std, alpha=0.2)
        ax.plot(parameters, accuracy_scores, '-*', label='train accuracy', alpha=0.9)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(parameter_name, fontsize=14)
        ax.set_ylabel('Accuracy', fontsize=14)
        ax.set_xticks(parameters)
        ax.legend()
        fig.savefig(rf'{self.output_file_path}/experiment_{self.classifier_name}_{parameter_name}.png')
        plt.close(fig)
