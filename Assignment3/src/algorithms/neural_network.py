from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.neural_network import MLPClassifier

from src.algorithms.base import BaseClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.utils import timing


class NeuralNetwork(BaseClassifier):
    def __init__(self,
                 data_x: pd.DataFrame,
                 data_y: pd.DataFrame,
                 test_size: float = 0.3,
                 apply_data_scaling: bool = True,
                 apply_data_balancing: bool = True,
                 dimensionality_reducer: bool = True,
                 output_directory: str = ""):

        super().__init__(data_x=data_x,
                         data_y=data_y,
                         test_size=test_size,
                         apply_data_scaling=apply_data_scaling,
                         apply_data_balancing=apply_data_balancing,
                         dimensionality_reducer=dimensionality_reducer,
                         output_directory=output_directory)

        self.classifier_name = type(self).__name__

    def _set_classifier(self, parameters):
        self.classifier = MLPClassifier(max_iter=4000,
                                        random_state=self.seed)

        if parameters:
            self.classifier.set_params(**parameters)

    @timing
    def run_experiments(self, parameter_groups, dataset_name, dataset_algorithm_settings=None, cv=5):
        # # https://scikit-learn.org/stable/modules/cross_validation.html
        # accuracy = round(accuracy_score(self.test_y, self.predicted_y), 5)
        # print(f"Accuracy: {accuracy}")

        # score = round(roc_auc_score(self.test_y, self.predicted_y), 5)
        # print(f"ROC Score: {score}")

        # https://scikit-learn.org/stable/modules/cross_validation.html
        formatted_params = {'classifier__activation': dataset_algorithm_settings['activation'],
                            'classifier__alpha': dataset_algorithm_settings['alpha'],
                            'classifier__hidden_layer_sizes': dataset_algorithm_settings["hidden_layer_sizes"],
                            'classifier__learning_rate_init': dataset_algorithm_settings["learning_rate"],
                            'classifier__max_iter': dataset_algorithm_settings["max_iter"]}
        self.complete_processing_pipeline.set_params(**formatted_params)
        scores = cross_val_score(self.complete_processing_pipeline,
                                 self.train_x,
                                 self.train_y,
                                 cv=cv,
                                 scoring='f1_macro')
        # print(f"scores: {scores}")
        print("%0.2f f1_macro score with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

        self.plot_learning_curves(dataset_name)

    def plot_learning_curves(self, dataset_name, jobs=3, train_sizes=np.linspace(.1, 1.0, 5), cv=5):
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
                                           train_sizes,
                                           dataset_name)

        self._plot_training_time_versus_samples(mean_train_times, std_train_times, train_sizes, dataset_name)
        self._plot_training_time_versus_score( mean_train_times,mean_test_scores, std_test_scores, dataset_name)

    def _plot_training_learning_curve(self, mean_test_scores,
                                      mean_train_scores,
                                      std_test_scores,
                                      std_train_scores,
                                      train_sizes,
                                      dataset_name):
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
        fig.savefig(rf'{self.output_file_path}/learning_curve_{dataset_name}.png')  # save the figure to file
        plt.close(fig)

    def _plot_training_time_versus_samples(self, mean_train_times,
                                           std_train_times,
                                           train_sizes,
                                           dataset_name):

        title = f"Training Time - {self.classifier_name}"
        fig, ax = plt.subplots(nrows=1, ncols=1)

        ax.grid()
        ax.plot(train_sizes, mean_train_times, 'o-')
        ax.fill_between(train_sizes, mean_train_times - std_train_times, mean_train_times + std_train_times, alpha=0.1)
        ax.set_xlabel("Training samples")
        ax.set_ylabel("Training Time")
        ax.set_title(title)
        ax.legend(loc="best")
        fig.savefig(rf'{self.output_file_path}/training_time_{dataset_name}.png')  # save the figure to file
        plt.close(fig)

    def _plot_training_time_versus_score(self, mean_train_times,
                                         mean_test_scores,
                                         std_test_scores,
                                         dataset_name):

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
            rf'{self.output_file_path}/score_vs_training_time_{dataset_name}.png')  # save the figure to file
        plt.close(fig)