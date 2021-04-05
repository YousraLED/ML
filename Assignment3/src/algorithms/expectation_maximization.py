import itertools
import traceback

import matplotlib as mpl
from scipy import linalg
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.algorithms.base import BaseClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class ExpectationMaximization(BaseClassifier):
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
        self.classifier = GaussianMixture(random_state=self.seed, n_init=20, init_params='random')

        if parameters:
            self.classifier.set_params(**parameters)

    # Adapted from https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#sphx-glr-auto-examples-mixture-plot-gmm-selection-py
    def run_experiments(self, parameter_groups, dataset_name, dataset_algorithm_settings=None):
        lowest_bic = np.infty
        bic = []
        test_accuracies = []
        n_components_range = parameter_groups["n_components"]
        cv_types = parameter_groups["cv_types"]
        calculate_bic = parameter_groups["calculate_bic"]

        data_type_transformer_step = self._get_data_type_transformer_step()
        data_processing_steps = [data_type_transformer_step]
        pipeline = Pipeline(steps=data_processing_steps)
        pipeline.fit(self.train_x)
        X = pipeline.transform(self.train_x)

        best_gmm = None
        for cv_type in cv_types:
            for n_components in n_components_range:
                formatted_params = {'classifier__covariance_type': cv_type,
                                    'classifier__n_components': n_components,
                                    'classifier__reg_covar': dataset_algorithm_settings["reg_covar"]}

                self.complete_processing_pipeline.set_params(**formatted_params)
                self.train()
                self.predict()
                lb = preprocessing.LabelBinarizer()
                y_test = lb.fit_transform(self.test_y)
                test_accuracy = np.mean(self.predicted_y.ravel() == y_test.ravel()) * 100
                test_accuracies.append(test_accuracy)

                print('test_accuracy', test_accuracy)
                print(formatted_params)

                try:
                    label_encoder = LabelEncoder()
                    true_labels = label_encoder.fit_transform(self.train_y)

                    pipeline = self._get_preprocessor_pipeline().fit(self.train_x, self.train_y)
                    #pipeline = self.complete_processing_pipeline.fit(self.train_x, self.train_y)
                    data = pipeline.transform(self.train_x)
                    r, c = data.shape

                    if c >= 2:
                        pca_df = pd.DataFrame(data[:, 0:2], columns=["component_1", "component_2"])
                    else:
                        pca_df = pd.DataFrame(data[:, 0], columns=["component_1"])

                    #pca_df["predicted_cluster"] = self.complete_processing_pipeline["classifier"].labels_
                    #pca_df["true_label"] = label_encoder.inverse_transform(true_labels)

                    self.plot_components(pca_df.values, dataset_name + f"_{n_components}")
                except:
                    traceback.print_exc()

                if calculate_bic:
                    bic.append(self.classifier.bic(X))
                    if bic[-1] < lowest_bic:
                        lowest_bic = bic[-1]
                        best_gmm = self.classifier

        if calculate_bic:
            bic = np.array(bic)
            self.plot_bic(n_components_range, cv_types, bic, dataset_name)
        test_accuracies = np.array(test_accuracies)
        self.plot_accuracies(n_components_range, cv_types, test_accuracies, dataset_name)

    def plot_components(self, best_pca, dataset_name):
        lb = preprocessing.LabelBinarizer()
        train_y = lb.fit_transform(self.train_y)

        fig, ax = plt.subplots(1, 1)

        r, c = best_pca.shape

        if c >= 2:
            ax.scatter(best_pca[:, 0], best_pca[:, 1], c=train_y, cmap='rainbow')
        else:
            ax.scatter(best_pca[:, 0], best_pca[:, 0], c=train_y, cmap='rainbow')

        ax.set_title(f"1st vs 2nd PC for dataset {dataset_name}")
        ax.set_xlabel('First principal component')
        ax.set_ylabel('Second Principal Component')
        fig.savefig(rf'{self.output_file_path}/Part2-a_{self.classifier_name}_{dataset_name}_ComponentsScatter.png')
        plt.close(fig)

    def plot_bic(self, n_components_range, cv_types, bic, dataset_name):
        color_iter = itertools.cycle(['navy',
                                      'turquoise',
                                      'cornflowerblue',
                                      'darkorange'])
        bars = []

        # Plot the BIC scores
        plt.figure(figsize=(8, 6))
        fig, ax = plt.subplots(1, 1)
        for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
            xpos = np.array(n_components_range) + .2 * (i - 2)
            bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                          (i + 1) * len(n_components_range)],
                                width=.2, color=color))
        plt.xticks(n_components_range)
        plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
        plt.title(f'BIC score per model for dataset {dataset_name}')
        # xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + \
        #        .2 * np.floor(bic.argmin() / len(n_components_range))
        # plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
        ax.set_xlabel('Number of components')
        ax.legend([b[0] for b in bars], cv_types)
        fig.savefig(rf'{self.output_file_path}/Part1-b_{self.classifier_name}_GMMMSelection_bic.png')
        plt.close(fig)

    def plot_accuracies(self, n_components_range, cv_types, accuracies, dataset_name):
        color_iter = itertools.cycle(['navy',
                                      'turquoise',
                                      'cornflowerblue',
                                      'darkorange'])
        bars = []

        # Plot the BIC scores
        plt.figure(figsize=(8, 6))
        fig, ax = plt.subplots(1, 1)
        for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
            xpos = np.array(n_components_range) + .2 * (i - 2)
            bars.append(plt.bar(xpos, accuracies[i * len(n_components_range):
                                          (i + 1) * len(n_components_range)],
                                width=.2, color=color))
        plt.xticks(n_components_range)
        plt.ylim([accuracies.min() * 1.01 - .01 * accuracies.max(), accuracies.max()])
        plt.title(f'Test Accuracy per model for dataset {dataset_name}')
        ax.set_xlabel('Number of components')
        ax.legend([b[0] for b in bars], cv_types)
        fig.savefig(rf'{self.output_file_path}/Part1-b_{self.classifier_name}_{dataset_name}_GMMMSelection_accuracies.png')
        plt.close(fig)


    def plot_results(self, means, covariances, index, title):
        color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold', 'darkorange'])
        fig, ax = plt.subplots(1, 1) # plt.subplot(2, 1, 1 + index)
        for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):

            v, w = linalg.eigh(covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])
            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_artist(ell)

        # ax.set_xlim(-9., 5.)
        # ax.set_ylim(-3., 6.)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        plt.show()
        return

