from collections import defaultdict

from sklearn import preprocessing
from sklearn.metrics import pairwise_distances
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sps
from scipy.linalg import pinv

from src.algorithms.base import BaseClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class RandomProjection(BaseClassifier):
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
        self.classifier = GaussianRandomProjection(random_state=self.seed) #, whiten=True, max_iter=1000, tol=1e-2)

        if parameters:
            self.classifier.set_params(**parameters)

    def run_experiments(self, parameter_groups, dataset_name, dataset_algorithm_settings=None):
        n_components_range = dataset_algorithm_settings["n_components"]
        df_losses = pd.DataFrame()
        seeds = [2, 10, 25, 50, 75, 100]

        for seed in seeds:
            losses = []
            for n_components in n_components_range:
                formatted_params = {'classifier__n_components': n_components,
                                    'classifier__random_state': seed}
                self.complete_processing_pipeline.set_params(**formatted_params)
                self.train()

                data_type_transformer_step = self._get_data_type_transformer_step()
                data_processing_steps = [data_type_transformer_step]
                pipeline = Pipeline(steps=data_processing_steps)

                pipeline.fit(self.train_x)
                X = pipeline.transform(self.train_x)

                loss = self.reconstruction_error(self.classifier, X)
                losses.append(loss)
            df_losses[seed] = losses
        self.plot_losses(n_components_range, df_losses, dataset_name, seeds)

    def plot_components(self, x_ica, dataset_name, best_pcs):
        lb = preprocessing.LabelBinarizer()
        train_y = lb.fit_transform(self.train_y)

        fig, ax = plt.subplots(1, 1)
        ax.scatter(x_ica[:, best_pcs[0]], x_ica[:, best_pcs[1]], c=train_y, cmap='rainbow')
        ax.set_title(f"IC{best_pcs[0]} vs IC{best_pcs[1]} PC for dataset {dataset_name}")
        ax.set_xlabel(f'IC{best_pcs[0]}')
        ax.set_ylabel(f'IC{best_pcs[1]}')
        fig.savefig(rf'{self.output_file_path}/Part3-a_{self.classifier_name}_ComponentsScatter.png')
        plt.close(fig)

    def plot_losses(self, n_components_range, losses, dataset_name, seeds):
        fig, ax = plt.subplots(1, 1)
        ax.plot(n_components_range, losses, marker='o')
        ax.legend([f"seed={seed}" for seed in seeds])
        ax.set_title(f"Projection Loss vs Component for Dataset {dataset_name}")
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("Projection Loss")
        fig.savefig(rf'{self.output_file_path}/Part2-c_{self.classifier_name}_ProjectionLoss.png')
        plt.close(fig)

    @staticmethod
    def reconstruction_error(projections, X):
        W = projections.components_
        if sps.issparse(W):
            W = W.todense()
        p = pinv(W)
        reconstructed = ((p @ W) @ (X.T)).T  # Unproject projected data
        errors = np.square(X - reconstructed)
        return np.nanmean(errors)

    @staticmethod
    def pairwiseDistCorr(X1, X2):
        assert X1.shape[0] == X2.shape[0]

        d1 = pairwise_distances(X1)
        d2 = pairwise_distances(X2)
        return np.corrcoef(d1.ravel(), d2.ravel())[0, 1]

