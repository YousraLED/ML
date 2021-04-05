from sklearn import preprocessing
from sklearn.decomposition import FastICA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.algorithms.base import BaseClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class IndependentComponentAnalysis(BaseClassifier):
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
        self.classifier = FastICA(random_state=self.seed, whiten=True, max_iter=1000, tol=1e-2) #, svd_solver='randomized')

        if parameters:
            self.classifier.set_params(**parameters)

    def run_experiments(self, parameter_groups, dataset_name, dataset_algorithm_settings=None):
        n_components_range = dataset_algorithm_settings["n_components"]
        best_pcs = dataset_algorithm_settings["best_pcs"]
        kurts = []
        best_ica = None
        last_best_ica_kurt = 0
        for n_components in n_components_range:
            formatted_params = {'classifier__n_components': n_components}
            self.complete_processing_pipeline.set_params(**formatted_params)
            self.train()

            data_type_transformer_step = self._get_data_type_transformer_step()
            data_processing_steps = [data_type_transformer_step]
            pipeline = Pipeline(steps=data_processing_steps)

            pipeline.fit(self.train_x)
            X = pipeline.transform(self.train_x)

            x_ica = self.classifier.transform(X)
            df_x_ica = pd.DataFrame(x_ica)
            x_ica_kurt = df_x_ica.kurt(axis=0).abs().mean()
            print(n_components, x_ica_kurt)

            if x_ica_kurt > last_best_ica_kurt:
                best_ica = x_ica
            kurts.append(x_ica_kurt)

        self.plot_components(best_ica, dataset_name, best_pcs)
        self.plot_kurtosis(n_components_range, kurts, dataset_name)

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

    def plot_kurtosis(self, n_components_range, kurts, dataset_name):
        fig, ax = plt.subplots(1, 1)
        ax.plot(n_components_range, kurts, marker='o')
        ax.set_title(f"Kurtosis vs Component for Dataset {dataset_name}")
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("Kurtosis")
        fig.savefig(rf'{self.output_file_path}/Part3-b_{self.classifier_name}_KurtosisVsComponent.png')
        plt.close(fig)

