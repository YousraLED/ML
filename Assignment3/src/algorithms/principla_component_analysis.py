from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.algorithms.base import BaseClassifier
import pandas as pd
import matplotlib.pyplot as plt


class PrincipalComponentAnalysis(BaseClassifier):
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
        self.classifier = PCA(random_state=self.seed, whiten=True) #, svd_solver='randomized')

        if parameters:
            self.classifier.set_params(**parameters)

    def run_experiments(self, parameter_groups, dataset_name, dataset_algorithm_settings=None):
        n_components_range = dataset_algorithm_settings["n_components"]
        best_component_num = dataset_algorithm_settings["best_component_num"]
        best_pca = None
        losses = []
        for n_components in n_components_range:
            formatted_params = {'classifier__n_components': n_components}
            self.complete_processing_pipeline.set_params(**formatted_params)
            self.train()

            data_type_transformer_step = self._get_data_type_transformer_step()
            data_processing_steps = [data_type_transformer_step]
            pipeline = Pipeline(steps=data_processing_steps)

            pipeline.fit(self.train_x)
            X = pipeline.transform(self.train_x)

            # scaler = StandardScaler()
            # scaler.fit(X)
            # X = scaler.transform(X)
            x_pca = self.classifier.transform(X)
            x_projected = self.classifier.inverse_transform(x_pca)
            loss = ((X - x_projected) ** 2).mean()
            losses.append(loss)

            if n_components == best_component_num:
                best_pca = x_pca

            print(self.classifier.explained_variance_ratio_)
            print(self.classifier.n_components_ )

        self.plot_components(best_pca, dataset_name)
        self.plot_variance(n_components_range, dataset_name)
        self.plot_losses(n_components_range, losses, dataset_name)

    def plot_components(self, best_pca, dataset_name):
        lb = preprocessing.LabelBinarizer()
        train_y = lb.fit_transform(self.train_y)

        fig, ax = plt.subplots(1, 1)
        ax.scatter(best_pca[:, 0], best_pca[:, 1], c=train_y, cmap='rainbow')
        ax.set_title(f"1st vs 2nd PC for dataset {dataset_name}")
        ax.set_xlabel('First principal component')
        ax.set_ylabel('Second Principal Component')
        fig.savefig(rf'{self.output_file_path}/Part3-a_{self.classifier_name}_ComponentsScatter.png')
        plt.close(fig)

    def plot_variance(self, n_components_range, dataset_name):
        fig, ax = plt.subplots(1, 1)
        ax.plot(n_components_range, self.classifier.explained_variance_ratio_.cumsum(), marker='o')
        ax.set_title(f"Explained Variance By Component for Dataset {dataset_name}")
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("Cumulative Explained Variance")
        fig.savefig(rf'{self.output_file_path}/Part2-b_{self.classifier_name}_VarianceByComponent.png')
        plt.close(fig)

    def plot_losses(self, n_components_range, losses, dataset_name):
        fig, ax = plt.subplots(1, 1)
        ax.plot(n_components_range, losses, marker='o')
        ax.set_title(f"Projection Loss vs Component for Dataset {dataset_name}")
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("Projection Loss")
        fig.savefig(rf'{self.output_file_path}/Part2-c_{self.classifier_name}_ProjectionLoss.png')
        plt.close(fig)

