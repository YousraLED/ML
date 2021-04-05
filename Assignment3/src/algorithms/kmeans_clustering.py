import traceback

from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import _kmeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.algorithms.base import BaseClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances


class KmeansClustering(BaseClassifier):
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

    @staticmethod
    def _calculate_manhattan_distance(X, Y=None, Y_norm_squared=None, squared=False, X_norm_squared=None):
        return manhattan_distances(X, Y)

    @staticmethod
    def _calculate_euclidean_distance(X, Y=None, Y_norm_squared=None, squared=False, X_norm_squared=None):
        return euclidean_distances(X, Y, Y_norm_squared=Y_norm_squared, squared=squared, X_norm_squared=X_norm_squared)

    def set_distance_function(self, function_name='euclidean'):
        if function_name == 'manhattan':
            _kmeans.euclidean_distances = self._calculate_manhattan_distance
        elif function_name == 'euclidean':
            _kmeans.euclidean_distances = self._calculate_euclidean_distance

    def _set_classifier(self, parameters):
        self.classifier = KMeans(random_state=self.seed, init='k-means++', n_init=10)

        if parameters:
            self.classifier.set_params(**parameters)

    def run_experiments(self, parameter_groups, dataset_name, dataset_algorithm_settings=None):
        distance_function_names = ['euclidean', 'manhattan']
        n_clusters = parameter_groups["n_clusters"]
        all_wcss = pd.DataFrame(columns=distance_function_names)

        for distance_function_name in distance_function_names:
            wcss = []
            self.set_distance_function(function_name=distance_function_name)
            for parameter_name, parameter_group in parameter_groups.items():
                for parameter in parameter_group:
                    formatted_params = {f'classifier__{parameter_name}': parameter}

                    self.complete_processing_pipeline.set_params(**formatted_params)
                    self.train()
                    self.predict()

                    try:
                        label_encoder = LabelEncoder()
                        true_labels = label_encoder.fit_transform(self.train_y)

                        # pipeline = self._get_preprocessor_pipeline().fit(self.train_x,  self.train_y)
                        pipeline = self.complete_processing_pipeline.fit(self.train_x, self.train_y)
                        data = pipeline.transform(self.train_x)
                        r, c = data.shape

                        if c >= 2:
                            pca_df = pd.DataFrame(data[:, 0:2], columns=["component_1", "component_2"])
                        else:
                            pca_df = pd.DataFrame(data[:, 0], columns=["component_1"])

                        pca_df["predicted_cluster"] = self.complete_processing_pipeline["classifier"].labels_
                        pca_df["true_label"] = label_encoder.inverse_transform(true_labels)

                        self.plot_components(pca_df.values, dataset_name + f"_{parameter}")
                    except:
                        traceback.print_exc()

                    centroids = self.classifier.cluster_centers_
                    wcss.append(self.classifier.inertia_)
                    print(parameter)
                    print(centroids)
            all_wcss[distance_function_name] = wcss

        fig, ax = plt.subplots(1, 1)
        ax.plot(n_clusters, all_wcss, alpha=0.7)
        ax.set_title(f'The Elbow Method Graph for dataset: {dataset_name}')
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('WCSS')
        ax.legend(distance_function_names)
        fig.savefig(rf'{self.output_file_path}/Part1-a_{self.classifier_name}_{dataset_name}_ElbowMethodGraph.png')
        plt.close(fig)

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