from configobj import ConfigObj
import pandas as pd
import pandas_profiling as pdp


class EDA:
    def __init__(self, config_file_path, root_directory):
        self.root_directory = root_directory
        self.config_file_path = config_file_path
        self.config = ConfigObj(self.config_file_path)

        self.COLUMNS_TO_EXCLUDE = 'COLUMNS_TO_EXCLUDE'
        self.COLUMN_NAME_Y = 'COLUMN_NAME_Y'
        self.COLUMN_INDEX_Y = 'COLUMN_INDEX_Y'
        self.HEADRE_EXISTS = 'HEADRE_EXISTS'
        self.DELIMITER = 'DELIMITER'
        self.ROOT_DIRECTORY = 'ROOT_DIRECTORY'
        self.FILE_PATH = 'FILE_PATH'
        self.INCLUDE = 'INCLUDE'
        self.DATASETS = 'DATASETS'
        self.DATASET_NAME = 'DATASET_NAME'
        self.SCALER = 'SCALER'
        self.ENCODER = 'ENCODER'
        self.DIMENSIONALITY_REDUCER = 'DIMENSIONALITY_REDUCER'
        self.REQUIRES_BALANCING = 'REQUIRES_BALANCING'

        self.Y_DATA = 'Y_DATA'
        self.X_DATA = 'X_DATA'

        self.dataset_configurations = dict()
        self.datasets = dict()
        self._extract_dataset_configurations_from_file()

    def _extract_dataset_configurations_from_file(self):
        dataset_configurations = self.config[self.DATASETS]

        for dataset_name in dataset_configurations.keys():
            dataset_section = dataset_configurations[dataset_name]
            include = dataset_section.as_bool(self.INCLUDE)

            if include:
                columns_to_exclude = sorted(list(map(str, dataset_section[self.COLUMNS_TO_EXCLUDE])))
                header_exists = dataset_section.as_bool(self.HEADRE_EXISTS)
                requires_balancing = dataset_section.as_bool(self.REQUIRES_BALANCING)
                configuration = {
                    self.ROOT_DIRECTORY: dataset_section[self.ROOT_DIRECTORY],
                    self.FILE_PATH: dataset_section[self.FILE_PATH],
                    self.DELIMITER: dataset_section[self.DELIMITER],
                    self.HEADRE_EXISTS: header_exists,
                    self.COLUMN_INDEX_Y: dataset_section[self.COLUMN_INDEX_Y],
                    self.COLUMN_NAME_Y: dataset_section[self.COLUMN_NAME_Y],
                    self.COLUMNS_TO_EXCLUDE: columns_to_exclude,
                    self.SCALER: dataset_section[self.SCALER],
                    self.ENCODER: dataset_section[self.ENCODER],
                    self.DIMENSIONALITY_REDUCER: dataset_section[self.DIMENSIONALITY_REDUCER],
                    self.REQUIRES_BALANCING: requires_balancing
                }
                self.dataset_configurations[dataset_name] = configuration

    def load_data_using_configuration(self):
        for dataset_name, configuration in self.dataset_configurations.items():
            # print(configuration)
            file_path = fr"{self.root_directory}\datasets\{configuration[self.FILE_PATH]}"
            data = pd.read_csv(file_path, sep=configuration[self.DELIMITER])

            columns_to_exclude = configuration[self.COLUMNS_TO_EXCLUDE]
            if len(columns_to_exclude):
                data.drop(columns_to_exclude, axis=1, inplace=True)

            if configuration[self.COLUMN_NAME_Y]:
                column_y = configuration[self.COLUMN_NAME_Y]
                columns_x = [column for column in data.columns if column != column_y]

                y = data[column_y]
                x = data[columns_x]

            elif configuration[self.COLUMN_INDEX_Y]:
                column_indexes = range(0, len(data.columns))
                column_index_y = int(configuration[self.COLUMN_INDEX_Y])
                column_indexes_x = [column for column in column_indexes if column != column_index_y]

                # print(column_indexes)
                # print(column_index_y)
                # print(column_indexes_x)

                y = data.iloc[:, column_index_y]
                x = data.drop(data.columns[column_index_y], axis=1) #data.loc[:, column_indexes_x]

            # print(dataset_name)
            # print(data.head())
            # print(y.shape)
            # print(x.shape)
            # print(y.head())
            # print(x.head())

            dataset = {self.Y_DATA: y, self.X_DATA: x}

            self.datasets[dataset_name] = dataset

    def get_data(self):
        return self.datasets

    def analyze_data(self):
        for dataset_name, dataset in self.datasets.items():
            root_directory = self.dataset_configurations[dataset_name][self.ROOT_DIRECTORY]
            analysis_output_file_path = fr"{root_directory}\analysis_{dataset_name}.html"
            data = dataset[self.X_DATA]
            data[self.Y_DATA] = dataset[self.Y_DATA]
            profile = pdp.ProfileReport(data)
            profile.to_file(output_file=analysis_output_file_path)


if __name__ == '__main__':
    root_directory_path = r'..\datasets'
    config_file_path = "config_datasets.ini"  # fr"{root_directory_path}/config.ini"
    eda = EDA(config_file_path)
    eda.load_data_using_configuration()
    eda.analyze_data()
