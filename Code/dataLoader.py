

import numpy as np
import pandas

from config import NAN, LINE_LENGTH, DATA_DIR, DATA_FILENAME, DATASETS

import warnings
warnings.filterwarnings("ignore")

np.set_printoptions(linewidth=LINE_LENGTH)



class DataLoader:
    def __init__(self):
        self.__dataset = {
            'Hamberman': DataLoader.__load_hamberman_data('Hamberman'),
            'Iris': DataLoader.__load_iris_data('Iris'),
            'Wine': DataLoader.__load_wine_data('Wine'),
        }

        print('Datasets Loaded Successfully!')

    @staticmethod
    def __load_hamberman_data(dataset_name):
        data = np.array(pandas.read_csv(filepath_or_buffer=DATASETS[dataset_name]['Path'], sep=',', header=None))
        labels = [int(value) - 1 for value in data[:, len(data[0]) - 1]]
        return {'data': np.array(data), 'attributes': DATASETS[dataset_name]['Attributes'], 'transactions' : DataLoader.data_to_transactions(np.array(data), DATASETS[dataset_name]['Attributes'])}

    @staticmethod
    def __load_iris_data(dataset_name):
        data = np.array(pandas.read_csv(filepath_or_buffer=DATASETS[dataset_name]['Path'], sep=',', header=None))
        target = dict([(y, x + 1) for x, y in enumerate(sorted(set(data[:, len(data[0]) - 1])))])
        labels = [int(target[x]) - 1 for x in data[:, len(data[0]) - 1]]
        for d in data:
            d[-1] = int(target[d[-1]]) - 1
        return {'data': np.array(data), 'attributes': DATASETS[dataset_name]['Attributes'], 'transactions' : DataLoader.data_to_transactions(np.array(data), DATASETS[dataset_name]['Attributes'])}

    @staticmethod
    def __load_wine_data(dataset_name):
        data = np.array(pandas.read_csv(filepath_or_buffer=DATASETS[dataset_name]['Path'], sep=',', header=None))
        labels = [int(value) - 1 for value in data[:, 0]]
        return {'data': np.array(data), 'attributes': DATASETS[dataset_name]['Attributes'], 'transactions' : DataLoader.data_to_transactions(np.array(data), DATASETS[dataset_name]['Attributes'])}

    # This method should get also the attributes for the apriori rule
    @staticmethod
    def data_to_transactions(data, attributes):
        transactions =[]
        for d in data:
            if NAN in d:
                continue
            transactions.append(list(zip(attributes, d)))
        return transactions

    def get_dataset(self):
        return self.__dataset
