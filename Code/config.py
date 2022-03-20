import os
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import neighbors
from sklearn import linear_model

DATA_DIR = os.path.join("..", "Database")
DATA_FILENAME = "data.txt"

NAN = -1
LINE_LENGTH = 200

DATASETS = {
    'Hamberman' :
    {
            'Attributes' : ['Age', 'Operation Year', 'Positive axillary nodes', 'Survival status'],
            'Path' : os.path.join(DATA_DIR, 'Hamberman', DATA_FILENAME)
    },
    'Wine' :
    {
        'Attributes': ['Wine class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash  ', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline'],
        'Path' : os.path.join(DATA_DIR, 'Wine', DATA_FILENAME)
    },
    'Iris' :
    {
        'Attributes' : ['sepal length', 'sepal width', 'petal length', 'petal width', 'Iris type'],
        'Path' : os.path.join(DATA_DIR, 'Iris', DATA_FILENAME)
    }
}
RESULTS_DIR = os.path.join("..", "Results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "{}")

MODELS = {
    xgb.XGBRegressor(): 'XGBoost',
    SGDRegressor(): 'SGDRegressor',
    neighbors.KNeighborsRegressor(): 'KNeighborsRegressor',
    linear_model.LinearRegression(): 'LinearRegression',
    MLPRegressor(): 'Multi Layer Peceptron'
}

MISSING_RATIO = 0.10

MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.8