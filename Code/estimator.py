import numpy as np
from config import NAN
import warnings
warnings.filterwarnings("ignore")


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

class Estimator:
    def __init__(self, data):
        self.data = data
        self.complete_rows, self.incomplete_rows = self.__extract_rows()

    # Extract complete and incomplete rows
    def __extract_rows(self):
        rows, columns = len(self.data), len(self.data[0])
        complete_rows, incomplete_rows = [], []
        for i in range(rows):
            if NAN in self.data[i]:
                incomplete_rows.append(i)
            else:
                complete_rows.append(i)
        return np.array(complete_rows), np.array(incomplete_rows)

    # Estimate the missing values
    def estimate_missing_value(self, model):
        estimated_data = []
        complete_data = np.array([self.data[x] for x in self.complete_rows])
        incomplete_data = np.array([self.data[x] for x in self.incomplete_rows])

        # Iterate over each column(attribute)
        for column, value in enumerate(incomplete_data.transpose()):
            # If this column has a missing value, we want to use the complete dataset to predict the missing value at this attribute
            ind_rows = np.where(value == NAN)[0]
            if len(ind_rows) > 0:
                # Use the complete_data to predict
                # The column(attribute) with the missing value will be the predict label - (y)
                # The other columns(attributes) will be used as x train to train the model for y prediction
                x_train = np.delete(complete_data.transpose(), column, 0).transpose()
                y_train = np.array(complete_data[:, column])

                # Fit the model on the complete data (x -- y)
                model.fit(x_train, y_train)

                # Take all the rows from the incomplete data that has a missing value at the current predict attribute
                x_test = []
                x_test_temp = np.delete(incomplete_data.transpose(), column, 0).transpose()
                for i in ind_rows:
                    x_test.append(x_test_temp[i])

                # Predict the missing value for this row in the incomplete data
                predicted = model.predict(np.array(x_test))

                # Because we are using rules, although we had used a regression model, we want find the nearest class.
                for i, row_num in enumerate(ind_rows):
                    estimated_row = self.data[self.incomplete_rows[row_num]].copy()
                    estimated_row[column] = find_nearest(y_train, predicted[i]) # TODO, add comments
                    estimated_data.append(estimated_row)
        # Return the complete data and the estimated data
        return np.concatenate((complete_data, np.array(estimated_data)), axis=0)
