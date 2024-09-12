import numpy as np
import pandas as pd

class RidgeRegression:

    # assuming X is a 2D array of input features and y is 1D array of output feature
    def fit(self, X, y, learning_rate=0.01, iterations=1000, alpha=0.01):
        # Convert y to a numpy array if it's a Pandas Series
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y = y.values.flatten()  # Flatten in case it's a DataFrame with a single column
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.m, self.n = X.shape # number of samples and features
        self.w = np.zeros(self.n) # initialising weights to zero
        self.b = 0 # initialising bias to zero
        # gradient descent algorithm
        for _ in range(iterations):
            y_pred = self.predict(X) # calculating predictions to minimize loss bw actual and pred
            error = y_pred-y # loss between predictions and actual
            # calculate the gradients w.r.t current weights and bias
            dj_dw = (1/self.m)*(np.dot(X.T, error) + alpha*self.w)
            dj_db = (1/self.m) * np.sum(error)
            # update weights and bias
            self.w = self.w - learning_rate*dj_dw
            self.b = self.b - learning_rate*dj_db
        return self.b, self.w

    def predict(self, X):
        return np.dot(X, self.w) + self.b
