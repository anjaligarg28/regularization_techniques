import numpy as np
import pandas as pd

class LassoRegression:

    def soft_thresholding_operator(self, learning_rate, alpha):
        return np.sign(self.w) * np.maximum(np.abs(self.w) - learning_rate*alpha, 0)

    def fit(self, X, y, learning_rate=0.01, iterations=1000, alpha=0.1):
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
            dj_dw = (1/self.m)*(np.dot(X.T, error))
            dj_db = (1/self.m) * np.sum(error)
            # update weights and bias
            self.w = self.w - learning_rate*dj_dw
            self.b = self.b - learning_rate*dj_db
            # Apply soft-thresholding to the weights for L1 regularization
            self.w = self.soft_thresholding_operator(learning_rate, alpha)
        return self.b, self.w

    def predict(self, X):
        return np.dot(X, self.w) + self.b
