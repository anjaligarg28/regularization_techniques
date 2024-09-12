import numpy as np
import pandas as pd
"""
Gradient descent algorithm:
initialising weights in n+1 x 1 vector B
define loss function: J = (1/2m)*(mean squared error) = (1/2m)*(for i in range number of data points, error_of_i ka square)
J = (1/2m)*(XB-Y)t*(XB-Y)
J = (1/m)()
dJ/db = gradient = 
while epochs:
    # calculate gradient
    gradient = (1\n)*(Xt(XB-Y))
    B = B - learning_rate * gradient

Args:
    X (dataframe): independent features
    y (dataframe): dependent feature
    learning_rate (float): learning rate for weight updates
    epochs (int): number of iterations
"""
class GradientDescent:
    def __init__(self) -> None:
        pass

    def fit(self, X, y, learning_rate=0.01, iterations=1000):
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
        return self.b, self.w

    def predict(self, X):
        return np.dot(X, self.w) + self.b
