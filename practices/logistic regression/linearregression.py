import numpy as np

class LinearRegression:
    def fit(self, _X, _y):
        self.X = np.array(_X)
        self.y = np.array(_y)
        self.w = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.y)

    def predict(self, _x):
        x = np.array(_x)
        return x.dot(self.w)
    def get_w(self):
        return self.w
    def execute(self, _X, _y):
        self.fit(_X, _y)
        return self.predict(_X)