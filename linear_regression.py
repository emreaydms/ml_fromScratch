import numpy as np


class LRModel:
    def __init__(self):
        self.w = None

    def fit(self, x, y):
        raise NotImplementedError("This method should be implemented by the subclass")

    def predict(self, x):
        """
        Predict the target values for the given input data by using the weights.

        Parameters:
        - x (np.ndarray): The input data of shape (n_samples, n_features).

        Returns:
        - np.ndarray: The predicted target values of shape (n_samples,).
        """
        X = np.array(x)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples = X.shape[0]
        X_design = np.hstack([np.ones((n_samples, 1)), X])
        return X_design @ self.w


class LeastSquares(LRModel):
    def __init__(self):
        super().__init__()

    def fit(self, x, y):
        """
        Fit the Least Squares model to the training data.

        Parameters:
        - x (np.ndarray): The input data of shape (n_samples, n_features).
        - y (np.ndarray): The target values of shape (n_samples,).

        This method adds a column of ones to the input data to account for the intercept term,
        and then computes the weights using the normal equation.
        """
        X = np.array(x)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples = X.shape[0]
        X_design = np.hstack([np.ones((n_samples, 1)), X])
        # normal equation with pseudo-inverse
        self.w = np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y
        

class RidgeRegression(LRModel):
    def __init__(self, lambd=1.0):
        super().__init__()
        self.lambd = lambd

    def fit(self, x, y):
        """
        Fit the Ridge Regression model to the training data.

        Parameters:
        - x (np.ndarray): The input data of shape (n_samples, n_features).
        - y (np.ndarray): The target values of shape (n_samples,).

        This method adds a column of ones to the input data to account for the intercept term,
        and then computes the weights using the regularized normal equation.
        """
        X = np.array(x)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples, n_features = X.shape
        X_design = np.hstack([np.ones((n_samples, 1)), X])
        I = np.eye(n_features + 1)
        I[0, 0] = 0  # do not regularize intercept
        A = X_design.T @ X_design + self.lambd * I
        self.w = np.linalg.solve(A, X_design.T @ y)


class LassoRegression(LRModel):
    def __init__(self, alpha=1.0, lr=0.01, epochs=1000):
        super().__init__()
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs

    def fit(self, x, y):
        """
        Fit the Lasso Regression model to the training data.

        Parameters:
        - x (np.ndarray): The input data of shape (n_samples, n_features).
        - y (np.ndarray): The target values of shape (n_samples,).

        This method adds a column of ones to the input data to account for the intercept term,
        and then computes the weights using subgradient descent.
        """
        X = np.array(x)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples = X.shape[0]
        X_design = np.hstack([np.ones((n_samples, 1)), X])
        _, n_features = X_design.shape
        self.w = np.zeros(n_features)
        for _ in range(self.epochs):
            y_pred = X_design @ self.w
            error = y_pred - y
            grad = X_design.T @ error
            subgrad = self.alpha * np.sign(self.w)
            subgrad[0] = 0  # no penalty on intercept
            self.w -= self.lr * (grad + subgrad)


class ElasticNetRegression(LRModel):
    def __init__(self, alpha=1.0, l1_ratio=0.5, lr=0.01, epochs=1000):
        super().__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.lr = lr
        self.epochs = epochs

    def fit(self, x, y):
        """
        Fit the Elastic Net Regression model to the training data.

        Parameters:
        - x (np.ndarray): The input data of shape (n_samples, n_features).
        - y (np.ndarray): The target values of shape (n_samples,).

        This method adds a column of ones to the input data to account for the intercept term,
        and then computes the weights using a combination of L1 and L2 regularization terms.
        """
        X = np.array(x)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples = X.shape[0]
        X_design = np.hstack([np.ones((n_samples, 1)), X])
        _, n_features = X_design.shape
        self.w = np.zeros(n_features)
        for _ in range(self.epochs):
            y_pred = X_design @ self.w
            error = y_pred - y
            grad = X_design.T @ error
            # Elastic net penalty gradients
            l2_pen = self.alpha * (1 - self.l1_ratio) * self.w
            l1_pen = self.alpha * self.l1_ratio * np.sign(self.w)
            l2_pen[0] = 0
            l1_pen[0] = 0
            self.w -= self.lr * (grad + l2_pen + l1_pen)
