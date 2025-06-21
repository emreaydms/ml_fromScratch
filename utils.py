import numpy as np


class MinMaxScaler():
    def __init__(self, min_limit=0, max_limit=1):
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.data_min = None
        self.data_max = None

    def fit(self, x):
        """
        Calculate the minimum and maximum values for each feature in the dataset.

        Parameters:
        - x (np.ndarray): The input data of shape (n_samples, n_features).

        This method computes the minimum and maximum values for each feature (column-wise)
        in the input data and stores them in the instance attributes `data_min` and `data_max`.

        Returns:
        - self: The instance of the MinMaxScaler with the computed `data_min` and `data_max`.
        """
        self.data_min = np.min(x, axis=0)
        self.data_max = np.max(x, axis=0)
        return self

    def transform(self, x):
        """
        Scale the input data to the given range.

        Parameters:
        - x (np.ndarray): The input data of shape (n_samples, n_features).

        This method scales each feature to the range [min_limit, max_limit] using the
        computed minimum and maximum values from the `fit` method.

        Returns:
        - np.ndarray: The scaled data of shape (n_samples, n_features).
        """


        if self.data_min is None or self.data_max is None:
            raise ValueError("MinMaxScaler not fitted yet. Call 'fit' with training data.")
        data_range = self.data_max - self.data_min


        scale = (self.max_limit - self.min_limit) / np.where(data_range == 0, 1, data_range)
        x_scaled = (x - self.data_min) * scale + self.min_limit
        return x_scaled

    def fit_transform(self, x):
        """
        Fit the scaler and then transform the data.

        Parameters:
        - x (np.ndarray): The input data of shape (n_samples, n_features).

        This method first fits the scaler to the input data and then scales the data.

        Returns:
        - np.ndarray: The scaled data of shape (n_samples, n_features).
        """
        self.fit(x)
        return self.transform(x)
    
    def inverse_transform(self, x_scaled):
        """
        Reverse the scaling transformation.

        Parameters:
        - x_scaled (np.ndarray): The scaled data of shape (n_samples, n_features).

        This method reverses the scaling transformation to get the original data.

        Returns:
        - np.ndarray: The original data of shape (n_samples, n_features).
        """
        if self.data_min is None or self.data_max is None:
            raise ValueError("MinMaxScaler not fitted yet. Call 'fit' with training data.")
        data_range = self.data_max - self.data_min
        scale = (self.max_limit - self.min_limit) / np.where(data_range == 0, 1, data_range)
        x_orig = (x_scaled - self.min_limit) / scale + self.data_min
        return x_orig

class TrainTestSplit():
    def __init__(self, tt_ratio=0.5, shuffle=True):
        self.tt_ratio = tt_ratio
        self.shuffle = shuffle

    def shuffler(self, x, y):
        """
        Shuffle the input data and corresponding target values.

        Parameters:
        - x (np.ndarray): The input data of shape (n_samples, n_features).
        - y (np.ndarray): The target values of shape (n_samples,).

        This method shuffles the input data and the corresponding target values
        in unison, ensuring that the relationship between the input data and
        target values is maintained.

        Returns:
        - x (np.ndarray): The shuffled input data of shape (n_samples, n_features).
        - y (np.ndarray): The shuffled target values of shape (n_samples,).
        """
        if x.shape[0] != y.shape[0]:
            raise ValueError("Shapes of x and y must match.")
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)
        return x[idx], y[idx]
    
    def split(self, x, y):
        """
        Split the input data and corresponding target values into training and testing sets.

        Parameters:
        - x (np.ndarray): The input data of shape (n_samples, n_features).
        - y (np.ndarray): The target values of shape (n_samples,).

        This method splits the input data and the corresponding target values
        into training and testing sets based on the specified train-test ratio.

        Returns:
        - x_train (np.ndarray): The training input data of shape (n_train_samples, n_features).
        - x_test (np.ndarray): The testing input data of shape (n_test_samples, n_features).
        - y_train (np.ndarray): The training target values of shape (n_train_samples,).
        - y_test (np.ndarray): The testing target values of shape (n_test_samples,).
        """
        n_samples = x.shape[0]
        n_train = int(np.floor(n_samples * self.tt_ratio))
        x_train = x[:n_train]
        x_test = x[n_train:]
        y_train = y[:n_train]
        y_test = y[n_train:]
        return x_train, x_test, y_train, y_test
    
    def __call__(self, x, y):
        """
        Perform the train-test split on the input data and target values.

        Parameters:
        - x (np.ndarray): The input data of shape (n_samples, n_features).
        - y (np.ndarray): The target values of shape (n_samples,).

        This method first shuffles the input data and target values if specified,
        and then splits the data into training and testing sets based on the
        specified train-test ratio.

        Returns:
        - x_train (np.ndarray): The training input data of shape (n_train_samples, n_features).
        - x_test (np.ndarray): The testing input data of shape (n_test_samples, n_features).
        - y_train (np.ndarray): The training target values of shape (n_train_samples,).
        - y_test (np.ndarray): The testing target values of shape (n_test_samples,).
        """
        if self.shuffle:
            x, y = self.shuffler(x, y)
        return self.split(x, y)
