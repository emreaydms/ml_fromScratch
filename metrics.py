import numpy as np


def mean_squared_error(y_true, y_pred):
    """
    Calculate the mean squared error between true and predicted values.
    
    Parameters:
    - y_true: numpy array of true values
    - y_pred: numpy array of predicted values

    Returns:
    - float: The mean squared error between the true and predicted values.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    """
    Calculate the R2 score between true and predicted values.

    Parameters:
    - y_true (np.ndarray): Array of true values.
    - y_pred (np.ndarray): Array of predicted values.

    Returns:
    - float: The R2 score, which indicates the proportion of
        the variance in the dependent variable that is predictable
        from the independent variable(s).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0