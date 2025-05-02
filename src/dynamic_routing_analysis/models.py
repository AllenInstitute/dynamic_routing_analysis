import logging

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import solve
from sklearn.linear_model import ElasticNet as SklearnElasticNet
from sklearn.linear_model import LassoLars as SklearnLasso

logger = logging.getLogger(__name__) # debug < info < warning < error

class Ridge:
    def __init__(self, lam=None, W=None):
        self.lam = lam
        self.r2 = None
        self.mean_r2 = None
        self.W = W

    def fit(self, X, y):
        '''
        Analytical OLS solution with added L2 regularization penalty.
        Y: shape (n_timestamps * n_cells)
        X: shape (n_timestamps * n_kernel_params)
        lam (float): Strength of L2 regularization (hyperparameter to tune)
        '''

        # Compute the weights
        XtX = X.T @ X
        Xty = X.T @ y
        try:
            if self.lam == 0:
                self.W = np.dot(np.linalg.inv(XtX), Xty)
                self.W = self.W if ~np.isnan(np.sum(self.W)) else np.zeros(self.W.shape) # if weights contain NaNs, set to zeros
            else:
                a = XtX + self.lam * np.eye(X.shape[-1], dtype=np.float64)
                self.W = solve(a, Xty, assume_a='pos')
                # self.W = np.dot(np.linalg.inv(np.dot(X.T, X) + self.lam * np.eye(X.shape[-1])),
                #                 np.dot(X.T, y))
        except LinAlgError as e:
            logger.info(f"Matrix inversion failed due to a linear algebra error:{e}. Falling back to pseudo-inverse.")
            # Fallback to pseudo-inverse
            if self.lam == 0:
                self.W = np.dot(np.linalg.pinv(XtX), Xty)
            else:
                a = XtX + self.lam * np.eye(X.shape[-1], dtype=np.float64)
                self.W = solve(a, Xty, assume_a='pos')
                # self.W = np.dot(np.linalg.pinv(np.dot(X.T, X) + self.lam * np.eye(X.shape[-1])),
                #                 np.dot(X.T, y))
        except Exception as e:
            print("Unexpected error encountered:", e)
            raise  # Re-raise the exception to propagate unexpected errors

        assert ~np.isnan(np.sum(self.W)), 'Weights contain NaNs'

        self.mean_r2 = self.score(X, y)

        return self

    def get_params(self, deep=True):
        return {'lam': self.lam}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def score(self, X, y):
        '''
        Computes the fraction of variance in fit_trace_arr explained by the linear model y = X*W
        y: (n_timepoints, n_cells)
        W: (kernel_params, n_cells)
        X: (n_timepoints, n_kernel_params)
        '''
        # Y = X.values @ W.values
        y_pred = self.predict(X)
        var_total = np.var(y, axis=0)  # Total variance in the ophys trace for each cell
        var_resid = np.var(y - y_pred, axis=0)  # Residual variance in the difference between the model and data
        self.r2 = (var_total - var_resid) / var_total
        return np.nanmean(self.r2)  # Fraction of variance explained by linear model

    def predict(self, X):
        y = np.dot(X, self.W)
        return y


class Lasso:
    def __init__(self, lam=None, W=None):
        self.lam = lam  # Regularization strength (alpha in sklearn)
        self.W = W  # Model weights
        self.r2 = None  # R^2 for each output variable
        self.mean_r2 = None  # Mean R^2 across all output variables

    def fit(self, X, y):
        '''
        Fits the Lasso model using coordinate descent.

        Parameters:
        X: np.ndarray
            Input data matrix of shape (n_samples, n_features).
        y: np.ndarray
            Target matrix of shape (n_samples, n_targets).

        Returns:
        self
        '''
        try:
            # Use sklearn's Lasso for coordinate descent
            lasso = SklearnLasso(alpha=self.lam, fit_intercept=False, max_iter=1000)

            # If y has multiple targets, fit each one independently
            if y.ndim == 1:
                lasso.fit(X, y)
                self.W = lasso.coef_
            else:
                self.W = np.zeros((X.shape[1], y.shape[1]))
                for i in range(y.shape[1]):
                    lasso.fit(X, y[:, i])
                    self.W[:, i] = lasso.coef_

        except Exception as e:
            print("Unexpected error encountered:", e)
            raise  # Re-raise the exception to propagate unexpected errors

        assert ~np.isnan(np.sum(self.W)), 'Weights contain NaNs'

        self.mean_r2 = self.score(X, y)
        return self

    def get_params(self, deep=True):
        return {'lam': self.lam}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def score(self, X, y):
        '''
        Computes the fraction of variance in y explained by the linear model y = X*W.

        Parameters:
        X: np.ndarray
            Input data matrix of shape (n_samples, n_features).
        y: np.ndarray
            Target matrix of shape (n_samples, n_targets).

        Returns:
        float
            Mean R^2 across all targets.
        '''
        y_pred = self.predict(X)
        var_total = np.var(y, axis=0)  # Total variance in the target
        var_resid = np.var(y - y_pred, axis=0)  # Residual variance
        self.r2 = (var_total - var_resid) / var_total
        return np.nanmean(self.r2)

    def predict(self, X):
        '''
        Predicts outputs using the linear model.

        Parameters:
        X: np.ndarray
            Input data matrix of shape (n_samples, n_features).

        Returns:
        np.ndarray
            Predicted outputs of shape (n_samples, n_targets).
        '''
        return np.dot(X, self.W)


class ElasticNet:
    def __init__(self, lam = None, l1_ratio=None, W=None):
        self.lam = lam  # Regularization strength (alpha in sklearn)
        self.l1_ratio = l1_ratio  # Mix between L1 and L2 regularization
        self.W = W  # Model weights
        self.r2 = None  # R^2 for each output variable
        self.mean_r2 = None  # Mean R^2 across all output variables

    def fit(self, X, y):
        '''
        Fits the ElasticNet model using coordinate descent.

        Parameters:
        X: np.ndarray
            Input data matrix of shape (n_samples, n_features).
        y: np.ndarray
            Target matrix of shape (n_samples, n_targets).

        Returns:
        self
        '''
        try:
            # Use sklearn's ElasticNet for coordinate descent
            elastic_net = SklearnElasticNet(alpha=self.lam, l1_ratio=self.l1_ratio, fit_intercept=False, max_iter=1000)

            # If y has multiple targets, fit each one independently
            if y.ndim == 1:
                elastic_net.fit(X, y)
                self.W = elastic_net.coef_
            else:
                self.W = np.zeros((X.shape[1], y.shape[1]))
                for i in range(y.shape[1]):
                    elastic_net.fit(X, y[:, i])
                    self.W[:, i] = elastic_net.coef_

        except Exception as e:
            print("Unexpected error encountered:", e)
            raise  # Re-raise the exception to propagate unexpected errors

        assert ~np.isnan(np.sum(self.W)), 'Weights contain NaNs'

        self.mean_r2 = self.score(X, y)
        return self

    def get_params(self, deep=True):
        return {'lam': self.lam,'l1_ratio': self.l1_ratio}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def score(self, X, y):
        '''
        Computes the fraction of variance in y explained by the linear model y = X*W.

        Parameters:
        X: np.ndarray
            Input data matrix of shape (n_samples, n_features).
        y: np.ndarray
            Target matrix of shape (n_samples, n_targets).

        Returns:
        float
            Mean R^2 across all targets.
        '''
        y_pred = self.predict(X)
        var_total = np.var(y, axis=0)  # Total variance in the target
        var_resid = np.var(y - y_pred, axis=0)  # Residual variance
        self.r2 = (var_total - var_resid) / var_total
        return np.nanmean(self.r2)

    def predict(self, X):
        '''
        Predicts outputs using the linear model.

        Parameters:
        X: np.ndarray
            Input data matrix of shape (n_samples, n_features).

        Returns:
        np.ndarray
            Predicted outputs of shape (n_samples, n_targets).
        '''
        return np.dot(X, self.W)




class ReducedRankRegression:
    def __init__(self, rank=None, W=None):
        self.rank = rank # Desired rank for dimensionality reduction
        self.W = W  # Model weights
        self.r2 = None  # R^2 for each output variable
        self.mean_r2 = None  # Mean R^2 across all output variables

    def fit(self, X, y):
        '''
        Fits the Reduced Rank Regression model.

        Parameters:
        X: np.ndarray
            Input data matrix of shape (n_samples, n_features).
        y: np.ndarray
            Target matrix of shape (n_samples, n_targets).

        Returns:
        self
        '''
        try:
            # Compute the OLS solution for weights
            CXX = np.dot(X.T, X)

            # Compute the covariance matrix C_XY
            CXY = np.dot(X.T, y)

            # Perform Singular Value Decomposition (SVD) on C_XY @ W_OLS
            U, s, Vt = np.linalg.svd(np.dot(CXY.T, np.dot(np.linalg.pinv(CXX), CXY)))

            # Reduce the rank by keeping the top `rank` singular values
            if self.rank is not None:
                Vt = Vt[:self.rank.astype(int), :].T

            # Compute the weight matrix W
            self.W =  np.dot(np.linalg.pinv(CXX), np.dot(CXY, np.dot(Vt, Vt.T)))
        except Exception as e:
            print("Unexpected error encountered:", e)
            raise  # Re-raise the exception to propagate unexpected errors

        assert ~np.isnan(np.sum(self.W)), 'Weights contain NaNs'

        self.mean_r2 = self.score(X, y)
        return self

    def get_params(self, deep=True):
        return {'rank': self.rank}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def score(self, X, y):
        '''
        Computes the fraction of variance in y explained by the linear model y = X*W.

        Parameters:
        X: np.ndarray
            Input data matrix of shape (n_samples, n_features).
        y: np.ndarray
            Target matrix of shape (n_samples, n_targets).

        Returns:
        float
            Mean R^2 across all targets.
        '''
        y_pred = self.predict(X)
        var_total = np.var(y, axis=0)  # Total variance in the target
        var_resid = np.var(y - y_pred, axis=0)  # Residual variance
        self.r2 = (var_total - var_resid) / var_total
        return np.nanmean(self.r2)

    def predict(self, X):
        '''
        Predicts outputs using the linear model.

        Parameters:
        X: np.ndarray
            Input data matrix of shape (n_samples, n_features).

        Returns:
        np.ndarray
            Predicted outputs of shape (n_samples, n_targets).
        '''
        return np.dot(X, self.W)

