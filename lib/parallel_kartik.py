'''
Author - Kartik

File runs algorithms parallely.

Libraries: Daal4py, Sklearn

Algorithms: Ridge regression, KMeans, SVD
'''

import time
import numpy as np
import pandas as pd
import daal4py as d4p
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


class Parallel_k():

    def __init__(self, logger, latency, metrics):
            self.logger = logger
            self.latency = latency
            self.metrics = metrics

    # Ridge Regression
    def ridgeRegression(self, X_train, X_test, y_train, y_test, target):
        start_time = time.time()
        train_algo = d4p.ridge_regression_training(distributed=True, interceptFlag=True)
        train_result = train_algo.compute(X_train, y_train)
        predict_algo = d4p.ridge_regression_prediction()
        predict_result = predict_algo.compute(X_test, train_result.model)
        # stop_time = time.time()
        pd_predict = predict_result.prediction
        mse = mean_squared_error(y_test, pd_predict)
        r2score = r2_score(y_test, pd_predict)
        return (pd_predict, mse, r2score, time.time() - start_time)

    def KMeans(self, nClusters, X):
        kmeans_start_time = time.time()
        maxIter = 5
        init_algo = d4p.kmeans_init(nClusters=nClusters, distributed=True, method="randomDense")
        train_result = init_algo.compute(X)
        # The results provides the initial centroids
        assert train_result.centroids.shape[0] == nClusters
        # configure kmeans main object: we also request the cluster assignments
        algo = d4p.kmeans(nClusters, maxIter, assignFlag=True)
        # compute the clusters/centroids
        result = algo.compute(X, train_result.centroids)
        # Kmeans result objects provide assignments (if requested), centroids, goalFunction, nIterations and objectiveFunction
        assert result.centroids.shape[0] == nClusters
        assert result.assignments.shape == (X.shape[0], 1)
        assert result.nIterations <= maxIter
        return (result, time.time() - kmeans_start_time)

    def svd(self, data):
        svd_start_time = time.time()
        algo = d4p.svd(distributed=True)
        result = algo.compute(data)
        assert result.singularValues.shape == (1, data.shape[1])
        assert result.rightSingularMatrix.shape == (data.shape[1], data.shape[1])
        assert result.leftSingularMatrix.shape == data.shape

        if hasattr(data, 'toarray'):
            data = data.toarray()  # to make the next assertion work with scipy's csr_matrix
        assert np.allclose(data, np.matmul(np.matmul(result.leftSingularMatrix, np.diag(result.singularValues[0])),
                                           result.rightSingularMatrix))
        return (result, time.time() - svd_start_time)

#TO BE COMPLETED
    # def sklearn_ridge(self, X_train, X_test, y_train, y_test, target):
    #     ridge_start = time.time()
    #     model = Ridge(alpha=1.0)
    #     model.fit(X_train, y_train)
    #     y_pred = model.predict(X_test)
    #     skRidge_mse = mean_squared_error(y_test, y_pred)
    #     skRidge_r2score = r2_score(y_test, y_pred)
    #     return (y_pred, skRidge_mse, skRidge_r2score, time.time() - ridge_start)
    #
    # def sklearn_kmeans(self, nClusters, X):
    #     skKMeans_start = time.time()
    #     kmeans = KMeans(n_clusters=nClusters, random_state=0).fit(X)
    #     P = kmeans.predict(X)
    #     return (P, time.time() - skKMeans_start)