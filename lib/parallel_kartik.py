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
    def ridgeRegression(self, Data_Path, X_test,y_test, target):
        start_time = time.time()
        d4p.daalinit()
        file = Data_Path + str(d4p.my_procid()+1)+".csv"
        train_algo = d4p.ridge_regression_training(distributed=True, interceptFlag=True)
        data = pd.read_csv(file)
        X = data.drop(columns=target)
        y= data[target]
        train_result = train_algo.compute(X, y)
        if d4p.my_procid() == 0:
            predict_algo = d4p.ridge_regression_prediction()
            # now predict using the model from the training above
            predict_result = d4p.ridge_regression_prediction().compute(X_test, train_result.model)
            # The prediction result provides prediction
            assert predict_result.prediction.shape == (X_test.shape[0], y_test.shape[1])
        print('Ridge completed!')
        d4p.daalfini()
        mse = mean_squared_error(y_test, predict_result.prediction)
        r2score = r2_score(y_test, predict_result.prediction)
        return (predict_result.prediction, mse, r2score, time.time() - start_time)

    def KMeans(self, nClusters, Data_PAth):
        kmeans_start_time = time.time()
        maxIter = 25
        d4p.daalinit()
        data = Data_Path + str(d4p.my_procid()) + ".csv"
        init_algo = d4p.kmeans_init(nClusters=nClusters, distributed=True, method="plusPlusDense")
        # compute initial centroids
        centroids = init_algo.compute(data).centroids
        init_result = init_algo.compute(data)
        if d4p.my_procid() == 0:
            # configure kmeans main object
            algo = d4p.kmeans(nClusters, maxIter, distributed=True)
            # compute the clusters/centroids
            result = algo.compute(data, init_result.centroids)
            # The results provides the initial centroids
            assert result.centroids.shape[0] == nClusters
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