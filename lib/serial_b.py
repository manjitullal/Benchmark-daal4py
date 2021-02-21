'''
Author - Kartik 

File runs algorithms serially.

Libraries: Daal4py, Sklearn

Algorithms: Ridge regression, KMeans, SVD

'''

import time
import numpy as np
import pandas as pd
import daal4py as d4p
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans



class Serial_b():
    def __init__(self, logger, latency, metrics):
            self.logger = logger
            self.latency = latency
            self.metrics = metrics

    #Ridge Regression
    def ridgeRegression(self, X_train, X_test, y_train, y_test, target):
        '''
        Method for Ridge Regression

        '''
        
        # Configure a Ridge regression training object
        train_algo = d4p.ridge_regression_training(interceptFlag=True)
        self.logger.info('Training the Ridge Regression in pydaal Batch/Serial Mode')
        
        # time the computation time
        start_time = time.time()
        train_result = train_algo.compute(X_train, y_train)
        self.latency["Serial Ridge Regression Batch Time"] = time.time() - start_time

        predict_algo = d4p.ridge_regression_prediction()

        # Now train/compute, the result provides the model for prediction
        predict_result = predict_algo.compute(X_test, train_result.model)

        # stop_time = time.time()
        pd_predict = predict_result.prediction

        self.logger.info('Completed Ridge Regression in pydaal Batch/Serial Mode')

        # Compute metrics
        mse = mean_squared_error(y_test, pd_predict)
        r2score = r2_score(y_test, pd_predict)

        # Store the time taken and model metrics
        self.metrics["MSE For Serial Ridge regression Batch"] = mse
        self.metrics["R2 Score For Serial Ridge regression Batch"] = r2score

        return

    def kMeans(self, data, target):

        '''
        Method for serial running of Kmeans
        '''
        
        nClusters = 4
        maxIter = 25 #fixed maximum number of itertions
        data = data.drop(target, axis=1)


        init_algo = d4p.kmeans_init(nClusters=nClusters, method="plusPlusDense")
        self.logger.info('Training the KMeans in pydaal Batch/Serial Mode')

        train_result = init_algo.compute(data)

        # The results provides the initial centroids
        assert train_result.centroids.shape[0] == nClusters

        # configure kmeans main object: we also request the cluster assignments
        algo = d4p.kmeans(nClusters, maxIter)
        # compute the clusters/centroids

        kmeans_start_time = time.time()

        result = algo.compute(data, train_result.centroids)

        self.latency["Serial_KMeans_Batch_Time"] = time.time() - kmeans_start_time


        # Kmeans result objects provide assignments (if requested), centroids, goalFunction, nIterations and objectiveFunction
        assert result.centroids.shape[0] == nClusters
        assert result.assignments.shape == (data.shape[0], 1)
        assert result.nIterations <= maxIter

        self.logger.info('Completed KMeans in pydaal Batch/Serial Mode')

        return


    def svd(self, data, target):

        '''
        Method for serial execution of SVD
        '''
        
        data = data.drop(target, axis=1)

        algo = d4p.svd()
        self.logger.info('Training the SVD in pydaal Batch Mode')
        svd_start_time = time.time()

        result = algo.compute(data)

        self.latency["Serial_SVD_Batch_Time"] = time.time() - svd_start_time

        assert result.singularValues.shape == (1, data.shape[1])
        assert result.rightSingularMatrix.shape == (data.shape[1], data.shape[1])
        assert result.leftSingularMatrix.shape == data.shape

        if hasattr(data, 'toarray'):
            data = data.toarray()  # to make the next assertion work with scipy's csr_matrix
        assert np.allclose(data, np.matmul(np.matmul(result.leftSingularMatrix, np.diag(result.singularValues[0])),
                                             result.rightSingularMatrix))

        self.logger.info('Completed SVD in pydaal Batch/Serial Mode')

        
        return
