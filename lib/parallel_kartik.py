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
    
    #Initialization
    def __init__(self, logger, latency, metrics):
            self.logger = logger
            self.latency = latency
            self.metrics = metrics

    # daal4py Ridge Regression SPMD Mode
    def ridgeRegression(self, Data_Path,test_data_path, feature, target):
        start_time = time.time()

        #Initialize SPMD mode
        d4p.daalinit()

        file = Data_Path + str(d4p.my_procid()+1)+".csv"

        #Configure a Ridge regression training object
        train_algo = d4p.ridge_regression_training(distributed=True, interceptFlag=True)
        self.logger.info('Training the Ridge Regression in pydaal SPMD Mode')
        
        #training
        data = pd.read_csv(file)
        X = data.drop(columns=target)
        y= data[target]

        #test file setup
        test = pd.read_csv(test_data_path)
        X_test = test[feature]
        y_test = test[target]

        train_result = train_algo.compute(X, y)

        #Only process #0 reports results
        if d4p.my_procid() == 0:
            predict_algo = d4p.ridge_regression_prediction()
            # now predict using the model from the training above
            predict_result = predict_algo.compute(X_test, train_result.model)
            # The prediction result provides prediction
            assert predict_result.prediction.shape == (X_test.shape[0], y.shape[1])

        self.logger.info('Completed Ridge Regression in pydaal SPMD Mode')
        d4p.daalfini()

        #Compute metrics
        mse = mean_squared_error(y_test, predict_result.prediction)
        r2score = r2_score(y_test, predict_result.prediction)

        #Store the time taken and model metrics
        self.latency["Parallel Ridge Regression SPMD Time"] = time.time() - start_time
        self.metrics["MSE For Parallel Ridge regression SPMD"] = mse
        self.metrics["R2 Score For Parallel Ridge regression SPMD"] = r2score

        return predict_result.prediction



    #daal4py KMeans Clustering SPMD Mode
    def kMeans(self, nClusters, Data_Path):

        kmeans_start_time = time.time()
        maxIter = 25 #fixed maximum number of itertions

        # Initialize SPMD mode
        d4p.daalinit()

        #training setup
        file_path = Data_Path + str(d4p.my_procid()) + ".csv"
        data = pd.read_csv(file_path)
        init_algo = d4p.kmeans_init(nClusters=nClusters, distributed=True, method="plusPlusDense")

        self.logger.info('Training the KMeans in pydaal SPMD Mode')

        # compute initial centroids
        centroids = init_algo.compute(data).centroids
        init_result = init_algo.compute(data)

        # configure kmeans main object
        algo = d4p.kmeans(nClusters, maxIter, distributed=True)

        # compute the clusters/centroids
        result = algo.compute(data, init_result.centroids)

        # The result provides the initial centroids
        assert result.centroids.shape[0] == nClusters

        # result is available on all processes - but we print only on root
        if d4p.my_procid() == 0:
            print("KMeans completed", result)

        self.logger.info('Completed KMeans in pydaal SPMD Mode')

        d4p.daalfini()


        self.latency["Parallel_KMeans_SPMD_Time"] = time.time() - kmeans_start_time

        return result



    # daal4py SVD SPMD Mode
    def svd(self, Data_Path):

        svd_start_time = time.time()

        # Initialize SPMD mode
        d4p.daalinit()

        #Train setup
        file_path = Data_Path + str(d4p.my_procid()) + ".csv"
        data = pd.read_csv(file_path)

        algo = d4p.svd(distributed=True)
        self.logger.info('Training the SVD in pydaal SPMD Mode')

        #SVD result
        result = algo.compute(data)

        assert result.singularValues.shape == (1, data.shape[1])
        assert result.rightSingularMatrix.shape == (data.shape[1], data.shape[1])
        assert result.leftSingularMatrix.shape == data.shape

        if hasattr(data, 'toarray'):
            data = data.toarray()  # to make the next assertion work with scipy's csr_matrix

        # result is available on all processes - but we print only on root
        if d4p.my_procid() == 0:
            print("SVD completed", result)
            
        assert np.allclose(data, np.matmul(np.matmul(result.leftSingularMatrix, np.diag(result.singularValues[0])),
                                           result.rightSingularMatrix))

        self.logger.info('Completed SVD in pydaal SPMD Mode')
        d4p.daalfini()
        
        self.latency["Parallel_SVD_SPMD_Time"] = time.time() - svd_start_time

        return result

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