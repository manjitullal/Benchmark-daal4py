'''

Author - Abhishek Maheshwarappa

Linear regression 
Sklearn
'''


'''

serial

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
from sklearn.decomposition import PCA




class Serial_a():

    def __init__(self, logger, latency, metrics):
            self.logger = logger
            self.latency = latency
            self.metrics = metrics


    def linearRegression(self, X_train, X_test, y_train, y_test, target):

        start = time.time()

        # Configure a Linear regression training object
        train_algo  = d4p.linear_regression_training(method = 'qrDense')

        self.logger.info('Training the Linear Regression in pydaal Batch/Serial Mode')

        # Now train/compute, the result provides the model for prediction
        lm_trained = train_algo.compute(X_train, y_train)

        y_pred = d4p.linear_regression_prediction().compute(X_test, lm_trained.model).prediction

        self.logger.info('Completed Linear Regression in pydaal Batch/Serial Mode')

        #Compute metrics
        mse = mean_squared_error(y_test, y_pred)
        r2score = r2_score(y_test, y_pred)

        # Store the time taken and model metrics
        self.latency["Serial Linear Regression Batch Time"] = time.time() - start
        self.metrics['MSE_serial_linear_regression_pydaal'] = mse
        self.metrics['r2_score_serial_linear_regression_pydaal'] = r2score

        return



    def pca(self, data, target):

        start = time.time()

        data = data.drop(target, axis=1)

        # configure a PCA object

        self.logger.info('Training the serial PCA in  pydaal')

        algo = d4p.pca(resultsToCompute="mean|variance|eigenvalue",nComponents = 10, isDeterministic=True)

        self.logger.info('Training the PCA in pydaal Batch Mode')

        result = algo.compute(data)

        self.logger.info('Completed PCA in pydaal Batch/Serial Mode')

        self.latency["Serial_PCA_Batch_Time"] = time.time() - start

        return result


    def naiveBayes(self, X_train, X_test, y_train, y_test, target):
        start = time.time()

        # store unique target values
        category_count = len(y_train.unique())

        method ='defaultDense'

        # Configure a training object (20 classes)
        train_algo = d4p.multinomial_naive_bayes_training(category_count, method=method)
        self.logger.info('Training the Naive Bayes in pydaal Batch/Serial Mode')

        train_result = train_algo.compute(X_train, y_train)

        # Now let's do some prediction
        predict_algo = d4p.multinomial_naive_bayes_prediction(category_count, method=method)

        # now predict using the model from the training above
        presult = predict_algo.compute(X_test, train_result.model)

        # Prediction result provides prediction
        assert (presult.prediction.shape == (X_test.shape[0], 1))

        self.logger.info('Completed Naive Bayes in pydaal Batch/Serial Mode')

        # Store the time taken
        self.latency["Serial Naive Bayes Batch Time"] = time.time() - start

        return
        

























    # def serial_linear_sk_learn(self, X_train, X_test, y_train, y_test, target):
    #
    #
    #     regr = linear_model.LinearRegression()
    #
    #     start = time.time()
    #     # Train the model using the training sets
    #
    #     self.logger.info('Training the Linear Regression in Sk learn')
    #     model = regr.fit(X_train,y_train)
    #
    #     self.latency['Time for serial Linear Regression sk_learn'] = time.time() - start
    #
    #     # Make predictions using the testing set
    #
    #
    #     y_pred = regr.predict(X_test)
    #     self.logger.info('Predictions done successfully..!!!')
    #
    #     mse = mean_squared_error(y_test, y_pred)
    #
    #     self.metrics['MSE_serial_linear_regression_sk_learn'] = mse
    #
    #     r2score = r2_score(y_test, y_pred)
    #
    #     self.metrics['r2_score_serial_linear_regression_sk_learn'] = r2score
    #
    #     return y_pred, mse, r2_score

    # def serial_pca_sk_learn(self, data):
    #
    #     start = time.time()
    #
    #     pca = PCA(n_components=10)
    #
    #     self.logger.info('serial PCA in  SK_learn')
    #     result = pca.fit(data)
    #
    #     self.latency['Time for serial PCA sk_learn'] = time.time() - start
    #
    #
    #     return result


    


    