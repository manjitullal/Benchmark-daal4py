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




class serial_abhi():

    def __init__(self, logger, latency, metrics):
            self.logger = logger
            self.latency = latency
            self.metrics = metrics
    
    
    def serial_linear_sk_learn(self, X_train, X_test, y_train, y_test, target):

        
        regr = linear_model.LinearRegression()
        
        start = time.time()
        # Train the model using the training sets

        self.logger.info('Training the Linear Regression in Sk learn')
        model = regr.fit(X_train,y_train)

        self.latency['Time for serial Linear Regression sk_learn'] = time.time() - start

        # Make predictions using the testing set


        y_pred = regr.predict(X_test)
        self.logger.info('Predictions done successfully..!!!')

        mse = mean_squared_error(y_test, y_pred)

        self.metrics['MSE_serial_linear_regression_sk_learn'] = mse

        r2score = r2_score(y_test, y_pred)

        self.metrics['r2_score_serial_linear_regression_sk_learn'] = r2score

        return y_pred, mse, r2_score


    def serial_linear_pydaal(self, X_train, X_test, y_train, y_test, target):
        
        start = time.time()

        d4p_lm = d4p.linear_regression_training(method = 'qrDense')

        self.logger.info('Training the Linear Regression in pydaal')
        lm_trained = d4p_lm.compute(X_train, y_train )


        self.latency['Time for serial Linear Regression pydaal'] = time.time() - start

        y_pred = d4p.linear_regression_prediction().compute(X_test, lm_trained.model).prediction



        mse = mean_squared_error(y_test, y_pred)
        
        self.metrics['MSE_serial_linear_regression_pydaal'] = mse

        r2score = r2_score(y_test, y_pred)

        self.metrics['r2_score_serial_linear_regression_pydaal'] = r2score

        return y_pred, mse, r2score

    def serial_pca_sk_learn(self, data):
        
        start = time.time()
        
        pca = PCA(n_components=10)
        
        self.logger.info('serial PCA in  SK_learn')
        result = pca.fit(data)
        
        self.latency['Time for serial PCA sk_learn'] = time.time() - start


        return result


    def serial_pca_pydaal(self, data):

        start = time.time()

        zscore = d4p.normalization_zscore()
        # configure a PCA object

        self.logger.info('Training the serial PCA in  pydaal')

        algo = d4p.pca(resultsToCompute="mean|variance|eigenvalue",nComponents = 10, isDeterministic=True, normalization=zscore)

        result = algo.compute(data)

        self.latency['Time for serial PCA pydaal'] = time.time() - start


        return result


    