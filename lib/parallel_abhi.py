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


class parallel_abhi():

    def __init__(self, logger, latency, metrics):
            self.logger = logger
            self.latency = latency
            self.metrics = metrics
    
    
    def parallel_linear_sk_learn(self, X_train, X_test, y_train, y_test, target):

        
        regr = linear_model.LinearRegression(n_jobs = -1)
        
        start = time.time()
        # Train the model using the training sets

        self.logger.info('Training the parallel Linear Regression in Sk learn')
        model = regr.fit(X_train,y_train)

        self.latency['Time for parallel Linear Regression sk_learn'] = time.time() - start

        # Make predictions using the testing set


        y_pred = regr.predict(X_test)
        self.logger.info('Predictions done successfully..!!!')

        mse = mean_squared_error(y_test, y_pred)

        self.metrics['MSE_parallel_linear_regression_sk_learn'] = mse

        r2score = r2_score(y_test, y_pred)

        self.metrics['r2_score_parallel_linear_regression_sk_learn'] = r2score

        return y_pred, mse, r2_score


    def parallel_linear_pydaal(self, X_train, X_test, y_train, y_test, target):
        
        d4p.daalinit()

        start = time.time()

        d4p_lm = d4p.linear_regression_training(method = 'qrDense', distributed=True)

        self.logger.info('Training the Linear Regression in pydaal')
        lm_trained = d4p_lm.compute(X_train, y_train )


        self.latency['Time for serial parallel Regression pydaal'] = time.time() - start

        y_pred = d4p.linear_regression_prediction().compute(X_test, lm_trained.model).prediction



        mse = mean_squared_error(y_test, y_pred)
        
        self.metrics['MSE_serial_parallel_regression_pydaal'] = mse

        r2score = r2_score(y_test, y_pred)

        self.metrics['r2_score_parallel_regression_pydaal'] = r2score

        d4p.daalfini()

        return y_pred, mse, r2score
