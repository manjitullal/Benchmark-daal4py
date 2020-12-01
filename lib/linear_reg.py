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




class LinearReg():
    
    
    def serial_linear(self, X_train, X_test, y_train, y_test, target,n = 1):


        start = time.time()
        regr = linear_model.LinearRegression(n_jobs = n)

        # Train the model using the training sets
        model = regr.fit(X_train,y_train)

        print(time.time() - start)

        start = time.time()

        # Make predictions using the testing set
        y_pred = regr.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)

        r2score = r2_score(y_test, y_pred)

        return y_pred, mse, r2_score


    def serial_pydaal(self, X_train, X_test, y_train, y_test, target, distributed=False):
        start = time.time()

        d4p_lm = d4p.linear_regression_training(method = 'qrDense', distributed = distributed)
        lm_trained = d4p_lm.compute(X_train, y_train )


        print(time.time() - start)

        y_pred = d4p.linear_regression_prediction().compute(X_test, lm_trained.model).prediction

        mse = mean_squared_error(y_test, y_pred)


        r2score = r2_score(y_test, y_pred)

        return y_pred, mse, r2score



