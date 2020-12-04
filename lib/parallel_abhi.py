'''

Author - Abhishek Maheshwarappa

Linear regression 

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


class Parallel_a():

    #Initialization
    def __init__(self, logger, latency, metrics):
        self.logger = logger
        self.latency = latency
        self.metrics = metrics

    # daal4py Linear Regression SPMD Mode
    def linearRegression(self, Data_Path,test_data_path,  target, n):

        

        # Initialize SPMD mode
        d4p.daalinit(nthreads = n)

        # training setup
        file = Data_Path + str(d4p.my_procid() + 1) + ".csv"
        data = pd.read_csv(file)
        X = data.drop(columns=target)
        y = data[target]

        train_algo = d4p.linear_regression_training(method='qrDense', distributed=True)

        self.logger.info('Training the Linear Regression in pydaal SPMD Mode')

        start = time.time()

        train_result  = train_algo.compute(X, y)

        self.latency['Parallel_LinearRegression_Pydaal_Time'] = time.time() - start

        # test file setup
        test = pd.read_csv(test_data_path)

        y_test = test[target]
        X_test = test.drop(target,axis = 1)

        if d4p.my_procid() == 0:
            predict_algo = d4p.linear_regression_prediction()

            # now predict using the model from the training above
            predict_result = predict_algo.compute(X_test, train_result.model)

            # The prediction result provides prediction
            #assert predict_result.prediction.shape == (X_test.shape[0], y.shape[1])

        
        d4p.daalfini()

        self.logger.info('Completed Linear Regression in pydaal SPMD Mode')

        #Compute metrics
        mse = mean_squared_error(y_test, predict_result.prediction)
        r2score = r2_score(y_test, predict_result.prediction)

        # Store the time taken and model metrics
        self.metrics['MSE_Parallel_LinearRegression_Pydaal'] = mse
        self.metrics['r2score_Parallel_LinearRegression_Pydaal'] = r2score
        

        return 



    # daal4py PCA SPMD Mode
    def pca(self, Data_Path, target,n):



        # Initialize SPMD mode
        d4p.daalinit(nthreads = n)

        # Train setup
        file_path = Data_Path + str(d4p.my_procid()+1) + ".csv"
        data = pd.read_csv(file_path)
        data = data.drop(target, axis =1)

        # configure a PCA object
        algo = d4p.pca(method='svdDense', distributed=True)

        self.logger.info('Training the PCA in  pydaal SPMD Mode')

        start = time.time()

        result = algo.compute(data)
        self.latency['Parallel_PCA_SPMD_Time'] = time.time() - start

        # result is available on all processes - but we print only on root
        if d4p.my_procid() == 0:
            print("PCA completed", result)

       
        d4p.daalfini()

        self.logger.info('Completed PCA in pydaal SPMD Mode')

        
        return 



    # daal4py Naive Bayes SPMD Mode
    def naiveBayes(self, Data_Path,test_data_path, target, n):

        

        # Initialize SPMD mode
        d4p.daalinit(nthreads = n)

        # training setup
        file = Data_Path + str(d4p.my_procid() + 1) + ".csv"
        data = pd.read_csv(file)
        X = data.drop(columns=target)
        y = data[target]

        # test file setup
        test = pd.read_csv(test_data_path)
       
        y_test = test[target]
        X_test = test.drop(target,axis = 1)

        #store unique target values
        category_count = len(y.unique())
        print(category_count)

        # Configure a training object
        train_algo = d4p.multinomial_naive_bayes_training(2, distributed=True)
        self.logger.info('Training the Naive Bayes in pydaal SPMD Mode')

        start = time.time()

        train_result = train_algo.compute(X, y)
        self.latency['Parallel_NaiveBayes_Pydaal_Time'] = time.time() - start
        # Now let's do some prediction
        # It runs only on a single node
        if d4p.my_procid() == 0:
            predict_algo = d4p.multinomial_naive_bayes_prediction(category_count)

            # now predict using the model from the training above
            presult = predict_algo.compute(X_test, train_result.model)

            # Prediction result provides prediction
            #assert (presult.prediction.shape == (X_test.shape[0], 1))

            print('Naive Bayes Result', presult.prediction)

        
        d4p.daalfini()

        self.logger.info('Completed Naive Bayes in pydaal SPMD Mode')
        
        # Compute metrics - Define classification metrics
        # mse = mean_squared_error(y_test, predict_result.prediction)
        # r2score = r2_score(y_test, predict_result.prediction)

        # # Store the time taken and model metrics
        # self.metrics['MSE_Parallel_NaiveBayes_Pydaal'] = mse
        # self.metrics['r2score_Parallel_NaiveBayes_Pydaal'] = r2score


        return 










    # def parallel_linear_sk_learn(self, X_train, X_test, y_train, y_test, target):
    #     regr = linear_model.LinearRegression(n_jobs=-1)
    #
    #     start = time.time()
    #     # Train the model using the training sets
    #
    #     self.logger.info('Training the parallel Linear Regression in Sk learn')
    #     model = regr.fit(X_train, y_train)
    #
    #     self.latency['Time for parallel Linear Regression sk_learn'] = time.time() - start
    #
    #     # Make predictions using the testing set
    #
    #     y_pred = regr.predict(X_test)
    #     self.logger.info('Predictions done successfully..!!!')
    #
    #     mse = mean_squared_error(y_test, y_pred)
    #
    #     self.metrics['MSE_parallel_linear_regression_sk_learn'] = mse
    #
    #     r2score = r2_score(y_test, y_pred)
    #
    #     self.metrics['r2_score_parallel_linear_regression_sk_learn'] = r2score
    #
    #     return y_pred, mse, r2_score

    # def parallel_pca_sk_learn(self, data):
    #     start = time.time()
    #
    #     pca = PCA(n_components=10)
    #
    #     self.logger.info('serial PCA in  SK_learn')
    #     result = pca.fit(data)
    #
    #     self.latency['Time for serial PCA sk_learn'] = time.time() - start
    #
    #     return result