'''

Abhishek Maheshwarappa

'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from linear_reg import LinearReg
from lib.Numeric import Numeric 
from Serial_2 import Serial_2



url =  'https://raw.githubusercontent.com/Angelic-Interpretability/Model-Interpretability-Techniques/master/Datasets/insurance.csv'
data = pd.read_csv(url, error_bad_lines=False)

num = Numeric()

target = 'charges'

df, dict_df = num.convert_to_numeric(data, target, False)



feature = df.columns.tolist()
feature.remove(target)


X = df[feature]

y = df[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state =1693)


reg = LinearReg()

n_jobs = 1

y_pred, mse, r2score = reg.serial_linear(X_train, X_test, y_train, y_test, target, n_jobs)

print("SK learn MSE", mse,'\n')


y_pred, mse, r2score = reg.serial_pydaal(X_train, X_test, y_train, y_test, target)

print("Daal 4 py MSE", mse,'\n')

y_pred, mse, r2score = reg.serial_pydaal(X_train, X_test, y_train, y_test, target, True)

print("Dist Daal 4 py MSE", mse,'\n')


#Serial
serial = Serial_2()

ridge_predict, ridge_time = serial.ridgeRegression(X_train, X_test, y_train, y_test, target)
print("Serial ridge result", ridge_predict,'\n')
print("Serial ridge Time", ridge_time,'\n')

kmeans_result, kmeans_time = serial.KMeans(10, df) 
print("Serial kmeans result", kmeans_result,'\n')
print("Serial kmeans Time", kmeans_time,'\n')

svd_result, svd_time = serial.svd(df)
print("Serial svd result", svd_result,'\n')
print("Serial svd Time", svd_time,'\n')
