'''

Author : Abhishek Maheshwarappa and Kartik

'''
import time
import numpy as np
import pandas as pd
import daal4py as d4p
from sklearn.model_selection import train_test_split


d4p.daalinit(2) # initializes the distribution engine

print(d4p.num_procs())
# organizing variables used in the model for prediction
# each process gets its own data

key = 'insurance'

target = 'charges'

dist_data_path = './dist_data/' + key +'_'
test_data_path = './dist_data/' + key +'_test'+'.csv'

infile = dist_data_path + str(d4p.my_procid()+1) + ".csv"

# read data
indep_data = pd.read_csv(infile).drop([target], axis=1) # house characteristics
dep_data   = pd.read_csv(infile)[target] # house price

train_result = d4p.linear_regression_training(distributed=True).compute(indep_data, dep_data)

d4p.daalfini() # stops the distribution engine


import os

CPU_Pct=str(round(float(os.popen('''grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage }' ''').readline()),2))

#print results
print("CPU Usage = " + CPU_Pct)