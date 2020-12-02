'''

Author - Abhishek Maheshwarappa

'''

import numpy as np
import logging
import os
import sys
import datetime
import optparse
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split


from lib.serial_abhi import serial_abhi
from lib.parallel_abhi import parallel_abhi
from lib.Numeric import Numeric
from lib.Input_Output_files_functions import Input_Ouput_functions 

path = '/home/maheshwarappa.a/RC_Benchmark/HPC_AI_SKUNKWORKS/Benchmark-daal4py/data/'
files = os.listdir(path)
print("\n\n")
print("***************************************************************")
print("------------------------Data-------------------------")
print("\n")

for f in files:
	print(f)
print('\n\n')
print()

key = input('Which Data to train? \n')


data = pd.read_csv("data/" + key + ".csv", error_bad_lines=False)


# run folder which will be unique always
run_folder = '{}_'.format(
    key) + str(datetime.datetime.now()) + '_outputs'
# temprary folder location to export the results
temp_folder = "/home/maheshwarappa.a/RC_Benchmark/HPC_AI_SKUNKWORKS/Benchmark-daal4py/temp/"
# target folder to export all the result
target_dir = temp_folder + '/' + run_folder

# checking if the temp folder exists. Create one if not.
check_folder = os.path.isdir(target_dir)
if not check_folder:
    os.makedirs(target_dir)
    print("created folder : ", target_dir)


class mains():

    def __init__(self):
        # getting the current system time
        self.current_time = datetime.datetime.now().strftime("%Y-%m-%d__%H.%M")

        # declaring variables and data structures
        self.latency = dict()  # latency dictionary to hold execution times of individual functions

        # metric dictionary

        self.metrics = dict()

        # removing any existing log files if present
        if os.path.exists(target_dir + '/main.log'):
            os.remove(target_dir+ '/main.log')

        # get custom logger
        self.logger = self.get_loggers(target_dir)
    

    @staticmethod
    def get_loggers(temp_path):
        logger = logging.getLogger("HPC-AI skunkworks")  # name the logger as HPC-AI skunkworks
        logger.setLevel(logging.INFO)
        f_hand = logging.FileHandler(temp_path +'/'+ key+'.log')  # file where the custom logs needs to be handled
        f_hand.setLevel(logging.INFO)  # level to set for logging the errors
        f_format = logging.Formatter('%(asctime)s : %(process)d : %(levelname)s : %(message)s',
                                        datefmt='%d-%b-%y %H:%M:%S')
        # format in which the logs needs to be written
        f_hand.setFormatter(f_format)  # setting the format of the logs
        logger.addHandler(f_hand)  # setting the logging handler with the above formatter specification

        return logger

    def data_split(self, data):
        num = data.shape
        num_each = round(num[0]/3)-1

        l=0
        
        for i in range(3):
            df = data[l:num_each]
            l =+ num_each
            num_each =+ num_each
            filename = '/home/maheshwarappa.a/RC_Benchmark/HPC_AI_SKUNKWORKS/Benchmark-daal4py/dist_data/' + key +'_'+str(i+1)+'.csv'
            df.to_csv(filename, index = False)






    def main(self):




        self.logger.info("Intell DAAL4PY Logs initiated!")
        self.logger.info("Current time: " + str(self.current_time))



        num = Numeric(self.logger, self.latency)

        target = 'charges'

        df, dict_df = num.convert_to_numeric(data, target, False)

        self.data_split(df)



        feature = df.columns.tolist()
        feature.remove(target)


        X = df[feature]

        y = df[target]


        self.logger.info('spliting the data frame into Train and test')


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state =1693)


        serial = serial_abhi(self.logger, self.latency, self.metrics)

        parallel = parallel_abhi(self.logger, self.latency, self.metrics)



        y_pred, mse, r2score = serial.serial_linear_sk_learn(X_train, X_test, y_train, y_test, target)

        print("SK serial learn MSE", mse,'\n')


        y_pred, mse, r2score = serial.serial_linear_pydaal(X_train, X_test, y_train, y_test, target)

        print("Daal seial daal4py MSE", mse,'\n')

        # result_pca_pydaal = serial.serial_pca_pydaal(X)

        # result_pca_sk_learn = serial.serial_pca_sk_learn(X)

        y_pred, mse, r2score = parallel.parallel_linear_sk_learn(X_train, X_test, y_train, y_test, target)

        print("SK parallel learn MSE", mse,'\n')


        y_pred, mse, r2score = parallel.parallel_linear_pydaal(X_train, X_test, y_train, y_test, target)

        print("Daal parallel daal4py MSE", mse,'\n')

    


        io = Input_Ouput_functions(self.logger, self.latency)


        self.logger.info('Exporting the latency')
        file_name = target_dir + '/latency_stats.json'
        io.export_to_json(self.latency, file_name )

        self.logger.info('Exporting the Metrics')
        file_name = target_dir + '/metrics_stats.json'
        io.export_to_json(self.metrics, file_name )


        self.logger.info("Program completed normally")
        self.logger.handlers.clear()


if __name__ == "__main__":
    main = mains()
    main.main()