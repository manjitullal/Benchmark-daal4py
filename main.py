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


# from lib.serial_abhi import Serial_a
# from lib.serial_kartik import Serial_k
from lib.parallel_abhi import Parallel_a
from lib.parallel_kartik import Parallel_k
from lib.Numeric import Numeric
from lib.Input_Output_files_functions import Input_Ouput_functions 

path = './data/'
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

while(1):
    print("\n The columns present are\n\n", data.columns)
    list_cols = data.columns.to_list()
    print("\nChoose the target coulum\n")
    target = input()
    if target in list_cols :
        if data[target].isnull().sum()==0:
            break
        print("\n\nThe selected target Contains Null values, select other target")
    print("\nThe typed value is not present in the columns, try retyping it\n")

print("Run options")
print("1. Serial\n")
print("2. Parallel\n")
type_key = input("Want to run parallel or serial?")

# run folder which will be unique always
run_folder = '{}_'.format(
    key)+type_key + str(datetime.datetime.now()) + '_outputs'
# temprary folder location to export the results
temp_folder = "./temp/"
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

        '''
        This funtion helps to generate the data
        required for multiprocessing
        '''
        self.logger.info(" The Data spliting process started")
        num = data.shape
        num_each = round(num[0]/3)

        l=0
        nums = num_each

        for i in range(3):
            df = data[l:nums]
            l += num_each
            nums += num_each
            if nums > num[0]:
                nums = num[0]
            filename = './dist_data/' + key +'_'+str(i+1)+'.csv'
            df.to_csv(filename, index = False)
        self.logger.info("Data spliting process done successfuly!!!")




    def main(self):


        self.logger.info("Intell DAAL4PY Logs initiated!")
        self.logger.info("Current time: " + str(self.current_time))


        # creating object for numeric
        num = Numeric(self.logger, self.latency)
        df, dict_df = num.convert_to_numeric(data, target, False)

        # creating data for distrubuted processing in Pydaal
        msk = np.random.rand(len(df)) < 0.8
        train = df[msk]
        test = df[~msk]
        filename = './dist_data/' + key +'_test'+'.csv'
        test.to_csv(filename, index = False)
        self.data_split(train)



        feature = df.columns.tolist()
        feature.remove(target)

        # splitting data into train nd test for serial
        X = df[feature]
        y = df[target]
        self.logger.info('spliting the data frame into Train and test')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state =1693)

        if type_key == 'serial':
            self.logger.info(" Serial Execution starts ..!! ")
            
            self.logger.info('Serial Initialization')
            serial_a = Serial_a(self.logger, self.latency, self.metrics)
            serial_k = Serial_k(self.logger, self.latency, self.metrics)

            # linear Regression
            serial_a.regression(X_train, X_test, y_train, y_test, target)

            # Ridge Regression
            serial_k.ridgeregression(X_train, X_test, y_train, y_test, target)

            # Naive bayes
            serial_a.naivebayes(X_train, X_test, y_train, y_test, target)

            # K-means Regression
            serial_a.kmeans(df, target)

            # PCA Regression
            serial_a.pca(df, target)

            # SVD Regression
            serial_a.svd(df, target)

            self.logger.info(" Serial Execution ends..!! ")


        if type_key == 'parallel':
            self.logger.info(" Parallel Execution starts ..!! ")
            
            self.logger.info('Parallel Initialization')
            parallel_a = Parallel_a(self.logger, self.latency, self.metrics)
            parallel_k = Parallel_k(self.logger, self.latency, self.metrics)


            # path for distrubted data and test data

            dist_data_path = './dist_data/' + key +'_'
            test_data_path = './dist_data/' + key +'_test'+'.csv'

            # parallel linear regression
            parallel_a.linearRegression(dist_data_path, test_data_path, feature, target)
            
            # parallel ridge regression regression
            parallel_k.ridgeRegression(dist_data_path, test_data_path, feature, target)

            # # parallel linear regression
            # parallel_k.naiveBayes(dist_data_path, test_data_path, feature, target)

            # # parallel linear regression
            # parallel_k.kMeans(dist_data_path)

            # # parallel linear regression
            # parallel_k.pca(dist_data_path,target)

            # # parallel linear regression
            # parallel_k.svd(dist_data_path,target)


        self.logger.info(" Parallel Execution ends..!! ")
    


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