'''

Author - Abhishek Maheshwarappa & Kartik Kumar

Main function


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


from lib.serial_a import Serial_a
from lib.serial_b import Serial_b
from lib.parallel_a import Parallel_a
from lib.parallel_b import Parallel_b
from lib.Numeric import Numeric
from lib.Input_Output_files_functions import Input_Ouput_functions


'''
These are the code to handle and read any data 
from the user choice 
'''

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

# read the data from the user
key = input('Which Data to train? \n')

# get the number of thereads from the user
num = input("enter num of threads\n")

# read the data from the data floder
data = pd.read_csv("data/" + key + ".csv", error_bad_lines=False)

'''
Ask user to select the target
among the columns
'''

while(1):
    print("\n The columns present are\n\n", data.columns)
    list_cols = data.columns.to_list()
    print("\nChoose the target coulum\n")
    target = input()
    if target in list_cols:
        if data[target].isnull().sum() == 0:
            break
        else:
            data[target] = data[target].fillna(0)
            break

        print("\n\nThe selected target Contains Null values, select other target")

    print("\nThe typed value is not present in the columns, try retyping it\n")


print("you want classfication or not ?\n")
print("Note - PCA, SVD and Kmeans run irrespective of the classification or not\n")
classification = input('Classification: \n1 - True or 2 - False \n')


# ask the user wether he wants to run parallel or serial
print("Run options")
print("1 - Serial\n")
print("2 - Parallel\n")
type_key = input("Want to run parallel or serial?\n")

type_key_str = '_Serial_' if type_key == "1" else '_Parallel_'

# run folder which will be unique always
run_folder = '{}_'.format(key)+'_' + num + type_key_str + \
    str(datetime.datetime.now()) + '_outputs'
# temprary folder location to export the results
temp_folder = "./temp/"
# target folder to export all the result
target_dir = temp_folder + '/' + run_folder

numthreads = int(num)

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
        # latency dictionary to hold execution times of individual functions
        self.latency = dict()

        # metric dictionary

        self.metrics = dict()

        # removing any existing log files if present
        if os.path.exists(target_dir + '/main.log'):
            os.remove(target_dir + '/main.log')

        # get custom logger
        self.logger = self.get_loggers(target_dir)

    @staticmethod
    def get_loggers(temp_path):
        # name the logger as HPC-AI skunkworks
        logger = logging.getLogger("HPC-AI skunkworks")
        logger.setLevel(logging.INFO)
        # file where the custom logs needs to be handled
        f_hand = logging.FileHandler(temp_path + '/' + key+'.log')
        f_hand.setLevel(logging.INFO)  # level to set for logging the errors
        f_format = logging.Formatter('%(asctime)s : %(process)d : %(levelname)s : %(message)s',
                                     datefmt='%d-%b-%y %H:%M:%S')
        # format in which the logs needs to be written
        f_hand.setFormatter(f_format)  # setting the format of the logs
        # setting the logging handler with the above formatter specification
        logger.addHandler(f_hand)

        return logger

    def data_split(self, data):
        '''
        This funtion helps to generate the data
        required for multiprocessing
        '''
        self.logger.info(" The Data spliting process started")
        num = data.shape
        num_each = round(num[0]/3)

        l = 0
        nums = num_each

        for i in range(3):
            df = data[l:nums]
            l += num_each
            nums += num_each
            if nums > num[0]:
                nums = num[0]
            filename = './dist_data/' + key + '_'+str(i+1)+'.csv'
            df.to_csv(filename, index=False)
        self.logger.info("Data spliting process done successfuly!!!")

    def main(self):

        self.logger.info("Intell DAAL4PY Logs initiated!")
        self.logger.info("Current time: " + str(self.current_time))

        # creating object for numeric
        num = Numeric(self.logger, self.latency)

        flag = True if classification == '1' else False

        df, dict_df = num.convert_to_numeric(data, target, flag)
        print(df.shape)

        # creating data for distrubuted processing in Pydaal
        msk = np.random.rand(len(df)) < 0.8
        train = df[msk]
        test = df[~msk]
        filename = './dist_data/' + key + '_test'+'.csv'
        test.to_csv(filename, index=False)
        self.data_split(train)

        feature = df.columns.tolist()
        feature.remove(target)

        # checking if serial or not
        if type_key == '1':

            X_train = train[feature]
            y_train = train[target]
            X_test = test[feature]
            y_test = test[target]

            self.logger.info('spliting the data frame into Train and test')
            self.logger.info(" Serial Execution starts ..!! ")

            self.logger.info('Serial Initialization')
            serial_a = Serial_a(self.logger, self.latency, self.metrics)
            Serial_b = Serial_b(self.logger, self.latency, self.metrics)

            # Naive bayes
            if classification == '1':
                serial_a.naiveBayes(X_train, X_test, y_train, y_test, target)

            else:
                # linear Regression
                serial_a.linearRegression(
                    X_train, X_test, y_train, y_test, target)

                # Ridge Regression
                Serial_b.ridgeRegression(
                    X_train, X_test, y_train, y_test, target)

                # linear
                serial_a.serial_linear_sk_learn(
                    X_train, X_test, y_train, y_test, target)

            # K-means Regression
            Serial_b.kMeans(df, target)

            # PCA Regression
            serial_a.pca(df, target)

            # SVD Regression
            Serial_b.svd(df, target)

            self.logger.info(" Serial Execution ends..!! ")

        # check parallel or not
        if type_key == '2':
            self.logger.info(" Parallel Execution starts ..!! ")

            print('\n\n Select which algorithim to run?')
            print("1.Linear Regression - LR ")
            print("2.Ridge Regression - RR")
            print("3.Naive Bayes - NB")
            print("4.K Means - KM")
            print("5.PCA - P")
            print("6.SVD - S\n")

            Parallel_bey = input("Enter the code for the algo required\n\n")

            self.logger.info('Parallel Initialization')
            parallel_a = Parallel_a(self.logger, self.latency, self.metrics)
            Parallel_b = Parallel_b(self.logger, self.latency, self.metrics)

            # path for distrubted data and test data

            dist_data_path = './dist_data/' + key + '_'
            test_data_path = './dist_data/' + key + '_test'+'.csv'

            # parallel linear regression
            if Parallel_bey == 'LR':
                parallel_a.linearRegression(
                    dist_data_path, test_data_path,  target, numthreads)

            # parallel ridge regression regression
            elif Parallel_bey == "RR":
                Parallel_b.ridgeRegression(
                    dist_data_path, test_data_path,  target, numthreads)

            # parallel linear regression
            elif Parallel_bey == "NB":
                parallel_a.naiveBayes(
                    dist_data_path, test_data_path,  target, numthreads)

            # parallel linear regression
            elif Parallel_bey == "KM":
                Parallel_b.kMeans(dist_data_path, numthreads)

            # parallel linear regression
            elif Parallel_bey == "P":
                parallel_a.pca(dist_data_path, target, numthreads)

            # parallel linear regression
            elif Parallel_bey == "S":
                Parallel_b.svd(dist_data_path, target, numthreads)

        self.logger.info(" Parallel Execution ends..!! ")

        io = Input_Ouput_functions(self.logger, self.latency)

        self.logger.info('Exporting the latency')
        file_name = target_dir + '/latency_stats.json'
        io.export_to_json(self.latency, file_name)

        self.logger.info('Exporting the Metrics')
        file_name = target_dir + '/metrics_stats.json'
        io.export_to_json(self.metrics, file_name)

        self.logger.info("Program completed normally")
        self.logger.handlers.clear()


if __name__ == "__main__":
    main = mains()
    main.main()
