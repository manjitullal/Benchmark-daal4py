# **Benchmark-daal4py**


## **Fast-Scalable-and-Easy-Machine-Learning-With-DAAL4PY**


# **Guide to use the environment and Code**

## **Steps to setup environment to use Daal4py**

1. We have a environment.yml file locate it.

2. Go to anaconda or miniconda installed on the cluster and create the virtual environment.

3. While creating the environment use yml file provided

> 3.a. Create the environment from the environment.yml file:

```conda env create -f environment.yml```

> 3.b. Activate the new environment: 

```conda activate myenv```

> 3.c. Verify that the new environment was installed correctly:

```conda env list```


## **Steps to run the code**

1. Once the environment is set up it will allow you to run Daal4py
2. Navigate to the directory containing main.py file
3. There few datasets in the data folder and any other tabular data can be added
4. To run use the command 

```python main.py```


### **Custom logger**

This helps to track all the things happening in the entire script and log them to track any errors while debugging.

### **Latency**

This is a dictionary used to track all the time taken by different functions in the script and will be saved as a json file for every run.

### **Metrics**

This is the dictionary to save all the metrics of the trained model and will be saved as a json file for every run.

### **Create directory**

WE create a new temporary directory for every run where our results for that particular run will be saved.

 
Serial script and parallel script contains the following Algorithms

1. Linear regression
2. Ridge Regression 
3. PCA
4. SVD
5. Naive Bayes
6. K-Means

These scripts are all commented on for easy understanding.

### References ###

https://intelpython.github.io/daal4py/
