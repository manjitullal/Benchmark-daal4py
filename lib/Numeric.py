import pandas as pd
import numpy as np
import time
from sklearn import preprocessing


class Numeric():

    def __init__(self, logger, latency):
        self.logger = logger
        self.latency = latency

    def convert_to_numeric(self, df,target,classification = True):

        # getting categorical and continous dtypes
        categorical_columns = df.select_dtypes(include=['object']).columns.to_list()

        continous_columns = df.select_dtypes(exclude=['object']).columns.to_list()

        # handling missing values
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode())
        for col in continous_columns:
            df[col] = df[col].fillna(df[col].mean())

        if target in continous_columns: continous_columns.remove(target)


        for col in continous_columns:
            if len(df[col].unique())==2:
                continous_columns.remove(col)

        # Create x, where x the 'scores' column's values as floats
        x = df[continous_columns].values.astype(float)

        # Create a minimum and maximum processor object
        min_max_scaler = preprocessing.MinMaxScaler()

        # Create an object to transform the data to fit minmax processor
        x_scaled = min_max_scaler.fit_transform(x)

        # Run the normalizer on the dataframe
        df[continous_columns] = pd.DataFrame(x_scaled)

        # dict
        df_dict = {}

        print("before for", categorical_columns)



        for col in categorical_columns:
            print(col ,' - ', len(df[col].unique()))
            if len(df[col].unique())==2:
                df[col] = df[col].astype("category")
                df_dict[col] = dict(
                    enumerate(df[col].cat.categories))
                df[col] = df[col].cat.codes
            if len(df[col].unique())>10:
                # print("\nBefore",categorical_columns)
                # print("\ndf", df.columns)
                categorical_columns.remove(col)
                df = df.drop(col, axis = 1)
                # print("\nafter",categorical_columns)
                # print("\ndf", df.columns)
                # print("Col - ",col)

        if classification:
            categorical_columns.remove(target)

        if categorical_columns:
            print(categorical_columns)
            data=pd.get_dummies(df[categorical_columns])
            data = data.apply(np.int64)
            df=df.drop(categorical_columns ,axis = 1)
            df = df.join(data)

        return df,df_dict