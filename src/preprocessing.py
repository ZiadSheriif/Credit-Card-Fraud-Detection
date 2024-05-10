# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  RobustScaler

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def read_dataset(self):
        self.df = pd.read_csv(self.data_path)
        print(self.df.head())
        print(self.df.describe())
        
    def check_missing_values(self):
        missing_values = self.df.isnull().sum().max()
        print("Number of missing values:", missing_values)
        print('No Frauds', round(self.df['Class'].value_counts()[0] / len(self.df) * 100, 2), '% of the dataset')
        print('Frauds', round(self.df['Class'].value_counts()[1] / len(self.df) * 100, 2), '% of the dataset')
        
    def split_data(self):
        X = self.df.drop('Class', axis=1)
        y = self.df['Class']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    def scale_features(self):
        rob_scaler = RobustScaler()
        self.df['scaled_amount'] = rob_scaler.fit_transform(self.df['Amount'].values.reshape(-1,1))
        self.df['scaled_time'] = rob_scaler.fit_transform(self.df['Time'].values.reshape(-1,1))
        self.df.drop(['Time','Amount'], axis=1, inplace=True)
        
        scaled_amount = self.df['scaled_amount']
        scaled_time = self.df['scaled_time']
        
        self.df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
        self.df.insert(0, 'scaled_amount', scaled_amount)
        self.df.insert(1, 'scaled_time', scaled_time)
        
        print(self.df.head())
        
        return self.df
        
    def remove_outliers(self, df,features):
        self.df = df
        for feature in features:
            fraud_values = self.df[feature].loc[self.df['Class'] == 1].values
            q25, q75 = np.percentile(fraud_values, 25), np.percentile(fraud_values, 75)
            iqr = q75 - q25
            print(f'Quartile 25: {q25} | Quartile 75: {q75}')

            cut_off = iqr * 1.5
            lower, upper = q25 - cut_off, q75 + cut_off
            print(f'{feature} Lower: {lower}')
            print(f'{feature} Upper: {upper}')

            outliers = [x for x in fraud_values if x < lower or x > upper]
            print(f'{feature} outliers: {outliers}')
            print(f'Feature {feature} Outliers for Fraud Cases: {len(outliers)}')

            self.df = self.df.drop(self.df[(self.df[feature] > upper) | (self.df[feature] < lower)].index)
            print('Number of Instances after outliers removal: {}'.format(len(self.df)))
            print('----' * 44)

        return self.df

    def preprocess(self):
        self.read_dataset()
        self.check_missing_values()
        return self.df