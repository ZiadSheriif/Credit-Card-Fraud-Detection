

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

class DataSplitter:
    def __init__(self, df: pd.DataFrame, target: str):
        self.df = df
        self.target = target
        self.X = df.drop(target, axis=1)
        self.y = df[target]

    def split_data(self):
        sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

        for train_index, test_index in sss.split(self.X, self.y):
            print("Train:", train_index, "Test:", test_index)
            original_Xtrain, original_Xtest = self.X.iloc[train_index], self.X.iloc[test_index]
            original_ytrain, original_ytest = self.y.iloc[train_index], self.y.iloc[test_index]

        # Turn into an array
        original_Xtrain = original_Xtrain.values
        original_Xtest = original_Xtest.values
        original_ytrain = original_ytrain.values
        original_ytest = original_ytest.values

        # See if both the train and test label distribution are similarly distributed
        train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
        test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)
        print('-' * 100)

        print('Label Distributions: \n')
        print(train_counts_label/ len(original_ytrain))
        print(test_counts_label/ len(original_ytest))

        return original_Xtrain, original_Xtest, original_ytrain, original_ytest