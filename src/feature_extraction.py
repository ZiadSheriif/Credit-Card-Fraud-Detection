# feature_engineering.py

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
import time
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import pandas as pd

class FeatureExtraction:
    def __init__(self,df):
       self.df=df
        
    def oversample(self, X_train, y_train):
        return self.smote.fit_resample(X_train, y_train)
        
    def undersample(self):
        self.df=self.df.sample(frac=1)
        
        fraud_df = self.df.loc[self.df['Class'] == 1]
        non_fraud_df = self.df.loc[self.df['Class'] == 0][:492]
        normal_distributed_df = pd.concat([fraud_df, non_fraud_df])
        new_df = normal_distributed_df.sample(frac=1, random_state=42)
        self.df=new_df
        return new_df
        
        
    def reduce_dimensions(self):
        X = self.df.drop('Class', axis=1)
        y = self.df['Class']

        # T-SNE Implementation
        t0 = time.time()
        X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values)
        t1 = time.time()
        print("T-SNE took {:.2} s".format(t1 - t0))

        # PCA Implementation
        t0 = time.time()
        X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X.values)
        t1 = time.time()
        print("PCA took {:.2} s".format(t1 - t0))

        # TruncatedSVD
        t0 = time.time()
        X_reduced_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(X.values)
        t1 = time.time()
        print("Truncated SVD took {:.2} s".format(t1 - t0))

        return X_reduced_tsne, X_reduced_pca, X_reduced_svd
