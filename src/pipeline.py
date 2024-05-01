from preprocessing import *
from data_splitter import *
from train import *
from visualization import *
from feature_extraction import *


def run():
    # Data Preprocessing
    df=DataPreprocessor('../dataset/creditcard.csv')
    df.preprocess()
    
    # data distribution
    vis=DataDistribution(df.df)
    vis.plot_class_distribution()
    vis.plot_distributions()
    
    
    # Data scaling
    df.scale_features()
    
    # data splitting
    ds=DataSplitter(df.df,'Class')
    ds.split_data()
    
    # Feature Engineering
    fe=FeatureEngineer()
    
    
    

if __name__ == "__main__":
    run()