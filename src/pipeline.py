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
    scaled_df=df.scale_features()
    
    # data splitting
    ds=DataSplitter(scaled_df,'Class')
    ds.split_data()
    
    # Feature Engineering
    fe=FeatureExtraction(scaled_df)
    under_sampled_df= fe.undersample()
    
    vis.plot_subsample(under_sampled_df)
    vis.plot_corr(under_sampled_df)
    vis.plot_neg_corr(under_sampled_df, ['V10', 'V12', 'V14', 'V17'])
    vis.plot_pos_corr(under_sampled_df, ['V4', 'V2', 'V11', 'V19'])
    vis.plot_fraud_dist(under_sampled_df, ['V10', 'V12', 'V14'])
    
    
    # remove outliers
    features=['V14', 'V12', 'V10', ]
    df.remove_outliers(under_sampled_df,features)
    
    vis.plot_outlier_reduction(under_sampled_df, features)
    
    # TODOS: Complete the pipeline 
    # x_red_tsne, x_red_pca, x_red_svd=fe.reduce_dimensions()
    # vis.plot_clusters(x_red_tsne, x_red_pca, x_red_svd, under_sampled_df['Class'])
    
    
    

if __name__ == "__main__":
    run()