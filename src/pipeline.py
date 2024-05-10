from preprocessing import *
from data_splitter import *
from model import *
from visualization import *
from feature_extraction import *
from sklearn.model_selection import ShuffleSplit, StratifiedKFold

PATH = "../dataset/creditcard.csv"


def run():
    # Data Preprocessing
    df = DataPreprocessor(PATH)
    df.preprocess()
    
    
    X = df.df.drop('Class', axis=1)
    y = df.df['Class']

    stf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

    for train_index, test_index in stf.split(X, y):
        print("Train:", train_index, "Test:", test_index)
        original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
        original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]
        
    original_Xtrain = original_Xtrain.values
    original_Xtest = original_Xtest.values
    original_ytrain = original_ytrain.values
    original_ytest = original_ytest.values

    # data distribution
    vis = DataDistribution(df.df)
    vis.plot_class_distribution()
    vis.plot_distributions()

    # Data scaling
    scaled_df = df.scale_features()

    # data splitting
    ds = DataSplitter(scaled_df, "Class")
    ds.split_data()

    # Feature Engineering
    fe = FeatureExtraction(scaled_df)
    under_sampled_df = fe.undersample()

    # Data Visualization
    vis.plot_subsample(under_sampled_df)
    vis.plot_corr(under_sampled_df)
    vis.plot_neg_corr(under_sampled_df, ["V10", "V12", "V14", "V17"])
    vis.plot_pos_corr(under_sampled_df, ["V4", "V2", "V11", "V19"])
    vis.plot_fraud_dist(under_sampled_df, ["V10", "V12", "V14"])

    # remove outliers
    features = ["V14", "V12", "V10"]
    df.remove_outliers(under_sampled_df, features)

    # visualize the outlier reduction
    vis.plot_outlier_reduction(under_sampled_df, features)

    # Feature Extraction and Visualization
    x_red_tsne, x_red_pca, x_red_svd = fe.reduce_dimensions()
    vis.plot_clusters(x_red_tsne, x_red_pca, x_red_svd, under_sampled_df["Class"])

    # Initialize the model trainer
    train = ModelTrainer()
    X = under_sampled_df.drop("Class", axis=1)
    y = under_sampled_df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Turn the values into an array for feeding the classification algorithms.
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values

    # Train the models
    train.train_models(X_train, y_train)
    train.undersample_and_evaluate(under_sampled_df)

    # Evaluate the model
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
    train.plot_learning_curve(X_train, y_train, cv=cv, n_jobs=4)
    train.calculate_roc_auc_scores(X_train, y_train)
    train.graph_roc_curve_multiple(X_train, y_train)
    train.logistic_roc_curve()
    train.confusion_matrix(X_test, y_test)
    train.statistics_of_classifiers( y_test)
    
    # make over sampling in order not to skip features
    train.smote_on_logistic_regression(original_Xtrain, original_Xtest, original_ytrain, original_ytest,stf)
    train.sampling_by_smote(original_Xtrain, original_ytrain, original_Xtest, original_ytest, X_test, y_test)


if __name__ == "__main__":
    run()
