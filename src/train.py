# model_training.py

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score,StratifiedShuffleSplit,GridSearchCV
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt


class ModelTrainer:
    def __init__(self):
        self.models = {
            "Logistic Regression": {
                "model": LogisticRegression(),
                "params": {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            },
            "K Nearest Neighbors": {
                "model": KNeighborsClassifier(),
                "params": {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
            },
            "Support Vector Machine": {
                "model": SVC(),
                "params": {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
            },
            "Decision Tree": {
                "model": DecisionTreeClassifier(),
                "params": {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), "min_samples_leaf": list(range(5,7,1))}
            },
            "Random Forest": {
                "model": RandomForestClassifier(),
                "params": {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), "min_samples_leaf": list(range(5,7,1))}
            }
        }
        
        
        
        
    def train_models(self, X_train, y_train):
        results = {}
        for name, model_info in self.models.items():
            grid_search = GridSearchCV(model_info["model"], model_info["params"])
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            scores = cross_val_score(best_model, X_train, y_train, cv=5)
            results[name] = scores.mean()
            print(f"Classifier: {name} has a training score of {round(scores.mean(), 2) * 100} % accuracy score")
            print(f"{name} Cross Validation Score: {round(scores.mean() * 100, 2)}%")
        return results
        
        
    def undersample_and_evaluate(self, df):
        undersample_X = df.drop('Class', axis=1)
        undersample_y = df['Class']

        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

        undersample_accuracy = []
        undersample_precision = []
        undersample_recall = []
        undersample_f1 = []
        undersample_auc = []

        for train_index, test_index in sss.split(undersample_X, undersample_y):
            undersample_Xtrain, undersample_Xtest = undersample_X.iloc[train_index], undersample_X.iloc[test_index]
            undersample_ytrain, undersample_ytest = undersample_y.iloc[train_index], undersample_y.iloc[test_index]

            undersample_pipeline = imbalanced_make_pipeline(NearMiss(sampling_strategy='majority'), LogisticRegression())
            undersample_model = undersample_pipeline.fit(undersample_Xtrain, undersample_ytrain)
            undersample_prediction = undersample_model.predict(undersample_Xtest)

            undersample_accuracy.append(undersample_pipeline.score(undersample_Xtest, undersample_ytest))
            undersample_precision.append(precision_score(undersample_ytest, undersample_prediction))
            undersample_recall.append(recall_score(undersample_ytest, undersample_prediction))
            undersample_f1.append(f1_score(undersample_ytest, undersample_prediction))
            undersample_auc.append(roc_auc_score(undersample_ytest, undersample_prediction))

        return undersample_accuracy, undersample_precision, undersample_recall, undersample_f1, undersample_auc
        
        


    def plot_learning_curve(estimator1, estimator2, estimator3, estimator4, estimator5, X, y, ylim=None, cv=None,
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
        f, ((ax1, ax2), (ax3, ax4), (ax5, _)) = plt.subplots(3, 2, figsize=(28,22), sharey=True)
        if ylim is not None:
            plt.ylim(*ylim)
        # First Estimator
        train_sizes, train_scores, test_scores = learning_curve(
            estimator1, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="#ff9124")
        ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
        ax1.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
                label="Training score")
        ax1.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
                label="Cross-validation score")
        ax1.set_title("Logistic Regression Learning Curve", fontsize=14)
        ax1.set_xlabel('Training size (m)')
        ax1.set_ylabel('Score')
        ax1.grid(True)
        ax1.legend(loc="best")
        
        # Second Estimator 
        train_sizes, train_scores, test_scores = learning_curve(
            estimator2, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        ax2.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="#ff9124")
        ax2.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
        ax2.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
                label="Training score")
        ax2.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
                label="Cross-validation score")
        ax2.set_title("Knears Neighbors Learning Curve", fontsize=14)
        ax2.set_xlabel('Training size (m)')
        ax2.set_ylabel('Score')
        ax2.grid(True)
        ax2.legend(loc="best")
        
        # Third Estimator
        train_sizes, train_scores, test_scores = learning_curve(
            estimator3, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        ax3.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="#ff9124")
        ax3.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
        ax3.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
                label="Training score")
        ax3.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
                label="Cross-validation score")
        ax3.set_title("Support Vector Classifier \n Learning Curve", fontsize=14)
        ax3.set_xlabel('Training size (m)')
        ax3.set_ylabel('Score')
        ax3.grid(True)
        ax3.legend(loc="best")
        
        # Fourth Estimator
        train_sizes, train_scores, test_scores = learning_curve(
            estimator4, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        ax4.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="#ff9124")
        ax4.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
        ax4.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
                label="Training score")
        ax4.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
                label="Cross-validation score")
        ax4.set_title("Decision Tree Classifier \n Learning Curve", fontsize=14)
        ax4.set_xlabel('Training size (m)')
        ax4.set_ylabel('Score')
        ax4.grid(True)
        ax4.legend(loc="best")

        # Fifth Estimator
        train_sizes, train_scores, test_scores = learning_curve(
            estimator5, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        ax5.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="#ff9124")
        ax5.fill_between(train_sizes, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
        ax5.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
                    label="Training score")
        ax5.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
                    label="Cross-validation score")
        ax5.set_title("Random Forest Classifier \n Learning Curve", fontsize=14)
        ax5.set_xlabel('Training size (m)')
        ax5.set_ylabel('Score')
        ax5.grid(True)
        ax5.legend(loc="best")
        
        return plt

        
#! I think we will use SVM , Decision Tree and Random Forest