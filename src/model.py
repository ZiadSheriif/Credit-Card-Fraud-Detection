# model_training.py

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score,StratifiedShuffleSplit,GridSearchCV,cross_val_predict
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score,roc_curve
from sklearn.model_selection import learning_curve
import numpy as np
from typing import Counter
import matplotlib.pyplot as plt


class ModelTrainer:
    def __init__(self):
        self.estimator = {}
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
            self.estimator[name] = best_model
            scores = cross_val_score(best_model, X_train, y_train, cv=5)
            results[name] = scores.mean()
            print(f"Classifier: {name} has a training score of {round(scores.mean(), 2) * 100} % accuracy score")
            print(f"{name} Cross Validation Score: {round(scores.mean() * 100, 2)}%")
        return results
        
        
    def undersample_and_evaluate(self, df):
        undersample_X = df.drop('Class', axis=1)
        undersample_y = df['Class']

        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        
        # for train_index, test_index in sss.split(undersample_X, undersample_y):
        #     print("Train:", train_index, "Test:", test_index)
        #     undersample_Xtrain, undersample_Xtest = undersample_X.iloc[train_index], undersample_X.iloc[test_index]
        #     undersample_ytrain, undersample_ytest = undersample_y.iloc[train_index], undersample_y.iloc[test_index]

        undersample_accuracy = []
        undersample_precision = []
        undersample_recall = []
        undersample_f1 = []
        undersample_auc = []
        
        # Implementing NearMiss Technique 
        # Distribution of NearMiss (Just to see how it distributes the labels we won't use these variables)
        X_nearmiss, y_nearmiss = NearMiss().fit_resample(undersample_X.values, undersample_y.values)
        print('NearMiss Label Distribution: {}'.format(Counter(y_nearmiss)))

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
        
        


    def plot_learning_curve(self,X, y, ylim=None, cv=None,n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
        estimator1 = self.estimator["Logistic Regression"]
        estimator2 = self.estimator["K Nearest Neighbors"]
        estimator3 = self.estimator["Support Vector Machine"]
        estimator4 = self.estimator["Decision Tree"]
        estimator5 = self.estimator["Random Forest"]
        print("Estimators: ", estimator1, estimator2, estimator3, estimator4, estimator5)
        
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
        ax2.set_title("Knearst Neighbors Learning Curve", fontsize=14)
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
        plt.show()
        
    def calculate_roc_auc_scores(self,X_train, y_train):
        log_reg_pred = cross_val_predict(self.estimator["Logistic Regression"], X_train, y_train, cv=5, method="decision_function")
        knearst_pred = cross_val_predict(self.estimator["K Nearest Neighbors"], X_train, y_train, cv=5)
        svc_pred = cross_val_predict(self.estimator["Support Vector Machine"], X_train, y_train, cv=5, method="decision_function")
        tree_pred = cross_val_predict(self.estimator["Decision Tree"], X_train, y_train, cv=5)
        rf_pred = cross_val_predict(self.estimator["Random Forest"], X_train, y_train, cv=5)

        print('Logistic Regression: ', roc_auc_score(y_train, log_reg_pred))
        print('KNearst Neighbors: ', roc_auc_score(y_train, knearst_pred))
        print('Support Vector Classifier: ', roc_auc_score(y_train, svc_pred))
        print('Decision Tree Classifier: ', roc_auc_score(y_train, tree_pred))
        print('Random Forest Classifier: ', roc_auc_score(y_train, rf_pred))
        
    def graph_roc_curve_multiple(self, X_train, y_train):
        log_reg_pred = cross_val_predict(self.estimator["Logistic Regression"], X_train, y_train, cv=5, method="decision_function")
        knearst_pred = cross_val_predict(self.estimator["K Nearest Neighbors"], X_train, y_train, cv=5)
        svc_pred = cross_val_predict(self.estimator["Support Vector Machine"], X_train, y_train, cv=5, method="decision_function")
        tree_pred = cross_val_predict(self.estimator["Decision Tree"], X_train, y_train, cv=5)
        rf_pred = cross_val_predict(self.estimator["Random Forest"], X_train, y_train, cv=5)

        log_fpr, log_tpr, _ = roc_curve(y_train, log_reg_pred)
        knear_fpr, knear_tpr, _ = roc_curve(y_train, knearst_pred)
        svc_fpr, svc_tpr, _ = roc_curve(y_train, svc_pred)
        tree_fpr, tree_tpr, _ = roc_curve(y_train, tree_pred)
        rf_fpr, rf_tpr, _ = roc_curve(y_train, rf_pred)

        plt.figure(figsize=(16,8))
        plt.title('ROC Curve \n Top 5 Classifiers', fontsize=18)
        plt.plot(log_fpr, log_tpr, label='Logistic Regression Classifier Score: {:.4f}'.format(roc_auc_score(y_train, log_reg_pred)))
        plt.plot(knear_fpr, knear_tpr, label='KNearst Neighbors Classifier Score: {:.4f}'.format(roc_auc_score(y_train, knearst_pred)))
        plt.plot(svc_fpr, svc_tpr, label='Support Vector Classifier Score: {:.4f}'.format(roc_auc_score(y_train, svc_pred)))
        plt.plot(tree_fpr, tree_tpr, label='Decision Tree Classifier Score: {:.4f}'.format(roc_auc_score(y_train, tree_pred)))
        plt.plot(rf_fpr, rf_tpr, label='Random Forest Classifier Score: {:.4f}'.format(roc_auc_score(y_train, rf_pred)))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([-0.01, 1, 0, 1])
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                    arrowprops=dict(facecolor='#6E726D', shrink=0.05),
                    )
        plt.legend()
        plt.show()

        
#! I think we will use SVM , Decision Tree and Random Forest