import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from matplotlib import patches
import warnings
warnings.filterwarnings("ignore")

class DataDistribution:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def plot(self, x: str, y: str):
        plt.plot(self.data[x], self.data[y])
        plt.show()

    def plot_class_distribution(self, colors=["#0101DF", "#DF0101"]):
        sns.countplot(x='Class', data=self.data, palette=colors)
        plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
        plt.show()

    def plot_distributions(self):
        fig, ax = plt.subplots(1, 2, figsize=(18,4))

        amount_val = self.data['Amount'].values
        time_val = self.data['Time'].values

        sns.distplot(amount_val, ax=ax[0], color='r')
        ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
        ax[0].set_xlim([min(amount_val), max(amount_val)])

        sns.distplot(time_val, ax=ax[1], color='b')
        ax[1].set_title('Distribution of Transaction Time', fontsize=14)
        ax[1].set_xlim([min(time_val), max(time_val)])

        plt.show()
        
    def plot_subsample(self, new_df, colors=["#0101DF", "#DF0101"]):
        print('Distribution of the Classes in the subsample dataset')
        print(new_df['Class'].value_counts()/len(new_df))

        sns.countplot(x='Class', data=new_df, palette=colors)
        plt.title('Equally Distributed Classes', fontsize=14)
        plt.show()
        
    def plot_corr(self, new_df):
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24,20))

        # Entire DataFrame
        corr = self.data.corr()
        sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)
        ax1.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=14)

        sub_sample_corr = new_df.corr()
        sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax2)
        ax2.set_title('SubSample Correlation Matrix \n (use for reference)', fontsize=14)
        plt.show()
        
    def plot_neg_corr(self, new_df, features, colors=["#0101DF", "#DF0101"]):
        f, axes = plt.subplots(ncols=len(features), figsize=(20,4))

        for i, feature in enumerate(features):
            sns.boxplot(x="Class", y=feature, data=new_df, palette=colors, ax=axes[i])
            axes[i].set_title(f'{feature} vs Class Negative Correlation')

        plt.show()
        
    def plot_pos_corr(self, new_df, features, colors=["#0101DF", "#DF0101"]):
        f, axes = plt.subplots(ncols=len(features), figsize=(20,4))

        for i, feature in enumerate(features):
            sns.boxplot(x="Class", y=feature, data=new_df, palette=colors, ax=axes[i])
            axes[i].set_title(f'{feature} vs Class Positive Correlation')

        plt.show()
        
    def plot_fraud_dist(self, new_df, features, colors=["#FB8861", "#56F9BB", "#C5B3F9"]):
        f, axes = plt.subplots(1, len(features), figsize=(20, 6))

        for i, feature in enumerate(features):
            fraud_dist = new_df[feature].loc[new_df['Class'] == 1].values
            sns.distplot(fraud_dist, ax=axes[i], fit=norm, color=colors[i])
            axes[i].set_title(f'{feature} Distribution \n (Fraud Transactions)', fontsize=14)

        plt.show()
        
    def plot_outlier_reduction(self, new_df, features, colors=["#B3F9C5", "#f9c5b3"]):
        f, axes = plt.subplots(1, len(features), figsize=(20,6))

        for i, feature in enumerate(features):
            sns.boxplot(x="Class", y=feature, data=new_df, ax=axes[i], palette=colors)
            axes[i].set_title(f'{feature} Feature \n Reduction of outliers', fontsize=14)
            axes[i].annotate('Fewer extreme \n outliers', xy=(0.98, -17.5), xytext=(0, -12),
                             arrowprops=dict(facecolor='black'),
                             fontsize=14)

        plt.show()

    def plot_clusters(self, X_reduced_tsne, X_reduced_pca, X_reduced_svd, y):
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24,6))
        f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)

        blue_patch = patches.Patch(color='#0A0AFF', label='No Fraud')
        red_patch = patches.Patch(color='#AF0000', label='Fraud')

        # t-SNE scatter plot
        ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
        ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
        ax1.set_title('t-SNE', fontsize=14)
        ax1.grid(True)
        ax1.legend(handles=[blue_patch, red_patch])

        # PCA scatter plot
        ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
        ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
        ax2.set_title('PCA', fontsize=14)
        ax2.grid(True)
        ax2.legend(handles=[blue_patch, red_patch])

        # TruncatedSVD scatter plot
        ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
        ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
        ax3.set_title('Truncated SVD', fontsize=14)
        ax3.grid(True)
        ax3.legend(handles=[blue_patch, red_patch])

        plt.show()