import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

class Preprocessor:
    def __init__(self, df):
        self.df = df

    def handle_missing_values(self):
        """Fills any missing values in the DataFrame with the median of the respective column."""
        self.df = self.df.fillna(self.df.median())

    def normalize_amount(self):
        """Normalizes the 'Amount' column using StandardScaler.
        This scales the 'Amount' feature to have a mean of 0 and a standard deviation of 1."""
        scaler = StandardScaler()
        self.df['Amount'] = scaler.fit_transform(self.df['Amount'].values.reshape(-1,1))

    def convert_time(self):
        """Converts the 'Time' feature from seconds to hours,
        and takes the modulus with 24 to fit the time into a 24-hour format."""
        self.df['Time'] = self.df['Time'].apply(lambda x : x / 3600 % 24)

    def handle_class_imbalance(self):
        """Uses the Synthetic Minority Over-sampling Technique 
        (SMOTE) to handle the class imbalance in the dataset. 
        It creates synthetic samples of the minority 
        class (frauds) to balance the dataset.
        """
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(self.df.drop('Class', axis=1), self.df['Class'])
        self.df = pd.concat([pd.DataFrame(X_res), pd.DataFrame(y_res, columns=['Class'])], axis=1)

    def preprocess(self):
        self.handle_missing_values()
        self.normalize_amount()
        self.convert_time()
        self.handle_class_imbalance()
        return self.df