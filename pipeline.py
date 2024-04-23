from utils.data_loading import load_data
from src.preprocessing import Preprocessor

if __name__ == "__main__":
    path = "./dataset/creditcard.csv"
    data = load_data(path)
    # preprocessor = Preprocessor(data)
    # preprocessed_data = preprocessor.preprocess()
    print(data.describe())
    print("Count of Fraudulent:", data['Class'].value_counts()[1])
    print("Count of Non-Fraudulent:", data['Class'].value_counts()[0])
    print('Frauds', round(data['Class'].value_counts()[1]/len(data) * 100,2), '% of the dataset')
    print('Non-Frauds', round(data['Class'].value_counts()[0]/len(data) * 100,2), '% of the dataset')
