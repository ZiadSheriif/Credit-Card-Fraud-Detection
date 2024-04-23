from utils.data_loading import load_data
from src.preprocessing import Preprocessor

if __name__ == "__main__":
    path = "./dataset/creditcard.csv"
    data = load_data(path)
    preprocessor = Preprocessor(data)
    preprocessed_data = preprocessor.preprocess()
    print(preprocessed_data.head())
