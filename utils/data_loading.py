import pandas as pd


def load_data(file_path):
    return pd.read_csv(file_path)


# path_one = "../dataset/fraudTrain.csv"
# path_two = "../dataset/creditcard.csv"

# result = load_data(path_two)
# print(result.head())
