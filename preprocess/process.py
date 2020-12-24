import pandas as pd

def normalize(dataset):
    """
    Normalize feature values to [0, 1]
    :param dataset: a DataFrame object representing the given dataset
    :return: dataset: a DataFrame object containing normalized feature values from the given dataset
    All missing values in dataset are replaced with 0s
    """
    dataset = dataset.drop(["Unnamed: 0"], axis=1)
    dataset_class = dataset["clase"]
    dataset = dataset.drop(["clase"], axis=1)
    # Normalize features in dataset to the range [0, 1]
    dataset = (dataset - dataset.min()) / (dataset.max() - dataset.min())
    dataset = dataset.fillna(0)
    dataset["clase"] = dataset_class.values
    return dataset