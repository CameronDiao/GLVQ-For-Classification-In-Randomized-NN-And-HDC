import pandas as pd

def generate_pred(product_matrix):
    """
    Generate class type predictions based on product_matrix of dot product similarities
    :param product_matrix: a DataFrame object containing the dot product similarities of
    each data sample to each class type
    :return: a Series object containing the class predictions for each data sample
    """
    # Instantiate empty list of class predictions
    pred = []
    # Iterate through every testing sample
    for sample in product_matrix.index:
        # Predict class type of each testing sample based on product_matrix
        sample_pred = product_matrix.loc[sample]
        class_pred = sample_pred.idxmax(axis=1)
        pred.append(class_pred)
    # Convert list of class predictions to a Series object
    pred_series = pd.Series(pred, index=product_matrix.index)
    return pred_series