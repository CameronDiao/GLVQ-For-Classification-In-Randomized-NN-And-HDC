import numpy as np
import pandas as pd

def generate_pred(product_matrix):
    pred = []
    for sample in product_matrix.index:
        sample_pred = product_matrix.loc[sample]
        if (sample_pred[0.0] > sample_pred[1.0]):
            pred.append(0.0)
        else:
            pred.append(1.0)
    pred_series = pd.Series(pred, index=product_matrix.index)
    return pred_series