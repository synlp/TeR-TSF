import numpy as np


def WAPE(pred, true):
    numerator = np.sum(np.abs(pred - true))
    denominator = np.sum(np.abs(true))
    return numerator/denominator

def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def metric(pred, true, train=None, seasonality=6):

    mae = MAE(pred, true)
    wape = WAPE(pred, true)
    return mae, wape