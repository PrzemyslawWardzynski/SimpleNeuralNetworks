import numpy as np


def unipolar_to_bipolar(data):
    tmp = np.where(data <= 0.2, data - 1, data)
    tmp[:,0] = 0
    return tmp 

def to_bias(data):
    tmp = data.copy()
    tmp[:,0] = 1
    return tmp

def threshold_function(isUnipolar, z, threshold):

    result = (z > threshold).astype(int)
    if(isUnipolar):
        return result

    else:
        return result if result == 1 else -1

def predict(X, weights, threshold, isUnipolar):
    z = (X.T)@weights
    return threshold_function(isUnipolar, z, threshold)





