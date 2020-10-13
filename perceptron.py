import numpy as np
import random as random

UNIPOLAR = True
BIPOLAR = False
REPS = 10
MAX_EPOCHS = 10000

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



def train(X, Y, weights, threshold, stepAlpha, isUnipolar):

    epochs = 0
    X_transpose = np.transpose(X)
    delta = np.ones(X.shape[1])
  
    while(np.any(delta != 0) and epochs < MAX_EPOCHS):
        epochs += 1
        delta = np.ones(X.shape[1])
        delta[0] = X_transpose[0][0]

        for i in range(0,X.shape[1]):
            xi = X_transpose[i]
           
            z = xi @ weights
            delta[i] =  Y[i] - threshold_function(isUnipolar,z,threshold)
            if(delta[i] != 0):
                weights += stepAlpha * delta[i] * xi 
                weights = np.where(weights > 1.0, 1.0, weights)
                weights = np.where(weights < -1.0, -1.0, weights)
        

    return weights,epochs

#                   x0,x1,x2,y
data = np.array([
                    (0,0,0,0),
                    (0,0.01,0.02,0),
                    (0,0.02,0.02,0),
                    (0,0,0.01,0),

                    (0,0,1,0),
                    (0,0.01,0.99,0),
                    (0,0.02,0.98,0),
                    (0,0.01,1,0),

                    (0,1,0,0),
                    (0,0.98,0.01,0),
                    (0,1,0.02,0),
                    (0,0.99,0.02,0),
                    
                    (0,1,1,1),
                    (0,0.99,0.98,1),
                    (0,0.98,1,1),
                    (0,1,0.99,1),

                ])

def test_constant_threshold(data, weights, threshold, step, isUnipolar):

    if(isUnipolar == BIPOLAR):
        data = unipolar_to_bipolar(data)

    x = data[:,:3].T
    y = data[:,3]


    testList = []

    for i in range(0,threshold.shape[0]):

        avgEpochs = 0
        avgWeights = np.zeros(3)

        for j in range(0,REPS):

            
            weights_copy = weights.copy()
            
            w,epochs = train(x,y,weights_copy,threshold[i],step,isUnipolar)
            
            avgEpochs += epochs
            avgWeights += w

            if(epochs == MAX_EPOCHS):
                break
        
        
        testList.append("{0};{1}".format(threshold[i],avgEpochs/(j+1)))

    print(testList)

#TEST1 - theta threshold experiment

#weights = np.array([0,0.1,0.1])
#threshold = np.array([0.,0.4,0.8,1.2,1.6,2])
#step = 0.01
#test_constant_threshold(data, weights, threshold, step, UNIPOLAR)

#TEST2 - bias , weigths range experiment



def test_initial_weights(data, weight_range, bias, step, isUnipolar):

    if(isUnipolar == BIPOLAR):
        data = unipolar_to_bipolar(data)

    data = to_bias(data)

    x = data[:,:3].T
    y = data[:,3]


    testList = []

    for i in range(0,weight_range.shape[0]):

        avgEpochs = 0
        avgWeights = np.zeros(3)

        for j in range(0,REPS):

            weights = np.array([
                            bias,
                            random.uniform(-1*weight_range[i],weight_range[i]),
                            random.uniform(-1*weight_range[i],weight_range[i])
                            ])
           
            weights_copy = weights.copy()
            
            w,epochs = train(x,y,weights_copy,0,step,isUnipolar)
            
            avgEpochs += epochs
            avgWeights += w

            if(epochs == MAX_EPOCHS):
                break
        
        
        testList.append("{0};{1}".format(weight_range[i],avgEpochs/(j+1)))

    print(testList)

#weight_range = np.arange(1,0,-0.1)
#bias = -0.5
#step = 0.01
#test_initial_weights(data,weight_range,bias,step,UNIPOLAR)

#TEST3 - step experiment

def test_step(data, weights, bias, step, isUnipolar):

    if(isUnipolar == BIPOLAR):
        data = unipolar_to_bipolar(data)

    data = to_bias(data)

    x = data[:,:3].T
    y = data[:,3]


    testList = []

    for i in range(0,step.shape[0]):

        avgEpochs = 0
        avgWeights = np.zeros(3)

        for j in range(0,REPS):

            
            weights_copy = weights.copy()
            weights_copy[0] = bias
            
            w,epochs = train(x,y,weights_copy,0,step[i],isUnipolar)
            
            avgEpochs += epochs
            avgWeights += w

            if(epochs == MAX_EPOCHS):
                break
        
        
        testList.append("{0};{1}".format(step[i],avgEpochs/(j+1)))

    print(testList)

#bias = 0.5
#weights= np.array([bias,0.1,0.01])
#step = np.arange(0.01,1,0.05)
#test_step(data,weights,bias,step,UNIPOLAR)

#TEST4 - threshold function (unipolar/bipolar) experiment

def test_activation_function(data, weights, bias, step, isUnipolar):

    if(isUnipolar == BIPOLAR):
        data = unipolar_to_bipolar(data)

    data = to_bias(data)

    

    x = data[:,:3].T
    y = data[:,3]

   

    testList = []
    avgEpochs = 0
    avgWeights = np.zeros(3)

    for j in range(0,REPS):

        
        weights_copy = weights.copy()
        weights_copy[0] = bias
        
        w,epochs = train(x,y,weights_copy,0,step,isUnipolar)
        
        avgEpochs += epochs
        avgWeights += w

        if(epochs == MAX_EPOCHS):
            break
        
        
    testList.append("{0};{1}".format(isUnipolar,avgEpochs/(j+1)))

    print(testList)

bias = 0.5
weights= np.array([bias,0.1,0.1])
step = 0.01
test_activation_function(data,weights,bias,step,UNIPOLAR)
test_activation_function(data,weights,bias,step,BIPOLAR)
