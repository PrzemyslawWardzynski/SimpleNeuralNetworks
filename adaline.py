import numpy as np
import random as random
from utility import *

REPS = 10
MAX_EPOCHS = 10000

def train(X, Y, weights, stepMi, acceptableError):

    
    X_transpose = np.transpose(X)
    epochs = 0
    lms = float("inf")
    
    while(lms/X.shape[1] > acceptableError and epochs < 10000):
        epochs += 1
        lms = 0.
        
        for i in range(0,X.shape[1]):
            
            xi = X_transpose[i]
           
            z = xi @ weights
            
            delta =  Y[i] - z
            lms += delta*delta
 
            weights += stepMi * delta * xi
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



data = unipolar_to_bipolar(data)
data = to_bias(data)

#TEST1 - bias , weights range experiment
print("TEST1")

def test_initial_weights(data, weight_range, bias, step, acceptableError):

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
            
            w,epochs = train(x,y,weights_copy,step,acceptableError)
            
            avgEpochs += epochs
            avgWeights += w

            if(epochs == MAX_EPOCHS):
                break
        
        
        testList.append("{0};{1}".format(weight_range[i],avgEpochs/(j+1)))

    print(testList)

weight_range = np.arange(1,0,-0.2)
bias = 0.5
step = 0.01
acceptableError = 0.3
test_initial_weights(data,weight_range,bias,step,acceptableError)
#TEST2 - step experiment
print("TEST2")

def test_step(data, weights, bias, step, acceptableError):

    x = data[:,:3].T
    y = data[:,3]

    testList = []

    for i in range(0,step.shape[0]):

        avgEpochs = 0
        avgWeights = np.zeros(3)

        for j in range(0,REPS):

            
            weights_copy = weights.copy()
            weights_copy[0] = bias
            
            w,epochs = train(x,y,weights_copy,step[i],acceptableError)
            
            avgEpochs += epochs
            avgWeights += w

            if(epochs == MAX_EPOCHS):
                break
        
        
        testList.append("{0:1.2};{1}".format(step[i],avgEpochs/(j+1)))

    print(testList)

bias = 0.5
weights= np.array([bias,0.1,0.01])
step = np.arange(0.01,1,0.1)
acceptableError = 0.3
test_step(data,weights,bias,step,acceptableError)



#TEST3 - error value experiment 
print("TEST3")

def test_acceptable_error(data, weights, bias, step, acceptableError):

    x = data[:,:3].T
    y = data[:,3]

    testList = []

    for i in range(0,acceptableError.shape[0]):

        avgEpochs = 0
        avgWeights = np.zeros(3)

        for j in range(0,REPS):

            
            weights_copy = weights.copy()
            weights_copy[0] = bias
            
            w,epochs = train(x,y,weights_copy,step,acceptableError[i])
            
            avgEpochs += epochs
            avgWeights += w

            if(epochs == MAX_EPOCHS):
                break
        
        
        testList.append("{0:1.2};{1}".format(acceptableError[i],avgEpochs/(j+1)))

    print(testList)

bias = 0.5
weights= np.array([bias,0.1,0.01])
step = 0.41
acceptableError = np.arange(0.1,0.5,0.05)
test_acceptable_error(data,weights,bias,step,acceptableError)