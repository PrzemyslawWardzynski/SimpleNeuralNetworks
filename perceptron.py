import numpy as np

def train(X, Y, weights, tresholdTheta, stepAlpha):

    epochs = 0
    X_transpose = np.transpose(X)
    delta = np.ones(X.shape[1]) 
    
    while(np.any(delta != 0)):
        epochs += 1
        delta = np.ones(X.shape[1])
        for i in range(0,X.shape[1]):
            
            xi = X_transpose[i]
           
            z = xi @ weights
            delta[i] =  Y[i] - (z > tresholdTheta).astype(int)
            
            weights += stepAlpha * delta[i] * xi 
            weights = np.where(weights > 1.0, 1.0, weights)
            weights = np.where(weights < -1.0, -1.0, weights)
            
        

    return weights,epochs




x = np.array([(0,0),(0,1),(1,0),(1,1)]).T
w = np.array([-0.7,-0.6])
y = np.array([0,0,0,1])
treshold = 1
stepAlpha = 1

w,epochs = train(x,y,w,treshold,stepAlpha)
print(w)
print(epochs)

