import numpy as np

def train(X, Y, weights, stepMi, acceptableError):

    
    X_transpose = np.transpose(X)
    epochs = 0
    lms = 999999.
    
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

bias = 0.5
x = np.array([(1,-1,-1),(1,-1,1),(1,1,-1),(1,1,1)]).T
w = np.array([0.8,0.6,0.7])
y = np.array([-1,-1,-1,1])
stepAlpha = 0.05
acceptableError = 0.3
w,epochs = train(x,y,w,stepAlpha,acceptableError)
print("Weights")
print(w)
print("Epochs")
print(epochs)


print((x.T @ w > 0).astype(int))