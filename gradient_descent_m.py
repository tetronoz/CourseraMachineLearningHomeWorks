import numpy as np
from pylab import plot,show, xlabel, ylabel, xlim, ylim, hold, legend, scatter, title
import sys

def featureNormalize(x):
    x_norm = x
    mu = []
    sigma = []

    for i in range(x.shape[1]):
        m = np.mean(x[:,i])
        s = np.std(x[:,i])
        mu.append(m)
        sigma.append(s)
        x_norm[:,i] = (x_norm[:,i] - m)/s
    
    return x_norm, mu, sigma

def computecost(x, y, theta):
    m = y.size
    hypothesis = x.dot(theta)
    err = (hypothesis - y)
    j = (1.0 / (2 * m)) * err.T.dot(err)

    return j

def gradientDescentMulti(x, y, theta, alpha, num_iters):
    m = y.size
    
    j_history = np.zeros(shape=(num_iters, 1))
    
    for i in range(num_iters):
        hypothesis = x.dot(theta)
        theta_size = theta.size
        for z in range(theta_size):
            temp = x[:,z]
            temp.shape = (m, 1)
            err1 = (hypothesis - y) * temp
            theta[z][0] = theta[z][0] - alpha * (1.0 / m) * err1.sum()
        j_history[i, 0] = computecost(x, y, theta)
    
    return theta, j_history

if __name__ == '__main__':
    
    # Load Data
    data = np.loadtxt('ex1data2.txt', delimiter=',')
    x = data[:,0:2]
    y = data[:, 2]
    m = y.size
    y.shape = (m, 1)
        
    # Scale feature and set them to zero mean
    
    alpha = 0.01
    num_iters = 400
    theta = np.zeros(shape=(3,1))
    
    x_norm, mu, sigma = featureNormalize(x)
    
    x = np.ones(shape=(m,3))
    x[:,1:3] = x_norm
    
    theta, j_history = gradientDescentMulti(x, y, theta, alpha, num_iters)
    
    print theta, j_history
    plot(np.arange(num_iters), j_history)
    xlabel('Number of iterations')
    ylabel('Cost J')
    show()
 
    #Predict price of a 1650 sq-ft 3 br house
    price = np.array([1.0,   ((1650.0 - mu[0]) / sigma[0]), ((3 - mu[1]) / sigma[1])]).dot(theta)
    print 'Predicted price of a 1650 sq-ft, 3 br house: %f' % (price)


    
