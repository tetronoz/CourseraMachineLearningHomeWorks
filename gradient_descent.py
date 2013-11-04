import numpy as np
from pylab import plot,show, xlabel, ylabel, xlim, ylim, hold, legend, scatter, title

def compute_cost(x, y, theta):
    m = y.size
    hypothesis = x.dot(theta).flatten()
    j = (1.0 / (2 * m)) * ((hypothesis - y) ** 2).sum()
    return j

def gradient_descent(x, y, theta, alpha, iterations):
    m = y.size
    j_history = np.zeros(shape=(iterations, 1))
    
    for i in range(iterations):
        hypothesis = x.dot(theta).flatten()
        
        err1 = (hypothesis - y) * x[:,0]
        err2 = (hypothesis - y) * x[:,1]
        
        theta[0][0] = theta[0][0] - alpha * (1.0 / m) * err1.sum()
        theta[1][0] = theta[1][0] - alpha * (1.0 / m) * err2.sum()
        
        j_history[i, 0] = compute_cost(x, y, theta)
    return theta, j_history

if __name__ == '__main__':
    data = np.loadtxt('ex1data1.txt', dtype='float', delimiter=',')
    x = data[:,0]
    y = data[:,1]
    m = y.size
    ylabel('Profit in $10,000s')
    xlabel('Population of City in 10,000s')
    plot(x,y,'rx')
    show()
    x = np.ones(shape=(m, 2))
    x[:,1] = data[:,0]

    theta = np.zeros(shape=(2,1))
    iterations = 1500
    alpha = 0.01

    print compute_cost(x, y, theta)
    theta, j_history = gradient_descent(x, y, theta, alpha, iterations)
    predict1 = np.array([1, 3.5]).dot(theta).flatten()
    print 'For population = 35,000, we predict a profit of %f' % (predict1 * 10000)
    predict2 = np.array([1, 7.0]).dot(theta).flatten()
    print 'For population = 70,000, we predict a profit of %f' % (predict2 * 10000)
    ylabel('Profit in $10,000s')
    xlabel('Population of City in 10,000s')
    result = x.dot(theta).flatten()
    plot(data[:, 0], data[:, 1],'rx')
    plot(data[:, 0], result)
    legend(['Training data', 'Linear regression'], loc='lower right')
    show()

