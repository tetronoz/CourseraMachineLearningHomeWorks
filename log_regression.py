import numpy as np
from numpy import  r_, c_
from pylab import plot, show, xlabel, ylabel,legend, axis
from math import exp
from scipy import optimize

def plotData(x, y):
    pos = (y == 1).nonzero()[:1]
    neg = (y == 0).nonzero()[:1]

    plot(x[pos, 0].T, x[pos, 1].T, 'k+')
    plot(x[neg, 0].T, x[neg, 1].T, 'ro')

def sigmoid(x):
    #g = np.zeros(shape=x.shape)
    return 1.0 / (1.0 + np.e ** (-1.0 * x))

def costfunction(theta, x, y):    
    m = x.shape[0]
    y.shape = (m,1)
    hypothesis = sigmoid(x.dot(theta.T))
    hypothesis.shape = (m,1)
    j = (1.0 / m) * (-y * np.log(hypothesis) - (1 - y) * np.log(1 - hypothesis)).sum()
    g = computegrad(theta, x, y)
    return j#, g

def computegrad(theta, x, y):
    hypothesis = sigmoid(x.dot(theta.T))
    d = hypothesis - y
    theta_size = theta.size
    g = np.zeros(theta_size)
    m = y.size
    for i in range(theta_size):
        temp = (x[:,i])
        temp.shape = (m,1)
        g[i] = (1.0 / m) * (d * temp).sum() * -1
    return g

def plotDecisionBoundary(theta, x, y):
    plotData(x[:, 1:3], y)
    if x.shape[1] <= 3:
        plot_x = r_[x[:,2].min()-2,  x[:,2].max()+2]
        plot_y = (-1./theta[2]) * (theta[1]*plot_x + theta[0])
        plot(plot_x, plot_y)
        ylabel("Exam 2 score")
        xlabel("Exam 1 score")
        legend(['Admitted', 'Not admitted', 'Decision Boundary'])
        axis([30, 100, 30, 100])
    else:
        pass

def predict(theta, x):
    v = sigmoid(x.dot(theta))
    v.shape = (x.shape[0],1)
    p = v >= 0.5
    return p
            

if __name__ == '__main__':
    data = np.loadtxt('ex2data1.txt', dtype='float', delimiter=',')
    
    x = data[:, :2]
    y = data[:,2]
    m, n = data.shape
    
    #Plotting data
    plotData(x, y)

    ylabel("Exam 2 score")
    xlabel("Exam 1 score")
    legend(['Admitted', 'Not admitted'], loc='upper right')
    show()
    x = c_[np.ones(m), x]
    
    initial_theta = np.zeros(shape=(1,n))
    cost = costfunction(initial_theta, x, y)
    print 'Cost at initial theta (zeros): %f' % cost
    #print 'Gradient at initial theta (zeros): ' + str(grad) 
    
    options = {'full_output': True, 'maxiter': 400}
    theta, cost, _, _, _ = optimize.fmin(lambda t:costfunction(t, x, y), initial_theta, **options)
    print 'Cost at theta found by fminunc: %f' % cost
    print 'theta: %s' % theta
    
    plotDecisionBoundary(theta, x, y)
    show()
    prob = sigmoid(np.array([1,45,85]).dot(theta))
    print 'For a student with scores 45 and 85, we predict an admission probability of %f' % prob
    p = predict(theta, x)
    print 'Train Accuracy:', (p == y).mean() * 100