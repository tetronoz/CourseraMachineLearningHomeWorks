import numpy as np
from numpy import  r_, c_, mat
from pylab import plot, show, xlabel, ylabel, legend, axis
from scipy import optimize

def plotdata(x, y):  
    plot(x[y.flatten() == 1, 0], x[y.flatten() == 1, 1], "k+")
    plot(x[y.flatten() == 0, 0], x[y.flatten() == 0, 1], "yo")

def mapfeature(x1, x2):
    degree = 7
    m = x1.shape[0]
    
    out = np.ones(shape=(m,1))
    for i in range(1,degree):
        for j in range(i+1):
            t = (x1 ** (i-j)) * (x2 ** j)
            t.shape = (m,1)
            out = np.c_[out, t]
            
    return out            

def sigmoid(x):
    return 1.0 / (1.0 + np.e ** (-1.0 * x))

def costfunctionreg(theta, x, y, l):
    m = y.size
    y.shape = (x.shape[0],1)
    j = 0
    theta = theta.reshape((x.shape[1],1))
    
    th = theta[1:, 0]
    
    hypothesis = sigmoid(x.dot(theta))
    j = 1.0 / m * (-y * np.log (hypothesis) - (1 - y) * np.log (1 - hypothesis)).sum() + l / 2 * m *(th.sum()) 
    
    return j

def plotdecisionboundary(theta, x, y):
    plotdata(x[:, 1:3], y)
    if x.shape[1] <= 3:
        plot_x = r_[x[:,2].min()-2,  x[:,2].max()+2]
        plot_y = (-1./theta[2]) * (theta[1]*plot_x + theta[0])
        plot(plot_x, plot_y)
        ylabel("Exam 2 score")
        xlabel("Exam 1 score")
        legend(['Admitted', 'Not admitted', 'Decision Boundary'])
        axis([30, 100, 30, 100])
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros(shape=(len(u), len(v)))
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = (mapfeature(u[i], v[j])) * theta
 
        z = z.T
        contour(u, v, z)
        title('lambda = %f' % l)
        xlabel('Microchip Test 1')
        ylabel('Microchip Test 2')
        legend(['y = 1', 'y = 0', 'Decision boundary'])

if __name__ == '__main__':
    
    data = np.loadtxt('ex2data2.txt', delimiter=',')
    x = data[:, 0:2]
    y = data[:,2]
    
    plotdata(x, y)
    xlabel('Microchip Test 1')
    ylabel('Microchip Test 2')
    legend(['y = 1', 'y = 0'], loc='upper right')
    #show()

    #=========== Part 1: Regularized Logistic Regression ============
    #Add Polynomial Features
    x = mapfeature(x[:,0], x[:,1])
    
    #initial_theta = np.zeros(shape=(x.shape[1], 1))
    initial_theta = np.zeros(shape=(x.shape[1]))
    l = 1
    cost = costfunctionreg(initial_theta, x, y, l)
    print "Cost at initial theta (zeros): %f\n" % cost
    options = {'full_output': True, 'maxiter': 400}
    theta, cost, _, _, _ = optimize.fmin(lambda t:costfunctionreg(t, x, y, l), initial_theta, **options)
    #print 'Cost at theta found by fminunc: %f' % cost
    #print 'theta: %s' % theta
    
    plotdecisionboundary(theta, x, y)
    show()
