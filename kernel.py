## The following is a collection of kernel density functions

import numpy as np

def mvkernel(x,kernel):
    """
PURPOSE: Calculate product kernel values at points x

INPUTS
x       points of kernel evaluation (3)x(n)x(dim of x)
kernel  Scalar value indicating which kernel to use where
           1 = Rectangular
           2 = Triangular
           3 = Biweight
           4 = Silverman's
           5 = Epanechnikov order 2
           6 = Epanechnikov order 2 alt
           7 = Epanechnikov order 4
           8 = Epanechnikov order 6
           9 = Gaussian order 2
           10 = Gaussian order 4
           11 = Gaussian order 6

OUTPUTS
ker     product of kernel values along axis = 0
    """

    if kernel == 1:
        # Rectangular Kernel
        ker = 1/2*(np.absolute(x)<1)
    elif kernel == 2:
        # Triangular Kernel
        ker = (1-np.absolute(x))*(np.absolute(x)<1)
    elif kernel == 3:
        # Biweight Kernel
        ker = (15/16*(1-x**2)**2)*(np.absolute(x)< 1)
    elif kernel == 4:
        # Silvermans Kernel
        ker = 0.5*np.exp(-np.absolute(x)/np.sqrt(2))*np.sin(np.absolute(x)/np.sqrt(2) + np.pi/4)
    elif kernel == 5:
        # Epanechnikov order 2
        ker = 0.75*(1-x**2)*(np.absolute(x)<=1)
    elif kernel == 6:
        # Epanechnikov order 2 (alt)?
        ker = (0.75/np.sqrt(5))*(1-0.2*x**2)*(np.absolute(x)<=np.sqrt(5))
    elif kernel == 7:
        # Epanechnikov order 4
        ker  = (0.75/np.sqrt(5))*(15/8-7/8*x**2)*(1-0.2*x**2)*(np.absolute(x)<=np.sqrt(5))
    elif kernel == 8:
        # Epanechnikov order 6
        ker  = ((0.75/np.sqrt(5))*(175/64-105/32*x**2+ 231/320*x**4)
                                 *(1-0.2*x**2)*(np.absolute(x)<=np.sqrt(5)))
    elif kernel == 9:
        # Gaussian order 2
        ker  = (1/np.sqrt(2*np.pi))*np.exp(-0.5*x**2)
    elif kernel == 10:
        # Gaussian order 4
        ker  = (3/2 -1/2*x**2)*(1/np.sqrt(2*np.pi))*np.exp(-0.5*x**2)
    elif kernel == 11:
        # Gaussian order 6
        ker  = (15/8-5/4*x**2+1/8*x**4)*(1/np.sqrt(2*np.pi))*np.exp(-0.5*x**2)
    else:
        print('Incorrect Kernel Number')

    if x.ndim > 2:
        # Value of each product kernel
        ker = np.prod(ker,axis = 0)

    return ker

def mvden(x,p,h,kernel):

    """
Purpose: Multivariate Rosenblatt Kernel Density Estimator at points p based on sample x

INPUTS
x       np.array of data having order (number of observations)x(dimension of X)
P       np.array of points of evaluation having order (number of points)x(dimension of X)
h       Bandwidth vector of order (1)x(dimension of X)
kernel  Scalar value indicating which kernel to use where
           1 = Rectangular
           2 = Triangular
           3 = Biweight
           4 = Silverman's
           5 = Epanechnikov order 2
           6 = Epanechnikov order 2 alt
           7 = Epanechnikov order 4
           8 = Epanechnikov order 6
           9 = Gaussian order 2
           10 = Gaussian order 4
           11 = Gaussian order 6

OUTPUTS
den      Density at each point of evaluation (number of observation)x1
    """

    x = np.array(x)
    p = np.array(p)
    h = np.array(h)

    if x.ndim == 1:
            m0 = (x.reshape(1,x.shape[0])-p.reshape(p.shape[0],1))/h
    elif x.ndim == 2:
        m0 = np.zeros((x.shape[1],p.shape[0],x.shape[0]))
        for i in range(0,x.shape[1]):
            m0[i,:,:] = (x[:,i].reshape(1,x.shape[0])-p[:,i].reshape(p.shape[0],1))/h[i]

    ker = mvkernel(m0,kernel)/(x.shape[0]*np.prod(h))
    den = np.dot(ker,np.ones(x.shape[0]))

    return den
