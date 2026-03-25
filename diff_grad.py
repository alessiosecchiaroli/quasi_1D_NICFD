import matplotlib.pyplot as plt
import numpy as np

''' 
This file contains two functions:

- diff: calculate the gradient of 'f' with respect to 'h' using forward, backward or central numerical scheme
- diff02: calculate the gradient of 'f'with respect to 'h' using a 2nd order spatial accurate scheme

'''

def diff(f,h,method):

    grad = np.zeros_like(f)


    # print(grad)
    if method == 'central':

        grad[0] = (f[1]-f[0])/(h[1]-h[0])
        # print(grad)
        grad[-1] = (f[-1]-f[-2])/(h[-1]-h[-2])
        # print(grad)

        for i in range(1,len(f)-1):

            grad[i] = (f[i+1]-f[i-1])/(h[i+1]-h[i-1])

        return grad

    if method == 'forward':

        grad[-1] = (f[-1] - f[-2]) / (h[-1] - h[-2])

        for i in range(len(f)-1):
            grad[i] = (f[i + 1] - f[i]) / (h[i + 1] - h[i])

        return grad

    if method == 'backward':

        grad[0] = (f[1] - f[0]) / (h[1] - h[0])

        for i in range(1,len(f)):

            grad[i] = (f[i] - f[i-1])/(h[i] - h[i-1])

        return grad

def diff02(f,h):
    df = np.zeros_like (f)
    # Second order forward difference for first element
    df[0] = -(3 * f[0] - 4 * f[1] + f[2]) / (h[2] - h[0] )

    # Central difference interior elements
    df[1:-1] = (f[2:] - f[0:-2]) / (h[2:] - h[0:-2])

    # Backwards difference final element
    df[-1] = (3 * f[-1] - 4 * f[-2] + f[-3]) / (h[-1] - h[-3])

    return df

# TEST
# f = np.random.rand(611,1)
# h = np.random.rand(611,1)
#
# g1 = diff(f,h,'central')
# g2 = diff(f,h,'forward')
# g3 = diff(f,h,'backward')
#
# plt.figure()
#
# plt.plot(g1,color='red')
# plt.plot(g2, color='green')
# plt.plot(g3, color='blue')

# plt.show()
