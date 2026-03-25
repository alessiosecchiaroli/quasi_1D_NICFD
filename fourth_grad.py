import numpy as np


# 4th order accurate gradient function based on 2nd order version from http://projects.scipy.org/scipy/numpy/browser/trunk/numpy/lib/function_base.py
def gradientO4(f, *varargs):
   
    """ 
    Calculate the fourth-order-accurate gradient of an N-dimensional scalar function.
    Uses central differences on the interior and first differences on boundaries
    to give the same shape.
    Inputs:
      f -- An N-dimensional array giving samples of a scalar function
      varargs -- 0, 1, or N scalars giving the sample distances in each direction
    Outputs:
      N arrays of the same shape as f giving the derivative of f with respect
       to each dimension.

    """
    N = len (f.shape)  # number of dimensions
    n = len (varargs)
    if n == 0:
        dx = [1.0] * N
    elif n == 1:
        dx = [varargs[0]] * N
    elif n == N:
        dx = list (varargs)
    else:
        raise SyntaxError ("invalid number of arguments")

    # use central differences on interior and first differences on endpoints

    # print dx
    outvals = []

    # create slice objects --- initially all are [:, :, ..., :]
    slice0 = [slice (None)] * N
    slice1 = [slice (None)] * N
    slice2 = [slice (None)] * N
    slice3 = [slice (None)] * N
    slice4 = [slice (None)] * N

    otype = f.dtype.char
    if otype not in ['f', 'd', 'F', 'D']:
        otype = 'd'

    for axis in range (N):
        # select out appropriate parts for this dimension
        out = np.zeros (f.shape, f.dtype.char)

        slice0[axis] = slice (2, -2)
        slice1[axis] = slice (None, -4)
        slice2[axis] = slice (1, -3)
        slice3[axis] = slice (3, -1)
        slice4[axis] = slice (4, None)
        # 1D equivalent -- out[2:-2] = (f[:4] - 8*f[1:-3] + 8*f[3:-1] - f[4:])/12.0
        out[slice0] = (f[slice1] - 8.0 * f[slice2] + 8.0 * f[slice3] - f[slice4]) / 12.0

        slice0[axis] = slice (None, 2)
        slice1[axis] = slice (1, 3)
        slice2[axis] = slice (None, 2)
        # 1D equivalent -- out[0:2] = (f[1:3] - f[0:2])
        out[slice0] = (f[slice1] - f[slice2])

        slice0[axis] = slice (-2, None)
        slice1[axis] = slice (-2, None)
        slice2[axis] = slice (-3, -1)
        ## 1D equivalent -- out[-2:] = (f[-2:] - f[-3:-1])
        out[slice0] = (f[slice1] - f[slice2])

        # divide by step size
        outvals.append (out / dx[axis])

        # reset the slice object in this dimension to ":"
        slice0[axis] = slice (None)
        slice1[axis] = slice (None)
        slice2[axis] = slice (None)
        slice3[axis] = slice (None)
        slice4[axis] = slice (None)

    if N == 1:
        return outvals[0]
    else:
        return outvals


def gradientO4_fixed(f, dx):

    """ 
    Simplified version, it doesn't use the slice function anymore.
    """
    f = np.asarray(f)
    n = len(f)
    out = np.zeros_like(f, dtype=float)

    if n < 5:
        raise ValueError("Need at least 5 points")

    # Interior (4th order central)
    out[2:-2] = (f[:-4] - 8*f[1:-3] + 8*f[3:-1] - f[4:]) / (12*dx)

    # Left boundary (4th order one-sided)
    out[0] = (-25*f[0] + 48*f[1] - 36*f[2] + 16*f[3] - 3*f[4]) / (12*dx)
    out[1] = (-3*f[0] - 10*f[1] + 18*f[2] - 6*f[3] + f[4]) / (12*dx)

    # Right boundary (4th order one-sided)
    out[-2] = (3*f[-1] + 10*f[-2] - 18*f[-3] + 6*f[-4] - f[-5]) / (12*dx)
    out[-1] = (25*f[-1] - 48*f[-2] + 36*f[-3] - 16*f[-4] + 3*f[-5]) / (12*dx)

    return out