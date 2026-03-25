import pandas as pd
import numpy as np
from diff_grad import diff
from diff_grad import diff02

# read a file with x and y coordinates and generate the calculation grid
# feed coordinates in meters

def area_definer(scale_factor=1.0):

    coordinates = pd.read_csv("coord/nozzle_nominal.csv")

    x = coordinates["x"].to_numpy() * scale_factor # [m]
    y = coordinates["y"].to_numpy() * scale_factor # [m]
    z = 20e-3 # [m]

    Area = y*z*2 # multiply by two to obtain the entire channel

    # dA = np.diff(Area)
    # dA = np.gradient(Area)
    # dA = diff(Area,x,"backward")
    dA = diff02(Area,x)

    return x, Area, dA


# # TEST
# import matplotlib.pyplot as plt
#
# XX, A, DA = area_definer()
#
# plt.plot(XX,A, label= 'area')
# plt.plot(XX,DA, label = 'dA')
# plt.legend()
# plt.xlabel('x-coordinate [m]')
# plt.ylabel('Area (x) [m2]')
# plt.show()