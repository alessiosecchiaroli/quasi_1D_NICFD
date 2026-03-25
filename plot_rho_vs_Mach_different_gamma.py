import numpy as np
import CoolProp.CoolProp as cp
import matplotlib.pyplot as plt
from Geometry_define import area_definer
import os
from scipy.integrate import solve_bvp
from scipy.optimize import fsolve

'''
This file, solve the isentropic relation in terms of density as a simple function of Mach.
Gamma is given.
Obviously it's not physical and it's only used to investigate the effect of gamma on the solution.
''' 

# Initial conditions
T0 = 240  # C
p0 = 10 # bar

# Geometry 
x_old, area_old, dA_old = area_definer(1)

x = np.linspace(np.min(x_old), np.max(x_old), 2 * int(len(x_old)))
area = np.interp(x, x_old,area_old)
dA = np.interp(x, x_old,dA_old)

# Define the folder path based on experimental conditions
folder_path = 'Comparison_ideal_vs_non_ideal'

# Create the folder if it does not exist
os.makedirs(folder_path, exist_ok=True)


# transform units
T0 = T0 + 273 # K
p0 = p0 * 1E5 # Pa

# fluid choice
ideal = "REFPROP::N2"
non_ideal = "REFPROP::MM"

rho0 = cp.PropsSI('D','T',T0,'P',p0,non_ideal)

def rho(rho0,M,gamma):

    rho = rho0 * (1+((gamma-1)/2) * M**2)**(-1/(gamma-1))

    return np.array(rho)

Mach = np.load('BOS_December_2025/BOS_12_5_1/Mach.npy')
# Mach = np.linspace(0,10,1000)
gamma = np.linspace(0.4,0.99,10)

plt.figure()

for g in gamma:
    rho_plot = rho(rho0,Mach,g)

    plt.plot(Mach,rho_plot/rho0, label=f'$\gamma$ = {g}')

plt.ylabel(r'$\rho/\rho_0$')
plt.xlabel('Mach')
plt.legend()
plt.show()

