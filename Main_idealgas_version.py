import numpy as np
import scipy.optimize
# from sympy import *
from scipy.integrate import solve_bvp
import CoolProp.CoolProp as cp
import pandas as pd
import os

from diff_grad import diff
from diff_grad import  diff02
from Geometry_define import area_definer
from ideal_gas import PVRT
from scipy.optimize import minimize
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from fourth_grad import gradientO4, gradientO4_fixed


# Quasi 1-D code to obtain thermodynamic properties along a nozzle solving the area-Mach relation for an ideal gas
# Set the geometry through Geometry_define
# USE REFPROP instad of HEOS

x_old, area_old, dA_old = area_definer(1)
# x = x_old
# area = area_old
# dA = dA_old

x = np.linspace(np.min(x_old), np.max(x_old), 2 * int(len(x_old)))
area = np.interp(x, x_old,area_old)
dA = np.interp(x, x_old,dA_old)


# Define the folder path based on experimental conditions
folder_path = 'Comparison_ideal_vs_non_ideal'

# Create the folder if it does not exist
os.makedirs(folder_path, exist_ok=True)

# Constants
R_u = 8.314  # J/mol/K
M_molar = 0.028  # kg/mol for MM (adjust if different)
R = R_u / M_molar  # J/kg/K


# Initial conditions
T0 = 220 + 273.15  # K
p0 = 10e5  # Pa
gamma_const = cp.PropsSI('CPMASS','T',T0,'P',p0,'N2')/cp.PropsSI('CVMASS','T',T0,'P',p0,'N2')


# Ideal gas properties
def ideal_density(p, T):
    return p / (R * T)

def ideal_speed_of_sound(T):
    return (gamma_const * R * T) ** 0.5

def isentropic_pressure_ratio(M):
    return (1 + (gamma_const - 1) / 2 * M**2) ** (-gamma_const / (gamma_const - 1))

def isentropic_temperature_ratio(M):
    return (1 + (gamma_const - 1) / 2 * M**2) ** -1

def isentropic_area_ratio(M):
    return (1/M) * ((2 / (gamma_const + 1)) * (1 + (gamma_const - 1)/2 * M**2)) ** ((gamma_const + 1)/(2*(gamma_const - 1)))

# velocity, function of Mach and speed of sound
u = lambda x, y : x * y  # x = M, y = c

rho0 = ideal_density(p0, T0)
c0 = ideal_speed_of_sound(T0)



M0 = 0.05
v0 = M0 * c0
mf = area[0] * rho0 * v0

MACH = np.zeros_like(area)
p = np.zeros_like(area)
T = np.zeros_like(area)
rho = np.zeros_like(area)
gamma = np.full_like(area, gamma_const)
c = np.zeros_like(area)

A_star = np.min(area)

for i, a in enumerate(area):
    # Solve for Mach from area/M relation
    func = lambda M: isentropic_area_ratio(M) - (a / A_star)

    # Use previous Mach to guess sub or supersonic branch
    guess = M0 if i == 0 else MACH[i-1]
    if MACH[i-1] > 0.9 and MACH[i-1] < 1.2:
        guess = 1.1
    M = fsolve(func, guess, xtol=1e-16)[0]

    T[i] = T0 * isentropic_temperature_ratio(M)
    p[i] = p0 * isentropic_pressure_ratio(M)
    rho[i] = ideal_density(p[i], T[i])
    c[i] = ideal_speed_of_sound(T[i])
    MACH[i] = M

    M0 = M  # update guess


#
x_to_int = np.linspace(x[0],x[-1],10 * len(x))
step = x_to_int[1]-x_to_int[0]

M_int = np.interp(x_to_int,x,MACH)
A_int = np.interp(x_to_int,x,area)
rho_int = np.interp(x_to_int,x,rho)
# drho = diff(rho,x,"central")
# drho = np.gradient(rho_int,x_to_int,edge_order=1)
drho = diff02(rho,x)
# drho = gradientO4_fixed(rho_int,step)



plt.plot(x*1000, np.abs(drho))
plt.ylabel('Density gradient [kg/m4]')
plt.xlabel('Nozzle length [mm]')
plt.grid(True)
plt.savefig(os.path.join(folder_path, 'density_gradient_plot.pdf'), dpi=300, bbox_inches='tight')
plt.show()

fluid_speed = u(MACH,c)

plt.plot(x*1000, fluid_speed)
plt.ylabel('Speed [m/s]')
plt.xlabel('Nozzle length [mm]')
plt.grid(True)
plt.savefig(os.path.join(folder_path, 'velocity.pdf'), dpi=300, bbox_inches='tight')
plt.show()

# plt.plot(x*1000, gamma)
# plt.ylabel('Ratio of specific heats')
# plt.xlabel('Nozzle length [mm]')
# plt.grid(True)
# plt.show()

# plt.plot(M_int,rho_int/dens0)
# plt.show()

# ADIMENSIONAL PLOTS
# plt.plot(MACH,A_star/area, label='A*/A')
# plt.plot(MACH,p/p0, label= 'p/p0')
# plt.plot(MACH,T/T0, label='T/T0')
# plt.plot(MACH,rho/dens0, label='rho/rho0')
# plt.plot(MACH,c/c0, label='c/c0')
# plt.legend()
# plt.xlabel('Mach')
# plt.grid(True)
# plt.show()

plt.plot(M_int,A_star/A_int, label='A*/A')
plt.plot(MACH,p/p0, label= 'p/p0')
plt.plot(MACH,T/T0, label='T/T0')
plt.plot(M_int,rho_int/rho0, label='rho/rho0')
plt.plot(MACH,c/c0, label='c/c0')
plt.legend()
plt.xlabel('Mach')
plt.grid(True)
plt.savefig(os.path.join(folder_path, 'adimensional.pdf'), dpi=300, bbox_inches='tight')
plt.show()

# DIMENSIONAL RESULTS
plt.figure(figsize=(16, 9))  # Adjust the size to fit your screen resolution
plt.subplot(4,1,1)
plt.plot(x*1000,MACH) #, label='M')
plt.grid(True)
plt.ylabel('Mach')
plt.subplot(4,1,2)
plt.plot(x*1000, p) #, label = 'p')
plt.grid(True)
plt.ylabel('Pressure [Pa]')
plt.subplot(4,1,3)
plt.plot(x*1000,T) #, label= 'T')
plt.grid(True)
plt.ylabel('Temperature [K]')
plt.subplot(4,1,4)
plt.plot(x*1000,rho) #, label = 'rho')
plt.grid(True)
plt.ylabel('Density [kg/m3]')
# plt.legend()
plt.xlabel('Nozzle length [mm]')

# Adjust layout to prevent overlapping
plt.tight_layout()

plt.savefig(os.path.join(folder_path, 'subplots.pdf'), dpi=300, bbox_inches='tight')
plt.show()


# Save the NumPy variable inside the folder
np.save(os.path.join(folder_path, 'x_coordinates.npy'), x)
np.save(os.path.join(folder_path, 'Density'), rho)
np.save(os.path.join(folder_path, 'Mach'), MACH)
np.save(os.path.join(folder_path, 'Pressure'), p)
np.save(os.path.join(folder_path, 'Temperature'), T)
np.save(os.path.join(folder_path, 'Area'), area)
np.save(os.path.join(folder_path, 'Speed'), fluid_speed)
np.save(os.path.join(folder_path, 'Gamma'), gamma)
np.save(os.path.join(folder_path, 'c'), c)

np.save(os.path.join(folder_path, 'M_int'), M_int)
# np.save(os.path.join(folder_path, 'A_int'), A_int)
# np.save(os.path.join(folder_path, 'A_star'), A_star)
np.save(os.path.join(folder_path, 'rho_int'), rho_int)
np.save(os.path.join(folder_path, 'dens0'), rho0)
np.save(os.path.join(folder_path, 'c0'), c0)
# np.save(os.path.join(folder_path, 'p0'), p0)
# np.save(os.path.join(folder_path, 'T0'), T0)
