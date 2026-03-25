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


# Quasi 1-D code to obtain thermodynamic properties along a nozzle solving the area-Mach relation for a non-ideal gas
# It make use of the generalized isentropic equations from Negerstigt 2023.
# Set the geometry through Geometry_define
# USE REFPROP instad of HEOS


x_old, area_old, dA_old = area_definer(1)
# x = x_old
# area = area_old
# dA = dA_old

x = np.linspace(np.min(x_old), np.max(x_old), 2 * int(len(x_old)))
area = np.interp(x, x_old,area_old)
dA = np.interp(x, x_old,dA_old)

# Experiments conditions
T0 = 220 # Celsius
p0 = 12 #bar

# Define the folder path based on experimental conditions
folder_path = 'CO2_test'

# Create the folder if it does not exist
os.makedirs(folder_path, exist_ok=True)

# Experiment conditions x coolprop
T0 = T0 + 273.15 # Kelvin
p0 = p0 * 1E5 #Pascal

# Fluid
fluid = "REFPROP::MM"

Z0 = cp.PropsSI('Z','T', T0,'P', p0, fluid )
gamma0 = cp.PropsSI('CPMASS','T', T0,'P', p0, fluid )/cp.PropsSI('CVMASS','T', T0,'P', p0, fluid )
R = 8.3125 # J/(K * mol)
dens0, c0 = PVRT(T0, p0,fluid)
isot_comp_0 = cp.PropsSI('ISOTHERMAL_COMPRESSIBILITY','T', T0,'P', p0, fluid)
gamma_pv_0 = gamma0 / p0 / isot_comp_0


# velocity, function of Mach and speed of sound
u = lambda x, y : x * y  # x = M, y = c

# initialize via Mach number
M0 = 0.05
v0 = u(M0,c0)

mf = area[0] * dens0 * v0 #[kg/m3] mass flow

p = np.zeros_like(area)
T = np.zeros_like(area)
rho = np.zeros_like(area)
gamma = np.zeros_like(area)
c = np.zeros_like(area)
gamma_pv = np.zeros_like(area)

A_star = np.min(area)
#
# plt.scatter(x,1 - A_star/area)
# plt.xlabel('x coordinate [mm]')
# plt.ylabel('1 - A*/A')
# plt.show()


def mach_area_rel(x,y):

    # y = area
    # x[0] = Mach, x[1] = p, x[2] = D, x[3] = gamma_pv
    f1 = (y / A_star) - (1/x[0]) * ((2 + (x[3] - 1) * x[0]**2)/(x[3]+1))**((x[3]+1) /2 /(x[3] - 1) )
    f2 = x[1] - p0 * (1 + (x[3] - 1) / 2 * x[0] ** 2) ** (-1 * (x[3]) / (x[3] - 1))
    # f3 = x[2] - T0 * ((1 + (x[3] - 1) / 2 * x[0] ** 2) ** (-1 * (gamma_tv-1)/(x[3] - 1)))
    f3 = x[2] - dens0 * (1 + (x[3] - 1) / 2 * x[0] ** 2) ** (-1 / (x[3] - 1))
    f4 = x[3] - (cp.PropsSI('CPMASS', 'D', x[2], 'P', x[1], fluid)/cp.PropsSI('CVMASS', 'D', x[2], 'P', x[1], fluid)) / (x[1] * cp.PropsSI('ISOTHERMAL_COMPRESSIBILITY','D', x[2],'P', x[1], fluid))

    return np.array([f1, f2, f3, f4])


MACH = np.zeros_like(area)
# MACH = np.linspace(0.05,2,len(area))
tol = 1e-3

for i, a in enumerate(area):

    print(i)
    print(M0)

    if i == 0:
        x0 = [M0, p0, dens0, gamma_pv_0]
    else:
        x0 = [M0, p[i-1], rho[i-1], gamma_pv[i-1]]
    
    y = a
    mach, pp, Rr, gammaa_pv = fsolve(mach_area_rel,x0, args=(a),xtol=1e-16,maxfev=10000)

    # # to help the solver reach the wanted solution
    # # if Mach is decreasing, start the search from supersonic conditions and lower p,T, gamma

    if mach - MACH[i-1] < 0: # force Mach to be higher than 1
        mach, pp, Rr, gammaa_pv = fsolve (mach_area_rel, [2, p[i-1], rho[i-1], gamma_pv[i-1]], args=(a), xtol=1e-16, maxfev=10000)

    # gammaa = cp.PropsSI('CPMASS','T', Tt,'P', pp, fluid )/cp.PropsSI('CVMASS','T', Tt,'P', pp, fluid )
    MACH[i] = mach
    p[i] = pp
    rho[i] = Rr
    gamma_pv[i] = gammaa_pv
    T[i] = cp.PropsSI('T','D',rho[i],'P',p[i],fluid)
    c[i] = cp.PropsSI ('A','D', rho[i], 'P', p[i], fluid)
    gamma[i] = cp.PropsSI('CPMASS','D',rho[i],'P',p[i],fluid)/cp.PropsSI('CVMASS','D',rho[i],'P',p[i],fluid)

    M0 = mach
    # print(mach)
    if M0 > 0.35 and M0 < 0.95:
        M0 = M0 * (1+tol)
    elif M0 > 0.95:
        M0 = M0 * (1+1e1*tol)



#
x_to_int = np.linspace(x[0],x[-1],10 * len(x))
x_step = x_to_int[1] - x_to_int[0]
M_int = np.interp(x_to_int,x,MACH)
A_int = np.interp(x_to_int,x,area)
rho_int = np.interp(x_to_int,x,rho)
c_int = np.interp(x_to_int,x,c)
fluid_speed = u(M_int,c_int)


mass_flow = rho_int * fluid_speed * A_int

plt.plot(x_to_int,mass_flow)
plt.show()

# drho = diff(rho,x,"forward")
# drho = np.gradient(rho_int,x_to_int,edge_order=1)
# drho = diff02(rho,x)
# # drho_4 = gradientO4_fixed(rho_int,x_step)



# plt.plot(x*1000, -1*drho)
# # plt.plot(x_to_int*1000, -1*drho_4)
# plt.ylabel('Density gradient [kg/m4]')
# plt.xlabel('Nozzle length [mm]')
# plt.grid(True)
# plt.savefig(os.path.join(folder_path, 'density_gradient_plot.pdf'), dpi=300, bbox_inches='tight')
# plt.show()

# fluid_speed = u(MACH,c)

# plt.plot(x*1000, fluid_speed)
# plt.ylabel('Speed [m/s]')
# plt.xlabel('Nozzle length [mm]')
# plt.grid(True)
# plt.savefig(os.path.join(folder_path, 'velocity.pdf'), dpi=300, bbox_inches='tight')
# plt.show()


# plt.plot(M_int,A_star/A_int, label='A*/A')
# plt.plot(MACH,p/p0, label= 'p/p0')
# plt.plot(MACH,T/T0, label='T/T0')
# plt.plot(M_int,rho_int/dens0, label='rho/rho0')
# plt.plot(MACH,c/c0, label='c/c0')
# plt.legend()
# plt.xlabel('Mach')
# plt.grid(True)
# plt.savefig(os.path.join(folder_path, 'adimensional.pdf'), dpi=300, bbox_inches='tight')
# plt.show()

# # DIMENSIONAL RESULTS
# plt.figure(figsize=(16, 9))  # Adjust the size to fit your screen resolution
# plt.subplot(5,1,1)
# plt.plot(x*1000,MACH) #, label='M')
# plt.grid(True)
# plt.ylabel('Mach')
# plt.subplot(5,1,2)
# plt.plot(x*1000, p) #, label = 'p')
# plt.grid(True)
# plt.ylabel('Pressure [Pa]')
# plt.subplot(5,1,3)
# plt.plot(x*1000,T) #, label= 'T')
# plt.grid(True)
# plt.ylabel('Temperature [K]')
# plt.subplot(5,1,4)
# plt.plot(x*1000,rho) #, label = 'rho')
# plt.grid(True)
# plt.ylabel('Density [kg/m3]')
# plt.subplot(5,1,5)
# plt.plot(x*1000,gamma_pv)
# plt.grid(True)
# plt.ylabel('Gamma_pv')
# # plt.legend()
# plt.xlabel('Nozzle length [mm]')

# # Adjust layout to prevent overlapping
# plt.tight_layout()

# plt.savefig(os.path.join(folder_path, 'subplots.pdf'), dpi=300, bbox_inches='tight')
# plt.show()


# # Save the NumPy variable inside the folder
# np.save(os.path.join(folder_path, 'x_coordinates.npy'), x)
# np.save(os.path.join(folder_path, 'Density'), rho)
# np.save(os.path.join(folder_path, 'Mach'), MACH)
# np.save(os.path.join(folder_path, 'Pressure'), p)
# np.save(os.path.join(folder_path, 'Temperature'), T)
# np.save(os.path.join(folder_path, 'Area'), area)
# np.save(os.path.join(folder_path, 'Speed'), fluid_speed)
# np.save(os.path.join(folder_path, 'Gamma'), gamma)
# np.save(os.path.join(folder_path, 'Gamma_pv'), gamma_pv)
# np.save(os.path.join(folder_path, 'c'), c)

# np.save(os.path.join(folder_path, 'M_int'), M_int)
# np.save(os.path.join(folder_path, 'A_int'), A_int)
# np.save(os.path.join(folder_path, 'A_star'), A_star)
# np.save(os.path.join(folder_path, 'rho_int'), rho_int)
# np.save(os.path.join(folder_path, 'dens0'), dens0)
# np.save(os.path.join(folder_path, 'c0'), c0)
# np.save(os.path.join(folder_path, 'p0'), p0)
# np.save(os.path.join(folder_path, 'T0'), T0)
# np.save(os.path.join(folder_path, 'Z0'), Z0)

# plt.plot(x*1000, gamma)
# plt.show()