import numpy as np
from sympy import *
from scipy.integrate import solve_bvp
import CoolProp.CoolProp as cp
import pandas as pd

from diff_grad import diff
from diff_grad import  diff02
from Geometry_define import area_definer
from ideal_gas import PVRT
from scipy.optimize import minimize
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from fourth_grad import gradientO4


# Quasi 1-D code to obtain thermodynamic properties along a nozzle solving the area-Mach relation for an ideal gas
# Set the geometry through Geometry_define

x_old, area_old, dA_old = area_definer()
# x = x_old
# area = area_old
# dA = dA_old

x = np.linspace(np.min(x_old), np.max(x_old), 2 * int(len(x_old)))
area = np.interp(x, x_old,area_old)
dA = np.interp(x, x_old,dA_old)

# Experiments conditions
T0 = 220 # Celsius
p0 = 4 #bar
# Experiment conditions x coolprop
T0 = T0 + 273.15 # Kelvin
p0 = p0 * 1E5 #Pascal

# Fluid
fluid = "MM"
# Z0 = cp.PropsSI('Z','T', T0,'P', P0, 'MM' )
gamma0 = cp.PropsSI('CPMASS','T', T0,'P', p0, fluid )/cp.PropsSI('CVMASS','T', T0,'P', p0, fluid )
R = 8.3125 # J/(K * mol)
dens0, c0 = PVRT(T0, p0,fluid)

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

A_star = np.min(area)
#
plt.scatter(x,1 - A_star/area)
plt.xlabel('x coordinate [mm]')
plt.ylabel('1 - A*/A')
plt.show()


def mach_area_rel(x,y):

    f1 = (y / A_star) - (1/x[0]) * ((x[3] + 1 )/2)**(-1/2 * (x[3] + 1)/(x[3] - 1)) * (1+ (x[3] - 1) / 2 * x[0]**2)**((x[3]+1) /2 /(x[3] - 1) )
    f2 = x[1] - p0 * (1 + (x[3] - 1) / 2 * x[0] ** 2) ** (-1 * (x[3]) / (x[3] - 1))
    f3 = x[2] - T0 * (1 + (x[3] - 1) / 2 * x[0] ** 2) ** (-1)
    # f4 = x[3] - dens0 * (1 + (x[4] - 1) / 2 * x[0] ** 2) ** (-1 / (x[4] - 1))
    f4 = x[3] - cp.PropsSI('CPMASS', 'T', x[2], 'P', x[1], fluid)/cp.PropsSI('CVMASS', 'T', x[2], 'P', x[1], fluid)

    return np.array([f1, f2, f3, f4])


MACH = np.zeros_like(area)
# MACH = np.linspace(0.05,2,len(area))
tol = 1e-6

for i, a in enumerate(area):
    if i == 0:
        x0 = [M0, p0, T0, gamma0]
    else:
        x0 = [M0, p[i-1], T[i-1], gamma[i-1]]
    y = a
    mach, pp, Tt, gammaa = fsolve(mach_area_rel,x0, args=(y),xtol=1e-16,maxfev=10000)

    # to help the solver reach the wanted solution
    # if Mach' is negative, start from supersonic conditions
    if mach - MACH[i-1] < 0:
        mach, pp, Tt, gammaa = fsolve (mach_area_rel, [2, p[i-1], T[i-1], gamma[i-1]], args=(y), xtol=1e-16, maxfev=10000)
    # gammaa = cp.PropsSI('CPMASS','T', Tt,'P', pp, fluid )/cp.PropsSI('CVMASS','T', Tt,'P', pp, fluid )
    MACH[i] = mach
    p[i] = pp
    T[i] = Tt
    gamma[i] = gammaa
    rho[i] = cp.PropsSI('D','T',Tt,'P',pp,fluid)
    c[i] = cp.PropsSI ('A','T', Tt, 'P', pp, fluid)

    M0 = mach
    # print(mach)
    if M0 > 0.35:
        M0 = M0 * (1+tol)
    elif M0 > 0.95:
        M0 = M0 * (1+10*tol)



#
x_to_int = np.linspace(x[0],x[-1],10 * len(x))
M_int = np.interp(x_to_int,x,MACH)
A_int = np.interp(x_to_int,x,area)
rho_int = np.interp(x_to_int,x,rho)

# drho = diff(rho,x,"forward")
# drho = np.gradient(rho_int,x_to_int,edge_order=1)
drho = diff02(rho,x)
# drho = gradientO4(rho_int,x_to_int)

plt.plot(x*1000, np.abs(drho))
plt.ylabel('Density gradient [kg/m4]')
plt.xlabel('Nozzle length [mm]')
plt.grid(True)
plt.show()


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
plt.plot(M_int,rho_int/dens0, label='rho/rho0')
plt.plot(MACH,c/c0, label='c/c0')
plt.legend()
plt.xlabel('Mach')
plt.grid(True)
plt.show()

# DIMENSIONAL RESULTS
plt.figure()
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
plt.show()

np.save('x_coordinates',x)
np.save('Density',rho)