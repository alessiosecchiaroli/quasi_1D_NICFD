import numpy as np
import matplotlib.pyplot as plt
import os
from diff_grad import diff02

plt.rcParams.update({'font.size': 16})

# # Enable LaTeX for rendering
plt.rc('text', usetex=True)

# # Set the font to Computer Modern Roman
plt.rc('font', family='serif', serif=['Computer Modern Roman'])

ideal_gas_path = os.path.join('Throat study/220C_9bar')
NICFD_path = os.path.join('Throat study/220C_9bar_refthroat')

# Universal variables
T0 = np.load(os.path.join(NICFD_path, 'T0.npy'))
p0 = np.load(os.path.join(NICFD_path, 'p0.npy'))
A_int = np.load(os.path.join(NICFD_path, 'A_int.npy'))
A_star = np.load(os.path.join(NICFD_path, 'A_star.npy'))
Z0 = np.load(os.path.join(NICFD_path, 'Z0.npy'))

# Load ideal gas data
ideal_x = np.load(os.path.join(ideal_gas_path, 'x_coordinates.npy'))
ideal_rho = np.load(os.path.join(ideal_gas_path, 'Density.npy'))
ideal_MACH = np.load(os.path.join(ideal_gas_path, 'Mach.npy'))
ideal_p = np.load(os.path.join(ideal_gas_path, 'Pressure.npy'))
ideal_T = np.load(os.path.join(ideal_gas_path, 'Temperature.npy'))
ideal_area = np.load(os.path.join(ideal_gas_path, 'Area.npy'))
ideal_speed = np.load(os.path.join(ideal_gas_path, 'Speed.npy'))
ideal_gamma = np.load(os.path.join(ideal_gas_path, 'Gamma.npy'))
ideal_c = np.load(os.path.join(ideal_gas_path, 'c.npy'))
ideal_c0 = np.load(os.path.join(ideal_gas_path, 'c0.npy'))
ideal_rho0 = np.load(os.path.join(ideal_gas_path, 'dens0.npy'))
ideal_M_int = np.load(os.path.join(ideal_gas_path, 'M_int.npy'))
ideal_rho_int = np.load(os.path.join(ideal_gas_path, 'rho_int.npy'))

# Load NICFD data
nicfd_x = np.load(os.path.join(NICFD_path, 'x_coordinates.npy'))
nicfd_rho = np.load(os.path.join(NICFD_path, 'Density.npy'))
nicfd_MACH = np.load(os.path.join(NICFD_path, 'Mach.npy'))
nicfd_p = np.load(os.path.join(NICFD_path, 'Pressure.npy'))
nicfd_T = np.load(os.path.join(NICFD_path, 'Temperature.npy'))
nicfd_area = np.load(os.path.join(NICFD_path, 'Area.npy'))
nicfd_speed = np.load(os.path.join(NICFD_path, 'Speed.npy'))
nicfd_gamma = np.load(os.path.join(NICFD_path, 'Gamma.npy'))
nicfd_c = np.load(os.path.join(NICFD_path, 'c.npy'))
nicfd_c0 = np.load(os.path.join(NICFD_path, 'c0.npy'))
nicfd_rho0 = np.load(os.path.join(NICFD_path, 'dens0.npy'))
nicfd_M_int = np.load(os.path.join(NICFD_path, 'M_int.npy'))
nicfd_rho_int = np.load(os.path.join(NICFD_path, 'rho_int.npy'))

# calculate density gradients

nicfd_rho_grad = diff02(nicfd_rho, nicfd_x)
ideal_rho_grad = diff02(ideal_rho, ideal_x)

# plotting

# Density gradient comparison
plt.figure()
plt.plot(nicfd_x*1000, -1*nicfd_rho_grad, color='blue')
plt.plot(ideal_x*1000, -1*ideal_rho_grad, linestyle='dashed', color='blue')
# plt.legend()
plt.grid(True)
plt.ylabel('Density Gradient [kg/m3/m]')
plt.xlabel('Nozzle length [mm]')
plt.title(f'Density gradients [$Z_0 ={Z0:.2f}$]')
plt.tight_layout()
plt.show()


# DIMENSIONAL RESULTS
plt.figure(figsize=(16, 9)) # Adjust the size to fit your screen resolution
plt.subplot(4,1,1)
plt.plot(ideal_x*1000,ideal_MACH, linestyle='dashed')
plt.plot(nicfd_x*1000,nicfd_MACH)
# plt.legend()
plt.grid(True)
plt.ylabel('Mach')
plt.title(f'Dimensional comparisons between Ideal Gas and NICFD [$Z_0 ={Z0:.2f}$]')
plt.subplot(4,1,2)
plt.plot(ideal_x*1000, ideal_p, linestyle='dashed')
plt.plot(nicfd_x*1000, nicfd_p)
# plt.legend()
plt.grid(True)
plt.ylabel('Pressure [Pa]')
plt.subplot(4,1,3)
plt.plot(ideal_x*1000,ideal_T, linestyle='dashed')
plt.plot(nicfd_x*1000,nicfd_T)
# plt.legend()
plt.grid(True)
plt.ylabel('Temperature [K]')
plt.subplot(4,1,4)
plt.plot(ideal_x*1000,ideal_rho, linestyle='dashed')
plt.plot(nicfd_x*1000,nicfd_rho)
# plt.legend()
plt.grid(True)
plt.ylabel('Density [kg/m3]')
# plt.legend()
plt.xlabel('Nozzle length [mm]')
plt.tight_layout()
plt.show()

# Adimensional plots
# NICFD
plt.figure()
plt.plot(nicfd_M_int,A_star/A_int, color='blue',label='A*/A')
plt.plot(nicfd_MACH,nicfd_p/p0, color='orange', label= 'p/$p_0$')
plt.plot(nicfd_MACH,nicfd_T/T0, color='green', label='T/$T_0$')
plt.plot(nicfd_M_int,nicfd_rho_int/nicfd_rho0, color='purple', label=r'$\rho$/$\rho_0$')
plt.plot(nicfd_MACH,nicfd_c/nicfd_c0, color='red', label='c/$c_0$')
# Ideal Gas
plt.plot(ideal_M_int,A_star/A_int, linestyle='dashed', color='blue')
plt.plot(ideal_MACH,ideal_p/p0, linestyle='dashed', color='orange')
plt.plot(ideal_MACH,ideal_T/T0, linestyle='dashed', color='green')
plt.plot(ideal_M_int,ideal_rho_int/ideal_rho0, linestyle='dashed', color='purple')
plt.plot(ideal_MACH,ideal_c/ideal_c0, linestyle='dashed', color='red')

plt.legend(loc='lower left')
plt.title(f'Adimensional Comparison between Ideal Gas and NICFD [$Z_0 ={Z0:.2f}$]')
plt.xlabel('Mach')
plt.grid(True)
# plt.savefig(os.path.join(folder_path, 'adimensional.pdf'), dpi=300, bbox_inches='tight')
plt.show()


# Speed of sound comparison
# plt.figure(figsize=(16, 9))  # Adjust the size to fit your screen resolution
plt.figure()
plt.plot(nicfd_x*1000, nicfd_speed, color = 'purple', label='Fluid velocity')
plt.plot(nicfd_x*1000, nicfd_c, color = 'blue', label='Speed of Sound')
plt.plot(ideal_x*1000, ideal_speed, linestyle='dashed', color = 'purple')
plt.plot(ideal_x*1000, ideal_c, linestyle='dashed', color = 'blue')
plt.legend()
plt.grid(True)
plt.ylabel('Speed [m/s]')
plt.xlabel('Nozzle length [mm]')
plt.tight_layout()
plt.show()


# comparison of ratio of specific heats
plt.figure()
plt.plot(nicfd_x*1000, nicfd_gamma, color='blue', label='Ratio of specific heats NICFD')
plt.plot(ideal_x*1000, ideal_gamma, linestyle='dashed', color='blue', label='Ratio of specific heats Ideal Gas')
# plt.legend()
plt.grid(True)
plt.ylabel(r'$\gamma$')
plt.xlabel('Nozzle length [mm]')
plt.title(f'"Ratio of Specific Heats" [$Z_0 ={Z0:.2f}$]')
plt.tight_layout()
plt.show()
