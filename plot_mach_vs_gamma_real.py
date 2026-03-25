import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter

root_folder = 'BOS_DEC_CORR_2025'

sub_folders = {'BOS_12_5_1',
               'BOS_12_5_2',
               'BOS_12_5_3',
               'BOS_12_5_4',
               'BOS_12_5_5',
               'BOS_12_11_1',
               'BOS_12_11_2',
               'BOS_12_11_3',
            #    'BOS_12_11_4',
               'BOS_12_11_5',
               'BOS_12_11_6',
               'BOS_12_11_7',
               'BOS_12_12_1',
               'BOS_12_12_2',
               'BOS_12_12_3',
               'BOS_12_12_4',
               }

fluid = {'MM'}

plt.figure()

for name in sorted(sub_folders):

    mach = np.load(os.path.join(root_folder,name,'M_arr.npy'))
    gamma_pv = np.load(os.path.join(root_folder,name,'Gamma_pv_arr.npy'))
    rho = np.load(os.path.join(root_folder,name,'Rho_arr.npy'))
    u = np.load(os.path.join(root_folder,name,'U_arr.npy'))
    area = np.load(os.path.join(root_folder,name,'area.npy'))
    x = np.load(os.path.join(root_folder,name,'x.npy'))
    c = np.load(os.path.join(root_folder,name,'c_arr.npy'))
    T =  np.load(os.path.join(root_folder,name,'T_arr.npy'))
    p =  np.load(os.path.join(root_folder,name,'P_arr.npy'))
    dens0 = np.load(os.path.join(root_folder,name,'dens0.npy'))

    mass_flow_check = rho * u * area
    A_star = np.min(area)


    rho_grad_noise = np.gradient(rho,x)
    rho_grad = savgol_filter(rho_grad_noise,30,3)

    p_grad_nois = np.gradient(p,x)
    p_grad = savgol_filter(p_grad_nois,30,3)

    T_grad_noise = np.gradient(T,x)
    T_grad = savgol_filter(T_grad_noise,30,3)

    drho_dg = np.gradient(rho,gamma_pv)

    plt.plot(x,rho_grad/dens0) #,label=[r'$\gamma_{pv0}$ ='f'{gamma_pv[0]}'])
    # plt.plot(x,rho/dens0)#,label=[r'$\gamma_{pv0}$ ='f'{gamma_pv[0]}'])
    plt.plot(x,p_grad/p[0],linestyle = '--')
    plt.plot(x,T_grad/T[0],linestyle='-.')
    
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.plot(x*1000, mass_flow_check, label='Calculated m_dot')
    # # plt.axhline(mf, color='r', linestyle='--', label='Target m_dot')
    # plt.ylabel('Mass Flow [kg/s]')
    # plt.title('Verification of Mass Conservation')
    # plt.legend()

    # # --- 1. PREPARE ADIMENSIONAL VARIABLES ---
    # # Ensure these are arrays derived from the solver results
    # p_ratio = p / p[0]
    # T_ratio = T / T[0]
    # rho_ratio = rho / dens0
    # c_ratio = c / c[0]
    # area_ratio = A_star / area

    # # --- 2. GENERATE THE PLOT ---
    # plt.subplot(1,2,2)

    # # Plotting against Mach number (M_arr)
    # plt.plot(mach, area_ratio, label='$A^*/A$', linewidth=2)
    # plt.plot(mach, p_ratio, label='$p/p_0$', linestyle='--')
    # plt.plot(mach, T_ratio, label='$T/T_0$')
    # plt.plot(mach, rho_ratio, label='$\\rho/\\rho_0$', linestyle='-.')
    # plt.plot(mach, c_ratio, label='$c/c_0$', alpha=0.7)

    # # Formatting
    # plt.title(f'Isentropic Flow Properties for {fluid}', fontsize=14)
    # plt.xlabel('Mach Number [-]', fontsize=12)
    # plt.ylabel('Property Ratio [-]', fontsize=12)
    # plt.grid(True, which='both', linestyle=':', alpha=0.5)
    # plt.legend(frameon=True, loc='best')
    # plt.show()
    
# plt.ylabel(r'$\rho / \rho_0$')
plt.xlabel('x')
# plt.legend()
from matplotlib.lines import Line2D
custom_lines = [
    Line2D([0], [0], color='black', linestyle='-'),
    Line2D([0], [0], color='black', linestyle='--'),
    Line2D([0], [0], color='black', linestyle='-.')
]

plt.legend(custom_lines, ['Density', 'Pressure', 'Temperature'])
plt.show()