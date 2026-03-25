import numpy as np
import CoolProp.CoolProp as cp
import os
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from Geometry_define import area_definer # Assuming this remains the same


def isentropic_relations_non_ideal(T0_C,p0_bar,fluid):

    T0 = T0_C + 273.15
    p0 = p0_bar * 1e5

    x_old, area_old, _ = area_definer(1)
    x = np.linspace(np.min(x_old), np.max(x_old), 500)
    area = np.interp(x, x_old, area_old)
    A_star_idx = np.argmin(area)
    A_star = area[A_star_idx]

    # --- STAGNATION & CRITICAL PROPERTIES ---
    s0 = cp.PropsSI('S', 'T', T0, 'P', p0, fluid)
    h0 = cp.PropsSI('H', 'T', T0, 'P', p0, fluid)
    dens0 = cp.PropsSI('D', 'T', T0, 'P', p0, fluid)

    # Calculate Mass Flow (mf) at the throat (Choked Flow)
    # We find the state where Mach = 1 at the minimum area
    def find_choked_state(p_guess):
        rho = cp.PropsSI('D', 'S', s0, 'P', p_guess, fluid)
        h = cp.PropsSI('H', 'S', s0, 'P', p_guess, fluid)
        u = np.sqrt(2 * max(0, h0 - h))
        c = cp.PropsSI('A', 'S', s0, 'P', p_guess, fluid)
        return u - c # Zero when u = c (Mach 1)

    p_star = fsolve(find_choked_state, p0 * 0.6)[0]
    rho_star = cp.PropsSI('D', 'S', s0, 'P', p_star, fluid)
    u_star = cp.PropsSI('A', 'S', s0, 'P', p_star, fluid)
    mf = rho_star * u_star * A_star # The constant mass flow rate [kg/s]

    # --- SOLVER FUNCTION ---
    def nozzle_equations(p_guess, current_area):
        # Under isentropic assumption, P defines the whole state
        try:
            rho = cp.PropsSI('D', 'S', s0, 'P', p_guess, fluid)
            h = cp.PropsSI('H', 'S', s0, 'P', p_guess, fluid)
            u = np.sqrt(2 * max(0, h0 - h))
            # Mass balance: rho * u * A - mf = 0
            return rho * u * current_area - mf
        except:
            return 1e10 # Penalty for invalid CoolProp states

    # --- DATA INITIALIZATION ---
    P_arr = np.zeros_like(x)
    Rho_arr = np.zeros_like(x)
    U_arr = np.zeros_like(x)
    M_arr = np.zeros_like(x)
    T_arr = np.zeros_like(x)
    c_arr = np.zeros_like(x)
    gamma_pv_arr = np.zeros_like(x)

    # --- MAIN LOOP ---
    p_guess = p0 * 0.99
    for i in range(len(x)):
        # Switch guess strategy after the throat to find supersonic solution
        if i > A_star_idx:
            if i == A_star_idx + 1: p_guess = p_star * 0.9 # Kickstart supersonic branch
    
        sol = fsolve(nozzle_equations, p_guess, args=(area[i]))
        P_arr[i] = sol[0]
    
        # Update properties for next step
        Rho_arr[i] = cp.PropsSI('D', 'S', s0, 'P', P_arr[i], fluid)
        h_curr = cp.PropsSI('H', 'S', s0, 'P', P_arr[i], fluid)
        U_arr[i] = np.sqrt(2 * max(0, h0 - h_curr))
        c_curr = cp.PropsSI('A', 'S', s0, 'P', P_arr[i], fluid)
        M_arr[i] = U_arr[i] / c_curr
        c_arr[i] = c_curr
        T_arr[i] = cp.PropsSI('T', 'S', s0, 'P', P_arr[i], fluid)
        
        gamma_id = cp.PropsSI('CPMASS', 'S', s0, 'P', P_arr[i], fluid)/cp.PropsSI('CVMASS', 'S', s0, 'P', P_arr[i], fluid)
        gamma_pv_arr[i] = gamma_id/ P_arr[i] / cp.PropsSI('ISOTHERMAL_COMPRESSIBILITY', 'S', s0, 'P', P_arr[i], fluid)

        p_guess = P_arr[i] # Continuity in guessing

    # --- VERIFICATION & PLOTTING ---
    mass_flow_check = Rho_arr * U_arr * area

    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.plot(x*1000, mass_flow_check, label='Calculated m_dot')
    # plt.axhline(mf, color='r', linestyle='--', label='Target m_dot')
    # plt.ylabel('Mass Flow [kg/s]')
    # plt.title('Verification of Mass Conservation')
    # plt.legend()

    # # --- 1. PREPARE ADIMENSIONAL VARIABLES ---
    # # Ensure these are arrays derived from the solver results
    # p_ratio = P_arr / p0
    # T_arr = np.array([cp.PropsSI('T', 'S', s0, 'P', p_val, fluid) for p_val in P_arr])
    # T_ratio = T_arr / T0
    # rho_ratio = Rho_arr / dens0
    # c_arr = np.array([cp.PropsSI('A', 'S', s0, 'P', p_val, fluid) for p_val in P_arr])
    # c_ratio = c_arr / cp.PropsSI('A', 'T', T0, 'P', p0, fluid)
    # area_ratio = A_star / area

    # # --- 2. GENERATE THE PLOT ---
    # plt.subplot(1,2,2)

    # # Plotting against Mach number (M_arr)
    # plt.plot(M_arr, area_ratio, label='$A^*/A$', linewidth=2)
    # plt.plot(M_arr, p_ratio, label='$p/p_0$', linestyle='--')
    # plt.plot(M_arr, T_ratio, label='$T/T_0$')
    # plt.plot(M_arr, rho_ratio, label='$\\rho/\\rho_0$', linestyle='-.')
    # plt.plot(M_arr, c_ratio, label='$c/c_0$', alpha=0.7)

    # # Formatting
    # plt.title(f'Isentropic Flow Properties for {fluid}', fontsize=14)
    # plt.xlabel('Mach Number [-]', fontsize=12)
    # plt.ylabel('Property Ratio [-]', fontsize=12)
    # plt.grid(True, which='both', linestyle=':', alpha=0.5)
    # plt.legend(frameon=True, loc='best')
    # plt.show()

    return x, area, dens0, P_arr, T_arr, Rho_arr, M_arr, U_arr, c_arr, gamma_pv_arr