import os
import numpy as np
from general_solver import isentropic_relations_non_ideal

'''
This file uses "general_solver.py" to calculate the quasi-1D solution for a converging diverging nozzle operating with non-ideal vapor.

'''

# ---  FLUID CHOICE ---
fluid = "REFPROP::MM"

# ---  EXPERIMENTS SETTINGS ---
# Using a list of dictionaries for easier iteration and readability
experiments = [
    {'exp_name': 'BOS_12_5_1', 'T0': 222,   'p0': 7.9},
    {'exp_name': 'BOS_12_5_2', 'T0': 220.6, 'p0': 11.1},
    {'exp_name': 'BOS_12_5_3', 'T0': 252,   'p0': 15.1},
    {'exp_name': 'BOS_12_5_4', 'T0': 252.2, 'p0': 11.1},
    {'exp_name': 'BOS_12_5_5', 'T0': 250.7, 'p0': 8},
    {'exp_name': 'BOS_12_11_1','T0': 221.8, 'p0': 8.1},
    {'exp_name': 'BOS_12_11_2','T0': 219.8, 'p0': 11.04},
    {'exp_name': 'BOS_12_11_3','T0': 248.5, 'p0': 17.5},
    {'exp_name': 'BOS_12_11_4','T0': 248.5, 'p0': 18.95},
    {'exp_name': 'BOS_12_11_5','T0': 254, 'p0': 14.9},
    {'exp_name': 'BOS_12_11_6','T0': 252.9, 'p0': 11.06},
    {'exp_name': 'BOS_12_11_7','T0': 251, 'p0': 9.08},
    {'exp_name': 'BOS_12_12_1','T0': 220, 'p0': 2},
    {'exp_name': 'BOS_12_12_2','T0': 220, 'p0': 4},
    {'exp_name': 'BOS_12_12_3','T0': 250, 'p0': 2},
    {'exp_name': 'BOS_12_12_4','T0': 250, 'p0': 4},
]

# ---  ITERATION LOOP ---
for exp in experiments:
    # Extract values
    name = exp['exp_name']
    T0_in = exp['T0']
    p0_in = exp['p0'] + 1.01325 # Assuming p0 is barg and you need bar (absolute)

    print(f"Processing Experiment: {name} | T0: {T0_in} C | p0: {p0_in} bar")

    # Call your function
    # Note: Ensure your function handles the T(C)->T(K) and p(bar)->p(Pa) conversions
    x, area, dens0, P_arr, T_arr, Rho_arr, M_arr, U_arr, c_arr, gamma_pv_arr = isentropic_relations_non_ideal(T0_in, p0_in, fluid)

    # Setup folder
    folder_path = f'BOS_DEC_CORR_2025/{name}'
    os.makedirs(folder_path, exist_ok=True)

    # ---  SAVING VARIABLES ---
    # Dictionary mapping filenames to variables for a cleaner save loop
    data_to_save = {
        "x": x,
        "area": area,
        "P_arr": P_arr,
        "T_arr": T_arr,
        "Rho_arr": Rho_arr,
        "M_arr": M_arr,
        "U_arr": U_arr,
        "c_arr": c_arr,
        "gamma_pv_arr": gamma_pv_arr,
        "dens0": np.array([dens0]) # Save scalar as array to keep npy format consistent
    }

    for filename, data in data_to_save.items():
        save_file = os.path.join(folder_path, f"{filename}.npy")
        np.save(save_file, data)

    print(f"Successfully saved all variables to {folder_path}")

print("\nAll experiments processed.")