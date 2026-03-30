import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter

'''
This file use pandas and seaborn to perform data analysis of the results obtained from "Main_v1.py"
'''


# Load the dataset
cwd = os.getcwd()

root_folder = os.path.join(cwd, 'BOS_DEC_CORR_2025')


# automated way to load all the folders and files in the root folder
# folders = {}

# for i in range(1, 5):
#     folder_name = f'BOS_12_5_{i}'
#     folder_path = os.path.join(root_folder, folder_name)
#     folders[folder_name] = folder_path

# for i in range(1, 7):
#     folder_name = f'BOS_12_11_{i}'
#     folder_path = os.path.join(root_folder, folder_name)
#     folders[folder_name] = folder_path

# hard-coded

folder = { 
         'BOS_12_5_1': os.path.join(root_folder, 'BOS_12_5_1'),
         'BOS_12_5_2': os.path.join(root_folder, 'BOS_12_5_2'),
         'BOS_12_5_3': os.path.join(root_folder, 'BOS_12_5_3'),
         'BOS_12_5_4': os.path.join(root_folder, 'BOS_12_5_4'),
         'BOS_12_5_5': os.path.join(root_folder, 'BOS_12_5_5'),
         'BOS_12_11_1': os.path.join(root_folder, 'BOS_12_11_1'),     
         'BOS_12_11_2': os.path.join(root_folder, 'BOS_12_11_2'),         
         'BOS_12_11_3': os.path.join(root_folder, 'BOS_12_11_3'),             
         'BOS_12_11_5': os.path.join(root_folder, 'BOS_12_11_5'),
         'BOS_12_11_6': os.path.join(root_folder, 'BOS_12_11_6'),
         'BOS_12_11_7': os.path.join(root_folder, 'BOS_12_11_7'),
         'BOS_12_12_1': os.path.join(root_folder, 'BOS_12_12_1'),
         'BOS_12_12_2': os.path.join(root_folder, 'BOS_12_12_2'),
         'BOS_12_12_3': os.path.join(root_folder, 'BOS_12_12_3'),
         'BOS_12_12_4': os.path.join(root_folder, 'BOS_12_12_4')
         }

# Load the files for each folder
data = {}
for name, path in folder.items():
    data[name] = {}
    try:
        data[name]['x_coordinates'] = np.load(os.path.join(path, 'x.npy'))
        data[name]['Gamma_pv'] = np.load(os.path.join(path, 'Gamma_pv_arr.npy'))
        data[name]['Density'] = np.load(os.path.join(path, 'Rho_arr.npy'))
        data[name]['Temperature'] = np.load(os.path.join(path, 'T_arr.npy'))
        data[name]['Pressure'] = np.load(os.path.join(path, 'P_arr.npy'))
        data[name]['Velocity'] = np.load(os.path.join(path, 'U_arr.npy'))
        data[name]['Mach'] = np.load(os.path.join(path, 'M_arr.npy'))
        data[name]['c'] = np.load(os.path.join(path, 'c_arr.npy'))
        data[name]['dens0'] = np.load(os.path.join(path, 'dens0.npy'))
        data[name]['area'] = np.load(os.path.join(path,'area.npy'))

        # data[name]['x_coordinates'] = np.load(os.path.join(path, 'x_coordinates.npy'))
        # data[name]['Gamma_pv'] = np.load(os.path.join(path, 'Gamma_pv.npy'))
        # data[name]['Density'] = np.load(os.path.join(path, 'Density.npy'))
        # data[name]['Temperature'] = np.load(os.path.join(path, 'Temperature.npy'))
        # data[name]['Pressure'] = np.load(os.path.join(path, 'Pressure.npy'))
        # data[name]['Velocity'] = np.load(os.path.join(path, 'Speed.npy'))
        # data[name]['Mach'] = np.load(os.path.join(path, 'Mach.npy'))
        # data[name]['c'] = np.load(os.path.join(path, 'c.npy'))
        # data[name]['dens0'] = np.load(os.path.join(path, 'dens0.npy'))
        # data[name]['area'] = np.load(os.path.join(path,'Area.npy'))

        print(f"Loaded files for {name}")
    except FileNotFoundError as e:
        print(f"File not found in {name}: {e}")
        data[name] = None


# Compile data into a long format DataFrame
rows = []
for name in data.keys():
    if data[name] is not None:
        x_coords = data[name]['x_coordinates']
        gamma_pv = data[name]['Gamma_pv']
        dens0 = data[name]['dens0']
        density = data[name]['Density']
        temperature = data[name]['Temperature']
        pressure = data[name]['Pressure']
        Mach = data[name]['Mach']
        c = data[name]['c']
        velocity = data[name]['Velocity']
        area = data[name]['area']
        # Assume x_coords, gamma_pv, density are arrays of same length, dens0 is scalar
        dens0_val = dens0.item() if dens0.ndim == 0 else dens0[0]


        for i in range(len(x_coords)):
            rows.append({
                'Folder': name,
                'x_coordinates': x_coords[i],
                'Gamma_pv': gamma_pv[i],
                'Density': density[i],
                'Temperature': temperature[i],
                'Pressure': pressure[i],
                'Mach': Mach[i],
                'Speed': velocity[i],
                'c': c[i],
                'dens0': dens0_val,
                'area' : area[i]
            })

csv_version = pd.DataFrame(rows)

csv_version['norm_density'] = csv_version['Density'] / csv_version.groupby('Folder')['Density'].transform('first')
csv_version['norm_temperature'] = csv_version['Temperature'] / csv_version.groupby('Folder')['Temperature'].transform('first')
csv_version['norm_pressure'] = csv_version['Pressure'] / csv_version.groupby('Folder')['Pressure'].transform('first')

csv_version['norm_dens_grad'] = (csv_version.groupby('Folder')['Density'].diff() / csv_version.groupby('Folder')['x_coordinates'].diff())/csv_version.groupby('Folder')['Density'].transform('first')  # Calculate density gradient for each folder
csv_version['norm_p_grad'] = (csv_version.groupby('Folder')['Pressure'].diff() / csv_version.groupby('Folder')['x_coordinates'].diff())/csv_version.groupby('Folder')['Pressure'].transform('first')  # Calculate pressure gradient for each folder
csv_version['norm_T_grad'] = (csv_version.groupby('Folder')['Temperature'].diff() / csv_version.groupby('Folder')['x_coordinates'].diff())/csv_version.groupby('Folder')['Temperature'].transform('first')  # Calculate temperature gradient for each folder

csv_version['beta'] = csv_version.groupby('Folder')['Pressure'].transform('first')/csv_version.groupby('Folder')['Pressure'].transform('last') 
csv_version['alfa'] = csv_version.groupby('Folder')['Density'].transform('first')/csv_version.groupby('Folder')['Density'].transform('last')

csv_version['mean_gamma_pv'] = csv_version.groupby('Folder')['Gamma_pv'].transform('mean')


for name in csv_version['Folder'].unique():

    print('Experiment:', name)
    print('Beta (P0/Pf):', csv_version[csv_version['Folder'] == name]['beta'].iloc[0])
    print('Alfa (rho0/rhof):', csv_version[csv_version['Folder'] == name]['alfa'].iloc[0])


# save the compiled data to a csv file
csv_version.to_csv(os.path.join(root_folder, 'compiled_data.csv'), index=False)

# plt.figure()
# plt.subplot(1,2,1)
# sns.lineplot(data=csv_version, x='x_coordinates',y='norm_pressure',errorbar=(lambda x: (x.min(), x.max())))
# sns.lineplot(data=csv_version, x='x_coordinates',y='norm_density',errorbar=(lambda x: (x.min(), x.max())))
# sns.lineplot(data=csv_version, x='x_coordinates',y='norm_temperature',errorbar=(lambda x: (x.min(), x.max())))
# plt.show()
## --- PLOT THE NORMALIZED TRENDS OF p,T, rho
# plt.figure()

# # plt.subplot(2,1,1)
# sns.lineplot(data=csv_version, x='x_coordinates', y='norm_density', hue='mean_gamma_pv',linestyle='-')
# sns.lineplot(data=csv_version, x='x_coordinates', y='norm_pressure', hue='mean_gamma_pv', linestyle='--')
# sns.lineplot(data=csv_version, x='x_coordinates', y='norm_temperature', hue='mean_gamma_pv', linestyle='-.')
# plt.title('Normalized variables vs x_coordinates')
# plt.title('Normalized Gamma_pv vs x_coordinates for Different Folders')
# plt.xlabel('x_coordinates [mm]')
# # Remove the automatic legend
# plt.legend().remove()
# plt.ylabel('Normalized variables')
# plt.legend()


# custom_lines = [
#     Line2D([0], [0], color='black', linestyle='-'),
#     Line2D([0], [0], color='black', linestyle='--'),
#     Line2D([0], [0], color='black', linestyle='-.')
# ]

# plt.legend(custom_lines, ['Density', 'Pressure', 'Temperature'])

# plt.subplot(2,1,2)
plt.figure()
# sns.lineplot(data=csv_version, x='x_coordinates', y=savgol_filter(csv_version['norm_dens_grad'],30,3), hue='mean_gamma_pv',linestyle='-')
# sns.lineplot(data=csv_version, x='x_coordinates', y=savgol_filter(csv_version['norm_p_grad'],30,3), hue='mean_gamma_pv', linestyle='--')
# sns.lineplot(data=csv_version, x='x_coordinates', y='norm_T_grad', hue='mean_gamma_pv', linestyle='-.')

sns.lineplot(data=csv_version, x='x_coordinates', y=savgol_filter(csv_version['norm_dens_grad'],30,3),errorbar=(lambda x: (x.min(), x.max())),linestyle='-',label=r'$\rho$')
sns.lineplot(data=csv_version, x='x_coordinates', y=savgol_filter(csv_version['norm_p_grad'],30,3),errorbar=(lambda x: (x.min(), x.max())), linestyle='--',label='p')
sns.lineplot(data=csv_version, x='x_coordinates', y='norm_T_grad',errorbar=(lambda x: (x.min(), x.max())),linestyle='-.',label='T')

plt.title('Normalized gradients vs x_coordinates')
# plt.title('Normalized Gamma_pv vs x_coordinates for Different Folders')
plt.xlabel('x_coordinates [mm]')
# # Remove the automatic legend
# plt.legend().remove()
plt.ylabel('Normalized gradients')
plt.legend()


plt.show()


csv_version['mean_gamma_pv'] = csv_version.groupby('Folder')['Gamma_pv'].transform('mean')

# csv_version['mean_p'] = csv_version.groupby('x')['Pressure'].transform('mean')
# csv_version['std_p'] = csv_version.groupby('x')['Pressure'].transform('std')

# csv_version['mean_t'] = csv_version.groupby('x')['Temperature'].transform('mean')
# csv_version['std_t'] = csv_version.groupby('x')['Temperature'].transform('std')

# csv_version['mean_r'] = csv_version.groupby('x')['Density'].transform('mean')
# csv_version['std_r'] = csv_version.groupby('x')['Density'].transform('std')




## ---- PLOT gamma-pv and Mach number evolution along the nozzle, and average gamma-pv

# plt.figure(figsize=(15, 12))
# plt.subplot(3,1,1)
# sns.lineplot(data=csv_version, x='x_coordinates', y='Gamma_pv', hue='Folder')
# plt.ylabel(r'$\gamma_{pv}$')
# # plt.show()
# plt.legend().remove()
# plt.subplot(3,1,2)
# sns.lineplot(data=csv_version, x='x_coordinates', y='Mach', hue='Folder', palette='tab10', linestyle='--')
# plt.ylabel(r'Mach')
# plt.xlabel('x_coordinates [mm]')
# plt.legend().remove()

# plt.subplot(3,1,3)
# sns.barplot(data=csv_version, x='Folder', y='mean_gamma_pv')
# plt.xlabel('Experiment')
# plt.ylabel(r'Mean $\gamma_{pv}$') 
# plt.ylim(0.7,1)

# plt.show()

# # --- PLOT THE NORMALIZED GRADIENS T,p,rho

# plt.figure()
# plt.subplot(1,2,1)
# sns.lineplot(data=csv_version, x='x_coordinates',y='norm_pressure')
# sns.lineplot(data=csv_version, x='x_coordinates',y='norm_density')
# sns.lineplot(data=csv_version, x='x_coordinates',y='norm_temperature')
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.subplot(1,2,2)
# sns.lineplot(data=csv_version, x='x_coordinates', y='norm_dens_grad',
#              hue='Folder', linestyle='-')

# sns.lineplot(data=csv_version, x='x_coordinates', y='norm_p_grad',
#              hue='Folder', linestyle='--')

# sns.lineplot(data=csv_version, x='x_coordinates', y='norm_T_grad',
#              hue='Folder', linestyle='-.')

# plt.title(r'Normalized variables (T, p, $\rho$ )')
# plt.xlabel('x_coordinates [mm]')
# # plt.tight_layout()

# # Remove the automatic legend
# plt.legend().remove()
# plt.ylabel('Normalized Gradients')

# # Add your custom legend
# # from matplotlib.lines import Line2D

# custom_lines = [
#     Line2D([0], [0], color='black', linestyle='-'),
#     Line2D([0], [0], color='black', linestyle='--'),
#     Line2D([0], [0], color='black', linestyle='-.')
# ]

# plt.legend(custom_lines, ['Density', 'Pressure', 'Temperature'])

# plt.show()
