import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.signal import savgol_filter
from matplotlib.lines import Line2D


'''
This script is used to analyze where the deviation between Main_gammapv and MAIN_v1 start to occur.
'''

wrong_df = pd.read_csv('BOS_December_2025/compiled_data.csv')
good_df = pd.read_csv('BOS_DEC_CORR_2025/compiled_data.csv')

print(wrong_df.head())
print(good_df.head())


folder_filter = ['BOS_12_5_1', 'BOS_12_5_3', 'BOS_12_5_4', 'BOS_12_5_5', 'BOS_12_11_1', 'BOS_12_11_5', 'BOS_12_11_6', 'BOS_12_11_7','BOS_12_12_1', 'BOS_12_12_2', 'BOS_12_12_3', 'BOS_12_12_4']

plt.figure(figsize=(12, 6))

# shapes = len(wrong_df['Folder'].unique())
# print(shapes)


for folder in sorted(folder_filter):
    # plt.figure(figsize=(12, 6))
    sns.lineplot(data=wrong_df[wrong_df['Folder'] == folder], x='x_coordinates', y='Speed', label='wrong', color='red',alpha=1)
    sns.lineplot(data=good_df[good_df['Folder'] == folder], x='x_coordinates', y='Speed', label='correct', color='blue',alpha=0.5)
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # print(folder)

    # plt.subplot(3,1,1)
    # sns.lineplot(data=wrong_df[wrong_df['Folder'] == folder], x='x_coordinates', y=savgol_filter(wrong_df[wrong_df['Folder'] == folder]['norm_dens_grad'], 51, 3), label='wrong', color='red')
    # sns.lineplot(data=good_df[good_df['Folder'] == folder], x='x_coordinates', y=savgol_filter(good_df[good_df['Folder'] == folder]['norm_dens_grad'], 51, 3), label='correct', color='blue')

    # plt.subplot(3,1,2)
    # sns.lineplot(data=wrong_df[wrong_df['Folder'] == folder], x='x_coordinates', y=savgol_filter(wrong_df[wrong_df['Folder'] == folder]['norm_p_grad'], 51, 3), label='wrong', color='red')
    # sns.lineplot(data=good_df[good_df['Folder'] == folder], x='x_coordinates', y=savgol_filter(good_df[good_df['Folder'] == folder]['norm_p_grad'], 51, 3), label='correct', color='blue')

    # plt.subplot(3,1,3)
    # sns.lineplot(data=wrong_df[wrong_df['Folder'] == folder], x='x_coordinates', y='norm_T_grad', label='wrong', color='red')
    # sns.lineplot(data=good_df[good_df['Folder'] == folder], x='x_coordinates', y='norm_T_grad', label='correct', color='blue')


# plt.subplot(3,1,1)
# plt.ylabel('norm_dens_grad')
# plt.legend().remove()
# plt.grid(True)

# plt.subplot(3,1,2)
# plt.ylabel('norm_p_grad')
# plt.legend().remove()
# plt.grid(True)

# plt.subplot(3,1,3)
# plt.ylabel('norm_T_grad')
# plt.legend().remove()
plt.grid(True)

custom_lines = [
    Line2D([0], [0], color='red', linestyle='-'),
    Line2D([0], [0], color='blue', linestyle='-'),
    ]

plt.legend(custom_lines, ['Wrong', 'Correct'])
plt.ylabel('u [m/s]')
plt.show()
