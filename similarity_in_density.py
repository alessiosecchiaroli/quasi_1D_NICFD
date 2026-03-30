import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import CoolProp.CoolProp as cp

# load the dataframe
folder = 'BOS_DEC_CORR_2025'
df = pd.read_csv(os.path.join(folder,'compiled_data.csv'))


# fluid informations
fluid = 'REFPROP::MM'
MolarMass = cp.PropsSI('M',fluid)
R_general = cp.PropsSI('GAS_CONSTANT',fluid)

R = R_general/MolarMass

print(R)

# print(df.head())

df['Compressibility'] = df['Pressure'] / df['Density'] / df['Temperature'] / R
df['Big Gamma'] = df.apply(
    lambda row: cp.PropsSI(
        'FUNDAMENTAL_DERIVATIVE_OF_GAS_DYNAMICS',
        'T', row['Temperature'],
        'P', row['Pressure'],   # NOTE: uppercase 'P'
        fluid
    ),
    axis=1
)

# print(df.head(2))

df['Astar_over_A'] = df.groupby('Folder')['area'].transform('min')/df['area']

df['Rho_over_rhoT'] = df['Density']/df.groupby('Folder')['Density'].transform('first')

df['Z0'] = df.groupby('Folder')['Compressibility'].transform('first')

print(df.head(2))


plt.figure()
plt.subplot(3,1,1)
sns.lineplot(data=df, x='Rho_over_rhoT', y='Astar_over_A', hue = 'Folder')
plt.ylabel(r'$A^*/A$')
plt.xlabel('')
plt.legend().remove()

plt.subplot(3,1,2)
sns.lineplot(data=df, x='Rho_over_rhoT', y='Compressibility',hue = 'Folder')
plt.ylabel('$Z$')
plt.xlabel('')
plt.legend().remove()

plt.subplot(3,1,3)
sns.lineplot(data=df, x='Rho_over_rhoT', y='Big Gamma',hue = 'Folder')
plt.ylabel(r'$\Gamma$')
plt.xlabel(r'$\rho / \rho_0$')
plt.legend().remove()

plt.show()