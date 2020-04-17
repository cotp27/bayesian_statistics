# -*- coding: utf-8 -*-
# Libreria
import seaborn as sns
import pandas as pd
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Funcion para estima MCO
def mco_func(X1, Y1):
    # Se estima el Beta
    invXX = inv(np.dot(X1.T, X1))
    XTY = np.dot(X1.T, Y1)
    Beta = np.dot(invXX, XTY) 
    return Beta

# Figura 8.1
# Datos
dict_dat = {}
dict_dat['edad'] = [23, 22, 22, 25, 27, 20, 31, 23, 27, 28, 22, 24]

dict_dat['capac_pulm'] = [-0.87, -10.74, -3.27, -1.97, 7.50, 
              -7.25, 17.05, 4.96, 10.40, 11.05, 0.26,2.51]

dict_dat['tip_ejer'] = ['running']*6 + ['aerobic']*6

oxi_df = pd.DataFrame(dict_dat)

# Matriz de informacion
x1 = np.array([1]* len(dict_dat['edad'])) 
x2 = np.array([0 if i=='running' else 1 for i in dict_dat['tip_ejer']])
x3 = np.array(dict_dat['edad'])
x4 = np.multiply(x2,x3)

# Forma matricial
X = np.array([x1, x2, x3, x4]).T
Y = dict_dat['capac_pulm']


# Se construyen los graficos
ed_min = min(x3)
ed_max = max(x3)
domi  = np.linspace(ed_min, ed_max, 10)

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
#----------------------------------------------------------------------------#
# Beta3=0; Beta4=0
model1 = [True, True, False, False]
Beta1 = mco_func(X[:,model1], Y)
y_1_1 = np.multiply((Beta1[0]*1 + Beta1[1] * 1), ([1]*len(domi)))
y_1_2 = np.multiply((Beta1[0]*1), ([1]*len(domi)))
sns.scatterplot(x="edad", y="capac_pulm", hue="tip_ejer", data=oxi_df, ax=axs[0][0])
axs[0,0].plot(domi, y_1_1)
axs[0,0].plot(domi, y_1_2)
axs[0][0].legend_.remove()

#----------------------------------------------------------------------------#
# Beta2=0; Beta4=0
model2 = [True, False, True, False]
Beta2 = mco_func(X[:,model2], Y)
y_2_1 = np.add(np.multiply((Beta2[0]*1), ([1]*len(domi))), np.multiply(Beta2[1], domi))
y_2_2 = np.add(np.multiply((Beta2[0]*1), ([1]*len(domi))), np.multiply(Beta2[1], domi)) 
sns.scatterplot(x="edad", y="capac_pulm", hue="tip_ejer", data=oxi_df, ax=axs[0][1])
axs[0,1].plot(domi, y_2_1)
axs[0,1].plot(domi, y_2_2)
axs[0][1].legend_.remove()

#----------------------------------------------------------------------------#
# Beta4=0
model3 = [True, True, True, False]
Beta3 = mco_func(X[:,model3], Y)
y_3_1 = np.add(np.multiply((Beta3[0]*1 + Beta3[1]*1), ([1]*len(domi))), np.multiply(Beta3[2], domi))
y_3_2 = np.add(np.multiply((Beta3[0]*1), ([1]*len(domi))), np.multiply(Beta3[2], domi)) 
sns.scatterplot(x="edad", y="capac_pulm", hue="tip_ejer", data=oxi_df, ax=axs[1][0])
axs[1,0].plot(domi, y_3_1)
axs[1,0].plot(domi, y_3_2)
axs[1][0].legend_.remove()

#----------------------------------------------------------------------------#
# Todas las variables
model4 = [True, True, True, True]
Beta4 = mco_func(X[:,model4], Y)
y_4_1 = np.add(np.add(np.multiply((Beta4[0]*1 + Beta4[1]*1), ([1]*len(domi))), np.multiply(Beta4[2], domi)) , np.multiply(Beta4[3], domi))
y_4_2 = np.add(np.multiply((Beta4[0]*1), ([1]*len(domi))), np.multiply(Beta4[2], domi)) 
sns.scatterplot(x="edad", y="capac_pulm", hue="tip_ejer", data=oxi_df, ax=axs[1][1])
axs[1,1].plot(domi, y_4_1)
axs[1,1].plot(domi, y_4_2)
axs[1][1].legend_.remove()

