# -*- coding: utf-8 -*-
# Libreria
import seaborn as sns
import pandas as pd
import numpy as np

# Figura 8.1
# Datos
dict_dat = {}
dict_dat['edad'] = [23, 22, 22, 25, 27, 20, 31, 23, 27, 28, 22, 24]

dict_dat['capac_pulm'] = [-0.87, -10.74, -3.27, -1.97, 7.50, 
              -7.25, 17.05, 4.96, 10.40, 11.05, 0.26,2.51]

dict_dat['tip_ejer'] = ['running']*6 + ['aerobic']*6

# Se crea el grafico
oxi_df = pd.DataFrame(dict_dat)
ax = sns.scatterplot(x="edad", y="capac_pulm", hue="tip_ejer", data=oxi_df)

